#!/usr/bin/env python3
from __future__ import annotations

import math
from typing import Any

import numpy as np

from box_lab_common import (
    MODELS_DIR,
    SESSIONS_DIR,
    fill_nan_series,
    load_trial_data,
    now_text,
    prediction_confidence,
    prediction_scores,
    read_manifests,
    resample_vector,
    resolve_npz_path,
    safe_label,
    time_window_segments,
    timestamp,
    write_json,
)
from get_range_profile import read_frame
from point_cloud_viewer import PointCloud, point_cloud_from_tlvs


INPUT_TYPE = "mmwave_posture_point_cloud_2d"
STAT_NAMES = ("mean", "std", "min", "max", "p10", "p50", "p90", "start", "end", "delta", "slope")


def read_point_cloud_frame(port: Any, frame_timeout_s: float) -> tuple[int, PointCloud]:
    frame_number, tlvs = read_frame(port, frame_timeout_s)
    return frame_number, point_cloud_from_tlvs(tlvs)


def cloud_arrays(cloud: PointCloud) -> tuple[np.ndarray, np.ndarray]:
    if len(cloud.x) == 0:
        return np.empty((0, 3), dtype=float), np.empty(0, dtype=float)
    return np.column_stack((cloud.x, cloud.y, cloud.z)), np.asarray(cloud.velocity, dtype=float)


def pack_point_cloud_frames(
    xyz_frames: list[np.ndarray],
    velocity_frames: list[np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    frame_count = len(xyz_frames)
    point_count = np.array([len(frame) for frame in xyz_frames], dtype=np.uint16)
    max_points = int(np.max(point_count)) if len(point_count) else 0
    points_xyz = np.full((frame_count, max_points, 3), np.nan, dtype=float)
    points_velocity = np.full((frame_count, max_points), np.nan, dtype=float)

    for index, xyz in enumerate(xyz_frames):
        count = len(xyz)
        if count == 0:
            continue
        points_xyz[index, :count, :] = xyz
        velocity = velocity_frames[index]
        points_velocity[index, : min(count, len(velocity))] = velocity[:count]

    return point_count, points_xyz, points_velocity


def filter_posture_points(xyz: np.ndarray, feature_params: dict[str, Any]) -> np.ndarray:
    xyz = np.asarray(xyz, dtype=float)
    if xyz.ndim != 2 or xyz.shape[1] != 3 or xyz.size == 0:
        return np.empty((0, 3), dtype=float)

    finite = np.all(np.isfinite(xyz), axis=1)
    xyz = xyz[finite]
    if xyz.size == 0:
        return np.empty((0, 3), dtype=float)

    x_limit = float(feature_params.get("x_limit_m", 2.0))
    min_range = float(feature_params.get("min_range_m", 0.2))
    max_range = float(feature_params.get("max_range_m", 5.0))

    mask = (
        (np.abs(xyz[:, 0]) <= x_limit)
        & (xyz[:, 1] >= min_range)
        & (xyz[:, 1] <= max_range)
    )
    return xyz[mask]


def series_stats(values: np.ndarray, time_s: np.ndarray) -> list[float]:
    values = fill_nan_series(np.asarray(values, dtype=float))
    if values.size == 0:
        return [0.0] * len(STAT_NAMES)

    finite = values[np.isfinite(values)]
    if finite.size == 0:
        values = np.zeros_like(values, dtype=float)
        finite = values

    if time_s.size != values.size:
        time_s = np.linspace(0.0, values.size - 1, values.size)
    duration = max(float(time_s[-1] - time_s[0]), 1e-9) if values.size > 1 else 1.0
    slope = float((values[-1] - values[0]) / duration)

    return [
        float(np.mean(values)),
        float(np.std(values)),
        float(np.min(values)),
        float(np.max(values)),
        float(np.percentile(finite, 10)),
        float(np.percentile(finite, 50)),
        float(np.percentile(finite, 90)),
        float(values[0]),
        float(values[-1]),
        float(values[-1] - values[0]),
        slope,
    ]


def append_series_features(
    features: list[float],
    names: list[str],
    prefix: str,
    values: np.ndarray,
    time_s: np.ndarray,
    trajectory_points: int = 0,
) -> None:
    features.extend(series_stats(values, time_s))
    names.extend(f"{prefix}_{stat}" for stat in STAT_NAMES)
    if trajectory_points > 0:
        trajectory = resample_vector(values, trajectory_points)
        features.extend(float(value) for value in trajectory)
        names.extend(f"{prefix}_t{index:02d}" for index in range(trajectory_points))


def normalized_histogram(values: np.ndarray, bins: int, value_range: tuple[float, float]) -> np.ndarray:
    if values.size == 0:
        return np.zeros(bins, dtype=float)
    hist, _edges = np.histogram(values, bins=bins, range=value_range)
    total = float(np.sum(hist))
    return hist.astype(float) / total if total > 0 else hist.astype(float)


def normalized_histogram2d(
    x_values: np.ndarray,
    y_values: np.ndarray,
    x_bins: int,
    y_bins: int,
    x_range: tuple[float, float],
    y_range: tuple[float, float],
) -> np.ndarray:
    if x_values.size == 0 or y_values.size == 0:
        return np.zeros((x_bins, y_bins), dtype=float)
    hist, _x_edges, _y_edges = np.histogram2d(
        x_values,
        y_values,
        bins=(x_bins, y_bins),
        range=(x_range, y_range),
    )
    total = float(np.sum(hist))
    return hist.astype(float) / total if total > 0 else hist.astype(float)


def extract_posture_feature_vector(
    data: dict[str, np.ndarray],
    feature_params: dict[str, Any],
) -> tuple[list[float] | None, list[str]]:
    if "points_xyz" not in data:
        return None, []

    points_xyz = np.asarray(data["points_xyz"], dtype=float)
    if points_xyz.ndim != 3 or points_xyz.shape[2] != 3:
        return None, []

    frame_count = points_xyz.shape[0]
    if frame_count < 2:
        return None, []

    time_s = np.asarray(
        data.get("time_s", np.linspace(0.0, frame_count - 1, frame_count)),
        dtype=float,
    )
    if time_s.size != frame_count:
        time_s = np.linspace(0.0, frame_count - 1, frame_count)

    counts = np.zeros(frame_count, dtype=float)
    centroid = np.full((frame_count, 2), np.nan, dtype=float)
    spread = np.full((frame_count, 2), np.nan, dtype=float)
    minimum = np.full((frame_count, 2), np.nan, dtype=float)
    maximum = np.full((frame_count, 2), np.nan, dtype=float)
    all_points: list[np.ndarray] = []

    for frame_index in range(frame_count):
        frame_points = filter_posture_points(points_xyz[frame_index], feature_params)
        counts[frame_index] = len(frame_points)
        if len(frame_points) == 0:
            continue
        frame_xy = frame_points[:, :2]
        all_points.append(frame_xy)
        centroid[frame_index] = np.mean(frame_xy, axis=0)
        spread[frame_index] = np.std(frame_xy, axis=0)
        minimum[frame_index] = np.min(frame_xy, axis=0)
        maximum[frame_index] = np.max(frame_xy, axis=0)

    width = maximum[:, 0] - minimum[:, 0]
    depth = maximum[:, 1] - minimum[:, 1]
    active_fraction = float(np.mean(counts > 0.0))

    trajectory_points = int(feature_params.get("trajectory_points", 20))
    features: list[float] = [active_fraction]
    names: list[str] = ["active_frame_fraction"]

    append_series_features(features, names, "point_count", counts, time_s, trajectory_points)
    axes = ("x", "y")
    for axis_index, axis_name in enumerate(axes):
        append_series_features(
            features,
            names,
            f"centroid_{axis_name}",
            centroid[:, axis_index],
            time_s,
            trajectory_points,
        )
        append_series_features(
            features,
            names,
            f"spread_{axis_name}",
            spread[:, axis_index],
            time_s,
        )
        append_series_features(
            features,
            names,
            f"min_{axis_name}",
            minimum[:, axis_index],
            time_s,
        )
        append_series_features(
            features,
            names,
            f"max_{axis_name}",
            maximum[:, axis_index],
            time_s,
        )

    append_series_features(features, names, "body_width_x", width, time_s)
    append_series_features(features, names, "body_depth_y", depth, time_s)

    if all_points:
        stacked = np.vstack(all_points)
    else:
        stacked = np.empty((0, 2), dtype=float)

    x_limit = float(feature_params.get("x_limit_m", 2.0))
    min_range = float(feature_params.get("min_range_m", 0.2))
    max_range = float(feature_params.get("max_range_m", 5.0))
    x_bins = int(feature_params.get("x_bins", 8))
    y_bins = int(feature_params.get("y_bins", 12))
    xy_x_bins = int(feature_params.get("xy_x_bins", 8))
    xy_y_bins = int(feature_params.get("xy_y_bins", 12))

    hist_specs = [
        ("x_hist", stacked[:, 0] if len(stacked) else np.empty(0), x_bins, (-x_limit, x_limit)),
        ("y_hist", stacked[:, 1] if len(stacked) else np.empty(0), y_bins, (min_range, max_range)),
    ]
    for prefix, values, bins, value_range in hist_specs:
        hist = normalized_histogram(values, bins, value_range)
        features.extend(float(value) for value in hist)
        names.extend(f"{prefix}_b{index:02d}" for index in range(bins))

    xy_hist = normalized_histogram2d(
        stacked[:, 0] if len(stacked) else np.empty(0),
        stacked[:, 1] if len(stacked) else np.empty(0),
        xy_x_bins,
        xy_y_bins,
        (-x_limit, x_limit),
        (min_range, max_range),
    )
    features.extend(float(value) for value in xy_hist.ravel())
    names.extend(
        f"xy_occupancy_x{x_index:02d}_y{y_index:02d}"
        for x_index in range(xy_x_bins)
        for y_index in range(xy_y_bins)
    )

    if not all(math.isfinite(value) for value in features):
        features = [0.0 if not math.isfinite(value) else value for value in features]
    return features, names


def slice_posture_segment(
    data: dict[str, np.ndarray],
    start_index: int,
    end_index: int,
) -> dict[str, np.ndarray]:
    segment: dict[str, np.ndarray] = {
        "points_xyz": np.asarray(data["points_xyz"])[start_index:end_index],
    }
    if "point_count" in data:
        segment["point_count"] = np.asarray(data["point_count"])[start_index:end_index]
    if "points_velocity" in data:
        segment["points_velocity"] = np.asarray(data["points_velocity"])[start_index:end_index]
    if "time_s" in data:
        time_s = np.asarray(data["time_s"], dtype=float)[start_index:end_index]
        segment["time_s"] = time_s - time_s[0] if time_s.size else time_s
    return segment
