#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import math
import re
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from get_range_profile import RANGE_PROFILE_MAJOR, RANGE_PROFILE_MINOR, read_frame
from point_cloud_viewer import PointCloud, point_cloud_from_tlvs


LAB_DIR = Path(__file__).resolve().parent
MODELS_DIR = LAB_DIR / "models"
SESSIONS_DIR = LAB_DIR / "sessions"
INPUT_TYPE = "mmwave_box_contents_2d"
PRESENCE_INPUT_TYPE = "mmwave_box_presence"


def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def now_text() -> str:
    return datetime.now().isoformat(timespec="seconds")


def safe_label(value: str) -> str:
    label = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value).strip())
    return label.strip("_") or "unknown"


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def read_manifest(dataset_dir: Path) -> list[dict[str, str]]:
    manifest = dataset_dir / "trials.csv"
    if manifest.exists():
        with manifest.open(newline="") as file:
            return list(csv.DictReader(file))

    rows: list[dict[str, str]] = []
    for metadata_path in sorted(dataset_dir.rglob("trial_metadata.json")):
        try:
            metadata = json.loads(metadata_path.read_text())
        except json.JSONDecodeError:
            continue
        rows.append({key: "" if value is None else str(value) for key, value in metadata.items()})
    return rows


def source_dataset_name(dataset_dir: Path) -> str:
    metadata_path = dataset_dir / "dataset_metadata.json"
    if metadata_path.exists():
        try:
            metadata = json.loads(metadata_path.read_text())
            return str(metadata.get("dataset_name") or dataset_dir.name)
        except json.JSONDecodeError:
            pass
    return dataset_dir.name


def read_manifests(dataset_dirs: list[Path]) -> tuple[list[dict[str, str]], list[str]]:
    rows: list[dict[str, str]] = []
    missing: list[str] = []
    for dataset_dir in dataset_dirs:
        dataset_rows = read_manifest(dataset_dir)
        if not dataset_rows:
            missing.append(str(dataset_dir))
            continue
        source_name = source_dataset_name(dataset_dir)
        for row in dataset_rows:
            item = dict(row)
            item["_dataset_dir"] = str(dataset_dir)
            item["_source_dataset"] = source_name
            rows.append(item)
    return rows, missing


def resolve_session_dir(row: dict[str, str]) -> Path:
    value = row.get("session_dir", "")
    session_dir = Path(value) if value else Path("")
    if session_dir.is_absolute():
        return session_dir
    dataset_dir = Path(row.get("_dataset_dir", "."))
    return (dataset_dir / session_dir).resolve()


def resolve_npz_path(row: dict[str, str]) -> Path:
    value = row.get("npz_path", "")
    if value:
        npz_path = Path(value)
        if npz_path.is_absolute():
            return npz_path
        dataset_dir = Path(row.get("_dataset_dir", "."))
        candidate = (dataset_dir / npz_path).resolve()
        if candidate.exists():
            return candidate
    return resolve_session_dir(row) / "trial_data.npz"


def load_trial_data(npz_path: Path) -> dict[str, np.ndarray]:
    with np.load(npz_path) as npz:
        return {key: npz[key] for key in npz.files}


def db_scale(values: np.ndarray) -> np.ndarray:
    return 10.0 * np.log10(np.maximum(values, 0.0) + 1.0)


def fill_nan_series(values: np.ndarray, fallback: float = 0.0) -> np.ndarray:
    output = np.asarray(values, dtype=float).copy()
    if output.size == 0:
        return output
    finite = np.isfinite(output)
    if not finite.any():
        output[:] = fallback
        return output
    if finite.all():
        return output
    indices = np.arange(output.size)
    output[~finite] = np.interp(indices[~finite], indices[finite], output[finite])
    return output


def resample_vector(values: np.ndarray, target_count: int) -> np.ndarray:
    values = fill_nan_series(np.asarray(values, dtype=float))
    if target_count <= 0:
        raise ValueError("target_count must be positive")
    if values.size == 0:
        return np.zeros(target_count, dtype=float)
    if values.size == 1:
        return np.full(target_count, float(values[0]), dtype=float)
    source_x = np.linspace(0.0, 1.0, values.size)
    target_x = np.linspace(0.0, 1.0, target_count)
    return np.interp(target_x, source_x, values)


def resample_matrix(matrix: np.ndarray, target_rows: int, target_cols: int) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=float)
    if target_rows <= 0 or target_cols <= 0:
        raise ValueError("target_rows and target_cols must be positive")
    if matrix.ndim != 2:
        raise ValueError("matrix must be 2D")
    if matrix.shape[0] == 0 or matrix.shape[1] == 0:
        return np.zeros((target_rows, target_cols), dtype=float)

    time_resampled = np.column_stack(
        [resample_vector(matrix[:, col], target_rows) for col in range(matrix.shape[1])]
    )
    if matrix.shape[1] == 1:
        return np.repeat(time_resampled, target_cols, axis=1)

    source_x = np.linspace(0.0, 1.0, matrix.shape[1])
    target_x = np.linspace(0.0, 1.0, target_cols)
    return np.vstack([np.interp(target_x, source_x, row) for row in time_resampled])


def robust_normalize(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return np.zeros_like(values, dtype=float)
    scale = float(np.percentile(np.abs(finite), 95))
    if not math.isfinite(scale) or scale <= 1e-9:
        scale = float(np.std(finite))
    if not math.isfinite(scale) or scale <= 1e-9:
        scale = 1.0
    return np.clip(values / scale, -6.0, 6.0)


def box_presence_motion_matrix(
    data: dict[str, np.ndarray],
    feature_params: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    if "range_profile" not in data or "range_m" not in data:
        return None

    profile = np.asarray(data["range_profile"], dtype=float)
    range_m = np.asarray(data["range_m"], dtype=float)
    if profile.ndim != 2 or range_m.ndim != 1:
        return None
    if profile.shape[0] < 2:
        return None

    cols = min(profile.shape[1], range_m.size)
    profile = profile[:, :cols]
    range_m = range_m[:cols]

    background = None
    if "range_background" in data:
        candidate = np.asarray(data["range_background"], dtype=float)
        if candidate.ndim == 1 and candidate.size >= cols:
            background = candidate[:cols]

    min_range = float(feature_params.get("min_range_m", 0.0))
    max_range = float(feature_params.get("max_range_m", 0.80))
    mask = (range_m >= min_range) & (range_m <= max_range)
    if not np.any(mask):
        return None

    profile = profile[:, mask]
    range_m = range_m[mask]
    if background is not None:
        background = background[mask]

    if bool(feature_params.get("db", True)):
        profile = db_scale(profile)
        if background is not None:
            background = db_scale(background)

    if background is not None:
        motion = profile - background[np.newaxis, :]
    else:
        fallback_frames = max(1, int(feature_params.get("fallback_background_frames", 5)))
        count = min(fallback_frames, profile.shape[0])
        motion = profile - np.median(profile[:count], axis=0)[np.newaxis, :]

    time_s = np.asarray(
        data.get("time_s", np.linspace(0.0, profile.shape[0] - 1, profile.shape[0])),
        dtype=float,
    )
    if time_s.size != profile.shape[0]:
        time_s = np.linspace(0.0, profile.shape[0] - 1, profile.shape[0])
    return range_m, time_s, motion


def extract_box_presence_feature_vector(
    data: dict[str, np.ndarray],
    feature_params: dict[str, Any],
) -> tuple[list[float] | None, list[str]]:
    payload = box_presence_motion_matrix(data, feature_params)
    if payload is None:
        return None, []

    _range_m, _time_s, motion = payload
    if bool(feature_params.get("normalize", False)):
        motion = robust_normalize(motion)

    frames = int(feature_params.get("resample_frames", 24))
    bins = int(feature_params.get("resample_bins", 32))
    image = resample_matrix(motion, frames, bins)

    features = [float(value) for value in image.ravel()]
    names = [
        f"presence_range_image_t{row:02d}_r{col:02d}"
        for row in range(frames)
        for col in range(bins)
    ]
    return features, names


def time_window_segments(
    time_s: np.ndarray,
    window_seconds: float,
    overlap: float,
    min_frames: int,
) -> list[tuple[int, int]]:
    time_s = np.asarray(time_s, dtype=float)
    if time_s.size < min_frames:
        return []
    if window_seconds <= 0:
        raise ValueError("window_seconds must be positive")
    if not 0.0 <= overlap < 1.0:
        raise ValueError("overlap must be in [0, 1)")

    step_seconds = window_seconds * (1.0 - overlap)
    first_time = float(time_s[0])
    last_time = float(time_s[-1])
    segments: list[tuple[int, int]] = []
    start_time = first_time

    while start_time <= last_time:
        end_time = start_time + window_seconds
        start_index = int(np.searchsorted(time_s, start_time, side="left"))
        end_index = int(np.searchsorted(time_s, end_time, side="right"))
        if end_index - start_index >= min_frames:
            segments.append((start_index, end_index))
        if end_time >= last_time:
            break
        start_time += step_seconds

    return segments


def slice_presence_segment(
    data: dict[str, np.ndarray],
    start_index: int,
    end_index: int,
) -> dict[str, np.ndarray]:
    segment: dict[str, np.ndarray] = {
        "range_m": np.asarray(data["range_m"]),
        "range_profile": np.asarray(data["range_profile"])[start_index:end_index],
    }
    if "range_background" in data:
        segment["range_background"] = np.asarray(data["range_background"])
    if "time_s" in data:
        time_s = np.asarray(data["time_s"], dtype=float)[start_index:end_index]
        segment["time_s"] = time_s - time_s[0] if time_s.size else time_s
    if "point_count" in data:
        segment["point_count"] = np.asarray(data["point_count"])[start_index:end_index]
    if "points_xyz" in data:
        segment["points_xyz"] = np.asarray(data["points_xyz"])[start_index:end_index]
    return segment


def range_profile_features(
    data: dict[str, np.ndarray],
    feature_params: dict[str, Any],
) -> tuple[np.ndarray | None, list[str]]:
    if "range_m" not in data:
        return None, []
    range_m = np.asarray(data["range_m"], dtype=float)
    if "mean_range_profile" in data:
        profile = np.asarray(data["mean_range_profile"], dtype=float)
    elif "range_profile" in data:
        profile = np.mean(np.asarray(data["range_profile"], dtype=float), axis=0)
    else:
        return None, []

    cols = min(range_m.size, profile.size)
    range_m = range_m[:cols]
    profile = profile[:cols]
    background = None
    if "range_background" in data:
        candidate = np.asarray(data["range_background"], dtype=float)
        if candidate.ndim == 1 and candidate.size >= cols:
            background = candidate[:cols]

    min_range = float(feature_params.get("min_range_m", 0.0))
    max_range = float(feature_params.get("max_range_m", 0.60))
    mask = (range_m >= min_range) & (range_m <= max_range)
    if not np.any(mask):
        return None, []

    profile = profile[mask]
    if background is not None:
        background = background[mask]
    if bool(feature_params.get("db", True)):
        profile = db_scale(profile)
        if background is not None:
            background = db_scale(background)
    if background is not None:
        profile = profile - background

    range_bins = int(feature_params.get("range_bins", 64))
    features = resample_vector(profile, range_bins)
    names = [f"box_delta_range_profile_b{index:02d}" for index in range(range_bins)]
    return features, names


def point_slice_features(
    data: dict[str, np.ndarray],
    feature_params: dict[str, Any],
) -> tuple[np.ndarray, list[str]]:
    names = [
        "slice_count_mean",
        "slice_count_std",
        "slice_count_max",
        "slice_count_total",
        "slice_centroid_x",
        "slice_centroid_y",
        "slice_spread_x",
        "slice_spread_y",
        "slice_min_x",
        "slice_max_x",
        "slice_min_y",
        "slice_max_y",
    ]

    count = np.asarray(data.get("slice_point_count", []), dtype=float)
    xyz = np.asarray(data.get("slice_points_xyz", []), dtype=float)
    if count.size == 0:
        return np.zeros(len(names), dtype=float), names

    count_stats = [
        float(np.mean(count)),
        float(np.std(count)),
        float(np.max(count)),
        float(np.sum(count)),
    ]

    points: list[np.ndarray] = []
    if xyz.ndim == 3:
        usable_frames = min(xyz.shape[0], count.size)
        for frame_index in range(usable_frames):
            point_count = max(0, min(int(count[frame_index]), xyz.shape[1]))
            if point_count == 0:
                continue
            frame_points = xyz[frame_index, :point_count, :]
            finite = np.all(np.isfinite(frame_points), axis=1)
            frame_points = frame_points[finite]
            if len(frame_points):
                points.append(frame_points)

    if not points:
        return np.array(count_stats + [0.0] * 8, dtype=float), names

    all_points = np.vstack(points)[:, :2]
    centroid = np.mean(all_points, axis=0)
    spread = np.std(all_points, axis=0)
    extras = [
        float(centroid[0]),
        float(centroid[1]),
        float(spread[0]),
        float(spread[1]),
        float(np.min(all_points[:, 0])),
        float(np.max(all_points[:, 0])),
        float(np.min(all_points[:, 1])),
        float(np.max(all_points[:, 1])),
    ]
    return np.array(count_stats + extras, dtype=float), names


def extract_box_feature_vector(
    data: dict[str, np.ndarray],
    feature_params: dict[str, Any],
) -> tuple[list[float] | None, list[str]]:
    profile, profile_names = range_profile_features(data, feature_params)
    if profile is None:
        return None, []

    features = list(float(value) for value in profile)
    names = list(profile_names)

    if bool(feature_params.get("include_points", True)):
        point_features, point_names = point_slice_features(data, feature_params)
        features.extend(float(value) for value in point_features)
        names.extend(point_names)

    return features, names


def range_profile_from_tlvs(tlvs: list[tuple[int, bytes]]) -> np.ndarray | None:
    for tlv_type, payload in tlvs:
        if tlv_type in {RANGE_PROFILE_MAJOR, RANGE_PROFILE_MINOR}:
            return np.frombuffer(payload, dtype="<u4").copy()
    return None


def read_box_data_frame(
    port: Any,
    frame_timeout_s: float,
    expected_range_profile_bytes: int,
) -> tuple[int, np.ndarray, PointCloud]:
    while True:
        frame_number, tlvs = read_frame(
            port,
            frame_timeout_s,
            expected_range_profile_bytes,
        )
        profile = range_profile_from_tlvs(tlvs)
        if profile is not None:
            return frame_number, profile, point_cloud_from_tlvs(tlvs)


def filter_point_slice(
    cloud: PointCloud,
    point_distance_m: float,
    point_window_m: float,
    x_limit_m: float | None,
    _z_limit_m: float | None,
) -> np.ndarray:
    if len(cloud.x) == 0:
        return np.empty((0, 3), dtype=float)

    xyz = np.column_stack((cloud.x, cloud.y, cloud.z))
    mask = np.abs(xyz[:, 1] - point_distance_m) <= point_window_m
    if x_limit_m is not None and x_limit_m > 0:
        mask &= np.abs(xyz[:, 0]) <= x_limit_m
    return xyz[mask]


def pack_slice_points(frames: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    frame_count = len(frames)
    counts = np.array([len(frame) for frame in frames], dtype=np.uint16)
    max_points = int(np.max(counts)) if len(counts) else 0
    points = np.full((frame_count, max_points, 3), np.nan, dtype=float)
    for index, frame_points in enumerate(frames):
        count = len(frame_points)
        if count:
            points[index, :count, :] = frame_points
    return counts, points


def estimator_classes(model: Any) -> list[Any] | None:
    classes = getattr(model, "classes_", None)
    if classes is not None:
        return list(classes)
    if hasattr(model, "steps") and model.steps:
        final_estimator = model.steps[-1][1]
        classes = getattr(final_estimator, "classes_", None)
        if classes is not None:
            return list(classes)
    return None


def prediction_confidence(model: Any, features: list[float], prediction: Any) -> float | None:
    if not hasattr(model, "predict_proba"):
        return None
    probabilities = model.predict_proba([features])[0]
    classes = estimator_classes(model)
    if classes is None:
        return float(max(probabilities))
    if prediction not in classes:
        return None
    return float(probabilities[classes.index(prediction)])


def prediction_scores(model: Any, features: list[float]) -> list[tuple[str, float]]:
    if not hasattr(model, "predict_proba"):
        return []
    probabilities = model.predict_proba([features])[0]
    classes = estimator_classes(model)
    if classes is None:
        classes = list(range(len(probabilities)))
    scores = [
        (str(label), float(score))
        for label, score in zip(classes, probabilities)
    ]
    return sorted(scores, key=lambda item: item[1], reverse=True)


def majority_vote(predictions: list[Any]) -> tuple[Any | None, float | None, dict[str, int]]:
    if not predictions:
        return None, None, {}
    counts = Counter(predictions)
    max_count = max(counts.values())
    tied_labels = {label for label, count in counts.items() if count == max_count}
    voted_prediction = None
    for label in reversed(predictions):
        if label in tied_labels:
            voted_prediction = label
            break
    vote_fraction = max_count / len(predictions)
    return voted_prediction, vote_fraction, {str(label): int(count) for label, count in counts.items()}
