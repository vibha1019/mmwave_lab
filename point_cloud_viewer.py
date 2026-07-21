#!/usr/bin/env python3
from __future__ import annotations

import argparse
import struct
import time
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np
import serial

from get_range_profile import (
    load_configuration,
    read_frame,
    send_configuration,
    write_cli_command,
)


POINT_CLOUD_FLOAT = 1
POINT_CLOUD_FIXED_TYPES = {301, 1020}


@dataclass(frozen=True)
class PointCloud:
    x: np.ndarray
    y: np.ndarray
    z: np.ndarray
    velocity: np.ndarray


def empty_point_cloud() -> PointCloud:
    empty = np.array([], dtype=float)
    return PointCloud(empty, empty, empty, empty)


def decode_float_points(payload: bytes) -> PointCloud:
    count = len(payload) // 16
    if count == 0:
        return empty_point_cloud()

    values = np.frombuffer(payload[: count * 16], dtype="<f4").reshape(count, 4)
    return PointCloud(
        x=values[:, 0].astype(float),
        y=values[:, 1].astype(float),
        z=values[:, 2].astype(float),
        velocity=values[:, 3].astype(float),
    )


def decode_fixed_points(payload: bytes) -> PointCloud:
    if len(payload) < 20:
        return empty_point_cloud()

    xyz_unit, doppler_unit, _snr_unit, _noise_unit = struct.unpack_from(
        "<ffff",
        payload,
        0,
    )
    num_major_points, _num_minor_points = struct.unpack_from("<HH", payload, 16)

    x_values: list[float] = []
    y_values: list[float] = []
    z_values: list[float] = []
    velocities: list[float] = []

    offset = 20
    for _ in range(num_major_points):
        if offset + 10 > len(payload):
            break

        x, y, z, doppler, _snr, _noise = struct.unpack_from(
            "<hhhhBB",
            payload,
            offset,
        )
        x_values.append(x * xyz_unit)
        y_values.append(y * xyz_unit)
        z_values.append(z * xyz_unit)
        velocities.append(doppler * doppler_unit)
        offset += 10

    return PointCloud(
        x=np.array(x_values, dtype=float),
        y=np.array(y_values, dtype=float),
        z=np.array(z_values, dtype=float),
        velocity=np.array(velocities, dtype=float),
    )


def point_cloud_from_tlvs(tlvs: list[tuple[int, bytes]]) -> PointCloud:
    for tlv_type, payload in tlvs:
        if tlv_type == POINT_CLOUD_FLOAT:
            return decode_float_points(payload)
        if tlv_type in POINT_CLOUD_FIXED_TYPES:
            return decode_fixed_points(payload)

    return empty_point_cloud()


def radial_ranges(cloud: PointCloud) -> np.ndarray:
    return np.sqrt(cloud.x**2 + cloud.y**2)


def xy_points(cloud: PointCloud) -> np.ndarray:
    if len(cloud.x) == 0:
        return np.empty((0, 2), dtype=float)
    return np.column_stack((cloud.x, cloud.y))


def estimate_tracked_velocity(
    cloud: PointCloud,
    previous_cloud: PointCloud | None,
    previous_time: float | None,
    now: float,
    max_match_distance_m: float,
) -> np.ndarray:
    tracked = np.full(len(cloud.x), np.nan, dtype=float)
    if (
        previous_cloud is None
        or previous_time is None
        or len(cloud.x) == 0
        or len(previous_cloud.x) == 0
    ):
        return tracked

    dt = now - previous_time
    if dt <= 0:
        return tracked

    current_xy = xy_points(cloud)
    previous_xy = xy_points(previous_cloud)
    deltas = current_xy[:, np.newaxis, :] - previous_xy[np.newaxis, :, :]
    distances = np.sqrt(np.sum(deltas * deltas, axis=2))
    nearest_indices = np.argmin(distances, axis=1)
    nearest_distances = distances[np.arange(len(cloud.x)), nearest_indices]
    matched = nearest_distances <= max_match_distance_m

    current_ranges = radial_ranges(cloud)
    previous_ranges = radial_ranges(previous_cloud)
    tracked[matched] = (
        current_ranges[matched] - previous_ranges[nearest_indices[matched]]
    ) / dt
    return tracked


def color_values(
    cloud: PointCloud,
    tracked_velocity: np.ndarray,
    color_by: str,
) -> np.ndarray:
    if color_by == "range":
        return radial_ranges(cloud)
    if color_by == "x":
        return cloud.x
    if color_by == "tracked-velocity":
        return tracked_velocity
    return cloud.velocity


def is_velocity_color(color_by: str) -> bool:
    return color_by in {"velocity", "tracked-velocity"}


def make_color_norm(
    color_by: str,
    xy_limit: float,
    velocity_limit: float,
) -> colors.Normalize:
    if color_by == "range":
        return colors.Normalize(vmin=0.0, vmax=xy_limit)
    if color_by == "x":
        return colors.Normalize(vmin=-xy_limit, vmax=xy_limit)

    limit = velocity_limit if velocity_limit > 0 else 1.0
    return colors.Normalize(vmin=-limit, vmax=limit)


def colorbar_label(color_by: str) -> str:
    if color_by == "range":
        return "range (m)"
    if color_by == "x":
        return "x left/right (m)"
    if color_by == "tracked-velocity":
        return "tracked radial velocity (m/s)"
    return "radial velocity (m/s)"


def velocity_range_text(values: np.ndarray) -> str:
    finite = values[np.isfinite(values)]
    if len(finite) == 0:
        return "n/a"
    return f"{np.min(finite):+.2f}..{np.max(finite):+.2f}"


def value_range_text(values: np.ndarray) -> str:
    finite = values[np.isfinite(values)]
    if len(finite) == 0:
        return "n/a"
    return f"{np.min(finite):.2f}..{np.max(finite):.2f}"


def summarize_cloud(
    frame_number: int,
    cloud: PointCloud,
    tracked_velocity: np.ndarray,
) -> str:
    title = f"frame {frame_number}: {len(cloud.x)} points"
    if not len(cloud.x):
        return title

    ranges = radial_ranges(cloud)
    title += f", median range {np.median(ranges):.2f} m"
    title += f", x {value_range_text(cloud.x)} m"
    title += f", y {value_range_text(cloud.y)} m"
    title += f", doppler {velocity_range_text(cloud.velocity)} m/s"
    title += f", tracked {velocity_range_text(tracked_velocity)} m/s"
    return title


def stop_and_drain(port: serial.Serial) -> None:
    port.reset_input_buffer()

    deadline = time.monotonic() + 4.0
    last_rx = time.monotonic()
    next_stop = 0.0
    text_tail = ""
    while time.monotonic() < deadline:
        now = time.monotonic()
        if now >= next_stop:
            port.write(b"sensorStop 0\r\n")
            port.flush()
            next_stop = now + 0.25

        waiting = port.in_waiting
        if waiting:
            data = port.read(waiting)
            text_tail = (text_tail + data.decode("ascii", errors="ignore"))[-512:]
            last_rx = time.monotonic()
        elif "done" in text_tail.lower() and time.monotonic() - last_rx > 0.25:
            break
        elif "mmwdemo:/>" in text_tail.lower() and time.monotonic() - last_rx > 0.5:
            break
        else:
            time.sleep(0.02)

    port.reset_input_buffer()


def remove_leading_sensor_stop(commands: list[str]) -> list[str]:
    if commands and commands[0].split()[0] == "sensorStop":
        return commands[1:]
    return commands


def configure_radar(
    port: serial.Serial,
    cfg_path: Path,
) -> None:
    commands = load_configuration(cfg_path)
    stop_and_drain(port)
    send_configuration(
        port,
        remove_leading_sensor_stop(commands),
        use_cfg_baud_rate=False,
    )


def set_2d_axes(axis, xy_limit: float) -> None:
    axis.set_xlim(-xy_limit, xy_limit)
    axis.set_ylim(0.0, xy_limit)
    axis.set_aspect("equal", adjustable="box")
    axis.set_xlabel("x left/right (m)")
    axis.set_ylabel("y range (m)")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Real-time top-down x/y point-cloud viewer for xWRL6432."
    )
    parser.add_argument("--port", required=True)
    parser.add_argument(
        "--cfg",
        type=Path,
        default=Path("xwrL64xx-evm/point_cloud.cfg"),
    )
    parser.add_argument("--baud", type=int, default=115200)
    parser.add_argument("--frame-timeout", type=float, default=5.0)
    parser.add_argument("--frames", type=int, default=0)
    parser.add_argument("--xy-limit", type=float, default=4.0)
    parser.add_argument(
        "--color-by",
        choices=["tracked-velocity", "velocity", "range", "x"],
        default="tracked-velocity",
    )
    parser.add_argument(
        "--velocity-limit",
        type=float,
        default=0.0,
        help="Fixed +/- velocity color scale in m/s. Use 0 for auto scale.",
    )
    parser.add_argument("--point-size", type=float, default=35.0)
    parser.add_argument("--track-gate", type=float, default=0.25)
    parser.add_argument(
        "--no-config",
        action="store_true",
        help="Read an already-running point-cloud stream without sending a cfg.",
    )
    args = parser.parse_args()

    with serial.Serial(args.port, args.baud, timeout=0.2) as port:
        if args.no_config:
            port.reset_input_buffer()
        else:
            print(f"Using cfg: {args.cfg}")
            try:
                configure_radar(port, args.cfg)
            except (RuntimeError, ValueError) as error:
                raise SystemExit(f"Could not configure radar: {error}") from None

        plt.ion()
        figure = plt.figure()
        axis = figure.add_subplot(111)
        set_2d_axes(axis, args.xy_limit)
        axis.grid(True)
        cmap = plt.get_cmap("coolwarm" if is_velocity_color(args.color_by) else "viridis")
        norm = make_color_norm(
            args.color_by,
            args.xy_limit,
            args.velocity_limit,
        )

        scatter = axis.scatter(
            [],
            [],
            c=[],
            cmap=cmap,
            norm=norm,
            s=args.point_size,
        )
        colorbar = figure.colorbar(scatter, ax=axis, pad=0.12)
        colorbar.set_label(colorbar_label(args.color_by))

        frame_count = 0
        consecutive_warnings = 0
        previous_cloud: PointCloud | None = None
        previous_time: float | None = None

        try:
            while plt.fignum_exists(figure.number):
                try:
                    frame_number, tlvs = read_frame(port, args.frame_timeout)
                except (TimeoutError, ValueError, RuntimeError) as error:
                    consecutive_warnings += 1
                    print(f"\nFrame parse warning: {error}")
                    if consecutive_warnings >= 3:
                        raise RuntimeError(
                            "No valid point-cloud frames received."
                        ) from error
                    continue

                consecutive_warnings = 0
                now = time.monotonic()
                cloud = point_cloud_from_tlvs(tlvs)
                tracked = estimate_tracked_velocity(
                    cloud,
                    previous_cloud,
                    previous_time,
                    now,
                    args.track_gate,
                )
                values = color_values(cloud, tracked, args.color_by)
                plot_values = np.nan_to_num(values, nan=0.0)

                if (
                    is_velocity_color(args.color_by)
                    and args.velocity_limit <= 0
                    and np.any(np.isfinite(values))
                ):
                    finite_values = values[np.isfinite(values)]
                    limit = max(0.1, float(np.max(np.abs(finite_values))) * 1.1)
                    norm.vmin = -limit
                    norm.vmax = limit
                    colorbar.update_normal(scatter)

                offsets = (
                    np.column_stack((cloud.x, cloud.y))
                    if len(cloud.x)
                    else np.empty((0, 2))
                )
                scatter.set_offsets(offsets)
                scatter.set_array(plot_values)
                point_colors = (
                    cmap(norm(plot_values)) if len(plot_values) else np.empty((0, 4))
                )
                scatter.set_facecolors(point_colors)
                scatter.set_edgecolors(point_colors)

                title = summarize_cloud(frame_number, cloud, tracked)
                axis.set_title(title)
                print("\r" + title, end="", flush=True)
                figure.canvas.draw_idle()
                plt.pause(0.001)

                previous_cloud = cloud
                previous_time = now
                frame_count += 1
                if args.frames and frame_count >= args.frames:
                    break

            print()

        except KeyboardInterrupt:
            print("\nStopping.")
        finally:
            if not args.no_config:
                print("> sensorStop 0")
                stop_and_drain(port)


if __name__ == "__main__":
    main()
