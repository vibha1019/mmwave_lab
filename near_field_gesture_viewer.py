#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np
import serial

from get_range_profile import (
    RANGE_PROFILE_MAJOR,
    RANGE_PROFILE_MINOR,
    load_configuration,
    parse_range_config,
    read_frame,
    send_configuration,
    write_cli_command,
)
from point_cloud_viewer import (
    PointCloud,
    empty_point_cloud,
    estimate_tracked_velocity,
    point_cloud_from_tlvs,
)


DEFAULT_CFG = Path("xwrL64xx-evm/near_field_hand_50cm.cfg")


def range_profile_from_tlvs(tlvs: list[tuple[int, bytes]]) -> np.ndarray | None:
    for tlv_type, payload in tlvs:
        if tlv_type in {RANGE_PROFILE_MAJOR, RANGE_PROFILE_MINOR}:
            return np.frombuffer(payload, dtype="<u4").astype(float)
    return None


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


def warm_reset_demo(port: serial.Serial, timeout_s: float = 8.0) -> None:
    """Reload the flashed demo so the UART is back at a clean CLI prompt."""
    print("> sensorWarmRst")
    port.reset_input_buffer()
    write_cli_command(port, "sensorWarmRst")

    chunks: list[bytes] = []
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        waiting = port.in_waiting
        if waiting:
            chunks.append(port.read(waiting))
            text = b"".join(chunks[-8:]).decode("ascii", errors="ignore").lower()
            if "mmwdemo:/>" in text:
                port.reset_input_buffer()
                return
        else:
            time.sleep(0.05)

    text = b"".join(chunks).decode("ascii", errors="ignore")
    if "not supported" in text.lower() or "error" in text.lower():
        raise RuntimeError(f"sensorWarmRst failed: {text.strip()}")

    raise RuntimeError("Timed out waiting for mmwDemo prompt after sensorWarmRst.")


def remove_leading_sensor_stop(commands: list[str]) -> list[str]:
    if commands and commands[0].split()[0] == "sensorStop":
        return commands[1:]
    return commands


def read_near_field_frame(
    port: serial.Serial,
    frame_timeout: float,
    expected_range_profile_bytes: int,
) -> tuple[int, np.ndarray, PointCloud]:
    while True:
        frame_number, tlvs = read_frame(
            port,
            frame_timeout,
            expected_range_profile_bytes,
        )
        profile = range_profile_from_tlvs(tlvs)
        if profile is not None:
            return frame_number, profile, point_cloud_from_tlvs(tlvs)


def filter_front_roi(
    cloud: PointCloud,
    x_limit_m: float,
    y_max_m: float,
) -> PointCloud:
    if len(cloud.x) == 0:
        return empty_point_cloud()

    mask = (
        (np.abs(cloud.x) <= x_limit_m)
        & (cloud.y >= 0.0)
        & (cloud.y <= y_max_m)
    )
    return PointCloud(
        x=cloud.x[mask],
        y=cloud.y[mask],
        z=cloud.z[mask],
        velocity=cloud.velocity[mask],
    )


def estimate_peak(
    ranges_m: np.ndarray,
    difference: np.ndarray,
    peak_ratio: float,
) -> tuple[float, float]:
    if len(difference) == 0:
        return math.nan, 0.0

    peak_index = int(np.argmax(difference))
    peak_strength = float(difference[peak_index])
    typical = float(np.median(difference)) + 1.0
    if peak_strength < peak_ratio * typical:
        return math.nan, peak_strength

    left = max(0, peak_index - 1)
    right = min(len(difference), peak_index + 2)
    weights = difference[left:right]
    weight_sum = float(np.sum(weights))
    if weight_sum <= 0:
        return float(ranges_m[peak_index]), peak_strength

    refined_range = float(np.sum(ranges_m[left:right] * weights) / weight_sum)
    return refined_range, peak_strength


def db_scale(values: np.ndarray) -> np.ndarray:
    return 10.0 * np.log10(np.maximum(values, 0.0) + 1.0)


def append_record(
    records: dict[str, list],
    frame_number: int,
    elapsed_s: float,
    profile_clip: np.ndarray,
    diff_clip: np.ndarray,
    peak_range_m: float,
    peak_strength: float,
    cloud: PointCloud,
    tracked_velocity: np.ndarray,
) -> None:
    records["frame_number"].append(frame_number)
    records["time_s"].append(elapsed_s)
    records["range_profile"].append(profile_clip.copy())
    records["range_diff"].append(diff_clip.copy())
    records["peak_range_m"].append(peak_range_m)
    records["peak_strength"].append(peak_strength)
    records["num_points"].append(len(cloud.x))
    if len(cloud.x):
        records["points_xyz"].append(np.column_stack((cloud.x, cloud.y, cloud.z)))
        records["points_raw_doppler"].append(cloud.velocity.copy())
        records["points_tracked_velocity"].append(tracked_velocity.copy())
    else:
        records["points_xyz"].append(np.empty((0, 3), dtype=float))
        records["points_raw_doppler"].append(np.empty(0, dtype=float))
        records["points_tracked_velocity"].append(np.empty(0, dtype=float))


def save_recording(
    path: Path,
    label: str,
    cfg_path: Path,
    range_m: np.ndarray,
    background: np.ndarray,
    records: dict[str, list],
) -> None:
    frame_count = len(records["frame_number"])
    if frame_count == 0:
        print("No frames recorded.")
        return

    max_points = max(records["num_points"]) if records["num_points"] else 0
    points_xyz = np.full((frame_count, max_points, 3), np.nan, dtype=float)
    points_raw_doppler = np.full((frame_count, max_points), np.nan, dtype=float)
    points_tracked_velocity = np.full((frame_count, max_points), np.nan, dtype=float)

    for index, xyz in enumerate(records["points_xyz"]):
        count = len(xyz)
        if count == 0:
            continue
        points_xyz[index, :count, :] = xyz
        points_raw_doppler[index, :count] = records["points_raw_doppler"][index]
        points_tracked_velocity[index, :count] = records[
            "points_tracked_velocity"
        ][index]

    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        label=np.array(label),
        cfg_path=np.array(str(cfg_path)),
        created_unix_s=np.array(time.time()),
        range_m=range_m,
        background=background,
        frame_number=np.array(records["frame_number"], dtype=np.uint32),
        time_s=np.array(records["time_s"], dtype=float),
        range_profile=np.vstack(records["range_profile"]),
        range_diff=np.vstack(records["range_diff"]),
        peak_range_m=np.array(records["peak_range_m"], dtype=float),
        peak_strength=np.array(records["peak_strength"], dtype=float),
        num_points=np.array(records["num_points"], dtype=np.uint16),
        points_xyz=points_xyz,
        points_raw_doppler=points_raw_doppler,
        points_tracked_velocity=points_tracked_velocity,
    )
    print(f"Saved {frame_count} frames to {path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Near-field range-profile and point-cloud viewer for hand gestures."
        )
    )
    parser.add_argument("--port", required=True)
    parser.add_argument("--cfg", type=Path, default=DEFAULT_CFG)
    parser.add_argument("--baud", type=int, default=115200)
    parser.add_argument("--background-frames", type=int, default=20)
    parser.add_argument("--min-range", type=float, default=0.0)
    parser.add_argument("--max-range", type=float, default=0.50)
    parser.add_argument("--x-limit", type=float, default=0.15)
    parser.add_argument("--history-frames", type=int, default=80)
    parser.add_argument("--peak-ratio", type=float, default=2.0)
    parser.add_argument("--point-size", type=float, default=55.0)
    parser.add_argument("--track-gate", type=float, default=0.08)
    parser.add_argument("--velocity-limit", type=float, default=0.75)
    parser.add_argument("--frame-timeout", type=float, default=5.0)
    parser.add_argument("--frames", type=int, default=0)
    parser.add_argument("--duration", type=float, default=0.0)
    parser.add_argument("--record", type=Path)
    parser.add_argument("--label", default="")
    parser.add_argument(
        "--no-config",
        action="store_true",
        help="Read an already-running stream without sending the cfg.",
    )
    parser.add_argument(
        "--no-warm-reset",
        action="store_true",
        help="Skip the default demo warm reset before sending the cfg.",
    )
    args = parser.parse_args()

    if args.background_frames < 1:
        raise SystemExit("--background-frames must be at least 1.")
    if args.history_frames < 2:
        raise SystemExit("--history-frames must be at least 2.")
    if args.min_range >= args.max_range:
        raise SystemExit("--min-range must be smaller than --max-range.")

    commands = load_configuration(args.cfg)
    range_config = parse_range_config(commands)
    if range_config is None:
        raise SystemExit("Could not compute range-bin spacing from cfg.")

    expected_bytes = range_config.num_range_bins * 4
    full_range_m = np.arange(range_config.num_range_bins) * range_config.bin_spacing_m
    range_mask = (full_range_m >= args.min_range) & (full_range_m <= args.max_range)
    if not np.any(range_mask):
        raise SystemExit("The requested range window has no FFT bins.")
    range_m = full_range_m[range_mask]

    records: dict[str, list] = {
        "frame_number": [],
        "time_s": [],
        "range_profile": [],
        "range_diff": [],
        "peak_range_m": [],
        "peak_strength": [],
        "num_points": [],
        "points_xyz": [],
        "points_raw_doppler": [],
        "points_tracked_velocity": [],
    }

    with serial.Serial(args.port, args.baud, timeout=0.2) as port:
        if args.no_config:
            port.reset_input_buffer()
        else:
            print(f"Using cfg: {args.cfg}")
            stop_and_drain(port)
            if args.no_warm_reset:
                pass
            else:
                warm_reset_demo(port)
            try:
                send_configuration(
                    port,
                    remove_leading_sensor_stop(commands),
                    use_cfg_baud_rate=False,
                )
            except (RuntimeError, ValueError) as error:
                raise SystemExit(f"Could not configure radar: {error}") from None

        print(
            "Estimated range-bin spacing: "
            f"{range_config.bin_spacing_m:.4f} m "
            f"({range_config.num_range_bins} bins)."
        )
        print(
            f"Keep the first {args.background_frames} frames empty "
            "to capture the background."
        )

        try:
            background_frames = []
            while len(background_frames) < args.background_frames:
                _frame_number, profile, _cloud = read_near_field_frame(
                    port,
                    args.frame_timeout,
                    expected_bytes,
                )
                background_frames.append(profile)
        except KeyboardInterrupt:
            print("\nStopping.")
            if not args.no_config:
                print("> sensorStop 0")
                stop_and_drain(port)
            return
        except (TimeoutError, ValueError, RuntimeError) as error:
            if not args.no_config:
                print("> sensorStop 0")
                stop_and_drain(port)
            raise SystemExit(f"Could not capture background: {error}") from None

        background = np.median(np.vstack(background_frames), axis=0)
        background_clip = background[range_mask]
        print("Background captured. Move a hand in the first 50 cm.")

        plt.ion()
        figure = plt.figure(figsize=(12, 7))
        grid = figure.add_gridspec(
            2,
            2,
            width_ratios=(1.25, 1.0),
            height_ratios=(1.0, 1.0),
            wspace=0.28,
            hspace=0.35,
        )
        profile_axis = figure.add_subplot(grid[0, 0])
        waterfall_axis = figure.add_subplot(grid[1, 0])
        cloud_axis = figure.add_subplot(grid[:, 1])

        (profile_line,) = profile_axis.plot(range_m, np.zeros_like(range_m))
        profile_axis.set_xlim(args.min_range, args.max_range)
        profile_axis.set_ylim(0.0, 10.0)
        profile_axis.set_xlabel("range (m)")
        profile_axis.set_ylabel("change strength (dB)")
        profile_axis.grid(True)

        history = np.zeros((args.history_frames, len(range_m)), dtype=float)
        waterfall = waterfall_axis.imshow(
            history,
            aspect="auto",
            origin="lower",
            extent=(range_m[0], range_m[-1], -args.history_frames, 0),
            cmap="magma",
            vmin=0.0,
            vmax=10.0,
        )
        waterfall_axis.set_xlabel("range (m)")
        waterfall_axis.set_ylabel("recent frames")
        figure.colorbar(waterfall, ax=waterfall_axis, pad=0.02)

        cloud_axis.set_xlim(-args.x_limit, args.x_limit)
        cloud_axis.set_ylim(0.0, args.max_range)
        cloud_axis.set_aspect("equal", adjustable="box")
        cloud_axis.set_xlabel("x left/right (m)")
        cloud_axis.set_ylabel("y range (m)")
        cloud_axis.grid(True)

        cmap = plt.get_cmap("coolwarm")
        norm = colors.Normalize(
            vmin=-args.velocity_limit,
            vmax=args.velocity_limit,
        )
        scatter = cloud_axis.scatter(
            [],
            [],
            c=[],
            cmap=cmap,
            norm=norm,
            s=args.point_size,
        )
        colorbar = figure.colorbar(scatter, ax=cloud_axis, pad=0.12)
        colorbar.set_label("tracked radial velocity (m/s)")

        frame_count = 0
        consecutive_warnings = 0
        previous_cloud: PointCloud | None = None
        previous_time: float | None = None
        run_start = time.monotonic()

        try:
            while plt.fignum_exists(figure.number):
                try:
                    frame_number, profile, cloud = read_near_field_frame(
                        port,
                        args.frame_timeout,
                        expected_bytes,
                    )
                except (TimeoutError, ValueError, RuntimeError) as error:
                    consecutive_warnings += 1
                    print(f"\nFrame parse warning: {error}")
                    if consecutive_warnings >= 3:
                        raise RuntimeError("No valid frames received.") from error
                    continue

                consecutive_warnings = 0
                now = time.monotonic()
                elapsed_s = now - run_start
                difference = np.maximum(profile - background, 0.0)
                diff_clip = difference[range_mask]
                display_profile = db_scale(diff_clip)
                peak_range_m, peak_strength = estimate_peak(
                    range_m,
                    diff_clip,
                    args.peak_ratio,
                )

                roi_cloud = filter_front_roi(
                    cloud,
                    args.x_limit,
                    args.max_range,
                )
                tracked_velocity = estimate_tracked_velocity(
                    roi_cloud,
                    previous_cloud,
                    previous_time,
                    now,
                    args.track_gate,
                )
                velocity_colors = np.nan_to_num(tracked_velocity, nan=0.0)
                point_colors = (
                    cmap(norm(velocity_colors))
                    if len(velocity_colors)
                    else np.empty((0, 4))
                )

                profile_line.set_ydata(display_profile)
                y_max = max(10.0, float(np.percentile(display_profile, 98)) * 1.2)
                profile_axis.set_ylim(0.0, y_max)

                history = np.roll(history, -1, axis=0)
                history[-1, :] = display_profile
                waterfall.set_data(history)
                waterfall.set_clim(0.0, max(10.0, float(np.percentile(history, 99))))

                offsets = (
                    np.column_stack((roi_cloud.x, roi_cloud.y))
                    if len(roi_cloud.x)
                    else np.empty((0, 2))
                )
                scatter.set_offsets(offsets)
                scatter.set_array(velocity_colors)
                scatter.set_facecolors(point_colors)
                scatter.set_edgecolors(point_colors)

                if math.isfinite(peak_range_m):
                    peak_text = f"peak {peak_range_m * 100:.1f} cm"
                else:
                    peak_text = "no clear hand peak"

                status = (
                    f"frame {frame_number}: {peak_text}, "
                    f"ROI points {len(roi_cloud.x)}"
                )
                profile_axis.set_title(status)
                cloud_axis.set_title(
                    "front ROI: x +/-"
                    f"{args.x_limit * 100:.0f} cm, "
                    f"range < {args.max_range * 100:.0f} cm"
                )
                print("\r" + status, end="", flush=True)

                if args.record is not None:
                    append_record(
                        records,
                        frame_number,
                        elapsed_s,
                        profile[range_mask],
                        diff_clip,
                        peak_range_m,
                        peak_strength,
                        roi_cloud,
                        tracked_velocity,
                    )

                figure.canvas.draw_idle()
                plt.pause(0.001)

                previous_cloud = roi_cloud
                previous_time = now
                frame_count += 1
                if args.frames and frame_count >= args.frames:
                    break
                if args.duration and elapsed_s >= args.duration:
                    break

            print()

        except KeyboardInterrupt:
            print("\nStopping.")
        finally:
            if args.record is not None:
                save_recording(
                    args.record,
                    args.label,
                    args.cfg,
                    range_m,
                    background_clip,
                    records,
                )
            if not args.no_config:
                print("> sensorStop 0")
                stop_and_drain(port)


if __name__ == "__main__":
    main()
