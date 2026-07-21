#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import numpy as np
import serial

from box_lab_common import (
    PRESENCE_INPUT_TYPE,
    now_text,
    read_box_data_frame,
    safe_label,
    timestamp,
    write_json,
)
from get_range_profile import load_configuration, parse_range_config, send_configuration
from near_field_gesture_viewer import (
    remove_leading_sensor_stop,
    stop_and_drain,
    warm_reset_demo,
)


DEFAULT_CFG = Path("xwrL64xx-evm/near_field_hand_50cm.cfg")
DEFAULT_LABELS = ("empty", "object")

FIELDNAMES = [
    "dataset_name",
    "label",
    "input_type",
    "trail_index",
    "attempt_index",
    "duration_s",
    "capture_duration_s",
    "frame_count",
    "mean_frame_rate_hz",
    "range_bin_count",
    "range_bin_spacing_m",
    "mean_point_count",
    "max_point_count",
    "npz_path",
    "session_dir",
    "started_at",
    "finished_at",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect moving-box binary empty/object mmWave recordings."
    )
    parser.add_argument("--port", required=True, help="EVM CLI/data serial port.")
    parser.add_argument("--cfg", type=Path, default=DEFAULT_CFG)
    parser.add_argument("--baud", type=int, default=115200)
    parser.add_argument(
        "--label",
        action="append",
        help="Condition label. Can be repeated or comma separated. Default: empty,object.",
    )
    parser.add_argument(
        "--trails",
        "--trials",
        "--trails",
        dest="trails",
        metavar="trails",
        type=int,
        default=3,
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=30.0,
        help="Seconds of raw data saved for each occasion.",
    )
    parser.add_argument(
        "--background-frames",
        type=int,
        default=30,
        help="Empty-scene frames captured once before all trails.",
    )
    parser.add_argument(
        "--min-frames",
        type=int,
        default=20,
        help="Minimum valid frames required before an occasion can be kept.",
    )
    parser.add_argument(
        "--dataset-name",
        default=f"box_presence_dataset_{timestamp()}",
    )
    parser.add_argument("--out-root", default="datasets")
    parser.add_argument("--frame-timeout", type=float, default=5.0)
    parser.add_argument("--auto-accept", action="store_true")
    parser.add_argument("--no-config", action="store_true")
    parser.add_argument("--no-warm-reset", action="store_true")
    return parser.parse_args()


def normalize_labels(values: list[str] | None) -> list[str]:
    if not values:
        return list(DEFAULT_LABELS)

    labels: list[str] = []
    for value in values:
        for part in value.split(","):
            label = part.strip()
            if label:
                labels.append(label)
    if not labels:
        raise SystemExit("At least one label is required.")
    return labels


def append_manifest(manifest_path: Path, row: dict) -> None:
    exists = manifest_path.exists()
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("a", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDNAMES)
        if not exists:
            writer.writeheader()
        writer.writerow({name: row.get(name, "") for name in FIELDNAMES})


def prompt_background_ready(frame_count: int) -> None:
    print()
    print("=" * 56)
    print("Background calibration")
    print("Remove the box from the radar view and keep the scene still.")
    input(f"Press Enter to capture {frame_count} empty-scene background frames...")


def prompt_ready(label: str, trail_index: int, trails: int, duration_s: float) -> None:
    print()
    print("=" * 56)
    print(f"Condition: {label}")
    print(f"Occasion {trail_index}/{trails}")
    print("Bring the closed box into view, then slowly move it during capture.")
    print("Change distance and direction gently; avoid fast gestures.")
    input(f"Press Enter to record {duration_s:.0f} seconds...")


def prompt_keep_recording() -> bool:
    while True:
        answer = input("Keep this recording? [Y/n] ").strip().lower()
        if answer in {"", "y", "yes"}:
            return True
        if answer in {"n", "no", "r", "redo"}:
            return False
        print("Please answer Y or N.")


def cloud_xyz(cloud) -> np.ndarray:
    if len(cloud.x) == 0:
        return np.empty((0, 3), dtype=float)
    return np.column_stack((cloud.x, cloud.y, cloud.z))


def capture_background(
    port: serial.Serial,
    frame_count: int,
    frame_timeout_s: float,
    expected_range_profile_bytes: int,
) -> np.ndarray:
    frames: list[np.ndarray] = []
    consecutive_warnings = 0

    while len(frames) < frame_count:
        try:
            _frame_number, profile, _cloud = read_box_data_frame(
                port,
                frame_timeout_s,
                expected_range_profile_bytes,
            )
        except (TimeoutError, ValueError, RuntimeError) as error:
            consecutive_warnings += 1
            print(f"\nBackground frame warning: {error}", flush=True)
            if consecutive_warnings >= 3:
                raise RuntimeError("Could not capture a stable background.") from error
            continue

        consecutive_warnings = 0
        frames.append(profile.astype(float))
        print(f"\rBackground frames: {len(frames)}/{frame_count}", end="", flush=True)

    print()
    return np.median(np.vstack(frames), axis=0)


def capture_trail(
    port: serial.Serial,
    duration_s: float,
    frame_timeout_s: float,
    expected_range_profile_bytes: int,
) -> dict:
    frame_number: list[int] = []
    time_s: list[float] = []
    range_profile: list[np.ndarray] = []
    point_frames: list[np.ndarray] = []
    consecutive_warnings = 0
    start = time.monotonic()
    next_status = start

    while time.monotonic() - start < duration_s:
        try:
            number, profile, cloud = read_box_data_frame(
                port,
                frame_timeout_s,
                expected_range_profile_bytes,
            )
        except (TimeoutError, ValueError, RuntimeError) as error:
            consecutive_warnings += 1
            print(f"\nFrame parse warning: {error}", flush=True)
            if consecutive_warnings >= 3:
                raise RuntimeError("No valid frames received.") from error
            continue

        consecutive_warnings = 0
        now = time.monotonic()
        elapsed = now - start
        frame_number.append(number)
        time_s.append(elapsed)
        range_profile.append(profile)
        point_frames.append(cloud_xyz(cloud))

        if now >= next_status:
            remaining = max(0.0, duration_s - elapsed)
            print(
                f"\rRecording: {elapsed:5.1f}/{duration_s:.1f} s, "
                f"{len(frame_number)} frames, {remaining:4.1f} s left",
                end="",
                flush=True,
            )
            next_status = now + 1.0

    print()
    point_count = np.array([len(points) for points in point_frames], dtype=np.uint16)
    max_points = int(np.max(point_count)) if len(point_count) else 0
    points_xyz = np.full((len(point_frames), max_points, 3), np.nan, dtype=float)
    for index, points in enumerate(point_frames):
        if len(points):
            points_xyz[index, : len(points), :] = points

    return {
        "frame_number": np.array(frame_number, dtype=np.uint32),
        "time_s": np.array(time_s, dtype=float),
        "range_profile": np.vstack(range_profile) if range_profile else np.empty((0, 0)),
        "point_count": point_count,
        "points_xyz": points_xyz,
        "capture_duration_s": float(time.monotonic() - start),
    }


def save_trail(
    dataset_dir: Path,
    args: argparse.Namespace,
    label: str,
    trail_index: int,
    attempt_index: int,
    capture: dict,
    started_at: str,
    finished_at: str,
    range_m: np.ndarray,
    range_background: np.ndarray,
    range_config,
    cfg_text: str,
) -> dict:
    session_name = f"box_presence_{safe_label(label)}_{trail_index:03d}"
    session_dir = dataset_dir / "sessions" / session_name
    session_dir.mkdir(parents=True, exist_ok=True)
    npz_path = session_dir / "trial_data.npz"

    frame_count = len(capture["frame_number"])
    duration_s = max(float(capture["capture_duration_s"]), 1e-9)
    point_count = np.asarray(capture["point_count"], dtype=np.uint16)
    mean_point_count = float(np.mean(point_count)) if len(point_count) else 0.0
    max_point_count = int(np.max(point_count)) if len(point_count) else 0

    np.savez_compressed(
        npz_path,
        dataset_name=np.array(args.dataset_name),
        label=np.array(label),
        input_type=np.array(PRESENCE_INPUT_TYPE),
        cfg_path=np.array(str(args.cfg)),
        cfg_text=np.array(cfg_text),
        created_unix_s=np.array(time.time()),
        frame_number=capture["frame_number"],
        time_s=capture["time_s"],
        range_m=range_m,
        range_background=range_background,
        range_profile=capture["range_profile"],
        point_count=point_count,
        points_xyz=capture["points_xyz"],
    )

    metadata = {
        "dataset_name": args.dataset_name,
        "label": label,
        "input_type": PRESENCE_INPUT_TYPE,
        "trail_index": trail_index,
        "attempt_index": attempt_index,
        "duration_s": args.duration,
        "capture_duration_s": duration_s,
        "frame_count": frame_count,
        "mean_frame_rate_hz": frame_count / duration_s,
        "range_bin_count": int(range_config.num_range_bins),
        "range_bin_spacing_m": float(range_config.bin_spacing_m),
        "range_background": "median empty-scene profile captured once before all trails",
        "mean_point_count": mean_point_count,
        "max_point_count": max_point_count,
        "npz_path": str(npz_path),
        "session_dir": str(session_dir),
        "started_at": started_at,
        "finished_at": finished_at,
    }
    write_json(session_dir / "trial_metadata.json", metadata)
    return metadata


def make_dataset_metadata(args: argparse.Namespace, labels: list[str], range_config) -> dict:
    return {
        "dataset_name": args.dataset_name,
        "labels": labels,
        "input_type": PRESENCE_INPUT_TYPE,
        "collection_mode": "moving_closed_box_long_recordings",
        "trails_per_label": args.trails,
        "duration_s": args.duration,
        "background_frames": args.background_frames,
        "range_bin_count": int(range_config.num_range_bins),
        "range_bin_spacing_m": float(range_config.bin_spacing_m),
        "cfg_path": str(args.cfg),
        "created_at": now_text(),
        "npz_arrays": {
            "time_s": "frame timestamps relative to occasion start",
            "range_m": "range axis for range_profile columns",
            "range_background": "median empty-scene profile captured once per dataset",
            "range_profile": "raw per-frame range-profile TLV, shape frames x bins",
            "point_count": "number of valid point-cloud rows per frame",
            "points_xyz": "NaN-padded firmware point cloud; saved for inspection, not used by default features",
        },
    }


def main() -> int:
    args = parse_args()
    labels = normalize_labels(args.label)
    if args.trails < 1:
        raise SystemExit("--trails must be at least 1.")
    if args.duration <= 0:
        raise SystemExit("--duration must be positive.")
    if args.background_frames < 1:
        raise SystemExit("--background-frames must be at least 1.")
    if args.min_frames < 1:
        raise SystemExit("--min-frames must be at least 1.")

    commands = load_configuration(args.cfg)
    range_config = parse_range_config(commands)
    if range_config is None:
        raise SystemExit("Could not compute range-bin spacing from cfg.")

    expected_bytes = range_config.num_range_bins * 4
    range_m = np.arange(range_config.num_range_bins) * range_config.bin_spacing_m
    cfg_text = args.cfg.read_text(encoding="utf-8", errors="ignore")

    dataset_dir = Path(args.out_root).expanduser().resolve() / args.dataset_name
    manifest_path = dataset_dir / "trials.csv"
    background_path = dataset_dir / "background.npz"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    write_json(
        dataset_dir / "dataset_metadata.json",
        make_dataset_metadata(args, labels, range_config),
    )

    print(f"Dataset folder: {dataset_dir}")
    print(f"Manifest: {manifest_path}")
    print(f"Labels: {', '.join(labels)}")
    print(
        "Range bins: "
        f"{range_config.num_range_bins}, spacing {range_config.bin_spacing_m:.4f} m"
    )

    try:
        serial_port = serial.Serial(args.port, args.baud, timeout=0.2)
    except serial.SerialException as error:
        raise SystemExit(f"Could not open serial port {args.port}: {error}") from None

    interrupted = False
    accepted_total = 0
    with serial_port as port:
        if args.no_config:
            port.reset_input_buffer()
        else:
            print(f"Using cfg: {args.cfg}")
            stop_and_drain(port)
            if not args.no_warm_reset:
                warm_reset_demo(port)
            try:
                send_configuration(
                    port,
                    remove_leading_sensor_stop(commands),
                    use_cfg_baud_rate=False,
                )
            except (RuntimeError, ValueError) as error:
                raise SystemExit(f"Could not configure radar: {error}") from None

        try:
            prompt_background_ready(args.background_frames)
            background_started_at = now_text()
            range_background = capture_background(
                port,
                args.background_frames,
                args.frame_timeout,
                expected_bytes,
            )
            background_finished_at = now_text()
            np.savez_compressed(
                background_path,
                range_m=range_m,
                range_background=range_background,
                background_frames=np.array(args.background_frames, dtype=np.uint16),
                started_at=np.array(background_started_at),
                finished_at=np.array(background_finished_at),
            )
            print(f"Background saved: {background_path}")

            for label in labels:
                trail_index = 1
                while trail_index <= args.trails:
                    attempt_index = 1
                    while True:
                        prompt_ready(label, trail_index, args.trails, args.duration)
                        started_at = now_text()
                        capture = capture_trail(
                            port,
                            args.duration,
                            args.frame_timeout,
                            expected_bytes,
                        )
                        finished_at = now_text()
                        frame_count = len(capture["frame_number"])
                        fps = frame_count / max(float(capture["capture_duration_s"]), 1e-9)
                        print(f"Captured {frame_count} frames at {fps:.1f} Hz.")

                        if frame_count < args.min_frames:
                            print(f"Discarded: only {frame_count} valid frames.")
                            attempt_index += 1
                            continue

                        keep = args.auto_accept or prompt_keep_recording()
                        if keep:
                            row = save_trail(
                                dataset_dir,
                                args,
                                label,
                                trail_index,
                                attempt_index,
                                capture,
                                started_at,
                                finished_at,
                                range_m,
                                range_background,
                                range_config,
                                cfg_text,
                            )
                            append_manifest(manifest_path, row)
                            accepted_total += 1
                            print(
                                f"Saved {label} occasion {trail_index}/"
                                f"{args.trails}: {row['session_dir']}"
                            )
                            trail_index += 1
                            break

                        print("Discarded recording. Redoing the same occasion.")
                        attempt_index += 1

        except KeyboardInterrupt:
            interrupted = True
            print("\nInterrupted. Stopping radar...")
        except RuntimeError as error:
            interrupted = True
            print(f"\nCapture failed: {error}")
        finally:
            if not args.no_config:
                print("> sensorStop 0")
                stop_and_drain(port)

    print(f"Accepted trails: {accepted_total}")
    print(f"Dataset complete: {dataset_dir}")
    return 1 if interrupted else 0


if __name__ == "__main__":
    raise SystemExit(main())
