#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import numpy as np
import serial

from get_range_profile import load_configuration, parse_range_config, send_configuration
from near_field_gesture_viewer import (
    remove_leading_sensor_stop,
    stop_and_drain,
    warm_reset_demo,
)
from posture_lab_common import (
    INPUT_TYPE,
    cloud_arrays,
    now_text,
    pack_point_cloud_frames,
    read_point_cloud_frame,
    safe_label,
    timestamp,
    write_json,
)


DEFAULT_CFG = Path("xwrL64xx-evm/point_cloud.cfg")
DEFAULT_POSTURES = (
    "empty",
    "sitting",
    "standing",
    "standing_arms_forward",
    "squat",
)
POSTURE_INSTRUCTIONS = {
    "empty": "Keep the space in front of the radar empty.",
    "sitting": "Sit in front of the radar and slowly vary position or angle.",
    "standing": "Stand naturally in front of the radar and slowly vary position or angle.",
    "standing_arms_forward": (
        "Stand with both arms extended straight forward and slowly vary position or angle."
    ),
    "squat": "Squat or crouch in front of the radar and slowly vary position or angle.",
}

FIELDNAMES = [
    "dataset_name",
    "collector",
    "posture",
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
        description="Collect continuous point-cloud recordings for posture recognition."
    )
    parser.add_argument("--port", required=True, help="EVM CLI/data serial port.")
    parser.add_argument("--cfg", type=Path, default=DEFAULT_CFG)
    parser.add_argument("--baud", type=int, default=115200)
    parser.add_argument("--collector", required=True, help="Student or collector ID.")
    parser.add_argument(
        "--posture",
        action="append",
        help=(
            "Posture label. Can be repeated or comma separated. Default: "
            "empty,sitting,standing,standing_arms_forward,squat."
        ),
    )
    parser.add_argument(
        "--trials",
        "--trails",
        dest="trails",
        metavar="TRIALS",
        type=int,
        default=3,
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=60.0,
        help="Seconds of point-cloud data saved for each trial.",
    )
    parser.add_argument(
        "--min-frames",
        type=int,
        default=20,
        help="Minimum valid radar frames required before a trial can be kept.",
    )
    parser.add_argument(
        "--dataset-name",
        default=f"posture_dataset_{timestamp()}",
    )
    parser.add_argument("--out-root", default="datasets")
    parser.add_argument("--frame-timeout", type=float, default=5.0)
    parser.add_argument("--auto-accept", action="store_true")
    parser.add_argument("--no-config", action="store_true")
    parser.add_argument("--no-warm-reset", action="store_true")
    return parser.parse_args()


def normalize_postures(values: list[str] | None) -> list[str]:
    if not values:
        return list(DEFAULT_POSTURES)

    postures: list[str] = []
    for value in values:
        for part in value.split(","):
            label = part.strip()
            if label:
                postures.append(label)
    if not postures:
        raise SystemExit("At least one posture label is required.")
    return postures


def append_manifest(manifest_path: Path, row: dict) -> None:
    exists = manifest_path.exists()
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("a", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDNAMES)
        if not exists:
            writer.writeheader()
        writer.writerow({name: row.get(name, "") for name in FIELDNAMES})


def prompt_ready(posture: str, trail_index: int, trails: int, duration_s: float) -> None:
    print()
    print("=" * 56)
    print(f"Posture: {posture}")
    print(f"Trial {trail_index}/{trails}")
    instruction = POSTURE_INSTRUCTIONS.get(posture.lower())
    if instruction:
        print(instruction)
    else:
        print("Hold the posture in front of the radar.")
        print("Slowly vary position or angle so the point cloud covers natural variation.")
    input(f"Press Enter to record {duration_s:.0f} seconds...")


def prompt_keep_recording() -> bool:
    while True:
        answer = input("Keep this recording? [Y/n] ").strip().lower()
        if answer in {"", "y", "yes"}:
            return True
        if answer in {"n", "no", "r", "redo"}:
            return False
        print("Please answer Y or N.")


def capture_trail(
    port: serial.Serial,
    duration_s: float,
    frame_timeout_s: float,
) -> dict:
    frame_numbers: list[int] = []
    time_s: list[float] = []
    xyz_frames: list[np.ndarray] = []
    velocity_frames: list[np.ndarray] = []
    consecutive_warnings = 0
    start = time.monotonic()
    next_status = start

    while time.monotonic() - start < duration_s:
        try:
            frame_number, cloud = read_point_cloud_frame(port, frame_timeout_s)
        except (TimeoutError, ValueError, RuntimeError) as error:
            consecutive_warnings += 1
            print(f"\nFrame parse warning: {error}", flush=True)
            if consecutive_warnings >= 3:
                raise RuntimeError("No valid frames received.") from error
            continue

        consecutive_warnings = 0
        now = time.monotonic()
        elapsed = now - start
        xyz, velocity = cloud_arrays(cloud)
        frame_numbers.append(frame_number)
        time_s.append(elapsed)
        xyz_frames.append(xyz)
        velocity_frames.append(velocity)

        if now >= next_status:
            remaining = max(0.0, duration_s - elapsed)
            mean_points = float(np.mean([len(frame) for frame in xyz_frames])) if xyz_frames else 0.0
            print(
                f"\rRecording: {elapsed:5.1f}/{duration_s:.1f} s, "
                f"{len(frame_numbers)} frames, mean points {mean_points:.1f}, "
                f"{remaining:4.1f} s left",
                end="",
                flush=True,
            )
            next_status = now + 1.0

    print()
    point_count, points_xyz, points_velocity = pack_point_cloud_frames(
        xyz_frames,
        velocity_frames,
    )
    return {
        "frame_number": np.array(frame_numbers, dtype=np.uint32),
        "time_s": np.array(time_s, dtype=float),
        "point_count": point_count,
        "points_xyz": points_xyz,
        "points_velocity": points_velocity,
        "capture_duration_s": float(time.monotonic() - start),
    }


def save_trail(
    dataset_dir: Path,
    args: argparse.Namespace,
    posture: str,
    trail_index: int,
    attempt_index: int,
    capture: dict,
    started_at: str,
    finished_at: str,
    range_config,
    cfg_text: str,
) -> dict:
    session_name = (
        f"posture_{safe_label(args.collector)}_"
        f"{safe_label(posture)}_{trail_index:03d}"
    )
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
        collector=np.array(args.collector),
        posture=np.array(posture),
        input_type=np.array(INPUT_TYPE),
        cfg_path=np.array(str(args.cfg)),
        cfg_text=np.array(cfg_text),
        created_unix_s=np.array(time.time()),
        frame_number=capture["frame_number"],
        time_s=capture["time_s"],
        point_count=point_count,
        points_xyz=capture["points_xyz"],
        points_velocity=capture["points_velocity"],
    )

    metadata = {
        "dataset_name": args.dataset_name,
        "collector": args.collector,
        "posture": posture,
        "input_type": INPUT_TYPE,
        "trail_index": trail_index,
        "attempt_index": attempt_index,
        "duration_s": args.duration,
        "capture_duration_s": duration_s,
        "frame_count": frame_count,
        "mean_frame_rate_hz": frame_count / duration_s,
        "range_bin_count": int(range_config.num_range_bins),
        "range_bin_spacing_m": float(range_config.bin_spacing_m),
        "mean_point_count": mean_point_count,
        "max_point_count": max_point_count,
        "npz_path": str(npz_path),
        "session_dir": str(session_dir),
        "started_at": started_at,
        "finished_at": finished_at,
    }
    write_json(session_dir / "trial_metadata.json", metadata)
    return metadata


def make_dataset_metadata(args: argparse.Namespace, postures: list[str], range_config) -> dict:
    return {
        "dataset_name": args.dataset_name,
        "collector": args.collector,
        "postures": postures,
        "input_type": INPUT_TYPE,
        "collection_mode": "continuous_point_cloud_posture_recordings",
        "trials_per_posture": args.trails,
        "duration_s": args.duration,
        "range_bin_count": int(range_config.num_range_bins),
        "range_bin_spacing_m": float(range_config.bin_spacing_m),
        "cfg_path": str(args.cfg),
        "created_at": now_text(),
        "npz_arrays": {
            "time_s": "frame timestamps relative to trial start",
            "point_count": "number of valid point-cloud rows per frame",
            "points_xyz": "NaN-padded firmware point cloud; z is retained but ignored by 2D features",
            "points_velocity": "NaN-padded point velocity from point-cloud TLV",
        },
    }


def main() -> int:
    args = parse_args()
    postures = normalize_postures(args.posture)
    if args.trails < 1:
        raise SystemExit("--trials must be at least 1.")
    if args.duration <= 0:
        raise SystemExit("--duration must be positive.")
    if args.min_frames < 1:
        raise SystemExit("--min-frames must be at least 1.")

    commands = load_configuration(args.cfg)
    range_config = parse_range_config(commands)
    if range_config is None:
        raise SystemExit("Could not compute range-bin spacing from cfg.")
    cfg_text = args.cfg.read_text(encoding="utf-8", errors="ignore")

    dataset_dir = Path(args.out_root).expanduser().resolve() / args.dataset_name
    manifest_path = dataset_dir / "trials.csv"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    write_json(
        dataset_dir / "dataset_metadata.json",
        make_dataset_metadata(args, postures, range_config),
    )

    print(f"Dataset folder: {dataset_dir}")
    print(f"Manifest: {manifest_path}")
    print(f"Collector: {args.collector}")
    print(f"Postures: {', '.join(postures)}")
    print(
        "Range bins from cfg: "
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
            for posture in postures:
                trail_index = 1
                while trail_index <= args.trails:
                    attempt_index = 1
                    while True:
                        prompt_ready(posture, trail_index, args.trails, args.duration)
                        started_at = now_text()
                        capture = capture_trail(port, args.duration, args.frame_timeout)
                        finished_at = now_text()
                        frame_count = len(capture["frame_number"])
                        fps = frame_count / max(float(capture["capture_duration_s"]), 1e-9)
                        mean_points = (
                            float(np.mean(capture["point_count"]))
                            if len(capture["point_count"])
                            else 0.0
                        )
                        print(
                            f"Captured {frame_count} frames at {fps:.1f} Hz, "
                            f"mean points {mean_points:.1f}."
                        )

                        if frame_count < args.min_frames:
                            print(f"Discarded: only {frame_count} valid frames.")
                            attempt_index += 1
                            continue

                        keep = args.auto_accept or prompt_keep_recording()
                        if keep:
                            row = save_trail(
                                dataset_dir,
                                args,
                                posture,
                                trail_index,
                                attempt_index,
                                capture,
                                started_at,
                                finished_at,
                                range_config,
                                cfg_text,
                            )
                            append_manifest(manifest_path, row)
                            accepted_total += 1
                            print(
                                f"Saved {posture} trial {trail_index}/"
                                f"{args.trails}: {row['session_dir']}"
                            )
                            trail_index += 1
                            break

                        print("Discarded recording. Redoing the same trial.")
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

    print(f"Accepted trials: {accepted_total}")
    print(f"Dataset complete: {dataset_dir}")
    return 1 if interrupted else 0


if __name__ == "__main__":
    raise SystemExit(main())
