#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import numpy as np
import serial

from box_lab_common import (
    INPUT_TYPE,
    filter_point_slice,
    now_text,
    pack_slice_points,
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
DEFAULT_CONTENTS = ("empty", "board", "marker_pen")

FIELDNAMES = [
    "dataset_name",
    "contents",
    "input_type",
    "trial_index",
    "attempt_index",
    "frames_per_trial",
    "frame_count",
    "mean_frame_rate_hz",
    "range_bin_count",
    "range_bin_spacing_m",
    "point_distance_m",
    "point_window_m",
    "mean_slice_point_count",
    "max_slice_point_count",
    "npz_path",
    "session_dir",
    "started_at",
    "finished_at",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect static mmWave box-content classification trials."
    )
    parser.add_argument("--port", required=True, help="EVM CLI/data serial port.")
    parser.add_argument("--cfg", type=Path, default=DEFAULT_CFG)
    parser.add_argument("--baud", type=int, default=115200)
    parser.add_argument(
        "--contents",
        "--item",
        action="append",
        help=(
            "Box content label. Can be repeated or comma separated. "
            "Default: empty,board,marker_pen."
        ),
    )
    parser.add_argument("--trials", "--trails", type=int, default=5)
    parser.add_argument(
        "--frames",
        type=int,
        default=20,
        help="Valid radar frames captured for each trial.",
    )
    parser.add_argument(
        "--background-frames",
        type=int,
        default=20,
        help="Empty-box frames captured once before all trials.",
    )
    parser.add_argument(
        "--point-distance",
        type=float,
        default=0.20,
        help="Forward y-distance for the saved point-cloud slice, in meters.",
    )
    parser.add_argument(
        "--point-window",
        type=float,
        default=0.05,
        help="Half-width around --point-distance for point-cloud slicing.",
    )
    parser.add_argument(
        "--x-limit",
        type=float,
        default=0.30,
        help="Optional left/right point slice limit. Use 0 to disable.",
    )
    parser.add_argument(
        "--z-limit",
        type=float,
        default=0.30,
        help="Accepted for old commands; ignored because point-cloud features are 2D x/y.",
    )
    parser.add_argument(
        "--dataset-name",
        default=f"box_contents_dataset_{timestamp()}",
    )
    parser.add_argument("--out-root", default="datasets")
    parser.add_argument("--frame-timeout", type=float, default=5.0)
    parser.add_argument("--auto-accept", action="store_true")
    parser.add_argument("--no-config", action="store_true")
    parser.add_argument("--no-warm-reset", action="store_true")
    return parser.parse_args()


def normalize_contents(values: list[str] | None) -> list[str]:
    if not values:
        return list(DEFAULT_CONTENTS)
    labels: list[str] = []
    for value in values:
        for part in value.split(","):
            label = part.strip()
            if label:
                labels.append(label)
    if not labels:
        raise SystemExit("At least one content label is required.")
    return labels


def append_manifest(manifest_path: Path, row: dict) -> None:
    exists = manifest_path.exists()
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("a", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDNAMES)
        if not exists:
            writer.writeheader()
        writer.writerow({name: row.get(name, "") for name in FIELDNAMES})


def prompt_ready(contents: str, trial_index: int, trials: int) -> None:
    print()
    print("=" * 56)
    print(f"Set box contents: {contents}")
    print(f"Trial {trial_index}/{trials}: open the box, place the object, close the box.")
    input("Press Enter when the box is closed and the scene is still...")


def prompt_keep_trial() -> bool:
    while True:
        answer = input("Keep this trial? [Y/n] ").strip().lower()
        if answer in {"", "y", "yes"}:
            return True
        if answer in {"n", "no", "r", "redo"}:
            return False
        print("Please answer Y or N.")


def prompt_background_ready(frame_count: int) -> None:
    print()
    print("=" * 56)
    print("Empty box calibration")
    print("Remove objects, close the empty box, and keep the scene still.")
    input(f"Press Enter to capture {frame_count} empty-box background frames...")


def capture_range_background(
    port: serial.Serial,
    frame_count: int,
    frame_timeout_s: float,
    expected_range_profile_bytes: int,
) -> dict:
    frame_numbers: list[int] = []
    time_s: list[float] = []
    range_profiles: list[np.ndarray] = []
    consecutive_warnings = 0
    start = time.monotonic()

    while len(frame_numbers) < frame_count:
        try:
            frame_number, profile, _cloud = read_box_data_frame(
                port,
                frame_timeout_s,
                expected_range_profile_bytes,
            )
        except (TimeoutError, ValueError, RuntimeError) as error:
            consecutive_warnings += 1
            print(f"\nBackground frame warning: {error}", flush=True)
            if consecutive_warnings >= 3:
                raise RuntimeError("Could not capture empty-box background.") from error
            continue

        consecutive_warnings = 0
        frame_numbers.append(frame_number)
        time_s.append(time.monotonic() - start)
        range_profiles.append(profile)
        print(f"\rBackground frames: {len(frame_numbers)}/{frame_count}", end="", flush=True)

    print()
    range_profile = np.vstack(range_profiles)
    return {
        "frame_number": np.array(frame_numbers, dtype=np.uint32),
        "time_s": np.array(time_s, dtype=float),
        "range_profile": range_profile,
        "range_background": np.mean(range_profile, axis=0),
    }


def capture_box_trial(
    port: serial.Serial,
    frame_count: int,
    frame_timeout_s: float,
    expected_range_profile_bytes: int,
    point_distance_m: float,
    point_window_m: float,
    x_limit_m: float,
    z_limit_m: float,
) -> dict:
    frame_numbers: list[int] = []
    time_s: list[float] = []
    range_profiles: list[np.ndarray] = []
    point_frames: list[np.ndarray] = []
    consecutive_warnings = 0
    start = time.monotonic()

    while len(frame_numbers) < frame_count:
        try:
            frame_number, profile, cloud = read_box_data_frame(
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
        frame_numbers.append(frame_number)
        time_s.append(time.monotonic() - start)
        range_profiles.append(profile)
        point_frames.append(
            filter_point_slice(
                cloud,
                point_distance_m,
                point_window_m,
                x_limit_m if x_limit_m > 0 else None,
                z_limit_m if z_limit_m > 0 else None,
            )
        )
        print(f"\rFrames: {len(frame_numbers)}/{frame_count}", end="", flush=True)

    print()
    slice_point_count, slice_points_xyz = pack_slice_points(point_frames)
    range_profile = np.vstack(range_profiles)
    return {
        "frame_number": np.array(frame_numbers, dtype=np.uint32),
        "time_s": np.array(time_s, dtype=float),
        "range_profile": range_profile,
        "mean_range_profile": np.mean(range_profile, axis=0),
        "slice_point_count": slice_point_count,
        "slice_points_xyz": slice_points_xyz,
        "capture_duration_s": float(time.monotonic() - start),
    }


def save_trial(
    dataset_dir: Path,
    args: argparse.Namespace,
    contents: str,
    trial_index: int,
    attempt_index: int,
    capture: dict,
    started_at: str,
    finished_at: str,
    range_m: np.ndarray,
    range_background: np.ndarray,
    range_config,
    cfg_text: str,
) -> dict:
    session_name = f"box_{safe_label(contents)}_trial_{trial_index:03d}"
    session_dir = dataset_dir / "sessions" / session_name
    session_dir.mkdir(parents=True, exist_ok=True)
    npz_path = session_dir / "trial_data.npz"

    frame_count = len(capture["frame_number"])
    duration_s = max(float(capture["capture_duration_s"]), 1e-9)
    point_counts = np.asarray(capture["slice_point_count"], dtype=np.uint16)
    mean_point_count = float(np.mean(point_counts)) if len(point_counts) else 0.0
    max_point_count = int(np.max(point_counts)) if len(point_counts) else 0

    np.savez_compressed(
        npz_path,
        dataset_name=np.array(args.dataset_name),
        contents=np.array(contents),
        input_type=np.array(INPUT_TYPE),
        cfg_path=np.array(str(args.cfg)),
        cfg_text=np.array(cfg_text),
        created_unix_s=np.array(time.time()),
        frame_number=capture["frame_number"],
        time_s=capture["time_s"],
        range_m=range_m,
        range_background=range_background,
        range_profile=capture["range_profile"],
        mean_range_profile=capture["mean_range_profile"],
        point_distance_m=np.array(args.point_distance),
        point_window_m=np.array(args.point_window),
        slice_point_count=point_counts,
        slice_points_xyz=capture["slice_points_xyz"],
    )

    metadata = {
        "dataset_name": args.dataset_name,
        "contents": contents,
        "input_type": INPUT_TYPE,
        "trial_index": trial_index,
        "attempt_index": attempt_index,
        "frames_per_trial": args.frames,
        "frame_count": frame_count,
        "mean_frame_rate_hz": frame_count / duration_s,
        "range_bin_count": int(range_config.num_range_bins),
        "range_bin_spacing_m": float(range_config.bin_spacing_m),
        "range_background": "mean empty-box profile captured once before all trials",
        "point_distance_m": args.point_distance,
        "point_window_m": args.point_window,
        "mean_slice_point_count": mean_point_count,
        "max_slice_point_count": max_point_count,
        "npz_path": str(npz_path),
        "session_dir": str(session_dir),
        "started_at": started_at,
        "finished_at": finished_at,
    }
    write_json(session_dir / "trial_metadata.json", metadata)
    return metadata


def make_dataset_metadata(args: argparse.Namespace, contents: list[str], range_config) -> dict:
    return {
        "dataset_name": args.dataset_name,
        "contents": contents,
        "input_type": INPUT_TYPE,
        "collection_mode": "fixed_box_static_frame_average",
        "trials_per_content": args.trials,
        "frames_per_trial": args.frames,
        "background_frames": args.background_frames,
        "point_distance_m": args.point_distance,
        "point_window_m": args.point_window,
        "x_limit_m": args.x_limit,
        "z_limit_m": args.z_limit,
        "point_cloud_axes": "2D x/y; raw z is saved but ignored by features",
        "range_bin_count": int(range_config.num_range_bins),
        "range_bin_spacing_m": float(range_config.bin_spacing_m),
        "cfg_path": str(args.cfg),
        "created_at": now_text(),
        "npz_arrays": {
            "range_background": "mean empty-box profile captured once before all trials",
            "mean_range_profile": "mean raw range profile over the trial frames",
            "range_profile": "raw per-frame range profile, shape frames x bins",
            "slice_points_xyz": "NaN-padded point cloud near point_distance_m; z retained but ignored",
            "slice_point_count": "valid point count per frame in slice_points_xyz",
        },
    }


def main() -> int:
    args = parse_args()
    contents_labels = normalize_contents(args.contents)
    if args.trials < 1:
        raise SystemExit("--trials must be at least 1.")
    if args.frames < 1:
        raise SystemExit("--frames must be at least 1.")
    if args.background_frames < 1:
        raise SystemExit("--background-frames must be at least 1.")
    if args.point_window <= 0:
        raise SystemExit("--point-window must be positive.")

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
        make_dataset_metadata(args, contents_labels, range_config),
    )

    print(f"Dataset folder: {dataset_dir}")
    print(f"Manifest: {manifest_path}")
    print(f"Contents labels: {', '.join(contents_labels)}")
    print(
        "Point slice: "
        f"y={args.point_distance:.2f} +/- {args.point_window:.2f} m"
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
            background = capture_range_background(
                port,
                args.background_frames,
                args.frame_timeout,
                expected_bytes,
            )
            range_background = background["range_background"]
            np.savez_compressed(
                background_path,
                range_m=range_m,
                range_background=range_background,
                frame_number=background["frame_number"],
                time_s=background["time_s"],
                range_profile=background["range_profile"],
            )
            print(f"Background saved: {background_path}")

            for contents in contents_labels:
                trial_index = 1
                while trial_index <= args.trials:
                    attempt_index = 1
                    while True:
                        prompt_ready(contents, trial_index, args.trials)
                        started_at = now_text()
                        capture = capture_box_trial(
                            port,
                            args.frames,
                            args.frame_timeout,
                            expected_bytes,
                            args.point_distance,
                            args.point_window,
                            args.x_limit,
                            args.z_limit,
                        )
                        finished_at = now_text()
                        print(
                            "Captured "
                            f"{len(capture['frame_number'])} frames, "
                            f"mean slice points {np.mean(capture['slice_point_count']):.1f}."
                        )

                        keep = args.auto_accept or prompt_keep_trial()
                        if keep:
                            row = save_trial(
                                dataset_dir,
                                args,
                                contents,
                                trial_index,
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
                                f"Saved {contents} trial {trial_index}/{args.trials}: "
                                f"{row['session_dir']}"
                            )
                            trial_index += 1
                            break

                        print("Discarded trial. Redoing the same content/trial.")
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
