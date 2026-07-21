#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import serial

from box_lab_common import (
    INPUT_TYPE,
    SESSIONS_DIR,
    extract_box_feature_vector,
    filter_point_slice,
    now_text,
    pack_slice_points,
    prediction_confidence,
    prediction_scores,
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Classify static box contents from short mmWave captures."
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--port", required=True)
    parser.add_argument("--cfg", type=Path, default=DEFAULT_CFG)
    parser.add_argument("--baud", type=int, default=115200)
    parser.add_argument("--frames", type=int, help="Default: 20, or model collection setting.")
    parser.add_argument(
        "--background-frames",
        type=int,
        help="Empty-box frames for calibration. Default: same as --frames.",
    )
    parser.add_argument("--point-distance", type=float, help="Default: model setting, or 0.20 m.")
    parser.add_argument("--point-window", type=float, help="Default: model setting, or 0.05 m.")
    parser.add_argument("--x-limit", type=float, default=0.30)
    parser.add_argument(
        "--z-limit",
        type=float,
        default=0.30,
        help="Accepted for old commands; ignored because point-cloud features are 2D x/y.",
    )
    parser.add_argument("--frame-timeout", type=float, default=5.0)
    parser.add_argument("--out-root", default=str(SESSIONS_DIR))
    parser.add_argument("--session-name")
    parser.add_argument(
        "--continuous",
        action="store_true",
        help="Keep classifying every --interval seconds without prompting.",
    )
    parser.add_argument("--interval", type=float, default=1.0)
    parser.add_argument("--no-config", action="store_true")
    parser.add_argument("--no-warm-reset", action="store_true")
    parser.add_argument("--no-save-captures", action="store_true")
    return parser.parse_args()


def cloud_xyz(cloud) -> np.ndarray:
    if len(cloud.x) == 0:
        return np.empty((0, 3), dtype=float)
    return np.column_stack((cloud.x, cloud.y, cloud.z))


def capture_box_window(
    port: serial.Serial,
    frame_count: int,
    frame_timeout_s: float,
    expected_range_profile_bytes: int,
    range_m: np.ndarray,
    point_distance_m: float,
    point_window_m: float,
    x_limit_m: float,
    z_limit_m: float,
    range_background: np.ndarray | None = None,
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

    slice_point_count, slice_points_xyz = pack_slice_points(point_frames)
    range_profile = np.vstack(range_profiles)
    capture = {
        "frame_number": np.array(frame_numbers, dtype=np.uint32),
        "time_s": np.array(time_s, dtype=float),
        "range_m": range_m,
        "range_profile": range_profile,
        "mean_range_profile": np.mean(range_profile, axis=0),
        "point_distance_m": np.array(point_distance_m),
        "point_window_m": np.array(point_window_m),
        "slice_point_count": slice_point_count,
        "slice_points_xyz": slice_points_xyz,
    }
    if range_background is not None:
        capture["range_background"] = range_background
    return capture


def calibrate_empty_box(
    port: serial.Serial,
    frame_count: int,
    frame_timeout_s: float,
    expected_range_profile_bytes: int,
    range_m: np.ndarray,
    point_distance_m: float,
    point_window_m: float,
    x_limit_m: float,
    z_limit_m: float,
    prompt: bool,
) -> np.ndarray:
    if prompt:
        print()
        print("=" * 56)
        print("Empty box calibration")
        print("Remove objects, close the empty box, and keep the scene still.")
        input(f"Press Enter to capture {frame_count} empty-box frames...")
    else:
        print(f"Capturing {frame_count} empty-box calibration frames...")

    capture = capture_box_window(
        port,
        frame_count,
        frame_timeout_s,
        expected_range_profile_bytes,
        range_m,
        point_distance_m,
        point_window_m,
        x_limit_m,
        z_limit_m,
    )
    return np.asarray(capture["mean_range_profile"], dtype=float)


def format_scores(scores: list[tuple[str, float]]) -> str:
    if not scores:
        return "scores unavailable"
    return ", ".join(f"{label}={score:.3f}" for label, score in scores)


def save_capture(path: Path, capture: dict, prediction: str, confidence: float | None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        prediction=np.array(prediction),
        confidence=np.array(np.nan if confidence is None else confidence),
        created_unix_s=np.array(time.time()),
        **capture,
    )


def main() -> int:
    args = parse_args()
    try:
        import joblib
    except ImportError as exc:
        raise SystemExit(f"Missing joblib/sklearn environment: {exc}") from exc

    model_path = Path(args.model).expanduser().resolve()
    payload = joblib.load(model_path)
    input_type = payload.get("input_type", INPUT_TYPE)
    if input_type != INPUT_TYPE:
        raise SystemExit(f"This evaluator cannot run model input_type: {input_type}")

    model = payload["model"]
    feature_params = payload.get("feature_params") or {}
    feature_names = payload.get("feature_names") or []
    classifier_label = payload.get("classifier_label", payload.get("classifier", "unknown"))
    frame_count = int(args.frames or payload.get("frames_per_trial", 20) or 20)
    background_frames = int(args.background_frames or frame_count)
    point_distance = float(
        args.point_distance
        if args.point_distance is not None
        else payload.get("point_distance_m", 0.20)
    )
    point_window = float(
        args.point_window
        if args.point_window is not None
        else payload.get("point_window_m", 0.05)
    )

    if frame_count < 1:
        raise SystemExit("--frames must be at least 1.")
    if background_frames < 1:
        raise SystemExit("--background-frames must be at least 1.")
    if point_window <= 0:
        raise SystemExit("--point-window must be positive.")

    commands = load_configuration(args.cfg)
    range_config = parse_range_config(commands)
    if range_config is None:
        raise SystemExit("Could not compute range-bin spacing from cfg.")
    expected_bytes = range_config.num_range_bins * 4
    range_m = np.arange(range_config.num_range_bins) * range_config.bin_spacing_m

    session_name = args.session_name or f"box_eval_{safe_label(model_path.stem)}_{timestamp()}"
    session_dir = Path(args.out_root).expanduser().resolve() / session_name
    session_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = session_dir / "box_predictions.csv"
    write_json(
        session_dir / "session_metadata.json",
        {
            "kind": "eval_box_realtime",
            "input_type": INPUT_TYPE,
            "model": str(model_path),
            "classifier_label": classifier_label,
            "feature_params": feature_params,
            "frames": frame_count,
            "background_frames": background_frames,
            "point_distance_m": point_distance,
            "point_window_m": point_window,
            "point_cloud_axes": "2D x/y; raw z is ignored by features",
            "cfg_path": str(args.cfg),
            "started_at": now_text(),
        },
    )

    print(f"Session folder: {session_dir}")
    print(f"Loaded classifier: {classifier_label}")
    print(f"Frames per prediction: {frame_count}")
    print(f"Point slice: y={point_distance:.2f} +/- {point_window:.2f} m")

    try:
        serial_port = serial.Serial(args.port, args.baud, timeout=0.2)
    except serial.SerialException as error:
        raise SystemExit(f"Could not open serial port {args.port}: {error}") from None

    interrupted = False
    prediction_count = 0
    with predictions_path.open("w", newline="") as prediction_file:
        writer = csv.DictWriter(
            prediction_file,
            fieldnames=[
                "index",
                "time_s",
                "prediction",
                "confidence",
                "scores_json",
                "mean_slice_point_count",
                "capture_npz",
            ],
        )
        writer.writeheader()
        start = time.monotonic()

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
                range_background = calibrate_empty_box(
                    port,
                    background_frames,
                    args.frame_timeout,
                    expected_bytes,
                    range_m,
                    point_distance,
                    point_window,
                    args.x_limit,
                    args.z_limit,
                    prompt=not args.continuous,
                )
                print("Empty-box calibration complete.")

                while True:
                    if args.continuous:
                        time.sleep(max(0.0, args.interval))
                    else:
                        print()
                        command = input(
                            "[Enter]=classify, b=empty-box calibration, q=quit: "
                        ).strip().lower()
                        if command in {"q", "quit", "exit"}:
                            break
                        if command in {"b", "bg", "background", "cal", "calibrate"}:
                            range_background = calibrate_empty_box(
                                port,
                                background_frames,
                                args.frame_timeout,
                                expected_bytes,
                                range_m,
                                point_distance,
                                point_window,
                                args.x_limit,
                                args.z_limit,
                                prompt=True,
                            )
                            print("Empty-box calibration updated.")
                            continue

                    capture = capture_box_window(
                        port,
                        frame_count,
                        args.frame_timeout,
                        expected_bytes,
                        range_m,
                        point_distance,
                        point_window,
                        args.x_limit,
                        args.z_limit,
                        range_background,
                    )
                    features, names = extract_box_feature_vector(capture, feature_params)
                    if features is None:
                        print("Could not extract features from this capture.")
                        continue
                    if feature_names and len(features) != len(feature_names):
                        print(
                            "Feature length mismatch: "
                            f"model expects {len(feature_names)}, got {len(features)}."
                        )
                        continue

                    prediction = model.predict([features])[0]
                    confidence = prediction_confidence(model, features, prediction)
                    scores = prediction_scores(model, features)
                    prediction_count += 1
                    capture_path = ""
                    if not args.no_save_captures:
                        path = session_dir / "captures" / f"capture_{prediction_count:04d}.npz"
                        save_capture(path, capture, str(prediction), confidence)
                        capture_path = str(path)

                    confidence_text = "" if confidence is None else f" ({confidence:.2f})"
                    print(f"prediction: {prediction}{confidence_text}")
                    print(f"scores: {format_scores(scores)}")
                    writer.writerow(
                        {
                            "index": prediction_count,
                            "time_s": f"{time.monotonic() - start:.3f}",
                            "prediction": prediction,
                            "confidence": "" if confidence is None else f"{confidence:.4f}",
                            "scores_json": json.dumps(scores),
                            "mean_slice_point_count": f"{np.mean(capture['slice_point_count']):.3f}",
                            "capture_npz": capture_path,
                        }
                    )
                    prediction_file.flush()

            except KeyboardInterrupt:
                interrupted = True
                print("\nStopping.")
            except RuntimeError as error:
                interrupted = True
                print(f"\nEvaluation failed: {error}")
            finally:
                if not args.no_config:
                    print("> sensorStop 0")
                    stop_and_drain(port)

    write_json(
        session_dir / "eval_box_summary.json",
        {
            "prediction_count": prediction_count,
            "predictions_csv": str(predictions_path),
            "interrupted": interrupted,
            "finished_at": now_text(),
        },
    )
    print(f"Predictions saved to: {predictions_path}")
    return 1 if interrupted else 0


if __name__ == "__main__":
    raise SystemExit(main())
