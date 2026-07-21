#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import time
from collections import deque
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
    SESSIONS_DIR,
    cloud_arrays,
    extract_posture_feature_vector,
    filter_posture_points,
    now_text,
    pack_point_cloud_frames,
    prediction_confidence,
    prediction_scores,
    read_point_cloud_frame,
    safe_label,
    timestamp,
    write_json,
)


DEFAULT_CFG = Path("xwrL64xx-evm/point_cloud.cfg")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run real-time posture recognition from mmWave point clouds."
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--port", required=True)
    parser.add_argument("--cfg", type=Path, default=DEFAULT_CFG)
    parser.add_argument("--baud", type=int, default=115200)
    parser.add_argument(
        "--duration",
        type=float,
        default=0.0,
        help="Evaluation duration in seconds. Use 0 to run until Ctrl+C.",
    )
    parser.add_argument("--window-seconds", type=float, help="Default: model setting.")
    parser.add_argument("--step-seconds", type=float, default=0.5)
    parser.add_argument("--min-window-frames", type=int, help="Default: model setting or 4.")
    parser.add_argument("--frame-timeout", type=float, default=5.0)
    parser.add_argument("--out-root", default=str(SESSIONS_DIR))
    parser.add_argument("--session-name")
    parser.add_argument("--no-plot", action="store_true")
    parser.add_argument("--no-save-frames", action="store_true")
    parser.add_argument("--no-config", action="store_true")
    parser.add_argument("--no-warm-reset", action="store_true")
    return parser.parse_args()


def records_to_posture_data(records: list[dict]) -> dict[str, np.ndarray] | None:
    if not records:
        return None
    point_count, points_xyz, points_velocity = pack_point_cloud_frames(
        [record["points_xyz"] for record in records],
        [record["points_velocity"] for record in records],
    )
    return {
        "frame_number": np.array([record["frame_number"] for record in records], dtype=np.uint32),
        "time_s": np.array([record["time_s"] for record in records], dtype=float),
        "point_count": point_count,
        "points_xyz": points_xyz,
        "points_velocity": points_velocity,
    }


def save_frame_stream(
    path: Path,
    records: list[dict],
    model_path: Path,
    cfg_path: Path,
) -> None:
    data = records_to_posture_data(records)
    if data is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        input_type=np.array(INPUT_TYPE),
        model_path=np.array(str(model_path)),
        cfg_path=np.array(str(cfg_path)),
        created_unix_s=np.array(time.time()),
        **data,
    )


def format_scores(scores: list[tuple[str, float]]) -> str:
    if not scores:
        return "scores unavailable"
    return ", ".join(f"{label}={score:.3f}" for label, score in scores)


def prediction_text(prediction, confidence) -> str:
    if prediction is None:
        return "n/a"
    if confidence is None:
        return str(prediction)
    return f"{prediction} ({confidence:.2f})"


class LivePosturePlot:
    def __init__(self, feature_params: dict):
        import matplotlib.pyplot as plt

        self.plt = plt
        self.feature_params = feature_params
        plt.ion()
        self.figure = plt.figure(figsize=(8.5, 6.5))
        self.axis = self.figure.add_subplot(111)
        x_limit = float(feature_params.get("x_limit_m", 2.0))
        min_range = float(feature_params.get("min_range_m", 0.2))
        max_range = float(feature_params.get("max_range_m", 5.0))
        self.axis.set_xlim(-x_limit, x_limit)
        self.axis.set_ylim(min_range, max_range)
        self.axis.set_aspect("equal", adjustable="box")
        self.axis.set_xlabel("x left/right (m)")
        self.axis.set_ylabel("y range (m)")
        self.axis.grid(True)
        self.scatter = self.axis.scatter([], [], c=[], cmap="viridis", s=45)
        self.scatter.set_clim(min_range, max_range)
        colorbar = self.figure.colorbar(self.scatter, ax=self.axis, pad=0.02)
        colorbar.set_label("y range (m)")
        self.figure.tight_layout()
        self.figure.show()

    def update(self, xyz: np.ndarray, prediction=None, confidence=None) -> None:
        filtered = filter_posture_points(np.asarray(xyz, dtype=float), self.feature_params)
        offsets = (
            filtered[:, :2]
            if len(filtered)
            else np.empty((0, 2), dtype=float)
        )
        colors = filtered[:, 1] if len(filtered) else np.empty(0, dtype=float)
        self.scatter.set_offsets(offsets)
        self.scatter.set_array(colors)
        title = "prediction: waiting"
        if prediction is not None:
            title = f"prediction: {prediction}"
            if confidence is not None:
                title += f" ({confidence:.2f})"
        title += f" | ROI points: {len(filtered)}"
        self.axis.set_title(title)
        self.figure.canvas.draw_idle()
        self.plt.pause(0.001)


def main() -> int:
    args = parse_args()
    if args.step_seconds <= 0:
        raise SystemExit("--step-seconds must be positive.")

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
    window_seconds = float(args.window_seconds or payload.get("window_seconds", 2.0))
    min_window_frames = int(args.min_window_frames or payload.get("min_segment_frames", 4) or 4)
    if window_seconds <= 0:
        raise SystemExit("--window-seconds must be positive.")
    if min_window_frames < 2:
        raise SystemExit("--min-window-frames must be at least 2.")

    commands = load_configuration(args.cfg)
    range_config = parse_range_config(commands)
    if range_config is None:
        raise SystemExit("Could not compute range-bin spacing from cfg.")

    session_name = args.session_name or f"posture_eval_{safe_label(model_path.stem)}_{timestamp()}"
    session_dir = Path(args.out_root).expanduser().resolve() / session_name
    session_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = session_dir / "posture_predictions.csv"
    frames_path = session_dir / "posture_realtime_frames.npz"
    summary_path = session_dir / "eval_posture_summary.json"
    write_json(
        session_dir / "session_metadata.json",
        {
            "kind": "eval_posture_realtime",
            "input_type": INPUT_TYPE,
            "model": str(model_path),
            "classifier_label": classifier_label,
            "feature_params": feature_params,
            "cfg_path": str(args.cfg),
            "duration_s": args.duration,
            "window_seconds": window_seconds,
            "step_seconds": args.step_seconds,
            "min_window_frames": min_window_frames,
            "range_bin_count": int(range_config.num_range_bins),
            "range_bin_spacing_m": float(range_config.bin_spacing_m),
            "started_at": now_text(),
        },
    )

    print(f"Session folder: {session_dir}")
    print(f"Loaded classifier: {classifier_label}")
    print(f"Window: {window_seconds:.2f} s; step: {args.step_seconds:.2f} s")

    try:
        serial_port = serial.Serial(args.port, args.baud, timeout=0.2)
    except serial.SerialException as error:
        raise SystemExit(f"Could not open serial port {args.port}: {error}") from None

    plot = None
    if not args.no_plot:
        try:
            plot = LivePosturePlot(feature_params)
        except ImportError as exc:
            print(f"Plot disabled; missing plotting dependency: {exc}")

    all_records: list[dict] = []
    recent_records: deque[dict] = deque()
    prediction_count = 0
    interrupted = False
    failed = False

    with predictions_path.open("w", newline="") as prediction_file:
        writer = csv.DictWriter(
            prediction_file,
            fieldnames=[
                "time_s",
                "frame_number",
                "window_frames",
                "prediction",
                "confidence",
                "scores_json",
                "point_count",
            ],
        )
        writer.writeheader()

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
                print("Reading point clouds. Use Ctrl+C to stop.")
                start_time = time.monotonic()
                last_prediction_time = start_time
                consecutive_warnings = 0
                feature_mismatch_reported = False
                display_prediction = None
                display_confidence = None

                while args.duration <= 0 or time.monotonic() - start_time < args.duration:
                    try:
                        frame_number, cloud = read_point_cloud_frame(
                            port,
                            args.frame_timeout,
                        )
                    except (TimeoutError, ValueError, RuntimeError) as error:
                        consecutive_warnings += 1
                        print(f"\nFrame parse warning: {error}", flush=True)
                        if consecutive_warnings >= 3:
                            raise RuntimeError("No valid frames received.") from error
                        continue

                    consecutive_warnings = 0
                    now = time.monotonic()
                    elapsed_s = now - start_time
                    xyz, velocity = cloud_arrays(cloud)
                    record = {
                        "time_s": elapsed_s,
                        "frame_number": frame_number,
                        "points_xyz": xyz,
                        "points_velocity": velocity,
                    }
                    all_records.append(record)
                    recent_records.append(record)
                    while recent_records and elapsed_s - recent_records[0]["time_s"] > window_seconds:
                        recent_records.popleft()

                    if (
                        now - last_prediction_time >= args.step_seconds
                        and len(recent_records) >= min_window_frames
                    ):
                        window_data = records_to_posture_data(list(recent_records))
                        if window_data is not None:
                            features, names = extract_posture_feature_vector(
                                window_data,
                                feature_params,
                            )
                        else:
                            features, names = None, []

                        if features is not None and feature_names and len(features) != len(feature_names):
                            if not feature_mismatch_reported:
                                print(
                                    "\nFeature length mismatch: "
                                    f"model expects {len(feature_names)}, got {len(features)}."
                                )
                                feature_mismatch_reported = True
                        elif features is not None:
                            prediction = model.predict([features])[0]
                            confidence = prediction_confidence(model, features, prediction)
                            scores = prediction_scores(model, features)
                            display_prediction = prediction
                            display_confidence = confidence
                            prediction_count += 1
                            last_prediction_time = now

                            writer.writerow(
                                {
                                    "time_s": f"{elapsed_s:.3f}",
                                    "frame_number": frame_number,
                                    "window_frames": len(recent_records),
                                    "prediction": prediction,
                                    "confidence": "" if confidence is None else f"{confidence:.4f}",
                                    "scores_json": json.dumps(scores),
                                    "point_count": len(xyz),
                                }
                            )
                            prediction_file.flush()
                            print(
                                "prediction: "
                                f"{prediction_text(prediction, confidence)} | "
                                f"scores: {format_scores(scores)}",
                                flush=True,
                            )

                    if plot is not None:
                        plot.update(xyz, display_prediction, display_confidence)

            except KeyboardInterrupt:
                interrupted = True
                print("\nInterrupted. Stopping radar...")
            except RuntimeError as error:
                failed = True
                print(f"\nEvaluation failed: {error}")
            finally:
                if not args.no_config:
                    print("> sensorStop 0")
                    stop_and_drain(port)

    if not args.no_save_frames:
        save_frame_stream(frames_path, all_records, model_path, args.cfg)

    write_json(
        summary_path,
        {
            "input_type": INPUT_TYPE,
            "model": str(model_path),
            "predictions_csv": str(predictions_path),
            "frames_npz": "" if args.no_save_frames else str(frames_path),
            "frame_count": len(all_records),
            "prediction_count": prediction_count,
            "interrupted": interrupted,
            "failed": failed,
            "finished_at": now_text(),
        },
    )
    print(f"Predictions saved to: {predictions_path}")
    if not args.no_save_frames:
        print(f"Frames saved to: {frames_path}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
