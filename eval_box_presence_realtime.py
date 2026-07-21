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

from box_lab_common import (
    PRESENCE_INPUT_TYPE,
    SESSIONS_DIR,
    box_presence_motion_matrix,
    db_scale,
    extract_box_presence_feature_vector,
    now_text,
    prediction_confidence,
    prediction_scores,
    read_box_data_frame,
    robust_normalize,
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
        description="Real-time empty/object box-presence evaluation with range plot."
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
    parser.add_argument("--background-frames", type=int, default=30)
    parser.add_argument(
        "--background-npz",
        type=Path,
        help=(
            "Use an existing background.npz from collection instead of capturing "
            "a new realtime background."
        ),
    )
    parser.add_argument("--frame-timeout", type=float, default=5.0)
    parser.add_argument("--out-root", default=str(SESSIONS_DIR))
    parser.add_argument("--session-name")
    parser.add_argument("--no-plot", action="store_true")
    parser.add_argument("--no-save-frames", action="store_true")
    parser.add_argument("--no-prompt", action="store_true")
    parser.add_argument("--no-config", action="store_true")
    parser.add_argument("--no-warm-reset", action="store_true")
    return parser.parse_args()


def cloud_xyz(cloud) -> np.ndarray:
    if len(cloud.x) == 0:
        return np.empty((0, 3), dtype=float)
    return np.column_stack((cloud.x, cloud.y, cloud.z))


def capture_background(
    port: serial.Serial,
    frame_count: int,
    frame_timeout_s: float,
    expected_range_profile_bytes: int,
    prompt: bool,
) -> np.ndarray:
    if prompt:
        print()
        print("=" * 56)
        print("Background calibration")
        print("Remove the box from the radar view and keep the scene still.")
        input(f"Press Enter to capture {frame_count} empty-scene frames...")
    else:
        print(f"Capturing {frame_count} empty-scene background frames...")

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


def load_background_npz(path: Path, expected_range_m: np.ndarray) -> np.ndarray:
    path = path.expanduser().resolve()
    expected_bins = expected_range_m.size
    try:
        with np.load(path) as npz:
            if "range_background" not in npz:
                raise ValueError("missing range_background array")
            background = np.asarray(npz["range_background"], dtype=float).reshape(-1)
            saved_range_m = (
                np.asarray(npz["range_m"], dtype=float).reshape(-1)
                if "range_m" in npz
                else None
            )
    except Exception as exc:
        raise RuntimeError(f"Could not load background from {path}: {exc}") from exc

    if background.size < expected_bins:
        raise RuntimeError(
            f"Background has {background.size} bins, but cfg expects {expected_bins}."
        )
    if saved_range_m is not None:
        saved_range_m = saved_range_m[:expected_bins]
        if saved_range_m.size < expected_bins or not np.allclose(
            saved_range_m,
            expected_range_m,
            rtol=1e-4,
            atol=1e-6,
        ):
            raise RuntimeError(
                "Background range axis does not match the current cfg. "
                "Use the same cfg as collection or capture a fresh background."
            )
    return background[:expected_bins]


def prompt_evaluation_ready(prompt: bool) -> None:
    if prompt:
        print()
        print("=" * 56)
        print("Realtime evaluation")
        print("Bring the closed box into view and move it slowly.")
        input("Press Enter to start prediction...")
    else:
        print("Starting prediction. Bring the closed box into view and move it slowly.")


def records_to_presence_data(
    records: list[dict],
    range_m: np.ndarray,
    range_background: np.ndarray,
) -> dict[str, np.ndarray] | None:
    if not records:
        return None

    point_count = np.array([len(record["points_xyz"]) for record in records], dtype=np.uint16)
    max_points = int(np.max(point_count)) if len(point_count) else 0
    points_xyz = np.full((len(records), max_points, 3), np.nan, dtype=float)
    for index, record in enumerate(records):
        points = record["points_xyz"]
        if len(points):
            points_xyz[index, : len(points), :] = points

    return {
        "frame_number": np.array([record["frame_number"] for record in records], dtype=np.uint32),
        "time_s": np.array([record["time_s"] for record in records], dtype=float),
        "range_m": range_m,
        "range_background": range_background,
        "range_profile": np.vstack([record["range_profile"] for record in records]),
        "point_count": point_count,
        "points_xyz": points_xyz,
    }


def save_frame_stream(
    path: Path,
    records: list[dict],
    range_m: np.ndarray,
    range_background: np.ndarray,
    model_path: Path,
    cfg_path: Path,
) -> None:
    data = records_to_presence_data(records, range_m, range_background)
    if data is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        input_type=np.array(PRESENCE_INPUT_TYPE),
        model_path=np.array(str(model_path)),
        cfg_path=np.array(str(cfg_path)),
        created_unix_s=np.array(time.time()),
        **data,
    )


def format_scores(scores: list[tuple[str, float]]) -> str:
    if not scores:
        return "scores unavailable"
    return ", ".join(f"{label}={score:.3f}" for label, score in scores)


def current_delta_profile(
    profile: np.ndarray,
    range_background: np.ndarray,
    range_m: np.ndarray,
    feature_params: dict,
) -> tuple[np.ndarray, np.ndarray]:
    cols = min(profile.size, range_background.size, range_m.size)
    profile = np.asarray(profile[:cols], dtype=float)
    background = np.asarray(range_background[:cols], dtype=float)
    range_m = np.asarray(range_m[:cols], dtype=float)
    mask = (
        (range_m >= float(feature_params.get("min_range_m", 0.0)))
        & (range_m <= float(feature_params.get("max_range_m", 0.80)))
    )
    if bool(feature_params.get("db", True)):
        delta = db_scale(profile) - db_scale(background)
    else:
        delta = profile - background
    return range_m[mask], delta[mask]


class LivePresenceRangePlot:
    def __init__(self, feature_params: dict):
        import matplotlib.pyplot as plt

        self.plt = plt
        self.feature_params = feature_params
        plt.ion()
        self.figure, (self.profile_axis, self.window_axis) = plt.subplots(
            2,
            1,
            figsize=(10, 6),
            gridspec_kw={"height_ratios": [1.0, 1.15]},
        )
        (self.profile_line,) = self.profile_axis.plot([], [], lw=2)
        self.profile_axis.set_xlabel("range (m)")
        self.profile_axis.set_ylabel("delta strength")
        self.profile_axis.grid(True)
        self.window_axis.set_xlabel("time in window (s)")
        self.window_axis.set_ylabel("range (m)")
        self.image = None
        self.colorbar = None
        self.figure.tight_layout()
        self.figure.show()

    def update(
        self,
        profile: np.ndarray,
        range_background: np.ndarray,
        range_m: np.ndarray,
        window_data: dict[str, np.ndarray] | None,
        prediction=None,
        confidence=None,
    ) -> None:
        plot_range_m, delta = current_delta_profile(
            profile,
            range_background,
            range_m,
            self.feature_params,
        )
        self.profile_line.set_data(plot_range_m, delta)
        if plot_range_m.size:
            self.profile_axis.set_xlim(float(plot_range_m[0]), float(plot_range_m[-1]))
        if delta.size:
            low = min(-3.0, float(np.percentile(delta, 2)) * 1.2)
            high = max(3.0, float(np.percentile(delta, 98)) * 1.2)
            if high <= low:
                high = low + 1.0
            self.profile_axis.set_ylim(low, high)

        title = "prediction: waiting"
        if prediction is not None:
            title = f"prediction: {prediction}"
            if confidence is not None:
                title += f" ({confidence:.2f})"
        self.profile_axis.set_title(title)

        if window_data is not None:
            payload = box_presence_motion_matrix(window_data, self.feature_params)
            if payload is not None:
                window_range_m, time_s, motion = payload
                if bool(self.feature_params.get("normalize", False)):
                    image = robust_normalize(motion).T
                    image_label = "normalized delta"
                    vmin, vmax = -3.0, 3.0
                else:
                    image = motion.T
                    image_label = "delta strength"
                    finite = image[np.isfinite(image)]
                    if finite.size:
                        vmin = min(-3.0, float(np.percentile(finite, 2)))
                        vmax = max(3.0, float(np.percentile(finite, 98)))
                    else:
                        vmin, vmax = -3.0, 3.0
                extent = (
                    float(time_s[0]),
                    float(time_s[-1]) if time_s.size > 1 else float(time_s[0] + 1.0),
                    float(window_range_m[0]),
                    float(window_range_m[-1]),
                )
                if self.image is None:
                    self.image = self.window_axis.imshow(
                        image,
                        aspect="auto",
                        origin="lower",
                        extent=extent,
                        cmap="coolwarm",
                        vmin=vmin,
                        vmax=vmax,
                    )
                    self.colorbar = self.figure.colorbar(self.image, ax=self.window_axis, pad=0.02)
                    self.colorbar.set_label(image_label)
                else:
                    self.image.set_data(image)
                    self.image.set_extent(extent)
                    self.image.set_clim(vmin, vmax)

        self.figure.canvas.draw_idle()
        self.plt.pause(0.001)


def prediction_text(prediction, confidence) -> str:
    if prediction is None:
        return "n/a"
    if confidence is None:
        return str(prediction)
    return f"{prediction} ({confidence:.2f})"


def main() -> int:
    args = parse_args()
    if args.step_seconds <= 0:
        raise SystemExit("--step-seconds must be positive.")
    if args.background_frames < 1:
        raise SystemExit("--background-frames must be at least 1.")

    try:
        import joblib
    except ImportError as exc:
        raise SystemExit(f"Missing joblib/sklearn environment: {exc}") from exc

    model_path = Path(args.model).expanduser().resolve()
    payload = joblib.load(model_path)
    input_type = payload.get("input_type", PRESENCE_INPUT_TYPE)
    if input_type != PRESENCE_INPUT_TYPE:
        raise SystemExit(f"This evaluator cannot run model input_type: {input_type}")

    model = payload["model"]
    feature_params = payload.get("feature_params") or {}
    feature_names = payload.get("feature_names") or []
    classifier_label = payload.get("classifier_label", payload.get("classifier", "unknown"))
    split_mode = payload.get("split_mode", "")
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
    expected_bytes = range_config.num_range_bins * 4
    range_m = np.arange(range_config.num_range_bins) * range_config.bin_spacing_m

    session_name = args.session_name or f"box_presence_eval_{safe_label(model_path.stem)}_{timestamp()}"
    session_dir = Path(args.out_root).expanduser().resolve() / session_name
    session_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = session_dir / "presence_predictions.csv"
    frames_path = session_dir / "presence_realtime_frames.npz"
    summary_path = session_dir / "eval_box_presence_summary.json"
    write_json(
        session_dir / "session_metadata.json",
        {
            "kind": "eval_box_presence_realtime",
            "input_type": PRESENCE_INPUT_TYPE,
            "model": str(model_path),
            "classifier_label": classifier_label,
            "feature_params": feature_params,
            "cfg_path": str(args.cfg),
            "duration_s": args.duration,
            "window_seconds": window_seconds,
            "step_seconds": args.step_seconds,
            "min_window_frames": min_window_frames,
            "background_frames": args.background_frames,
            "background_npz": "" if args.background_npz is None else str(args.background_npz),
            "range_bin_count": int(range_config.num_range_bins),
            "range_bin_spacing_m": float(range_config.bin_spacing_m),
            "started_at": now_text(),
        },
    )

    print(f"Session folder: {session_dir}")
    print(f"Loaded classifier: {classifier_label}")
    print(f"Window: {window_seconds:.2f} s; step: {args.step_seconds:.2f} s")
    if split_mode == "segment":
        print(
            "Warning: this model was validated with a segment-level split. "
            "That split can overstate accuracy because overlapping windows from "
            "the same recording can appear in both train and test."
        )
    if str(payload.get("classifier", "")).startswith("svm"):
        print(
            "Warning: SVM probabilities can be overconfident with small "
            "box-presence datasets. If live predictions stick on one class, "
            "try retraining with the default random_forest classifier."
        )

    try:
        serial_port = serial.Serial(args.port, args.baud, timeout=0.2)
    except serial.SerialException as error:
        raise SystemExit(f"Could not open serial port {args.port}: {error}") from None

    plot = None
    if not args.no_plot:
        try:
            plot = LivePresenceRangePlot(feature_params)
        except ImportError as exc:
            print(f"Plot disabled; missing plotting dependency: {exc}")

    all_records: list[dict] = []
    recent_records: deque[dict] = deque()
    prediction_count = 0
    interrupted = False
    failed = False
    range_background = np.zeros_like(range_m)

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
                if args.background_npz is not None:
                    range_background = load_background_npz(
                        args.background_npz,
                        range_m,
                    )
                    print(f"Loaded background: {args.background_npz}")
                else:
                    range_background = capture_background(
                        port,
                        args.background_frames,
                        args.frame_timeout,
                        expected_bytes,
                        prompt=not args.no_prompt,
                    )
                    print("Background captured.")
                prompt_evaluation_ready(prompt=not args.no_prompt)

                start_time = time.monotonic()
                last_prediction_time = start_time
                consecutive_warnings = 0
                feature_mismatch_reported = False
                display_prediction = None
                display_confidence = None

                while args.duration <= 0 or time.monotonic() - start_time < args.duration:
                    try:
                        frame_number, profile, cloud = read_box_data_frame(
                            port,
                            args.frame_timeout,
                            expected_bytes,
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
                    record = {
                        "time_s": elapsed_s,
                        "frame_number": frame_number,
                        "range_profile": profile,
                        "points_xyz": cloud_xyz(cloud),
                    }
                    all_records.append(record)
                    recent_records.append(record)
                    while recent_records and elapsed_s - recent_records[0]["time_s"] > window_seconds:
                        recent_records.popleft()

                    window_data = None
                    if len(recent_records) >= 2:
                        window_data = records_to_presence_data(
                            list(recent_records),
                            range_m,
                            range_background,
                        )

                    if (
                        now - last_prediction_time >= args.step_seconds
                        and len(recent_records) >= min_window_frames
                        and window_data is not None
                    ):
                        features, names = extract_box_presence_feature_vector(
                            window_data,
                            feature_params,
                        )
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
                        plot.update(
                            profile,
                            range_background,
                            range_m,
                            window_data,
                            display_prediction,
                            display_confidence,
                        )

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
        save_frame_stream(
            frames_path,
            all_records,
            range_m,
            range_background,
            model_path,
            args.cfg,
        )

    write_json(
        summary_path,
        {
            "input_type": PRESENCE_INPUT_TYPE,
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
