#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np

from posture_lab_common import (
    INPUT_TYPE,
    MODELS_DIR,
    extract_posture_feature_vector,
    load_trial_data,
    read_manifests,
    resolve_npz_path,
    slice_posture_segment,
    time_window_segments,
    timestamp,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train posture classifier from segmented mmWave point-cloud recordings."
    )
    parser.add_argument(
        "datasets",
        nargs="+",
        help="One or more dataset folders created by collect_posture_dataset.py.",
    )
    parser.add_argument("--window-seconds", type=float, default=2.0)
    parser.add_argument(
        "--overlap",
        type=float,
        default=0.5,
        help="Segment overlap fraction. Default 0.5 means 50%% overlap.",
    )
    parser.add_argument("--min-segment-frames", type=int, default=4)
    parser.add_argument("--test-size", type=float, default=0.25)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--group-by",
        choices=["recording", "collector"],
        default="recording",
        help="Use collector for cross-student evaluation when enough collectors exist.",
    )
    parser.add_argument("--min-range", type=float, default=0.2)
    parser.add_argument("--max-range", type=float, default=5.0)
    parser.add_argument("--x-limit", type=float, default=2.0)
    parser.add_argument("--trajectory-frames", type=int, default=20)
    parser.add_argument("--x-bins", type=int, default=8)
    parser.add_argument("--y-bins", type=int, default=12)
    parser.add_argument("--xy-x-bins", type=int, default=8)
    parser.add_argument("--xy-y-bins", type=int, default=12)
    parser.add_argument(
        "--classifier",
        choices=["random_forest", "svm_rbf", "svm_poly", "decision_tree", "knn"],
        default="random_forest",
    )
    parser.add_argument("--n-estimators", type=int, default=300)
    parser.add_argument("--svm-c", type=float, default=1.0)
    parser.add_argument("--svm-gamma", default="scale")
    parser.add_argument("--svm-degree", type=int, default=3)
    parser.add_argument("--decision-tree-max-depth", type=int)
    parser.add_argument("--knn-neighbors", type=int, default=5)
    parser.add_argument("--knn-weights", choices=["uniform", "distance"], default="distance")
    parser.add_argument("--model-out")
    parser.add_argument("--confusion-out")
    return parser.parse_args()


def feature_params_for_args(args: argparse.Namespace) -> dict:
    return {
        "min_range_m": float(args.min_range),
        "max_range_m": float(args.max_range),
        "x_limit_m": float(args.x_limit),
        "trajectory_points": int(args.trajectory_frames),
        "x_bins": int(args.x_bins),
        "y_bins": int(args.y_bins),
        "xy_x_bins": int(args.xy_x_bins),
        "xy_y_bins": int(args.xy_y_bins),
    }


def parse_svm_gamma(value: str) -> str | float:
    if value in {"scale", "auto"}:
        return value
    try:
        gamma = float(value)
    except ValueError as exc:
        raise SystemExit("--svm-gamma must be scale, auto, or a positive float.") from exc
    if gamma <= 0:
        raise SystemExit("--svm-gamma must be positive.")
    return gamma


def classifier_label(classifier: str) -> str:
    return {
        "random_forest": "Random Forest",
        "svm_rbf": "RBF SVM",
        "svm_poly": "Polynomial SVM",
        "decision_tree": "Decision Tree",
        "knn": "KNN",
    }[classifier]


def build_classifier(args: argparse.Namespace, train_count: int):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC
    from sklearn.tree import DecisionTreeClassifier

    if args.classifier == "random_forest":
        params = {
            "n_estimators": args.n_estimators,
            "random_state": args.random_state,
            "class_weight": "balanced",
        }
        return RandomForestClassifier(**params), params

    if args.classifier == "svm_rbf":
        params = {
            "kernel": "rbf",
            "C": args.svm_c,
            "gamma": parse_svm_gamma(args.svm_gamma),
            "class_weight": "balanced",
            "probability": True,
            "random_state": args.random_state,
        }
        return make_pipeline(StandardScaler(), SVC(**params)), params

    if args.classifier == "svm_poly":
        params = {
            "kernel": "poly",
            "degree": args.svm_degree,
            "C": args.svm_c,
            "gamma": parse_svm_gamma(args.svm_gamma),
            "class_weight": "balanced",
            "probability": True,
            "random_state": args.random_state,
        }
        return make_pipeline(StandardScaler(), SVC(**params)), params

    if args.classifier == "decision_tree":
        params = {
            "max_depth": args.decision_tree_max_depth,
            "random_state": args.random_state,
            "class_weight": "balanced",
        }
        return DecisionTreeClassifier(**params), params

    if args.classifier == "knn":
        requested_neighbors = max(1, int(args.knn_neighbors))
        actual_neighbors = min(requested_neighbors, int(train_count))
        params = {"n_neighbors": actual_neighbors, "weights": args.knn_weights}
        return make_pipeline(StandardScaler(), KNeighborsClassifier(**params)), params

    raise SystemExit(f"Unsupported classifier: {args.classifier}")


def scalar_text(value) -> str:
    if isinstance(value, np.ndarray):
        if value.shape == ():
            return str(value.item())
        if value.size == 1:
            return str(value.reshape(-1)[0])
    return str(value)


def label_from_row_or_data(row: dict[str, str], data: dict[str, np.ndarray]) -> str:
    label = row.get("posture") or row.get("label") or ""
    if label:
        return label
    if "posture" in data:
        return scalar_text(data["posture"])
    return ""


def collector_from_row_or_data(row: dict[str, str], data: dict[str, np.ndarray]) -> str:
    collector = row.get("collector") or ""
    if collector:
        return collector
    if "collector" in data:
        return scalar_text(data["collector"])
    return "unknown"


def build_examples(rows: list[dict[str, str]], args: argparse.Namespace, feature_params: dict):
    examples: list[list[float]] = []
    labels: list[str] = []
    recording_groups: list[str] = []
    collector_groups: list[str] = []
    segment_ids: list[str] = []
    feature_names: list[str] | None = None
    skipped: list[dict[str, str]] = []
    used_npz_paths: set[str] = set()
    collectors: set[str] = set()

    for row in rows:
        npz_path = resolve_npz_path(row)
        if not npz_path.exists():
            skipped.append({"npz_path": str(npz_path), "reason": "missing trial_data.npz"})
            continue

        try:
            data = load_trial_data(npz_path)
        except Exception as exc:
            skipped.append({"npz_path": str(npz_path), "reason": f"load error: {exc}"})
            continue

        label = label_from_row_or_data(row, data)
        collector = collector_from_row_or_data(row, data)
        if not label:
            skipped.append({"npz_path": str(npz_path), "reason": "missing posture label"})
            continue

        time_s = np.asarray(data.get("time_s", []), dtype=float)
        try:
            bounds = time_window_segments(
                time_s,
                args.window_seconds,
                args.overlap,
                args.min_segment_frames,
            )
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc
        if not bounds:
            skipped.append({"npz_path": str(npz_path), "reason": "no usable segments"})
            continue

        for segment_index, (start_index, end_index) in enumerate(bounds, start=1):
            try:
                segment = slice_posture_segment(data, start_index, end_index)
                features, names = extract_posture_feature_vector(segment, feature_params)
            except Exception as exc:
                skipped.append(
                    {
                        "npz_path": str(npz_path),
                        "reason": f"segment {segment_index} feature error: {exc}",
                    }
                )
                continue

            if features is None:
                skipped.append(
                    {
                        "npz_path": str(npz_path),
                        "reason": f"segment {segment_index} not usable",
                    }
                )
                continue
            if feature_names is None:
                feature_names = names
            elif len(names) != len(feature_names):
                skipped.append(
                    {
                        "npz_path": str(npz_path),
                        "reason": f"segment {segment_index} feature length mismatch",
                    }
                )
                continue

            examples.append(features)
            labels.append(label)
            recording_groups.append(str(npz_path))
            collector_groups.append(collector)
            segment_ids.append(f"{npz_path}#{segment_index:04d}")
            used_npz_paths.add(str(npz_path))
            collectors.add(collector)

    return (
        examples,
        labels,
        recording_groups,
        collector_groups,
        segment_ids,
        feature_names or [],
        skipped,
        used_npz_paths,
        collectors,
    )


def group_split(X: np.ndarray, y: np.ndarray, groups: np.ndarray, args: argparse.Namespace):
    from sklearn.model_selection import GroupShuffleSplit

    unique_groups = sorted(set(groups))
    unique_labels = sorted(set(y))
    if len(unique_groups) < 2:
        return None

    for offset in range(80):
        splitter = GroupShuffleSplit(
            n_splits=1,
            test_size=args.test_size,
            random_state=args.random_state + offset,
        )
        train_idx, test_idx = next(splitter.split(X, y, groups))
        if set(y[train_idx]) == set(unique_labels) and set(y[test_idx]) == set(unique_labels):
            return train_idx, test_idx
    return None


def split_examples(
    X: np.ndarray,
    y: np.ndarray,
    recording_groups: np.ndarray,
    collector_groups: np.ndarray,
    args: argparse.Namespace,
):
    from sklearn.model_selection import train_test_split

    if args.group_by == "collector":
        collector_split = group_split(X, y, collector_groups, args)
        if collector_split is not None:
            return collector_split[0], collector_split[1], "collector"

    recording_split = group_split(X, y, recording_groups, args)
    if recording_split is not None:
        mode = "recording" if args.group_by == "recording" else "recording_fallback"
        return recording_split[0], recording_split[1], mode

    indices = np.arange(len(y))
    class_counts = {label: int((y == label).sum()) for label in sorted(set(y))}
    stratify = y if min(class_counts.values()) >= 2 else None
    split_test_size = args.test_size
    if stratify is not None:
        n_classes = len(class_counts)
        n_samples = len(y)
        requested = int(math.ceil(args.test_size * n_samples)) if args.test_size < 1 else int(args.test_size)
        split_test_size = max(n_classes, requested)
        split_test_size = min(split_test_size, n_samples - n_classes)

    train_idx, test_idx = train_test_split(
        indices,
        test_size=split_test_size,
        random_state=args.random_state,
        stratify=stratify,
    )
    return train_idx, test_idx, "segment"


def default_model_path(dataset_dirs: list[Path], classifier: str) -> Path:
    name = f"{classifier}_posture_{timestamp()}.joblib"
    if len(dataset_dirs) == 1:
        return dataset_dirs[0] / "models" / name
    return MODELS_DIR / name


def main() -> int:
    args = parse_args()
    if args.window_seconds <= 0:
        raise SystemExit("--window-seconds must be positive.")
    if not 0.0 <= args.overlap < 1.0:
        raise SystemExit("--overlap must be in [0, 1).")
    if args.min_segment_frames < 2:
        raise SystemExit("--min-segment-frames must be at least 2.")
    if args.min_range >= args.max_range:
        raise SystemExit("--min-range must be smaller than --max-range.")
    if args.x_limit <= 0:
        raise SystemExit("--x-limit must be positive.")
    for name in ("trajectory_frames", "x_bins", "y_bins", "xy_x_bins", "xy_y_bins"):
        if int(getattr(args, name)) < 1:
            raise SystemExit(f"--{name.replace('_', '-')} must be at least 1.")

    try:
        import joblib
        import matplotlib.pyplot as plt
        from sklearn.metrics import (
            ConfusionMatrixDisplay,
            accuracy_score,
            classification_report,
            confusion_matrix,
        )
    except ImportError as exc:
        raise SystemExit(f"Training dependencies missing: {exc}") from exc

    dataset_dirs = [Path(item).expanduser().resolve() for item in args.datasets]
    rows, missing_datasets = read_manifests(dataset_dirs)
    if not rows:
        raise SystemExit("No trials found. Expected trials.csv or trial_metadata.json files.")

    feature_params = feature_params_for_args(args)
    (
        examples,
        labels,
        recording_groups,
        collector_groups,
        segment_ids,
        feature_names,
        skipped,
        used_npz_paths,
        collectors,
    ) = build_examples(rows, args, feature_params)
    if len(set(labels)) < 2:
        raise SystemExit("Need at least two posture labels to train.")
    if len(examples) < 4:
        raise SystemExit("Need at least four usable segments to train.")

    X = np.asarray(examples, dtype=float)
    y = np.asarray(labels)
    recording_group_array = np.asarray(recording_groups)
    collector_group_array = np.asarray(collector_groups)
    class_counts = {label: int((y == label).sum()) for label in sorted(set(labels))}
    train_idx, test_idx, split_mode = split_examples(
        X,
        y,
        recording_group_array,
        collector_group_array,
        args,
    )
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    if len(set(y_train)) < 2:
        raise SystemExit("Training split needs at least two posture labels.")

    model, classifier_params = build_classifier(args, train_count=len(X_train))
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    accuracy = float(accuracy_score(y_test, predictions))
    labels_order = sorted(set(y))
    matrix = confusion_matrix(y_test, predictions, labels=labels_order)
    report_text = classification_report(
        y_test,
        predictions,
        labels=labels_order,
        zero_division=0,
    )
    report_dict = classification_report(
        y_test,
        predictions,
        labels=labels_order,
        output_dict=True,
        zero_division=0,
    )

    model_out = (
        Path(args.model_out).expanduser().resolve()
        if args.model_out
        else default_model_path(dataset_dirs, args.classifier)
    )
    model_out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": model,
        "input_type": INPUT_TYPE,
        "feature_params": feature_params,
        "feature_names": feature_names,
        "classifier": args.classifier,
        "classifier_label": classifier_label(args.classifier),
        "classifier_params": classifier_params,
        "window_seconds": float(args.window_seconds),
        "overlap": float(args.overlap),
        "min_segment_frames": int(args.min_segment_frames),
        "group_by": args.group_by,
        "split_mode": split_mode,
        "class_counts": class_counts,
        "labels_order": labels_order,
        "collectors": sorted(collectors),
        "accuracy": accuracy,
        "classification_report": report_dict,
        "confusion_matrix": matrix.tolist(),
        "datasets": [str(path) for path in dataset_dirs],
        "trail_npz_paths": sorted(used_npz_paths),
        "segment_ids": segment_ids,
        "missing_datasets": missing_datasets,
        "skipped": skipped,
    }
    joblib.dump(payload, model_out)

    confusion_out = (
        Path(args.confusion_out).expanduser().resolve()
        if args.confusion_out
        else model_out.with_suffix(".confusion.png")
    )
    confusion_out.parent.mkdir(parents=True, exist_ok=True)
    display = ConfusionMatrixDisplay(matrix, display_labels=labels_order)
    display.plot(cmap="Blues", values_format="d")
    plt.title(f"Posture Classifier ({accuracy:.2%} accuracy)")
    plt.tight_layout()
    plt.savefig(confusion_out, dpi=180)
    plt.close()

    print(f"Collectors: {', '.join(sorted(collectors))}")
    print(f"Usable recordings: {len(used_npz_paths)}")
    print(f"Usable segments: {len(examples)}")
    print(f"Class counts: {class_counts}")
    print(f"Feature count: {len(feature_names)}")
    print(f"Split mode: {split_mode}; train={len(X_train)}, test={len(X_test)}")
    if split_mode == "segment":
        print("Warning: split fell back to segment-level because group split was too small.")
    if split_mode == "recording_fallback":
        print("Warning: collector split was not possible; used recording split instead.")
    if skipped:
        print(f"Skipped items: {len(skipped)}")
        for item in skipped[:10]:
            print(f"  - {item.get('npz_path', '')}: {item['reason']}")
    print(report_text)
    print(f"Accuracy: {accuracy:.3f}")
    print(f"Saved model: {model_out}")
    print(f"Saved confusion matrix: {confusion_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
