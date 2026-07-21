#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np

from box_lab_common import (
    MODELS_DIR,
    PRESENCE_INPUT_TYPE,
    extract_box_presence_feature_vector,
    load_trial_data,
    read_manifests,
    resolve_npz_path,
    slice_presence_segment,
    time_window_segments,
    timestamp,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train binary empty/object box-presence classifier from long recordings."
    )
    parser.add_argument(
        "datasets",
        nargs="+",
        help="One or more dataset folders created by collect_box_presence_dataset.py.",
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
        "--allow-segment-split",
        action="store_true",
        help=(
            "Allow train/test segments from the same recording. Use only for "
            "code smoke tests; it can make accuracy look unrealistically high."
        ),
    )
    parser.add_argument("--min-range", type=float, default=0.0)
    parser.add_argument("--max-range", type=float, default=1.00)
    parser.add_argument("--resample-frames", type=int, default=24)
    parser.add_argument("--resample-bins", type=int, default=32)
    parser.add_argument("--db", dest="db", action="store_true", default=True)
    parser.add_argument("--no-db", dest="db", action="store_false")
    parser.add_argument(
        "--normalize",
        dest="normalize",
        action="store_true",
        default=False,
        help="Robust-normalize each 2-second range image. Disabled by default.",
    )
    parser.add_argument("--no-normalize", dest="normalize", action="store_false")
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
        "resample_frames": int(args.resample_frames),
        "resample_bins": int(args.resample_bins),
        "db": bool(args.db),
        "normalize": bool(args.normalize),
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
    label = row.get("label") or row.get("contents") or ""
    if label:
        return label
    if "label" in data:
        return scalar_text(data["label"])
    return ""


def build_examples(rows: list[dict[str, str]], args: argparse.Namespace, feature_params: dict):
    examples: list[list[float]] = []
    labels: list[str] = []
    groups: list[str] = []
    segment_ids: list[str] = []
    feature_names: list[str] | None = None
    skipped: list[dict[str, str]] = []
    used_npz_paths: set[str] = set()

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
        if not label:
            skipped.append({"npz_path": str(npz_path), "reason": "missing label"})
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
                segment = slice_presence_segment(data, start_index, end_index)
                features, names = extract_box_presence_feature_vector(segment, feature_params)
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
            groups.append(str(npz_path))
            segment_ids.append(f"{npz_path}#{segment_index:04d}")
            used_npz_paths.add(str(npz_path))

    return examples, labels, groups, segment_ids, feature_names or [], skipped, used_npz_paths


def split_examples(X: np.ndarray, y: np.ndarray, groups: np.ndarray, args: argparse.Namespace):
    from sklearn.model_selection import train_test_split

    indices = np.arange(len(y))
    unique_labels = sorted(set(y))

    group_labels: dict[str, str] = {}
    for group, label in zip(groups, y):
        group = str(group)
        label = str(label)
        previous_label = group_labels.get(group)
        if previous_label is not None and previous_label != label:
            raise SystemExit(f"Recording group has multiple labels: {group}")
        group_labels[group] = label

    label_to_groups = {
        label: sorted(group for group, group_label in group_labels.items() if group_label == label)
        for label in unique_labels
    }
    too_small = {label: len(label_groups) for label, label_groups in label_to_groups.items() if len(label_groups) < 2}
    if not too_small:
        rng = np.random.default_rng(args.random_state)
        test_groups: set[str] = set()
        for label_groups in label_to_groups.values():
            shuffled = list(label_groups)
            rng.shuffle(shuffled)
            if args.test_size < 1:
                test_count = int(math.ceil(args.test_size * len(shuffled)))
            else:
                test_count = int(args.test_size)
            test_count = max(1, min(test_count, len(shuffled) - 1))
            test_groups.update(shuffled[:test_count])

        test_mask = np.array([str(group) in test_groups for group in groups], dtype=bool)
        train_idx = indices[~test_mask]
        test_idx = indices[test_mask]
        if set(y[train_idx]) == set(unique_labels) and set(y[test_idx]) == set(unique_labels):
            return train_idx, test_idx, "group"

    if not args.allow_segment_split:
        details = ", ".join(
            f"{label}={count} recording(s)"
            for label, count in sorted(too_small.items())
        )
        if not details:
            details = "the available recordings could not form a label-balanced group split"
        raise SystemExit(
            "Box-presence training now requires a recording-level train/test split. "
            f"Need at least two accepted recordings per label; found {details}. "
            "Collect more trails, combine multiple dataset folders, or pass "
            "--allow-segment-split only for a quick code smoke test."
        )

    class_counts = {label: int((y == label).sum()) for label in unique_labels}
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
    name = f"{classifier}_box_presence_{timestamp()}.joblib"
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
    if args.resample_frames < 2 or args.resample_bins < 2:
        raise SystemExit("--resample-frames and --resample-bins must be at least 2.")

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
    examples, labels, groups, segment_ids, feature_names, skipped, used_npz_paths = build_examples(
        rows,
        args,
        feature_params,
    )
    if len(set(labels)) < 2:
        raise SystemExit("Need at least two labels to train.")
    if len(examples) < 4:
        raise SystemExit("Need at least four usable segments to train.")

    X = np.asarray(examples, dtype=float)
    y = np.asarray(labels)
    group_array = np.asarray(groups)
    class_counts = {label: int((y == label).sum()) for label in sorted(set(labels))}
    train_idx, test_idx, split_mode = split_examples(X, y, group_array, args)
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    if len(set(y_train)) < 2:
        raise SystemExit("Training split needs at least two labels.")

    validation_model, validation_classifier_params = build_classifier(
        args,
        train_count=len(X_train),
    )
    validation_model.fit(X_train, y_train)
    predictions = validation_model.predict(X_test)
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
    final_model, classifier_params = build_classifier(args, train_count=len(X))
    final_model.fit(X, y)

    payload = {
        "model": final_model,
        "input_type": PRESENCE_INPUT_TYPE,
        "feature_params": feature_params,
        "feature_names": feature_names,
        "classifier": args.classifier,
        "classifier_label": classifier_label(args.classifier),
        "classifier_params": classifier_params,
        "validation_classifier_params": validation_classifier_params,
        "window_seconds": float(args.window_seconds),
        "overlap": float(args.overlap),
        "min_segment_frames": int(args.min_segment_frames),
        "class_counts": class_counts,
        "labels_order": labels_order,
        "accuracy": accuracy,
        "validation_accuracy": accuracy,
        "classification_report": report_dict,
        "confusion_matrix": matrix.tolist(),
        "split_mode": split_mode,
        "trained_on_all_segments": True,
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
    plt.title(f"Box Presence Classifier ({accuracy:.2%} accuracy)")
    plt.tight_layout()
    plt.savefig(confusion_out, dpi=180)
    plt.close()

    print(f"Usable trails: {len(used_npz_paths)}")
    print(f"Usable segments: {len(examples)}")
    print(f"Class counts: {class_counts}")
    print(f"Feature count: {len(feature_names)}")
    print(f"Split mode: {split_mode}; train={len(X_train)}, test={len(X_test)}")
    if split_mode == "segment":
        print(
            "Warning: segment-level split was explicitly allowed. Accuracy may be "
            "inflated because train and test windows can come from the same recording."
        )
    if skipped:
        print(f"Skipped items: {len(skipped)}")
        for item in skipped[:10]:
            print(f"  - {item.get('npz_path', '')}: {item['reason']}")
    print(report_text)
    print(f"Accuracy: {accuracy:.3f}")
    print("Saved model is retrained on all usable segments after validation.")
    print(f"Saved model: {model_out}")
    print(f"Saved confusion matrix: {confusion_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
