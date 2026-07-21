#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np

from box_lab_common import (
    INPUT_TYPE,
    MODELS_DIR,
    extract_box_feature_vector,
    load_trial_data,
    read_manifests,
    resolve_npz_path,
    timestamp,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a classifier for static mmWave box-content trials."
    )
    parser.add_argument(
        "datasets",
        nargs="+",
        help="One or more dataset folders created by collect_box_dataset.py.",
    )
    parser.add_argument("--test-size", type=float, default=0.25)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--min-range", type=float, default=0.0)
    parser.add_argument("--max-range", type=float, default=0.60)
    parser.add_argument("--range-bins", type=int, default=64)
    parser.add_argument("--db", dest="db", action="store_true", default=True)
    parser.add_argument("--no-db", dest="db", action="store_false")
    parser.add_argument("--include-points", dest="include_points", action="store_true", default=True)
    parser.add_argument("--no-points", dest="include_points", action="store_false")
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
        "range_bins": int(args.range_bins),
        "db": bool(args.db),
        "include_points": bool(args.include_points),
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


def build_examples(rows: list[dict[str, str]], feature_params: dict):
    examples: list[list[float]] = []
    labels: list[str] = []
    npz_paths: list[str] = []
    used_rows: list[dict[str, str]] = []
    skipped: list[dict[str, str]] = []
    feature_names: list[str] | None = None

    for row in rows:
        label = row.get("contents") or row.get("label") or ""
        npz_path = resolve_npz_path(row)
        if not npz_path.exists():
            skipped.append({"contents": label, "npz_path": str(npz_path), "reason": "missing trial_data.npz"})
            continue

        try:
            trial_data = load_trial_data(npz_path)
            features, names = extract_box_feature_vector(trial_data, feature_params)
        except Exception as exc:
            skipped.append({"contents": label, "npz_path": str(npz_path), "reason": f"feature error: {exc}"})
            continue

        if features is None:
            skipped.append({"contents": label, "npz_path": str(npz_path), "reason": "not enough usable data"})
            continue
        if feature_names is None:
            feature_names = names
        elif len(names) != len(feature_names):
            skipped.append({"contents": label, "npz_path": str(npz_path), "reason": "feature length mismatch"})
            continue

        examples.append(features)
        labels.append(label)
        npz_paths.append(str(npz_path))
        used_rows.append(row)

    return examples, labels, npz_paths, feature_names or [], skipped, used_rows


def single_numeric_value(rows: list[dict[str, str]], field: str, default: float) -> float:
    values: list[float] = []
    for row in rows:
        value = row.get(field)
        if value in (None, ""):
            continue
        try:
            values.append(float(value))
        except (TypeError, ValueError):
            continue
    unique = sorted(set(values))
    if len(unique) == 1:
        return unique[0]
    return default


def default_model_path(dataset_dirs: list[Path], classifier: str) -> Path:
    name = f"{classifier}_box_contents_{timestamp()}.joblib"
    if len(dataset_dirs) == 1:
        return dataset_dirs[0] / "models" / name
    return MODELS_DIR / name


def main() -> int:
    args = parse_args()
    if args.min_range >= args.max_range:
        raise SystemExit("--min-range must be smaller than --max-range.")
    if args.range_bins < 2:
        raise SystemExit("--range-bins must be at least 2.")

    dataset_dirs = [Path(item).expanduser().resolve() for item in args.datasets]
    rows, missing_datasets = read_manifests(dataset_dirs)
    if not rows:
        raise SystemExit("No trials found. Expected trials.csv or trial_metadata.json files.")

    feature_params = feature_params_for_args(args)
    examples, labels, npz_paths, feature_names, skipped, used_rows = build_examples(
        rows,
        feature_params,
    )
    if len(set(labels)) < 2:
        raise SystemExit("Need at least two box-content classes to train.")
    if len(examples) < 4:
        raise SystemExit("Need at least four usable trials to train.")

    try:
        import joblib
        import matplotlib.pyplot as plt
        from sklearn.metrics import (
            ConfusionMatrixDisplay,
            accuracy_score,
            classification_report,
            confusion_matrix,
        )
        from sklearn.model_selection import train_test_split
    except ImportError as exc:
        raise SystemExit(f"Training dependencies missing: {exc}") from exc

    X = np.asarray(examples, dtype=float)
    y = np.asarray(labels)
    class_counts = {label: int((y == label).sum()) for label in sorted(set(labels))}
    stratify = y if min(class_counts.values()) >= 2 else None
    split_test_size = args.test_size
    if stratify is not None:
        n_classes = len(class_counts)
        n_samples = len(y)
        requested = int(math.ceil(args.test_size * n_samples)) if args.test_size < 1 else int(args.test_size)
        split_test_size = max(n_classes, requested)
        split_test_size = min(split_test_size, n_samples - n_classes)

    indices = np.arange(len(y))
    train_idx, test_idx = train_test_split(
        indices,
        test_size=split_test_size,
        random_state=args.random_state,
        stratify=stratify,
    )
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    if len(set(y_train)) < 2:
        raise SystemExit("Training split needs at least two classes.")

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
        "frames_per_trial": int(single_numeric_value(used_rows, "frames_per_trial", 20)),
        "point_distance_m": float(single_numeric_value(used_rows, "point_distance_m", 0.20)),
        "point_window_m": float(single_numeric_value(used_rows, "point_window_m", 0.05)),
        "class_counts": class_counts,
        "labels_order": labels_order,
        "accuracy": accuracy,
        "classification_report": report_dict,
        "confusion_matrix": matrix.tolist(),
        "datasets": [str(path) for path in dataset_dirs],
        "trial_npz_paths": npz_paths,
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
    plt.title(f"Box Contents Classifier ({accuracy:.2%} accuracy)")
    plt.tight_layout()
    plt.savefig(confusion_out, dpi=180)
    plt.close()

    print(f"Usable trials: {len(examples)}")
    print(f"Class counts: {class_counts}")
    if skipped:
        print(f"Skipped trials: {len(skipped)}")
        for item in skipped[:10]:
            print(f"  - {item['contents']} {item['npz_path']}: {item['reason']}")
    print(f"Feature count: {len(feature_names)}")
    print(f"Split: train={len(X_train)}, test={len(X_test)}")
    print(report_text)
    print(f"Accuracy: {accuracy:.3f}")
    print(f"Saved model: {model_out}")
    print(f"Saved confusion matrix: {confusion_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
