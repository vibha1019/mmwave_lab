#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="TODO: plot average posture accuracy for each classifier."
    )
    parser.add_argument(
        "models",
        nargs="+",
        help="One or more .joblib model files saved by train_posture.py.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("posture_classifier_accuracy.png"),
        help="Output bar-chart image.",
    )
    return parser.parse_args()


def load_accuracy_records(model_paths: list[Path]) -> list[tuple[str, float]]:
    """Return (classifier_label, accuracy) records from saved posture models."""
    # TODO: import joblib, load each model payload, and read:
    # payload["classifier_label"] or payload["classifier"]
    # payload["accuracy"]
    raise NotImplementedError("TODO: load classifier names and accuracies.")


def average_accuracy_by_classifier(
    records: list[tuple[str, float]],
) -> list[tuple[str, float]]:
    """Average repeated runs for each classifier."""
    # TODO: group records by classifier name and compute the mean accuracy.
    raise NotImplementedError("TODO: average accuracy for each classifier.")


def plot_accuracy_bar_chart(summary: list[tuple[str, float]], out_path: Path) -> None:
    """Save a bar chart of average accuracy per classifier."""
    # TODO: use matplotlib to plot classifier names on x and accuracy on y.
    # Label the y-axis and keep the range between 0 and 1.
    raise NotImplementedError("TODO: plot and save the accuracy bar chart.")


def main() -> int:
    args = parse_args()
    model_paths = [Path(item).expanduser().resolve() for item in args.models]
    records = load_accuracy_records(model_paths)
    summary = average_accuracy_by_classifier(records)
    plot_accuracy_bar_chart(summary, args.out)
    print(f"Saved plot: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
