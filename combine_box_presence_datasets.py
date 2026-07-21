#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

from box_lab_common import LAB_DIR, PRESENCE_INPUT_TYPE, timestamp


FIELDNAMES = [
    "dataset_name",
    "source_dataset",
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
        description="Combine multiple box-presence dataset manifests."
    )
    parser.add_argument(
        "datasets",
        nargs="+",
        help="Dataset folders created by collect_box_presence_dataset.py.",
    )
    parser.add_argument(
        "--output",
        default=str(LAB_DIR / "datasets" / f"combined_box_presence_dataset_{timestamp()}"),
        help="Output dataset folder. Default: datasets/combined_box_presence_dataset_<timestamp>",
    )
    parser.add_argument(
        "--label",
        action="append",
        help="Keep only this label. Can be repeated or comma separated.",
    )
    parser.add_argument(
        "--allow-missing-session",
        action="store_true",
        help="Keep rows even if the referenced session_dir or npz_path does not exist.",
    )
    return parser.parse_args()


def read_rows(dataset_dir: Path) -> list[dict[str, str]]:
    manifest = dataset_dir / "trials.csv"
    if not manifest.exists():
        raise FileNotFoundError(f"missing trials.csv: {manifest}")
    with manifest.open(newline="") as file:
        return list(csv.DictReader(file))


def normalize_filter(values: list[str] | None) -> set[str] | None:
    if not values:
        return None
    normalized: set[str] = set()
    for value in values:
        for part in value.split(","):
            label = part.strip()
            if label:
                normalized.add(label)
    return normalized


def source_dataset_name(dataset_dir: Path) -> str:
    metadata_path = dataset_dir / "dataset_metadata.json"
    if metadata_path.exists():
        try:
            metadata = json.loads(metadata_path.read_text())
            return str(metadata.get("dataset_name") or dataset_dir.name)
        except json.JSONDecodeError:
            pass
    return dataset_dir.name


def resolve_path(dataset_dir: Path, value: str) -> Path:
    path = Path(value) if value else Path("")
    if path.is_absolute():
        return path
    return (dataset_dir / path).resolve()


def resolve_session_and_npz(dataset_dir: Path, row: dict[str, str]) -> tuple[Path, Path]:
    session_dir = resolve_path(dataset_dir, row.get("session_dir", ""))
    npz_value = row.get("npz_path", "")
    npz_path = resolve_path(dataset_dir, npz_value) if npz_value else session_dir / "trial_data.npz"
    return session_dir, npz_path


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_manifest = output_dir / "trials.csv"

    allowed_labels = normalize_filter(args.label)
    source_dirs = [Path(item).expanduser().resolve() for item in args.datasets]
    combined: list[dict[str, str]] = []
    skipped: list[dict[str, str]] = []
    label_counts: dict[str, int] = {}

    for dataset_dir in source_dirs:
        try:
            rows = read_rows(dataset_dir)
        except FileNotFoundError as exc:
            skipped.append({"dataset": str(dataset_dir), "reason": str(exc)})
            continue

        source_name = source_dataset_name(dataset_dir)
        for row in rows:
            label = row.get("label", "")
            if allowed_labels and label not in allowed_labels:
                continue

            session_dir, npz_path = resolve_session_and_npz(dataset_dir, row)
            if (
                not args.allow_missing_session
                and (not session_dir.exists() or not npz_path.exists())
            ):
                skipped.append(
                    {
                        "dataset": str(dataset_dir),
                        "session_dir": str(session_dir),
                        "npz_path": str(npz_path),
                        "reason": "missing session_dir or npz_path",
                    }
                )
                continue

            merged_row = {name: row.get(name, "") for name in FIELDNAMES}
            merged_row["dataset_name"] = output_dir.name
            merged_row["source_dataset"] = source_name
            merged_row["label"] = label
            merged_row["input_type"] = row.get("input_type", PRESENCE_INPUT_TYPE)
            merged_row["session_dir"] = str(session_dir)
            merged_row["npz_path"] = str(npz_path)
            combined.append(merged_row)
            label_counts[label] = label_counts.get(label, 0) + 1

    with output_manifest.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(combined)

    metadata = {
        "dataset_name": output_dir.name,
        "input_type": PRESENCE_INPUT_TYPE,
        "combined_from": [str(path) for path in source_dirs],
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "row_count": len(combined),
        "label_counts": label_counts,
        "label_filter": sorted(allowed_labels) if allowed_labels else None,
        "skipped": skipped,
    }
    (output_dir / "dataset_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n"
    )

    print(json.dumps(metadata, indent=2, sort_keys=True))
    if not combined:
        print("No rows were combined.", file=sys.stderr)
        return 1
    print(f"Combined manifest: {output_manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
