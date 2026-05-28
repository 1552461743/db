#!/usr/bin/env python3
"""Print basic frame count and duration info for a dataset CSV."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, Optional, Tuple


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Show frame count and duration for a CSV dataset.")
    parser.add_argument("csv_path", help="Path to the CSV file")
    return parser.parse_args()


def get_timestamp_seconds(row: dict) -> Optional[float]:
    if row.get("image_timestamp_ns"):
        return float(row["image_timestamp_ns"]) / 1_000_000_000.0
    if row.get("image_timestamp_sec"):
        return float(row["image_timestamp_sec"])
    return None


def update_group_bounds(
    group_bounds: Dict[str, Tuple[float, float]],
    group_key: str,
    timestamp_sec: float,
) -> None:
    if group_key not in group_bounds:
        group_bounds[group_key] = (timestamp_sec, timestamp_sec)
        return

    first_ts, last_ts = group_bounds[group_key]
    group_bounds[group_key] = (min(first_ts, timestamp_sec), max(last_ts, timestamp_sec))


def main() -> None:
    args = parse_args()
    csv_path = Path(args.csv_path).expanduser().resolve()
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    frame_count = 0
    first_ts = None
    last_ts = None
    group_bounds: Dict[str, Tuple[float, float]] = {}

    with csv_path.open("r", newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        if reader.fieldnames is None:
            raise RuntimeError(f"CSV has no header: {csv_path}")

        for row in reader:
            frame_count += 1
            timestamp_sec = get_timestamp_seconds(row)
            if timestamp_sec is None:
                continue
            if first_ts is None:
                first_ts = timestamp_sec
            last_ts = timestamp_sec

            group_key = (row.get("source_dataset_id") or "single_dataset").strip() or "single_dataset"
            update_group_bounds(group_bounds, group_key, timestamp_sec)

    span_duration_sec = 0.0
    if first_ts is not None and last_ts is not None:
        span_duration_sec = max(0.0, last_ts - first_ts)

    active_duration_sec = 0.0
    for group_first_ts, group_last_ts in group_bounds.values():
        active_duration_sec += max(0.0, group_last_ts - group_first_ts)

    print(f"CSV: {csv_path}")
    print(f"Frames: {frame_count}")
    if first_ts is None:
        print("Duration: unavailable (no image_timestamp_ns/image_timestamp_sec column)")
    else:
        if len(group_bounds) > 1:
            print(f"Source datasets: {len(group_bounds)}")
        print(f"Span duration (seconds): {span_duration_sec:.6f}")
        print(f"Span duration (minutes): {span_duration_sec / 60.0:.6f}")
        print(f"Active duration (seconds): {active_duration_sec:.6f}")
        print(f"Active duration (minutes): {active_duration_sec / 60.0:.6f}")


if __name__ == "__main__":
    main()
