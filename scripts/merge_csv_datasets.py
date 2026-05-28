#!/usr/bin/env python3
"""Merge multiple synced CSV datasets into one CSV file.

Designed for files like synced_dataset2.csv produced in this project.

Features:
- Accept multiple CSV files and/or directories as inputs
- If an input is a directory, recursively search for a target filename
- Merge headers by union, so slightly different CSV schemas can still merge
- Add source tracking columns for each merged row
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List, Sequence, Tuple


DEFAULT_SEARCH_NAME = "synced_dataset2.csv"
SOURCE_COLUMNS = ["merged_frame_index", "source_dataset_id", "source_csv_path"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge multiple synced CSV datasets into one file.")
    parser.add_argument(
        "inputs",
        nargs="+",
        help="CSV file paths and/or directories containing dataset CSV files",
    )
    parser.add_argument(
        "--output-csv",
        required=True,
        help="Path to merged output CSV",
    )
    parser.add_argument(
        "--search-name",
        default=DEFAULT_SEARCH_NAME,
        help=f"When an input is a directory, recursively search for this filename. Default: {DEFAULT_SEARCH_NAME}",
    )
    parser.add_argument(
        "--recursive-glob",
        default=None,
        help="Optional custom recursive glob pattern for directory inputs, e.g. '**/*.csv'",
    )
    parser.add_argument(
        "--drop-source-columns",
        action="store_true",
        help="Do not add merged_frame_index/source_dataset_id/source_csv_path columns",
    )
    return parser.parse_args()


def collect_csv_paths(inputs: Sequence[str], search_name: str, recursive_glob: str | None) -> List[Path]:
    collected: List[Path] = []

    for raw_input in inputs:
        path = Path(raw_input).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"Input path not found: {path}")

        if path.is_file():
            if path.suffix.lower() != ".csv":
                raise ValueError(f"Input file is not a CSV: {path}")
            collected.append(path)
            continue

        pattern = recursive_glob if recursive_glob else f"**/{search_name}"
        matches = sorted(path.glob(pattern))
        csv_matches = [match.resolve() for match in matches if match.is_file() and match.suffix.lower() == ".csv"]
        collected.extend(csv_matches)

    deduped = sorted(set(collected))
    if not deduped:
        raise RuntimeError("No CSV files found to merge")
    return deduped


def read_csv_rows(csv_path: Path) -> Tuple[List[str], List[Dict[str, str]]]:
    with csv_path.open("r", newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        if reader.fieldnames is None:
            raise RuntimeError(f"CSV has no header: {csv_path}")
        return list(reader.fieldnames), list(reader)


def build_merged_fieldnames(headers: Sequence[Sequence[str]], include_source_columns: bool) -> List[str]:
    merged: List[str] = []
    if include_source_columns:
        merged.extend(SOURCE_COLUMNS)

    for header in headers:
        for field in header:
            if field not in merged:
                merged.append(field)
    return merged


def main() -> None:
    args = parse_args()
    csv_paths = collect_csv_paths(args.inputs, args.search_name, args.recursive_glob)

    headers: List[List[str]] = []
    all_rows: List[Dict[str, str]] = []
    include_source_columns = not args.drop_source_columns
    merged_frame_index = 0

    for dataset_id, csv_path in enumerate(csv_paths):
        header, rows = read_csv_rows(csv_path)
        headers.append(header)

        for row in rows:
            merged_row = dict(row)
            if include_source_columns:
                merged_row["merged_frame_index"] = str(merged_frame_index)
                merged_row["source_dataset_id"] = str(dataset_id)
                merged_row["source_csv_path"] = str(csv_path)
            all_rows.append(merged_row)
            merged_frame_index += 1

    output_csv = Path(args.output_csv).expanduser().resolve()
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = build_merged_fieldnames(headers, include_source_columns)

    with output_csv.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in all_rows:
            complete_row = {field: row.get(field, "") for field in fieldnames}
            writer.writerow(complete_row)

    print(f"Merged CSV: {output_csv}")
    print(f"Input CSV count: {len(csv_paths)}")
    print(f"Merged rows: {len(all_rows)}")
    print("Source files:")
    for index, csv_path in enumerate(csv_paths):
        print(f"  [{index}] {csv_path}")


if __name__ == "__main__":
    main()
