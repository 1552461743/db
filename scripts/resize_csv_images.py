#!/usr/bin/env python3
"""Resize dataset images in place to 640x480 with aspect-ratio preservation.

Behavior:
- Accept one or more CSV files, image files, and/or directories
- If an input is a directory, it can be either:
  - a dataset directory containing CSV files
  - or an image directory containing image files
- Read image paths from CSV files, or directly from image directories/files
- Resize images to fit inside 640x480 without cropping
- Pad remaining area with a solid color
- Overwrite the original image files in place

CSV files are not modified.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
from pathlib import Path
from typing import Iterable, List, Sequence, Set, Tuple

import cv2
import numpy as np
from tqdm import tqdm


DEFAULT_SEARCH_NAME = "synced_dataset2.csv"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Resize dataset images in place to a fixed size without cropping.")
    parser.add_argument("inputs", nargs="+", help="CSV paths, image paths, and/or directories")
    parser.add_argument("--image-column", default="image_path", help="CSV image path column")
    parser.add_argument("--width", type=int, default=640, help="Target width")
    parser.add_argument("--height", type=int, default=480, help="Target height")
    parser.add_argument(
        "--num-workers",
        type=int,
        default=1,
        help="Number of parallel CPU workers for image processing",
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
        "--pad-color",
        nargs=3,
        type=int,
        default=[0, 0, 0],
        metavar=("B", "G", "R"),
        help="Padding color in BGR order",
    )
    return parser.parse_args()


def is_image_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES


def collect_inputs(
    inputs: Sequence[str],
    search_name: str,
    recursive_glob: str | None,
) -> Tuple[List[Path], Set[Path]]:
    collected: List[Path] = []
    direct_images: Set[Path] = set()

    for raw_input in inputs:
        path = Path(raw_input).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"Input path not found: {path}")

        if path.is_file():
            if path.suffix.lower() == ".csv":
                collected.append(path)
            elif is_image_file(path):
                direct_images.add(path)
            else:
                raise ValueError(f"Unsupported input file type: {path}")
            continue

        pattern = recursive_glob if recursive_glob else f"**/{search_name}"
        matches = sorted(path.glob(pattern))
        csv_matches = [match.resolve() for match in matches if match.is_file() and match.suffix.lower() == ".csv"]

        if csv_matches:
            collected.extend(csv_matches)
            continue

        image_matches = sorted(match.resolve() for match in path.glob("**/*") if is_image_file(match))
        if image_matches:
            direct_images.update(image_matches)
            continue

        raise RuntimeError(f"No CSV or image files found under directory: {path}")

    deduped = sorted(set(collected))
    if not deduped and not direct_images:
        raise RuntimeError("No CSV or image files found to process")
    return deduped, direct_images


def resolve_image_path(csv_path: Path, image_value: str) -> Path:
    image_path = Path(image_value).expanduser()
    if not image_path.is_absolute():
        image_path = (csv_path.parent / image_path).resolve()
    return image_path


def resize_with_letterbox(
    image: np.ndarray,
    target_width: int,
    target_height: int,
    pad_color: Tuple[int, int, int],
) -> np.ndarray:
    src_height, src_width = image.shape[:2]
    scale = min(target_width / src_width, target_height / src_height)

    resized_width = max(1, int(round(src_width * scale)))
    resized_height = max(1, int(round(src_height * scale)))
    resized = cv2.resize(image, (resized_width, resized_height), interpolation=cv2.INTER_AREA)

    canvas = np.full((target_height, target_width, 3), pad_color, dtype=np.uint8)
    x_offset = (target_width - resized_width) // 2
    y_offset = (target_height - resized_height) // 2
    canvas[y_offset:y_offset + resized_height, x_offset:x_offset + resized_width] = resized
    return canvas


def collect_image_paths_from_csv(
    csv_path: Path,
    image_column: str,
) -> Tuple[int, Set[Path]]:
    seen_paths = 0
    image_paths: Set[Path] = set()

    with csv_path.open("r", newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        if reader.fieldnames is None:
            raise RuntimeError(f"CSV has no header: {csv_path}")
        if image_column not in reader.fieldnames:
            raise KeyError(f"Image column '{image_column}' not found in CSV: {csv_path}")

        for row in reader:
            image_value = (row.get(image_column) or "").strip()
            if not image_value:
                continue

            image_path = resolve_image_path(csv_path, image_value)
            seen_paths += 1
            image_paths.add(image_path)

    return seen_paths, image_paths


def process_image(
    image_path: Path,
    target_width: int,
    target_height: int,
    pad_color: Tuple[int, int, int],
) -> None:
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Failed to read image: {image_path}")

    resized = resize_with_letterbox(image, target_width, target_height, pad_color)
    if not cv2.imwrite(str(image_path), resized):
        raise RuntimeError(f"Failed to overwrite image: {image_path}")


def process_images_in_parallel(
    image_paths: Iterable[Path],
    target_width: int,
    target_height: int,
    pad_color: Tuple[int, int, int],
    num_workers: int,
) -> int:
    image_paths = list(image_paths)
    if num_workers <= 1:
        for image_path in tqdm(image_paths, desc="Resizing images", unit="img"):
            process_image(image_path, target_width, target_height, pad_color)
        return len(image_paths)

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = [
            executor.submit(process_image, image_path, target_width, target_height, pad_color)
            for image_path in image_paths
        ]
        for future in tqdm(futures, desc="Resizing images", unit="img"):
            future.result()
    return len(image_paths)


def print_input_summary(csv_count: int, image_count: int) -> None:
    print(f"CSV files processed: {csv_count}")
    print(f"Unique resized images: {image_count}")


def main() -> None:
    args = parse_args()
    if args.num_workers < 1:
        raise ValueError("--num-workers must be >= 1")

    csv_paths, direct_images = collect_inputs(args.inputs, args.search_name, args.recursive_glob)
    pad_color = tuple(int(value) for value in args.pad_color)
    all_images: Set[Path] = set(direct_images)
    total_seen = 0

    for csv_path in csv_paths:
        seen_count, image_paths = collect_image_paths_from_csv(
            csv_path=csv_path,
            image_column=args.image_column,
        )
        total_seen += seen_count
        new_image_count = len(image_paths - all_images)
        all_images.update(image_paths)
        print(f"Processed CSV: {csv_path}")
        print(f"  Referenced images: {seen_count}")
        print(f"  Newly queued images: {new_image_count}")

    if direct_images:
        print(f"Direct image inputs: {len(direct_images)}")

    total_processed = process_images_in_parallel(
        image_paths=sorted(all_images),
        target_width=args.width,
        target_height=args.height,
        pad_color=pad_color,
        num_workers=args.num_workers,
    )

    print_input_summary(len(csv_paths), total_processed)
    if csv_paths:
        print(f"Referenced image rows: {total_seen}")
    print(f"Target size: {args.width}x{args.height}")
    print(f"Parallel workers: {args.num_workers}")
    print("Images overwritten in place.")


if __name__ == "__main__":
    main()
