#!/usr/bin/env python3
"""Extract supported ROS2 bag topics into an image-anchored CSV dataset.

This script is tailored to the three collection programs in this directory:
- image.py publishes a raw image topic (default: /dongbu)
- esp32.py publishes capacitance, IMU, magnetic field, and yaw topics
- two.cpp publishes normalized sensors and relative IMU transforms

Synchronization rule:
- Use each image frame as the anchor timestamp.
- For every other supported topic, pick the message whose timestamp is closest
  to that image timestamp.
- If a topic has no data, or an optional max delta threshold is exceeded, the
  corresponding CSV fields are left empty.

Notes:
- Messages with a ROS header use header.stamp for synchronization.
- Headerless messages (e.g. Float32MultiArray / Float32) fall back to the bag
  record timestamp.
"""

from __future__ import annotations

import argparse
import bisect
import csv
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import rosbag2_py
import yaml
from cv_bridge import CvBridge
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message
from sensor_msgs.msg import Image, Imu, MagneticField
from std_msgs.msg import Float32, Float32MultiArray
from tqdm import tqdm


FIXED_SUPPORTED_TOPICS = {
    "/sensor/capacitance",
    "/sensor/normalized",
    "/imu/relative_transform",
}

IMU_TOPIC_RE = re.compile(r"^/imu/channel_(\d+)$")
MAG_TOPIC_RE = re.compile(r"^/imu/channel_(\d+)/magnetic_field$")
YAW_TOPIC_RE = re.compile(r"^/imu/channel_(\d+)/yaw_tilt_deg$")


@dataclass
class TopicRecord:
    timestamp_ns: int
    payload: Dict[str, object]


@dataclass
class TopicSeries:
    topic: str
    type_name: str
    prefix: str
    records: List[TopicRecord] = field(default_factory=list)
    payload_keys: List[str] = field(default_factory=list)

    def add_record(self, timestamp_ns: int, payload: Dict[str, object]) -> None:
        if not self.payload_keys:
            self.payload_keys = list(payload.keys())
        else:
            for key in payload.keys():
                if key not in self.payload_keys:
                    self.payload_keys.append(key)
        self.records.append(TopicRecord(timestamp_ns=timestamp_ns, payload=payload))

    def sort_by_time(self) -> None:
        self.records.sort(key=lambda item: item.timestamp_ns)

    @property
    def timestamps(self) -> List[int]:
        return [record.timestamp_ns for record in self.records]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract ROS2 bag topics to a synchronized CSV using image timestamps as anchors."
    )
    parser.add_argument("bag_path", help="Path to the ROS2 bag directory")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory. Default: <bag_path>/csv_export",
    )
    parser.add_argument(
        "--image-topic",
        default="/dongbu",
        help="Image topic to use as synchronization anchor",
    )
    parser.add_argument(
        "--image-format",
        default="png",
        choices=["png", "jpg", "jpeg"],
        help="Saved image file format",
    )
    parser.add_argument(
        "--csv-name",
        default="synced_dataset.csv",
        help="Output CSV filename",
    )
    parser.add_argument(
        "--max-delta-ms",
        "--max-delta",
        type=float,
        default=None,
        help="Optional maximum allowed sync delta in milliseconds",
    )
    return parser.parse_args()


def resolve_bag_uri(bag_path: str) -> Path:
    path = Path(bag_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Bag path does not exist: {path}")
    if path.is_file() and path.name == "metadata.yaml":
        return path.parent
    if path.is_file():
        raise ValueError(
            "Please pass the ROS2 bag directory path (the folder containing metadata.yaml)."
        )
    metadata_path = path / "metadata.yaml"
    if not metadata_path.exists():
        raise FileNotFoundError(f"metadata.yaml not found in bag directory: {path}")
    return path


def infer_storage_id(bag_uri: Path) -> str:
    if list(bag_uri.glob("*.mcap")):
        return "mcap"
    if list(bag_uri.glob("*.db3")):
        return "sqlite3"
    raise RuntimeError(f"Unable to infer storage type for bag: {bag_uri}")


def is_supported_topic(topic: str, image_topic: str) -> bool:
    if topic == image_topic:
        return True
    if topic in FIXED_SUPPORTED_TOPICS:
        return True
    return bool(IMU_TOPIC_RE.match(topic) or MAG_TOPIC_RE.match(topic) or YAW_TOPIC_RE.match(topic))


def topic_prefix(topic: str, image_topic: str) -> str:
    if topic == image_topic:
        return "image"
    prefix = re.sub(r"[^0-9a-zA-Z]+", "_", topic).strip("_")
    return prefix or "topic"


def stamp_to_ns(stamp: object) -> int:
    sec = int(getattr(stamp, "sec", 0))
    nanosec = int(getattr(stamp, "nanosec", 0))
    return sec * 1_000_000_000 + nanosec


def message_timestamp_ns(msg: object, bag_timestamp_ns: int) -> int:
    header = getattr(msg, "header", None)
    if header is None:
        return int(bag_timestamp_ns)
    msg_stamp_ns = stamp_to_ns(header.stamp)
    return msg_stamp_ns if msg_stamp_ns > 0 else int(bag_timestamp_ns)


def sanitize_image_for_write(image_msg: Image, bridge: CvBridge):
    image = bridge.imgmsg_to_cv2(image_msg, desired_encoding="passthrough")
    encoding = image_msg.encoding.lower()

    if encoding == "rgb8":
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    elif encoding == "rgba8":
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2BGRA)
    elif encoding == "mono16":
        image = image.copy()

    return image


def flatten_message(topic: str, msg: object, image_topic: str) -> Dict[str, object]:
    prefix = topic_prefix(topic, image_topic)

    if isinstance(msg, Float32):
        return {f"{prefix}_data": float(msg.data)}

    if isinstance(msg, Float32MultiArray):
        return {f"{prefix}_{idx}": float(value) for idx, value in enumerate(msg.data)}

    if isinstance(msg, Imu):
        return {
            f"{prefix}_linear_acceleration_x": float(msg.linear_acceleration.x),
            f"{prefix}_linear_acceleration_y": float(msg.linear_acceleration.y),
            f"{prefix}_linear_acceleration_z": float(msg.linear_acceleration.z),
            f"{prefix}_angular_velocity_x": float(msg.angular_velocity.x),
            f"{prefix}_angular_velocity_y": float(msg.angular_velocity.y),
            f"{prefix}_angular_velocity_z": float(msg.angular_velocity.z),
        }

    if isinstance(msg, MagneticField):
        return {
            f"{prefix}_magnetic_field_x": float(msg.magnetic_field.x),
            f"{prefix}_magnetic_field_y": float(msg.magnetic_field.y),
            f"{prefix}_magnetic_field_z": float(msg.magnetic_field.z),
        }

    raise TypeError(f"Unsupported message type on topic {topic}: {type(msg)!r}")


def nearest_record(
    records: Sequence[TopicRecord],
    timestamps: Sequence[int],
    target_ns: int,
) -> Tuple[Optional[TopicRecord], Optional[int]]:
    if not records:
        return None, None

    index = bisect.bisect_left(timestamps, target_ns)
    candidates: List[TopicRecord] = []

    if index < len(records):
        candidates.append(records[index])
    if index > 0:
        candidates.append(records[index - 1])

    best = min(candidates, key=lambda item: abs(item.timestamp_ns - target_ns))
    return best, best.timestamp_ns - target_ns


def build_storage_reader(bag_uri: Path) -> rosbag2_py.SequentialReader:
    storage_id = infer_storage_id(bag_uri)
    storage_options = rosbag2_py.StorageOptions(uri=str(bag_uri), storage_id=storage_id)
    converter_options = rosbag2_py.ConverterOptions(
        input_serialization_format="cdr",
        output_serialization_format="cdr",
    )
    reader = rosbag2_py.SequentialReader()
    try:
        reader.open(storage_options, converter_options)
    except RuntimeError as exc:
        extra_files = []
        if storage_id == "sqlite3":
            extra_files = sorted(path.name for path in bag_uri.glob("*.db3"))
        elif storage_id == "mcap":
            extra_files = sorted(path.name for path in bag_uri.glob("*.mcap"))

        detail = str(exc)
        hint_parts = [
            f"Failed to open ROS2 bag: {bag_uri}",
            f"Storage type: {storage_id}",
        ]
        if extra_files:
            hint_parts.append(f"Bag files in directory: {', '.join(extra_files)}")
        if "malformed" in detail.lower():
            hint_parts.append(
                "The bag database looks corrupted, or metadata.yaml points to a damaged split file."
            )
        if len(extra_files) > 1:
            hint_parts.append(
                "There are multiple bag files in this directory. Check whether metadata.yaml points to the correct one."
            )
        hint_parts.append(f"Original rosbag2 error: {detail}")
        raise RuntimeError("\n".join(hint_parts)) from exc
    return reader


def read_total_message_count(bag_uri: Path) -> Optional[int]:
    metadata_path = bag_uri / "metadata.yaml"
    if not metadata_path.exists():
        return None

    try:
        with metadata_path.open("r", encoding="utf-8") as metadata_file:
            metadata = yaml.safe_load(metadata_file)
        bag_info = metadata.get("rosbag2_bagfile_information", {})
        total_messages = bag_info.get("message_count")
        if total_messages is None:
            return None
        return int(total_messages)
    except Exception:
        return None


def read_supported_topics(
    bag_uri: Path,
    image_topic: str,
    image_dir: Path,
    image_format: str,
) -> Tuple[Dict[str, TopicSeries], Dict[str, int]]:
    reader = build_storage_reader(bag_uri)
    bridge = CvBridge()
    total_messages = read_total_message_count(bag_uri)

    topic_type_map = {
        topic_info.name: topic_info.type for topic_info in reader.get_all_topics_and_types()
    }
    supported_topics = {
        topic: type_name
        for topic, type_name in topic_type_map.items()
        if is_supported_topic(topic, image_topic)
    }

    if image_topic not in supported_topics:
        raise RuntimeError(f"Image topic not found in bag: {image_topic}")

    message_classes = {
        topic: get_message(type_name) for topic, type_name in supported_topics.items()
    }
    series_map = {
        topic: TopicSeries(topic=topic, type_name=type_name, prefix=topic_prefix(topic, image_topic))
        for topic, type_name in supported_topics.items()
    }
    counts = {topic: 0 for topic in supported_topics}

    image_index = 0
    suffix = "jpg" if image_format == "jpeg" else image_format

    with tqdm(total=total_messages, desc="Reading bag", unit="msg") as progress:
        while reader.has_next():
            topic, serialized_data, bag_timestamp_ns = reader.read_next()
            progress.update(1)

            if topic not in supported_topics:
                continue

            msg = deserialize_message(serialized_data, message_classes[topic])
            timestamp_ns = message_timestamp_ns(msg, bag_timestamp_ns)
            counts[topic] += 1

            if topic == image_topic:
                image = sanitize_image_for_write(msg, bridge)
                image_name = f"frame_{image_index:06d}_{timestamp_ns}.{suffix}"
                image_path = image_dir / image_name
                if not cv2.imwrite(str(image_path), image):
                    raise RuntimeError(f"Failed to write image: {image_path}")

                payload = {
                    "image_path": str(image_path.resolve()),
                    "image_width": int(msg.width),
                    "image_height": int(msg.height),
                    "image_encoding": msg.encoding,
                }
                image_index += 1
            else:
                payload = flatten_message(topic, msg, image_topic)

            series_map[topic].add_record(timestamp_ns, payload)

    for series in series_map.values():
        series.sort_by_time()

    return series_map, counts


def write_synced_csv(
    csv_path: Path,
    series_map: Dict[str, TopicSeries],
    image_topic: str,
    max_delta_ms: Optional[float],
) -> int:
    image_series = series_map.get(image_topic)
    if image_series is None or not image_series.records:
        raise RuntimeError(f"No image data found on topic: {image_topic}")

    other_topics = sorted(topic for topic in series_map.keys() if topic != image_topic)
    image_records = image_series.records

    fieldnames: List[str] = [
        "frame_index",
        "image_timestamp_ns",
        "image_timestamp_sec",
    ]
    fieldnames.extend(image_series.payload_keys)

    for topic in other_topics:
        prefix = series_map[topic].prefix
        fieldnames.append(f"{prefix}_timestamp_ns")
        fieldnames.append(f"{prefix}_delta_ms")
        fieldnames.extend(series_map[topic].payload_keys)

    row_count = 0
    max_delta_ns = None if max_delta_ms is None else int(max_delta_ms * 1_000_000.0)
    timestamp_cache = {topic: series.timestamps for topic, series in series_map.items()}

    with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()

        for frame_index, image_record in enumerate(image_records):
            target_ns = image_record.timestamp_ns
            row: Dict[str, object] = {
                "frame_index": frame_index,
                "image_timestamp_ns": target_ns,
                "image_timestamp_sec": f"{target_ns / 1_000_000_000.0:.9f}",
            }
            row.update(image_record.payload)

            for topic in other_topics:
                series = series_map[topic]
                match, delta_ns = nearest_record(series.records, timestamp_cache[topic], target_ns)
                prefix = series.prefix

                if match is None or delta_ns is None:
                    continue

                if max_delta_ns is not None and abs(delta_ns) > max_delta_ns:
                    continue

                row[f"{prefix}_timestamp_ns"] = match.timestamp_ns
                row[f"{prefix}_delta_ms"] = f"{delta_ns / 1_000_000.0:.6f}"
                row.update(match.payload)

            writer.writerow(row)
            row_count += 1

    return row_count


def ensure_output_dirs(base_dir: Path) -> Tuple[Path, Path]:
    base_dir.mkdir(parents=True, exist_ok=True)
    image_dir = base_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    return base_dir, image_dir


def main() -> None:
    args = parse_args()
    bag_uri = resolve_bag_uri(args.bag_path)
    default_output_dir = bag_uri / "csv_export"
    output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else default_output_dir
    output_dir, image_dir = ensure_output_dirs(output_dir)
    csv_path = output_dir / args.csv_name

    series_map, counts = read_supported_topics(
        bag_uri=bag_uri,
        image_topic=args.image_topic,
        image_dir=image_dir,
        image_format=args.image_format,
    )
    row_count = write_synced_csv(
        csv_path=csv_path,
        series_map=series_map,
        image_topic=args.image_topic,
        max_delta_ms=args.max_delta_ms,
    )

    print(f"Bag directory: {bag_uri}")
    print(f"Output CSV: {csv_path}")
    print(f"Saved images: {image_dir}")
    print(f"Synchronized frames: {row_count}")
    print("Extracted topic counts:")
    for topic in sorted(counts.keys()):
        print(f"  {topic}: {counts[topic]}")


if __name__ == "__main__":
    main()
