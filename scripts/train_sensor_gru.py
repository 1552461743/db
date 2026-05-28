#!/usr/bin/env python3
"""Train a GRU regressor from suit sensor CSV data to HybrIK joint targets.

Default behavior:
- Inputs: capacitance, normalized capacitance, IMU accel/gyro, IMU magnetic,
  and yaw columns from the synchronized CSV.
- Targets: HybrIK SMPL joint xyz + joint rotation matrices.
- Temporal model: sliding window GRU, predicting the target of the last frame
  in each window.

The script saves:
- best_model.pt: checkpoint with model weights, selected columns, and scalers
- metrics.json: train/val/test metrics and configuration summary
- loss_curve.png: train/eval loss curves with run parameters
"""

from __future__ import annotations

import argparse
import cv2
import csv
import json
import random
import re
import socket
import sys
import time
from datetime import datetime
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

SUIT_ROOT = Path(__file__).resolve().parents[1]
RUNTIME_ROOT = SUIT_ROOT / "hybrik_runtime"
LEGACY_REPO_ROOT = Path(__file__).resolve().parents[3]
REPO_ROOT = (
    RUNTIME_ROOT
    if (RUNTIME_ROOT / "hybrik").exists()
    else SUIT_ROOT
    if (SUIT_ROOT / "hybrik").exists()
    else LEGACY_REPO_ROOT
)
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hybrik.models.layers.smpl.SMPL import SMPL_layer
from hybrik.models.layers.smpl.lbs import lbs
from hybrik.utils.render_pytorch3d import render_mesh_single_frame


INPUT_GROUP_PATTERNS = {
    "capacitance": re.compile(r"^sensor_capacitance_\d+$"),
    "normalized": re.compile(r"^sensor_normalized_\d+$"),
    "imu": re.compile(
        r"^imu_channel_\d+_(linear_acceleration|angular_velocity)_(x|y|z)$"
    ),
    "mag": re.compile(r"^imu_channel_\d+_magnetic_field_magnetic_field_(x|y|z)$"),
    "yaw": re.compile(r"^imu_channel_\d+_yaw_tilt_deg_data$"),
    "relative": re.compile(r"^imu_relative_transform_\d+$"),
}

GLOBAL_TARGET_PREFIXES = ("hybrik_transl_", "hybrik_cam_root_")
NON_JOINT_TARGET_PREFIXES = ("hybrik_status", "hybrik_bbox_")
SMPL_JOINT_NAMES = [
    "pelvis",
    "left_hip",
    "right_hip",
    "spine1",
    "left_knee",
    "right_knee",
    "spine2",
    "left_ankle",
    "right_ankle",
    "spine3",
    "left_foot",
    "right_foot",
    "neck",
    "left_collar",
    "right_collar",
    "jaw",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_thumb",
    "right_thumb",
]
UPPER_BODY_JOINT_NAMES = [
    "pelvis",
    "spine1",
    "spine2",
    "spine3",
    "neck",
    "left_collar",
    "right_collar",
    "jaw",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_thumb",
    "right_thumb",
]
TARGET_JOINT_PRESETS = {
    "all": SMPL_JOINT_NAMES,
    "upper_body": UPPER_BODY_JOINT_NAMES,
}


@dataclass
class SplitMetrics:
    loss: float
    mae: float
    rmse: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a GRU model that maps suit sensor sequences to HybrIK joint targets."
    )
    parser.add_argument("csv_path", help="Path to synced_dataset2.csv")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory to save the trained model. Default: <csv_dir>/gru_training",
    )
    parser.add_argument(
        "--input-groups",
        nargs="+",
        default=["capacitance", "normalized", "imu", "mag", "yaw"],
        choices=sorted(INPUT_GROUP_PATTERNS.keys()),
        help="Sensor column groups to use as inputs",
    )
    parser.add_argument(
        "--target-mode",
        default="both",
        choices=["xyz", "rotmat", "both"],
        help="Joint target type to learn",
    )
    parser.add_argument(
        "--target-joints",
        default="upper_body",
        choices=sorted(TARGET_JOINT_PRESETS.keys()),
        help="SMPL joint subset to learn. upper_body keeps lower-body joints out of the model output.",
    )
    parser.add_argument(
        "--include-global-targets",
        action="store_true",
        help="Also learn HybrIK transl/cam_root targets in addition to per-joint targets",
    )
    parser.add_argument("--seq-len", type=int, default=20, help="GRU input sequence length")
    parser.add_argument("--stride", type=int, default=1, help="Sliding window stride")
    parser.add_argument("--batch-size", type=int, default=64, help="Training batch size")
    parser.add_argument("--epochs", type=int, default=60, help="Training epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Adam learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Adam weight decay")
    parser.add_argument("--hidden-size", type=int, default=256, help="GRU hidden size")
    parser.add_argument("--num-layers", type=int, default=2, help="GRU layer count")
    parser.add_argument("--dropout", type=float, default=0.2, help="Dropout rate")
    parser.add_argument(
        "--bidirectional",
        action="store_true",
        help="Use a bidirectional GRU",
    )
    parser.add_argument("--train-ratio", type=float, default=0.8, help="Training frame ratio")
    parser.add_argument("--val-ratio", type=float, default=0.0, help="Validation frame ratio")
    parser.add_argument("--test-ratio", type=float, default=0.2, help="Test frame ratio")
    parser.add_argument(
        "--split-mode",
        default="chronological",
        choices=["chronological", "random_segment"],
        help="chronological: train/val/test follow time order; random_segment: val/test are random contiguous segments",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="PyTorch dataloader worker count",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Torch device, e.g. cpu or cuda:0. Default: auto-detect",
    )
    parser.add_argument(
        "--udp-replay-test",
        action="store_true",
        help="After loading the best checkpoint, replay test-set predictions over UDP at fixed FPS.",
    )
    parser.add_argument("--udp-host", default="127.0.0.1", help="UDP destination host for replay")
    parser.add_argument("--udp-port", type=int, default=5005, help="UDP destination port for replay")
    parser.add_argument("--udp-fps", type=float, default=30.0, help="Replay FPS for UDP test streaming")
    parser.add_argument(
        "--visualize-test-smpl",
        action="store_true",
        help="Replay test-set predictions in two windows: original image and predicted SMPL mesh.",
    )
    parser.add_argument(
        "--visualize-fps",
        type=float,
        default=10.0,
        help="FPS for SMPL test visualization replay",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def natural_sort_key(text: str) -> List[object]:
    parts = re.split(r"(\d+)", text)
    key: List[object] = []
    for part in parts:
        if part.isdigit():
            key.append(int(part))
        else:
            key.append(part)
    return key


def is_joint_xyz_column(name: str) -> bool:
    if not name.startswith("hybrik_"):
        return False
    if name == "hybrik_status":
        return False
    if name.startswith("hybrik_bbox_"):
        return False
    if name.startswith(GLOBAL_TARGET_PREFIXES):
        return False
    return bool(re.match(r"^hybrik_.+_(x|y|z)_m$", name))


def is_joint_rotmat_column(name: str) -> bool:
    if not name.startswith("hybrik_"):
        return False
    return bool(re.match(r"^hybrik_.+_rotmat_\d\d$", name))


def column_joint_name(name: str) -> str | None:
    xyz_match = re.match(r"^hybrik_(.+)_(x|y|z)_m$", name)
    if xyz_match:
        return xyz_match.group(1)
    rot_match = re.match(r"^hybrik_(.+)_rotmat_\d\d$", name)
    if rot_match:
        return rot_match.group(1)
    return None


def select_input_columns(header: Sequence[str], groups: Sequence[str]) -> List[str]:
    columns: List[str] = []
    for group in groups:
        pattern = INPUT_GROUP_PATTERNS[group]
        matches = [name for name in header if pattern.match(name)]
        columns.extend(sorted(matches, key=natural_sort_key))
    if not columns:
        raise RuntimeError("No input columns matched the selected input groups")
    return columns


def select_target_columns(
    header: Sequence[str],
    target_mode: str,
    include_global_targets: bool,
    target_joints: str,
) -> List[str]:
    columns: List[str] = []
    allowed_joints = set(TARGET_JOINT_PRESETS[target_joints])

    if target_mode in ("xyz", "both"):
        xyz_columns = [
            name
            for name in header
            if is_joint_xyz_column(name) and column_joint_name(name) in allowed_joints
        ]
        columns.extend(sorted(xyz_columns, key=natural_sort_key))

    if target_mode in ("rotmat", "both"):
        rot_columns = [
            name
            for name in header
            if is_joint_rotmat_column(name) and column_joint_name(name) in allowed_joints
        ]
        columns.extend(sorted(rot_columns, key=natural_sort_key))

    if include_global_targets:
        global_columns = [
            name
            for name in header
            if name.startswith("hybrik_transl_") or name.startswith("hybrik_cam_root_")
        ]
        columns = sorted(global_columns, key=natural_sort_key) + columns

    if not columns:
        raise RuntimeError("No target columns matched the selected target mode")
    return columns


def read_header(csv_path: Path) -> List[str]:
    with csv_path.open("r", newline="", encoding="utf-8") as csv_file:
        reader = csv.reader(csv_file)
        header = next(reader)
    return header


def load_numeric_rows(
    csv_path: Path,
    input_columns: Sequence[str],
    target_columns: Sequence[str],
) -> Tuple[np.ndarray, np.ndarray, List[str], int]:
    inputs: List[List[float]] = []
    targets: List[List[float]] = []
    image_paths: List[str] = []
    dropped_rows = 0

    needed_columns = list(input_columns) + list(target_columns)

    with csv_path.open("r", newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            if row is None:
                dropped_rows += 1
                continue

            if "hybrik_status" in row and row["hybrik_status"] not in ("", "ok"):
                dropped_rows += 1
                continue

            try:
                input_values = [float(row[column]) for column in input_columns]
                target_values = [float(row[column]) for column in target_columns]
            except (KeyError, TypeError, ValueError):
                dropped_rows += 1
                continue

            numeric_row = np.array(input_values + target_values, dtype=np.float32)
            if not np.isfinite(numeric_row).all():
                dropped_rows += 1
                continue

            inputs.append(input_values)
            targets.append(target_values)
            image_paths.append((row.get("image_path") or "").strip())

    if not inputs:
        raise RuntimeError(
            "No usable numeric rows found in CSV. Check whether the selected columns are present and complete."
        )

    _ = needed_columns
    return np.asarray(inputs, dtype=np.float32), np.asarray(targets, dtype=np.float32), image_paths, dropped_rows


def split_frame_ranges(
    num_frames: int,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    split_mode: str,
    seed: int,
) -> Dict[str, List[Tuple[int, int]]]:
    ratios = (train_ratio, val_ratio, test_ratio)
    if any(ratio < 0.0 or ratio >= 1.0 for ratio in ratios):
        raise ValueError("train_ratio, val_ratio, and test_ratio must be in [0,1)")
    if abs(sum(ratios) - 1.0) > 1e-6:
        raise ValueError("train_ratio + val_ratio + test_ratio must equal 1.0")

    val_count = int(num_frames * val_ratio)
    test_count = int(num_frames * test_ratio)
    train_count = num_frames - val_count - test_count

    if split_mode == "chronological":
        train_end = train_count
        val_end = train_count + val_count

        return {
            "train": [(0, train_end)],
            "val": [] if val_count == 0 else [(train_end, val_end)],
            "test": [] if test_count == 0 else [(val_end, num_frames)],
        }

    rng = random.Random(seed)
    held_out_segments: List[Tuple[str, int, int]] = []

    for split_name, split_count in sorted(
        [("val", val_count), ("test", test_count)], key=lambda item: item[1], reverse=True
    ):
        if split_count == 0:
            continue

        valid_starts: List[int] = []
        max_start = num_frames - split_count
        for start_idx in range(max_start + 1):
            end_idx = start_idx + split_count
            overlaps = any(not (end_idx <= seg_start or start_idx >= seg_end) for _, seg_start, seg_end in held_out_segments)
            if not overlaps:
                valid_starts.append(start_idx)

        if not valid_starts:
            raise RuntimeError(
                "Unable to sample non-overlapping random contiguous val/test segments with the current ratios."
            )

        start_idx = rng.choice(valid_starts)
        held_out_segments.append((split_name, start_idx, start_idx + split_count))

    held_out_segments.sort(key=lambda item: item[1])

    train_ranges: List[Tuple[int, int]] = []
    cursor = 0
    for _, seg_start, seg_end in held_out_segments:
        if cursor < seg_start:
            train_ranges.append((cursor, seg_start))
        cursor = seg_end
    if cursor < num_frames:
        train_ranges.append((cursor, num_frames))

    return {
        "train": train_ranges,
        "val": [(start, end) for split_name, start, end in held_out_segments if split_name == "val"],
        "test": [(start, end) for split_name, start, end in held_out_segments if split_name == "test"],
    }


def build_windows(
    features: np.ndarray,
    targets: np.ndarray,
    seq_len: int,
    stride: int,
    start_idx: int,
    end_idx: int,
) -> Tuple[np.ndarray, np.ndarray]:
    if end_idx - start_idx < seq_len:
        return np.empty((0, seq_len, features.shape[1]), dtype=np.float32), np.empty(
            (0, targets.shape[1]), dtype=np.float32
        )

    xs: List[np.ndarray] = []
    ys: List[np.ndarray] = []
    last_start = end_idx - seq_len

    for window_start in range(start_idx, last_start + 1, stride):
        window_end = window_start + seq_len
        xs.append(features[window_start:window_end])
        ys.append(targets[window_end - 1])

    return np.stack(xs), np.stack(ys)


def build_windows_from_ranges(
    features: np.ndarray,
    targets: np.ndarray,
    seq_len: int,
    stride: int,
    ranges: Sequence[Tuple[int, int]],
) -> Tuple[np.ndarray, np.ndarray]:
    xs_parts: List[np.ndarray] = []
    ys_parts: List[np.ndarray] = []

    for start_idx, end_idx in ranges:
        split_x, split_y = build_windows(features, targets, seq_len, stride, start_idx, end_idx)
        if len(split_x) == 0:
            continue
        xs_parts.append(split_x)
        ys_parts.append(split_y)

    if not xs_parts:
        return np.empty((0, seq_len, features.shape[1]), dtype=np.float32), np.empty(
            (0, targets.shape[1]), dtype=np.float32
        )

    return np.concatenate(xs_parts, axis=0), np.concatenate(ys_parts, axis=0)


def build_target_frame_indices_from_ranges(
    seq_len: int,
    stride: int,
    ranges: Sequence[Tuple[int, int]],
) -> np.ndarray:
    indices: List[int] = []
    for start_idx, end_idx in ranges:
        if end_idx - start_idx < seq_len:
            continue
        last_start = end_idx - seq_len
        for window_start in range(start_idx, last_start + 1, stride):
            indices.append(window_start + seq_len - 1)
    return np.asarray(indices, dtype=np.int64)


def compute_scalers(train_x: np.ndarray, train_y: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x_mean = train_x.reshape(-1, train_x.shape[-1]).mean(axis=0)
    x_std = train_x.reshape(-1, train_x.shape[-1]).std(axis=0)
    y_mean = train_y.mean(axis=0)
    y_std = train_y.std(axis=0)

    x_std = np.where(x_std < 1e-6, 1.0, x_std)
    y_std = np.where(y_std < 1e-6, 1.0, y_std)
    return x_mean, x_std, y_mean, y_std


def normalize_windows(array: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return (array - mean.reshape(1, 1, -1)) / std.reshape(1, 1, -1)


def normalize_targets(array: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return (array - mean.reshape(1, -1)) / std.reshape(1, -1)


class GRUPoseRegressor(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        output_size: int,
        dropout: float,
        bidirectional: bool,
    ) -> None:
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
            batch_first=True,
        )
        gru_output_size = hidden_size * (2 if bidirectional else 1)
        self.head = nn.Sequential(
            nn.Linear(gru_output_size, hidden_size),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, output_size),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        outputs, _ = self.gru(inputs)
        last_state = outputs[:, -1, :]
        return self.head(last_state)


def make_loader(
    x: np.ndarray,
    y: np.ndarray,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
) -> DataLoader:
    dataset = TensorDataset(torch.from_numpy(x).float(), torch.from_numpy(y).float())
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)


def compute_denorm_metrics(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    target_mean: torch.Tensor,
    target_std: torch.Tensor,
) -> Tuple[float, float]:
    pred_denorm = predictions * target_std + target_mean
    target_denorm = targets * target_std + target_mean
    mae = torch.mean(torch.abs(pred_denorm - target_denorm)).item()
    rmse = torch.sqrt(torch.mean((pred_denorm - target_denorm) ** 2)).item()
    return mae, rmse


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    device: torch.device,
    target_mean: torch.Tensor,
    target_std: torch.Tensor,
) -> SplitMetrics:
    model.train()
    total_loss = 0.0
    total_mae = 0.0
    total_rmse = 0.0
    total_samples = 0

    progress = tqdm(loader, desc="Train", leave=False)
    for batch_inputs, batch_targets in progress:
        batch_inputs = batch_inputs.to(device)
        batch_targets = batch_targets.to(device)

        optimizer.zero_grad(set_to_none=True)
        predictions = model(batch_inputs)
        loss = loss_fn(predictions, batch_targets)
        loss.backward()
        optimizer.step()

        batch_size = batch_inputs.shape[0]
        mae, rmse = compute_denorm_metrics(predictions, batch_targets, target_mean, target_std)
        total_loss += loss.item() * batch_size
        total_mae += mae * batch_size
        total_rmse += rmse * batch_size
        total_samples += batch_size
        progress.set_postfix(loss=f"{loss.item():.4f}")

    return SplitMetrics(
        loss=total_loss / max(total_samples, 1),
        mae=total_mae / max(total_samples, 1),
        rmse=total_rmse / max(total_samples, 1),
    )


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
    target_mean: torch.Tensor,
    target_std: torch.Tensor,
) -> SplitMetrics:
    model.eval()
    total_loss = 0.0
    total_mae = 0.0
    total_rmse = 0.0
    total_samples = 0

    for batch_inputs, batch_targets in loader:
        batch_inputs = batch_inputs.to(device)
        batch_targets = batch_targets.to(device)
        predictions = model(batch_inputs)
        loss = loss_fn(predictions, batch_targets)

        batch_size = batch_inputs.shape[0]
        mae, rmse = compute_denorm_metrics(predictions, batch_targets, target_mean, target_std)
        total_loss += loss.item() * batch_size
        total_mae += mae * batch_size
        total_rmse += rmse * batch_size
        total_samples += batch_size

    return SplitMetrics(
        loss=total_loss / max(total_samples, 1),
        mae=total_mae / max(total_samples, 1),
        rmse=total_rmse / max(total_samples, 1),
    )


def save_checkpoint(
    output_path: Path,
    model: nn.Module,
    args: argparse.Namespace,
    input_columns: Sequence[str],
    target_columns: Sequence[str],
    x_mean: np.ndarray,
    x_std: np.ndarray,
    y_mean: np.ndarray,
    y_std: np.ndarray,
    best_eval: SplitMetrics,
) -> None:
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "args": vars(args),
        "input_columns": list(input_columns),
        "target_columns": list(target_columns),
        "x_mean": x_mean.tolist(),
        "x_std": x_std.tolist(),
        "y_mean": y_mean.tolist(),
        "y_std": y_std.tolist(),
        "best_eval_metrics": asdict(best_eval),
    }
    torch.save(checkpoint, output_path)


def format_ranges(ranges: Sequence[Tuple[int, int]]) -> str:
    if not ranges:
        return "[]"
    return ", ".join(f"[{start}:{end})" for start, end in ranges)


def save_loss_plot(
    output_path: Path,
    history: Sequence[Dict[str, object]],
    args: argparse.Namespace,
    best_epoch: int,
    best_eval: SplitMetrics,
    has_val_split: bool,
    final_test: SplitMetrics,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epochs = [entry["epoch"] for entry in history]
    train_losses = [entry["train"]["loss"] for entry in history]
    eval_losses = [entry["eval"]["loss"] for entry in history]

    figure, axis = plt.subplots(figsize=(12, 7))
    axis.plot(epochs, train_losses, label="train_loss", linewidth=2)
    axis.plot(epochs, eval_losses, label="val_loss" if has_val_split else "eval_loss", linewidth=2)

    axis.axvline(best_epoch, color="tab:red", linestyle="--", linewidth=1.5, label=f"best_epoch={best_epoch}")
    axis.scatter([best_epoch], [best_eval.loss], color="tab:red", zorder=5)

    axis.set_title("GRU Training Loss Curve")
    axis.set_xlabel("Epoch")
    axis.set_ylabel("Loss")
    axis.grid(True, alpha=0.3)
    axis.legend()

    param_lines = [
        f"seq_len={args.seq_len}",
        f"stride={args.stride}",
        f"batch_size={args.batch_size}",
        f"epochs={args.epochs}",
        f"lr={args.lr}",
        f"weight_decay={args.weight_decay}",
        f"hidden_size={args.hidden_size}",
        f"num_layers={args.num_layers}",
        f"dropout={args.dropout}",
        f"bidirectional={args.bidirectional}",
        f"split_mode={args.split_mode}",
        f"train/val/test={args.train_ratio}/{args.val_ratio}/{args.test_ratio}",
        f"input_groups={','.join(args.input_groups)}",
        f"target_mode={args.target_mode}",
        f"target_joints={args.target_joints}",
        f"best_loss={best_eval.loss:.6f}",
        f"final_test_loss={final_test.loss:.6f}",
        f"final_test_mae={final_test.mae:.6f}",
        f"final_test_rmse={final_test.rmse:.6f}",
    ]
    figure.text(
        0.68,
        0.5,
        "\n".join(param_lines),
        fontsize=10,
        va="center",
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.9},
    )

    figure.tight_layout(rect=(0.0, 0.0, 0.66, 1.0))
    figure.savefig(output_path, dpi=200)
    plt.close(figure)


def build_udp_payload_vector(target_columns: Sequence[str], prediction_row: np.ndarray) -> List[float]:
    value_map = {column: float(value) for column, value in zip(target_columns, prediction_row.tolist())}

    has_transl = all(name in value_map for name in ["hybrik_transl_x_m", "hybrik_transl_y_m", "hybrik_transl_z_m"])
    payload: List[float] = []

    if has_transl:
        payload.extend(
            [
                value_map["hybrik_transl_x_m"],
                value_map["hybrik_transl_y_m"],
                value_map["hybrik_transl_z_m"],
            ]
        )

    for joint_name in SMPL_JOINT_NAMES:
        for axis in ("x", "y", "z"):
            column = f"hybrik_{joint_name}_{axis}_m"
            payload.append(value_map.get(column, 0.0))

    for joint_name in SMPL_JOINT_NAMES:
        for row in range(3):
            for col in range(3):
                column = f"hybrik_{joint_name}_rotmat_{row}{col}"
                payload.append(value_map.get(column, 1.0 if row == col else 0.0))

    return payload


def prediction_row_to_value_map(target_columns: Sequence[str], prediction_row: np.ndarray) -> Dict[str, float]:
    return {column: float(value) for column, value in zip(target_columns, prediction_row.tolist())}


def build_smpl_rotation_tensor(value_map: Dict[str, float]) -> np.ndarray:
    rotmats = np.zeros((len(SMPL_JOINT_NAMES), 3, 3), dtype=np.float32)
    for joint_index, joint_name in enumerate(SMPL_JOINT_NAMES):
        for row in range(3):
            for col in range(3):
                column = f"hybrik_{joint_name}_rotmat_{row}{col}"
                rotmats[joint_index, row, col] = value_map.get(column, 1.0 if row == col else 0.0)
    return rotmats


def create_smpl_renderer(device: torch.device) -> Dict[str, object]:
    h36m_jregressor = np.load(str(REPO_ROOT / "model_files/J_regressor_h36m.npy"))
    smpl_layer = SMPL_layer(
        str(REPO_ROOT / "model_files/basicModel_neutral_lbs_10_207_0_v1.0.0.pkl"),
        h36m_jregressor=h36m_jregressor,
        dtype=torch.float32,
    ).to(device)
    smpl_layer.eval()
    zero_betas = torch.zeros((1, 10), dtype=torch.float32, device=device)
    faces = torch.from_numpy(smpl_layer.faces.astype(np.int64)).to(device)
    return {
        "smpl_layer": smpl_layer,
        "zero_betas": zero_betas,
        "faces": faces,
    }


@torch.no_grad()
def render_prediction_as_smpl(
    renderer_state: Dict[str, object],
    rotmats: np.ndarray,
    device: torch.device,
    render_size: int = 720,
) -> np.ndarray:
    smpl_layer = renderer_state["smpl_layer"]
    zero_betas = renderer_state["zero_betas"]
    faces = renderer_state["faces"]

    pose_tensor = torch.from_numpy(rotmats).float().unsqueeze(0).to(device)
    vertices, joints, _, _ = lbs(
        zero_betas,
        pose_tensor,
        smpl_layer.v_template,
        smpl_layer.shapedirs,
        smpl_layer.posedirs,
        smpl_layer.J_regressor,
        smpl_layer.J_regressor_h36m,
        smpl_layer.parents,
        smpl_layer.lbs_weights,
        pose2rot=False,
        dtype=torch.float32,
    )
    vertices = vertices - joints[:, [0], :]
    translation = torch.tensor([[0.0, 0.0, 2.5]], dtype=torch.float32, device=device)

    render = render_mesh_single_frame(
        vertices=vertices,
        faces=faces,
        translation=translation,
        focal_length=1200.0,
        height=render_size,
        width=render_size,
        device=device,
    )
    render = render.detach().cpu().numpy()
    valid_mask = render[:, :, 3:4] > 0
    color = (render[:, :, :3] * 255.0).astype(np.uint8)
    background = np.full_like(color, 30, dtype=np.uint8)
    image = color * valid_mask.astype(np.uint8) + background * (~valid_mask).astype(np.uint8)
    return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)


@torch.no_grad()
def visualize_test_predictions_as_smpl(
    model: nn.Module,
    test_x: np.ndarray,
    test_frame_indices: np.ndarray,
    image_paths: Sequence[str],
    device: torch.device,
    y_mean: np.ndarray,
    y_std: np.ndarray,
    target_columns: Sequence[str],
    fps: float,
) -> int:
    if fps <= 0.0:
        raise ValueError("visualize_fps must be > 0")

    model.eval()
    interval = 1.0 / fps
    renderer_state = create_smpl_renderer(device)
    image_window = "Test Image"
    smpl_window = "Predicted SMPL"
    cv2.namedWindow(image_window, cv2.WINDOW_NORMAL)
    cv2.namedWindow(smpl_window, cv2.WINDOW_NORMAL)
    shown_count = 0

    try:
        progress = tqdm(range(len(test_x)), desc="SMPL visualize", leave=False)
        for sample_index in progress:
            start_time = time.perf_counter()
            batch_inputs = torch.from_numpy(test_x[sample_index : sample_index + 1]).float().to(device)
            prediction = model(batch_inputs).cpu().numpy()[0]
            prediction = prediction * y_std + y_mean
            value_map = prediction_row_to_value_map(target_columns, prediction)
            rotmats = build_smpl_rotation_tensor(value_map)
            smpl_image = render_prediction_as_smpl(renderer_state, rotmats, device)

            frame_index = int(test_frame_indices[sample_index]) if sample_index < len(test_frame_indices) else -1
            raw_image = None
            if 0 <= frame_index < len(image_paths):
                image_path = image_paths[frame_index]
                if image_path:
                    raw_image = cv2.imread(image_path, cv2.IMREAD_COLOR)

            if raw_image is None:
                raw_image = np.full((720, 1280, 3), 30, dtype=np.uint8)
                cv2.putText(
                    raw_image,
                    "image not found",
                    (40, 80),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )

            cv2.putText(
                raw_image,
                f"test sample {sample_index}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                smpl_image,
                f"test sample {sample_index}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

            cv2.imshow(image_window, raw_image)
            cv2.imshow(smpl_window, smpl_image)
            if cv2.waitKey(1) & 0xFF == 27:
                break

            shown_count += 1
            elapsed = time.perf_counter() - start_time
            remaining = interval - elapsed
            if remaining > 0.0:
                time.sleep(remaining)
    finally:
        try:
            cv2.destroyWindow(image_window)
            cv2.destroyWindow(smpl_window)
        except Exception:
            pass

    return shown_count


@torch.no_grad()
def replay_test_predictions_over_udp(
    model: nn.Module,
    test_x: np.ndarray,
    test_frame_indices: np.ndarray,
    image_paths: Sequence[str],
    device: torch.device,
    y_mean: np.ndarray,
    y_std: np.ndarray,
    target_columns: Sequence[str],
    udp_host: str,
    udp_port: int,
    udp_fps: float,
) -> int:
    if udp_fps <= 0.0:
        raise ValueError("udp_fps must be > 0")

    model.eval()
    interval = 1.0 / udp_fps
    udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sent_count = 0
    window_name = "GRU Test Replay"
    show_image = len(image_paths) > 0

    if show_image:
        try:
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        except Exception:
            show_image = False

    try:
        progress = tqdm(range(len(test_x)), desc="UDP replay", leave=False)
        for sample_index in progress:
            start_time = time.perf_counter()
            batch_inputs = torch.from_numpy(test_x[sample_index : sample_index + 1]).float().to(device)
            prediction = model(batch_inputs).cpu().numpy()[0]
            prediction = prediction * y_std + y_mean
            payload = {
                "seq": sample_index,
                "data": build_udp_payload_vector(target_columns, prediction),
            }
            udp_socket.sendto(json.dumps(payload).encode("utf-8"), (udp_host, udp_port))

            if show_image and sample_index < len(test_frame_indices):
                frame_index = int(test_frame_indices[sample_index])
                if 0 <= frame_index < len(image_paths):
                    image_path = image_paths[frame_index]
                    if image_path:
                        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
                        if image is not None:
                            cv2.imshow(window_name, image)
                            cv2.waitKey(1)

            sent_count += 1

            elapsed = time.perf_counter() - start_time
            remaining = interval - elapsed
            if remaining > 0.0:
                time.sleep(remaining)
    finally:
        udp_socket.close()
        if show_image:
            try:
                cv2.destroyWindow(window_name)
            except Exception:
                pass

    return sent_count


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    csv_path = Path(args.csv_path).expanduser().resolve()
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else csv_path.parent / "gru_training"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device) if args.device else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    header = read_header(csv_path)
    input_columns = select_input_columns(header, args.input_groups)
    target_columns = select_target_columns(header, args.target_mode, args.include_global_targets, args.target_joints)

    features, targets, image_paths, dropped_rows = load_numeric_rows(csv_path, input_columns, target_columns)
    num_frames = features.shape[0]
    active_split_count = 2 + int(args.val_ratio > 0.0)
    if num_frames < args.seq_len * active_split_count:
        raise RuntimeError(
            f"Not enough usable frames ({num_frames}) for seq_len={args.seq_len} and split count={active_split_count}"
        )

    frame_ranges = split_frame_ranges(
        num_frames,
        args.train_ratio,
        args.val_ratio,
        args.test_ratio,
        args.split_mode,
        args.seed,
    )
    train_x, train_y = build_windows_from_ranges(
        features, targets, args.seq_len, args.stride, frame_ranges["train"]
    )
    val_x, val_y = build_windows_from_ranges(
        features, targets, args.seq_len, args.stride, frame_ranges["val"]
    )
    test_x, test_y = build_windows_from_ranges(
        features, targets, args.seq_len, args.stride, frame_ranges["test"]
    )
    test_frame_indices = build_target_frame_indices_from_ranges(
        args.seq_len, args.stride, frame_ranges["test"]
    )

    has_val_split = len(val_x) > 0
    if len(train_x) == 0 or len(test_x) == 0 or (args.val_ratio > 0.0 and not has_val_split):
        raise RuntimeError(
            "At least one required split has no windows. Reduce seq_len or adjust split ratios."
        )

    x_mean, x_std, y_mean, y_std = compute_scalers(train_x, train_y)
    train_x = normalize_windows(train_x, x_mean, x_std)
    if has_val_split:
        val_x = normalize_windows(val_x, x_mean, x_std)
    test_x = normalize_windows(test_x, x_mean, x_std)
    train_y = normalize_targets(train_y, y_mean, y_std)
    if has_val_split:
        val_y = normalize_targets(val_y, y_mean, y_std)
    test_y = normalize_targets(test_y, y_mean, y_std)

    train_loader = make_loader(train_x, train_y, args.batch_size, True, args.num_workers)
    val_loader = make_loader(val_x, val_y, args.batch_size, False, args.num_workers) if has_val_split else None
    test_loader = make_loader(test_x, test_y, args.batch_size, False, args.num_workers)

    model = GRUPoseRegressor(
        input_size=train_x.shape[-1],
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        output_size=train_y.shape[-1],
        dropout=args.dropout,
        bidirectional=args.bidirectional,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, verbose=True
    )
    loss_fn = nn.MSELoss()

    target_mean_tensor = torch.from_numpy(y_mean).float().to(device)
    target_std_tensor = torch.from_numpy(y_std).float().to(device)

    best_eval = SplitMetrics(loss=float("inf"), mae=float("inf"), rmse=float("inf"))
    best_epoch = -1
    history: List[Dict[str, object]] = []
    checkpoint_path = output_dir / "best_model.pt"

    print(f"CSV: {csv_path}")
    print(f"Device: {device}")
    print(f"Input columns: {len(input_columns)}")
    print(f"Target columns: {len(target_columns)}")
    print(f"Usable frames: {num_frames}")
    print(f"Dropped CSV rows: {dropped_rows}")
    print(f"Split mode: {args.split_mode}")
    print(f"Train ranges: {format_ranges(frame_ranges['train'])}")
    if frame_ranges["val"]:
        print(f"Val ranges: {format_ranges(frame_ranges['val'])}")
    print(f"Test ranges: {format_ranges(frame_ranges['test'])}")
    if has_val_split:
        print(
            f"Windows train/val/test: {len(train_x)}/{len(val_x)}/{len(test_x)} | seq_len={args.seq_len}"
        )
    else:
        print(f"Windows train/test: {len(train_x)}/{len(test_x)} | seq_len={args.seq_len}")

    for epoch in range(1, args.epochs + 1):
        print(f"Epoch {epoch}/{args.epochs}")
        train_metrics = train_one_epoch(
            model,
            train_loader,
            optimizer,
            loss_fn,
            device,
            target_mean_tensor,
            target_std_tensor,
        )
        if has_val_split:
            eval_metrics = evaluate(
                model,
                val_loader,
                loss_fn,
                device,
                target_mean_tensor,
                target_std_tensor,
            )
            scheduler.step(eval_metrics.loss)
        else:
            eval_metrics = train_metrics
            scheduler.step(train_metrics.loss)

        history.append(
            {
                "epoch": epoch,
                "train": asdict(train_metrics),
                "eval": asdict(eval_metrics),
                "eval_split": "val" if has_val_split else "train",
                "lr": optimizer.param_groups[0]["lr"],
            }
        )

        if has_val_split:
            print(
                "  "
                f"train_loss={train_metrics.loss:.6f} train_mae={train_metrics.mae:.6f} train_rmse={train_metrics.rmse:.6f} | "
                f"val_loss={eval_metrics.loss:.6f} val_mae={eval_metrics.mae:.6f} val_rmse={eval_metrics.rmse:.6f}"
            )
        else:
            print(
                "  "
                f"train_loss={train_metrics.loss:.6f} train_mae={train_metrics.mae:.6f} train_rmse={train_metrics.rmse:.6f}"
            )

        if eval_metrics.loss < best_eval.loss:
            best_eval = eval_metrics
            best_epoch = epoch
            save_checkpoint(
                checkpoint_path,
                model,
                args,
                input_columns,
                target_columns,
                x_mean,
                x_std,
                y_mean,
                y_std,
                best_eval,
            )
            print(f"  Saved new best checkpoint to {checkpoint_path}")

    best_checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(best_checkpoint["model_state_dict"])
    test_metrics = evaluate(
        model,
        test_loader,
        loss_fn,
        device,
        target_mean_tensor,
        target_std_tensor,
    )

    smpl_visualized_frames = 0
    if args.visualize_test_smpl:
        print(f"Starting SMPL test visualization at {args.visualize_fps:.2f} FPS")
        smpl_visualized_frames = visualize_test_predictions_as_smpl(
            model=model,
            test_x=test_x,
            test_frame_indices=test_frame_indices,
            image_paths=image_paths,
            device=device,
            y_mean=y_mean,
            y_std=y_std,
            target_columns=target_columns,
            fps=args.visualize_fps,
        )
        print(f"SMPL visualization finished. Shown {smpl_visualized_frames} frames.")

    udp_replay_sent = 0
    if args.udp_replay_test:
        print(
            f"Starting UDP replay of test-set predictions to {args.udp_host}:{args.udp_port} at {args.udp_fps:.2f} FPS"
        )
        udp_replay_sent = replay_test_predictions_over_udp(
            model=model,
            test_x=test_x,
            test_frame_indices=test_frame_indices,
            image_paths=image_paths,
            device=device,
            y_mean=y_mean,
            y_std=y_std,
            target_columns=target_columns,
            udp_host=args.udp_host,
            udp_port=args.udp_port,
            udp_fps=args.udp_fps,
        )
        print(f"UDP replay finished. Sent {udp_replay_sent} frames.")

    metrics_summary = {
        "csv_path": str(csv_path),
        "output_dir": str(output_dir),
        "device": str(device),
        "input_columns": list(input_columns),
        "target_columns": list(target_columns),
        "num_input_features": len(input_columns),
        "num_target_features": len(target_columns),
        "usable_frames": int(num_frames),
        "dropped_rows": int(dropped_rows),
        "frame_ranges": {split_name: list(ranges) for split_name, ranges in frame_ranges.items()},
        "num_train_windows": int(len(train_x)),
        "num_val_windows": int(len(val_x)),
        "num_test_windows": int(len(test_x)),
        "best_epoch": int(best_epoch),
        "best_eval": asdict(best_eval),
        "best_eval_split": "val" if has_val_split else "train",
        "test": asdict(test_metrics),
        "udp_replay": {
            "enabled": bool(args.udp_replay_test),
            "host": args.udp_host,
            "port": int(args.udp_port),
            "fps": float(args.udp_fps),
            "sent_frames": int(udp_replay_sent),
        },
        "smpl_visualization": {
            "enabled": bool(args.visualize_test_smpl),
            "fps": float(args.visualize_fps),
            "shown_frames": int(smpl_visualized_frames),
        },
        "history": history,
        "args": vars(args),
    }

    metrics_path = output_dir / "metrics.json"
    with metrics_path.open("w", encoding="utf-8") as metrics_file:
        json.dump(metrics_summary, metrics_file, indent=2, ensure_ascii=False)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_path = output_dir / f"loss_curve_{timestamp}.png"
    save_loss_plot(plot_path, history, args, best_epoch, best_eval, has_val_split, test_metrics)

    print("Training finished.")
    print(f"Best checkpoint: {checkpoint_path}")
    print(f"Metrics summary: {metrics_path}")
    print(f"Loss curve: {plot_path}")
    if has_val_split:
        print(
            f"Best val loss={best_eval.loss:.6f}, val_mae={best_eval.mae:.6f}, val_rmse={best_eval.rmse:.6f}"
        )
    else:
        print(
            f"Best train loss={best_eval.loss:.6f}, train_mae={best_eval.mae:.6f}, train_rmse={best_eval.rmse:.6f}"
        )
    print(
        f"Test loss={test_metrics.loss:.6f}, test_mae={test_metrics.mae:.6f}, test_rmse={test_metrics.rmse:.6f}"
    )


if __name__ == "__main__":
    main()
