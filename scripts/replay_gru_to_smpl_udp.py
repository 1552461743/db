#!/usr/bin/env python3
"""Replay GRU checkpoint inference and stream direct SMPL-24 UDP payloads.

This script lets you choose a trained GRU model under `data/mod`, a dataset
under `data/data`, a frame range, and then runs inference frame-by-frame while
sending direct SMPL joint predictions over UDP.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import socket
import sys
import time
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

import cv2
import numpy as np
import torch
from torch import nn

SCRIPT_PATH = Path(__file__).resolve()
SUIT_ROOT = SCRIPT_PATH.parents[1]
REPO_ROOT = SCRIPT_PATH.parents[3]
MOD_ROOT = SUIT_ROOT / "mod"
DATASET_ROOT = SUIT_ROOT / "data"
GUI_RENDER_PATH = SUIT_ROOT / "gui/smpl_joint_pose_gui.py"

for import_root in (SUIT_ROOT, REPO_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from GM.smpl_udp_common import SMPL_JOINT_NAMES, SMPL_UDP_FORMAT


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
        return self.head(outputs[:, -1, :])


def load_module_from_path(module_name: str, module_path: Path):
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load module spec from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


GUI_RENDER = load_module_from_path("smpl_gui_render_replay", GUI_RENDER_PATH)


def resolve_device(device_arg: str | None) -> torch.device:
    if device_arg:
        requested = torch.device(device_arg)
        if requested.type == "cuda" and not torch.cuda.is_available():
            return torch.device("cpu")
        return requested
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def natural_sort_key(text: str) -> List[object]:
    import re

    parts = re.split(r"(\d+)", text)
    key: List[object] = []
    for part in parts:
        key.append(int(part) if part.isdigit() else part)
    return key


def preferred_dataset_csv(dataset_dir: Path) -> Path | None:
    candidates = [
        dataset_dir / "csv_export/synced_dataset2.csv",
        dataset_dir / "merged_dataset.csv",
        dataset_dir / "csv_export/synced_dataset.csv",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def discover_models() -> Dict[str, Path]:
    models: Dict[str, Path] = {}
    for entry in sorted(MOD_ROOT.iterdir(), key=lambda path: natural_sort_key(path.name)):
        checkpoint_path = entry / "best_model.pt"
        if entry.is_dir() and checkpoint_path.exists():
            models[entry.name] = checkpoint_path
    return models


def discover_datasets() -> Dict[str, Path]:
    datasets: Dict[str, Path] = {}
    for entry in sorted(DATASET_ROOT.iterdir(), key=lambda path: natural_sort_key(path.name)):
        if not entry.is_dir():
            continue
        csv_path = preferred_dataset_csv(entry)
        if csv_path is not None:
            datasets[entry.name] = csv_path
    return datasets


def prompt_choice(title: str, options: Dict[str, Path]) -> Tuple[str, Path]:
    items = list(options.items())
    print(title)
    for index, (name, path) in enumerate(items, start=1):
        print(f"  {index:2d}. {name:<12} {path}")
    while True:
        raw = input("Enter number: ").strip()
        try:
            selected = int(raw)
        except ValueError:
            print("Please enter a valid integer.")
            continue
        if 1 <= selected <= len(items):
            return items[selected - 1]
        print(f"Please choose between 1 and {len(items)}.")


def prompt_int(label: str, default: int, minimum: int | None = None) -> int:
    while True:
        raw = input(f"{label} [{default}]: ").strip()
        if not raw:
            value = default
        else:
            try:
                value = int(raw)
            except ValueError:
                print("Please enter an integer.")
                continue
        if minimum is not None and value < minimum:
            print(f"Please enter a value >= {minimum}.")
            continue
        return value


def prompt_float(label: str, default: float, minimum: float | None = None) -> float:
    while True:
        raw = input(f"{label} [{default}]: ").strip()
        if not raw:
            value = default
        else:
            try:
                value = float(raw)
            except ValueError:
                print("Please enter a number.")
                continue
        if minimum is not None and value < minimum:
            print(f"Please enter a value >= {minimum}.")
            continue
        return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Replay GRU checkpoint inference and stream direct SMPL-24 UDP payloads.")
    parser.add_argument("--model", default=None, help="Model name under data/mod or full checkpoint path")
    parser.add_argument("--dataset", default=None, help="Dataset name under data/data, dataset directory, or CSV path")
    parser.add_argument("--start-frame", type=int, default=None, help="Usable feature-frame index to start replaying")
    parser.add_argument("--frame-count", type=int, default=None, help="Number of frames to replay")
    parser.add_argument("--fps", type=float, default=None, help="UDP replay FPS")
    parser.add_argument("--udp-host", default="127.0.0.1", help="UDP destination host")
    parser.add_argument("--udp-port", type=int, default=5007, help="UDP destination port")
    parser.add_argument(
        "--udp-target",
        action="append",
        default=[],
        metavar="HOST:PORT",
        help="Additional UDP destination. Repeat to send each SMPL frame to multiple robot receivers.",
    )
    parser.add_argument("--device", default=None, help="Torch device, e.g. cpu or cuda:0")
    parser.add_argument("--hide-window", action="store_true", help="Disable original-image and SMPL preview windows")
    parser.add_argument(
        "--render-mode",
        default="skeleton",
        choices=["skeleton", "mesh"],
        help="SMPL preview mode. skeleton is much faster than mesh.",
    )
    parser.add_argument("--render-size", type=int, default=640, help="SMPL preview size in pixels")
    parser.add_argument("--list", action="store_true", help="List available models and datasets then exit")
    parser.add_argument("--interactive", action="store_true", help="Force interactive prompts even when arguments are provided")
    return parser.parse_args()


def parse_udp_target(target: str) -> Tuple[str, int]:
    if ":" not in target:
        raise ValueError(f"UDP target must be HOST:PORT, got: {target}")
    host, port_text = target.rsplit(":", 1)
    if not host:
        raise ValueError(f"UDP target host is empty: {target}")
    try:
        port = int(port_text)
    except ValueError as exc:
        raise ValueError(f"UDP target port must be an integer, got: {target}") from exc
    if not (0 < port < 65536):
        raise ValueError(f"UDP target port out of range, got: {target}")
    return host, port


def resolve_udp_targets(args: argparse.Namespace) -> List[Tuple[str, int]]:
    if args.udp_target:
        return [parse_udp_target(target) for target in args.udp_target]
    return [(str(args.udp_host), int(args.udp_port))]


def resolve_model_checkpoint(model_arg: str | None, interactive: bool) -> Tuple[str, Path]:
    models = discover_models()
    if not models:
        raise RuntimeError(f"No checkpoints found under {MOD_ROOT}")

    if interactive or not model_arg:
        return prompt_choice("Available models:", models)

    model_path = Path(model_arg).expanduser().resolve()
    if model_path.exists():
        if model_path.is_dir():
            model_path = model_path / "best_model.pt"
        if not model_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {model_path}")
        return model_path.parent.name, model_path

    if model_arg in models:
        return model_arg, models[model_arg]

    raise FileNotFoundError(f"Unknown model '{model_arg}'. Use --list to inspect available checkpoints.")


def resolve_dataset_csv(dataset_arg: str | None, interactive: bool) -> Tuple[str, Path]:
    datasets = discover_datasets()
    if not datasets:
        raise RuntimeError(f"No datasets found under {DATASET_ROOT}")

    if interactive or not dataset_arg:
        return prompt_choice("Available datasets:", datasets)

    dataset_path = Path(dataset_arg).expanduser().resolve()
    if dataset_path.exists():
        if dataset_path.is_file():
            return dataset_path.stem, dataset_path
        csv_path = preferred_dataset_csv(dataset_path)
        if csv_path is None:
            raise FileNotFoundError(f"Could not resolve a dataset CSV inside: {dataset_path}")
        return dataset_path.name, csv_path

    if dataset_arg in datasets:
        return dataset_arg, datasets[dataset_arg]

    raise FileNotFoundError(f"Unknown dataset '{dataset_arg}'. Use --list to inspect available datasets.")


def load_feature_rows(csv_path: Path, input_columns: Sequence[str]) -> Tuple[np.ndarray, List[str]]:
    rows: List[List[float]] = []
    image_paths: List[str] = []
    dropped = 0
    with csv_path.open("r", newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            if row is None:
                dropped += 1
                continue
            try:
                values = [float(row[column]) for column in input_columns]
            except (KeyError, TypeError, ValueError):
                dropped += 1
                continue
            numeric = np.asarray(values, dtype=np.float32)
            if not np.isfinite(numeric).all():
                dropped += 1
                continue
            rows.append(values)
            image_paths.append((row.get("image_path") or "").strip())
    if not rows:
        raise RuntimeError(f"No usable input rows found in CSV: {csv_path}")
    print(f"Usable feature rows: {len(rows)} | Dropped rows: {dropped}")
    return np.asarray(rows, dtype=np.float32), image_paths


def prediction_value_map(target_columns: Sequence[str], prediction: np.ndarray) -> Dict[str, float]:
    return {column: float(value) for column, value in zip(target_columns, prediction.tolist())}


def build_rotmats_from_value_map(value_map: Mapping[str, float]) -> np.ndarray:
    rotmats = np.zeros((len(SMPL_JOINT_NAMES), 3, 3), dtype=np.float32)
    for joint_index, joint_name in enumerate(SMPL_JOINT_NAMES):
        for row in range(3):
            for col in range(3):
                column = f"hybrik_{joint_name}_rotmat_{row}{col}"
                rotmats[joint_index, row, col] = value_map.get(column, 1.0 if row == col else 0.0)
    return rotmats


def build_rotmats_from_prediction(target_columns: Sequence[str], prediction: np.ndarray) -> np.ndarray:
    return build_rotmats_from_value_map(prediction_value_map(target_columns, prediction))


def build_smpl_payload_from_prediction(
    target_columns: Sequence[str],
    prediction: np.ndarray,
    smpl_state: dict[str, object],
    device: torch.device,
) -> Tuple[dict, np.ndarray]:
    value_map = prediction_value_map(target_columns, prediction)
    rotmats = build_rotmats_from_value_map(value_map)

    # Match the existing live ROS inference path: use rotmats to rebuild a
    # kinematically consistent SMPL skeleton instead of mixing in independently
    # regressed xyz targets.
    geometry = GUI_RENDER.compute_pose_geometry(smpl_state, rotmats, device)
    xyz = geometry["joints"].astype(np.float32)

    transl_columns = ["hybrik_transl_x_m", "hybrik_transl_y_m", "hybrik_transl_z_m"]
    has_transl = all(column in value_map for column in transl_columns)
    transl = np.zeros(3, dtype=np.float32)
    if has_transl:
        transl = np.array([value_map[column] for column in transl_columns], dtype=np.float32)

    payload = {
        "format": SMPL_UDP_FORMAT,
        "joint_names": list(SMPL_JOINT_NAMES),
        "has_translation": bool(has_transl),
        "transl": transl.tolist(),
        "joint_xyz": xyz.tolist(),
        "joint_rotmats": rotmats.tolist(),
    }
    return payload, rotmats


def load_checkpoint(checkpoint_path: Path, device: torch.device) -> Tuple[nn.Module, dict]:
    checkpoint = torch.load(str(checkpoint_path), map_location="cpu")
    input_columns = list(checkpoint["input_columns"])
    target_columns = list(checkpoint["target_columns"])
    checkpoint_args = checkpoint["args"]

    model = GRUPoseRegressor(
        input_size=len(input_columns),
        hidden_size=int(checkpoint_args["hidden_size"]),
        num_layers=int(checkpoint_args["num_layers"]),
        output_size=len(target_columns),
        dropout=float(checkpoint_args["dropout"]),
        bidirectional=bool(checkpoint_args["bidirectional"]),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    metadata = {
        "input_columns": input_columns,
        "target_columns": target_columns,
        "seq_len": int(checkpoint_args["seq_len"]),
        "x_mean": np.asarray(checkpoint["x_mean"], dtype=np.float32),
        "x_std": np.asarray(checkpoint["x_std"], dtype=np.float32),
        "y_mean": np.asarray(checkpoint["y_mean"], dtype=np.float32),
        "y_std": np.asarray(checkpoint["y_std"], dtype=np.float32),
    }
    return model, metadata


def load_frame_image(image_path_value: str) -> np.ndarray:
    if image_path_value:
        image_path = Path(image_path_value).expanduser()
        if image_path.exists():
            image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if image is not None:
                return image

    placeholder = np.full((720, 1280, 3), 30, dtype=np.uint8)
    cv2.putText(
        placeholder,
        "image not found",
        (40, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return placeholder


def annotate_image(image: np.ndarray, title: str, frame_index: int) -> np.ndarray:
    annotated = image.copy()
    cv2.putText(
        annotated,
        title,
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        annotated,
        f"frame {frame_index}",
        (20, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return annotated


def create_visualizer(device: torch.device, render_size: int) -> Dict[str, object]:
    smpl_state = GUI_RENDER.load_smpl_state(device)
    return {
        "smpl_state": smpl_state,
        "render_size": int(render_size),
        "camera_distance": 2.5,
        "focal_length": 1200.0,
        "raw_window": "Replay Frame",
        "smpl_window": "Replay Predicted SMPL",
    }


def show_prediction_frame(
    visualizer: Dict[str, object],
    rotmats: np.ndarray,
    image_path: str,
    frame_index: int,
    device: torch.device,
    render_mode: str,
) -> bool:
    geometry = GUI_RENDER.compute_pose_geometry(visualizer["smpl_state"], rotmats, device)
    smpl_image = GUI_RENDER.render_pose_preview(
        geometry=geometry,
        render_size=visualizer["render_size"],
        camera_distance=visualizer["camera_distance"],
        focal_length=visualizer["focal_length"],
        highlight_joint_index=None,
        show_mesh=(render_mode == "mesh"),
    )
    raw_image = load_frame_image(image_path)
    raw_image = annotate_image(raw_image, "Dataset Frame", frame_index)
    smpl_image = annotate_image(smpl_image, "Predicted SMPL", frame_index)

    cv2.imshow(visualizer["raw_window"], raw_image)
    cv2.imshow(visualizer["smpl_window"], smpl_image)
    return (cv2.waitKey(1) & 0xFF) != 27


def replay_predictions(
    model: nn.Module,
    metadata: dict,
    features: np.ndarray,
    image_paths: Sequence[str],
    start_frame: int,
    frame_count: int,
    device: torch.device,
    udp_targets: Sequence[Tuple[str, int]],
    fps: float,
    show_window: bool,
    render_mode: str,
    render_size: int,
) -> None:
    seq_len = metadata["seq_len"]
    valid_start = seq_len - 1
    if start_frame < valid_start:
        raise ValueError(
            f"start_frame={start_frame} is too early for seq_len={seq_len}. Minimum start frame is {valid_start}."
        )

    end_frame = min(start_frame + frame_count, features.shape[0])
    if end_frame <= start_frame:
        raise ValueError("Selected frame range is empty.")

    socket_client = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    interval = 1.0 / fps if fps > 0.0 else 0.0
    smpl_state = GUI_RENDER.load_smpl_state(device)
    visualizer = create_visualizer(device, render_size) if show_window else None

    if show_window:
        cv2.namedWindow(visualizer["raw_window"], cv2.WINDOW_NORMAL)
        cv2.namedWindow(visualizer["smpl_window"], cv2.WINDOW_NORMAL)

    target_text = ", ".join(f"{host}:{port}" for host, port in udp_targets)
    print(f"Replaying frames [{start_frame}, {end_frame}) -> UDP {target_text} | count={end_frame - start_frame} | fps={fps:.2f}")

    try:
        for target_frame in range(start_frame, end_frame):
            start_time = time.perf_counter()
            window = features[target_frame - seq_len + 1 : target_frame + 1]
            normalized = (window - metadata["x_mean"].reshape(1, -1)) / metadata["x_std"].reshape(1, -1)
            batch_inputs = torch.from_numpy(normalized[None]).float().to(device)

            with torch.no_grad():
                prediction = model(batch_inputs).cpu().numpy()[0]
            prediction = prediction * metadata["y_std"] + metadata["y_mean"]

            payload, rotmats = build_smpl_payload_from_prediction(
                metadata["target_columns"],
                prediction,
                smpl_state,
                device,
            )
            payload["seq"] = int(target_frame)
            payload["source"] = "gru_dataset_replay"
            packet = json.dumps(payload).encode("utf-8")
            for udp_host, udp_port in udp_targets:
                socket_client.sendto(packet, (udp_host, udp_port))

            sent_index = target_frame - start_frame + 1
            if sent_index == 1 or sent_index % 30 == 0 or target_frame == end_frame - 1:
                print(f"  sent frame {target_frame} ({sent_index}/{end_frame - start_frame})")

            if show_window:
                image_path = image_paths[target_frame] if target_frame < len(image_paths) else ""
                should_continue = show_prediction_frame(
                    visualizer,
                    rotmats,
                    image_path,
                    target_frame,
                    device,
                    render_mode,
                )
                if not should_continue:
                    print("Replay stopped by user (Esc).")
                    break

            elapsed = time.perf_counter() - start_time
            remaining = interval - elapsed
            if remaining > 0.0:
                time.sleep(remaining)
    finally:
        socket_client.close()
        if show_window:
            try:
                cv2.destroyWindow(visualizer["raw_window"])
                cv2.destroyWindow(visualizer["smpl_window"])
            except Exception:
                pass


def main() -> None:
    args = parse_args()
    models = discover_models()
    datasets = discover_datasets()

    if args.list:
        print("Models:")
        for name, path in models.items():
            print(f"  {name:<12} {path}")
        print("Datasets:")
        for name, path in datasets.items():
            print(f"  {name:<12} {path}")
        return

    interactive = args.interactive or args.model is None or args.dataset is None
    model_name, checkpoint_path = resolve_model_checkpoint(args.model, interactive)
    dataset_name, csv_path = resolve_dataset_csv(args.dataset, interactive)

    default_start = None
    default_count = 300
    default_fps = 30.0

    device = resolve_device(args.device)
    model, metadata = load_checkpoint(checkpoint_path, device)
    features, image_paths = load_feature_rows(csv_path, metadata["input_columns"])

    min_start_frame = metadata["seq_len"] - 1
    max_start_frame = features.shape[0] - 1
    if default_start is None:
        default_start = min_start_frame

    if interactive:
        print(f"Selected model: {model_name} -> {checkpoint_path}")
        print(f"Selected dataset: {dataset_name} -> {csv_path}")
        print(f"Device: {device}")
        print(f"seq_len: {metadata['seq_len']} | valid start frame >= {min_start_frame} | max frame index = {max_start_frame}")
        start_frame = prompt_int("Start frame", default_start, minimum=min_start_frame)
        frame_count = prompt_int("Frame count", default_count, minimum=1)
        fps = prompt_float("Replay FPS", default_fps, minimum=0.1)
    else:
        start_frame = args.start_frame if args.start_frame is not None else default_start
        frame_count = args.frame_count if args.frame_count is not None else default_count
        fps = args.fps if args.fps is not None else default_fps

    replay_predictions(
        model=model,
        metadata=metadata,
        features=features,
        image_paths=image_paths,
        start_frame=int(start_frame),
        frame_count=int(frame_count),
        device=device,
        udp_targets=resolve_udp_targets(args),
        fps=float(fps),
        show_window=not args.hide_window,
        render_mode=str(args.render_mode),
        render_size=int(args.render_size),
    )


if __name__ == "__main__":
    main()
