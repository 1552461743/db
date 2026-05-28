#!/usr/bin/env python3
"""Receive suit sensor state over UDP, run GRU inference, show SMPL, and send predictions over UDP.

Run this script inside the model conda environment where torch/HybrIK are available.
It does not require ROS2.
"""

from __future__ import annotations

import argparse
import json
import socket
import sys
import time
from collections import deque
from pathlib import Path
from typing import Deque, Dict, List, Sequence, Tuple

import cv2
import numpy as np
import torch
from torch import nn


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


class SensorPayloadIncomplete(ValueError):
    pass


class GRUPoseRegressor(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, output_size: int, dropout: float, bidirectional: bool) -> None:
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run live GRU inference from UDP sensor data.")
    parser.add_argument("--checkpoint-path", required=True, help="Path to best_model.pt")
    parser.add_argument("--listen-host", default="0.0.0.0", help="UDP host to listen on")
    parser.add_argument("--listen-port", type=int, default=5010, help="UDP port to listen on")
    parser.add_argument("--send-host", default="127.0.0.1", help="UDP host to send predictions to")
    parser.add_argument("--send-port", type=int, default=5005, help="UDP port to send predictions to")
    parser.add_argument(
        "--send-target",
        action="append",
        default=[],
        metavar="HOST:PORT",
        help="Additional UDP destination. Repeat to send each SMPL frame to multiple receivers.",
    )
    parser.add_argument("--device", default="cuda", help="Torch device")
    parser.add_argument("--render-hz", type=float, default=10.0, help="SMPL render FPS")
    parser.add_argument(
        "--payload-format",
        choices=["direct", "compact"],
        default="direct",
        help="direct sends GM-compatible SMPL JSON; compact sends the old {'data': [...]} payload",
    )
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


def resolve_send_targets(args: argparse.Namespace) -> List[Tuple[str, int]]:
    if args.send_target:
        return [parse_udp_target(target) for target in args.send_target]
    return [(str(args.send_host), int(args.send_port))]


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
    return {"smpl_layer": smpl_layer, "zero_betas": zero_betas, "faces": faces}


def render_smpl(renderer_state: Dict[str, object], rotmats: np.ndarray, device: torch.device) -> np.ndarray:
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
    vertex_min = vertices.amin(dim=1)
    vertex_max = vertices.amax(dim=1)
    vertex_center_xy = 0.5 * (vertex_min[:, :2] + vertex_max[:, :2])
    translation = torch.zeros((1, 3), dtype=torch.float32, device=device)
    translation[:, :2] = -vertex_center_xy
    translation[:, 2] = 3.2
    render = render_mesh_single_frame(vertices, faces, translation, 1200.0, 720, 720, device=device)
    render = render.detach().cpu().numpy()
    valid_mask = render[:, :, 3:4] > 0
    color = (render[:, :, :3] * 255.0).astype(np.uint8)
    background = np.full_like(color, 30, dtype=np.uint8)
    image = color * valid_mask.astype(np.uint8) + background * (~valid_mask).astype(np.uint8)
    return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)


def compute_smpl_joints(renderer_state: Dict[str, object], rotmats: np.ndarray, device: torch.device) -> np.ndarray:
    smpl_layer = renderer_state["smpl_layer"]
    zero_betas = renderer_state["zero_betas"]
    pose_tensor = torch.from_numpy(rotmats).float().unsqueeze(0).to(device)
    _, joints, _, _ = lbs(
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
    joints = joints - joints[:, [0], :]
    return joints[0].detach().cpu().numpy().astype(np.float32)


def build_rotmats_from_value_map(value_map: Dict[str, float]) -> np.ndarray:
    rotmats = np.zeros((len(SMPL_JOINT_NAMES), 3, 3), dtype=np.float32)
    for joint_index, joint_name in enumerate(SMPL_JOINT_NAMES):
        for row in range(3):
            for col in range(3):
                column = f"hybrik_{joint_name}_rotmat_{row}{col}"
                rotmats[joint_index, row, col] = value_map.get(column, 1.0 if row == col else 0.0)
    return rotmats


def sensor_list_value(payload: Dict[str, object], key: str, index: int, column: str) -> float:
    values = payload.get(key)
    if not isinstance(values, list) or index >= len(values):
        length = len(values) if isinstance(values, list) else "missing"
        raise SensorPayloadIncomplete(f"{column} needs {key}[{index}], got length {length}")
    return float(values[index])


def imu_list_value(channel_dict: Dict[str, object], key: str, axis_index: int, column: str) -> float:
    values = channel_dict.get(key)
    if not isinstance(values, list) or axis_index >= len(values):
        length = len(values) if isinstance(values, list) else "missing"
        raise SensorPayloadIncomplete(f"{column} needs imu {key}[{axis_index}], got length {length}")
    return float(values[axis_index])


def build_feature_vector(input_columns: Sequence[str], payload: Dict[str, object]) -> np.ndarray:
    imu_data = payload.get("imu", {})
    if not isinstance(imu_data, dict):
        raise SensorPayloadIncomplete("payload imu field is missing or invalid")
    values: List[float] = []

    for column in input_columns:
        if column.startswith("sensor_capacitance_"):
            index = int(column.split("_")[-1])
            values.append(sensor_list_value(payload, "capacitance", index, column))
            continue

        if column.startswith("sensor_normalized_"):
            index = int(column.split("_")[-1])
            values.append(sensor_list_value(payload, "normalized", index, column))
            continue

        if column.startswith("imu_relative_transform_"):
            index = int(column.split("_")[-1])
            values.append(sensor_list_value(payload, "relative_transform", index, column))
            continue

        if column.startswith("imu_channel_"):
            parts = column.split("_")
            channel = parts[2]
            channel_dict = imu_data.get(channel, {})
            if not isinstance(channel_dict, dict):
                raise SensorPayloadIncomplete(f"{column} needs imu channel {channel}")

            if "linear_acceleration" in column:
                axis = column[-1]
                axis_index = {"x": 0, "y": 1, "z": 2}[axis]
                values.append(imu_list_value(channel_dict, "linear_acceleration", axis_index, column))
                continue

            if "angular_velocity" in column:
                axis = column[-1]
                axis_index = {"x": 0, "y": 1, "z": 2}[axis]
                values.append(imu_list_value(channel_dict, "angular_velocity", axis_index, column))
                continue

            if "magnetic_field_magnetic_field" in column:
                axis = column[-1]
                axis_index = {"x": 0, "y": 1, "z": 2}[axis]
                values.append(imu_list_value(channel_dict, "magnetic_field", axis_index, column))
                continue

            if column.endswith("_yaw_tilt_deg_data"):
                if "yaw_tilt_deg" not in channel_dict:
                    raise SensorPayloadIncomplete(f"{column} needs imu yaw_tilt_deg")
                values.append(float(channel_dict["yaw_tilt_deg"]))
                continue

        raise KeyError(f"Unsupported input column mapping: {column}")

    return np.asarray(values, dtype=np.float32)


def prediction_to_udp_payload(
    target_columns: Sequence[str],
    prediction: np.ndarray,
    renderer_state: Dict[str, object],
    device: torch.device,
) -> Tuple[List[float], np.ndarray]:
    value_map = {column: float(value) for column, value in zip(target_columns, prediction.tolist())}
    payload: List[float] = []

    has_transl = all(name in value_map for name in ["hybrik_transl_x_m", "hybrik_transl_y_m", "hybrik_transl_z_m"])
    transl = np.zeros(3, dtype=np.float32)
    if has_transl:
        transl = np.array(
            [
                value_map["hybrik_transl_x_m"],
                value_map["hybrik_transl_y_m"],
                value_map["hybrik_transl_z_m"],
            ],
            dtype=np.float32,
        )
        payload.extend(transl.tolist())

    rotmats = build_rotmats_from_value_map(value_map)
    xyz = compute_smpl_joints(renderer_state, rotmats, device).reshape(-1).tolist()

    payload.extend(xyz)
    payload.extend(rotmats.reshape(-1).tolist())
    return payload, rotmats


def prediction_to_direct_smpl_payload(
    target_columns: Sequence[str],
    prediction: np.ndarray,
    seq: int,
    renderer_state: Dict[str, object],
    device: torch.device,
) -> Tuple[Dict[str, object], np.ndarray]:
    value_map = {column: float(value) for column, value in zip(target_columns, prediction.tolist())}
    has_transl = all(name in value_map for name in ["hybrik_transl_x_m", "hybrik_transl_y_m", "hybrik_transl_z_m"])
    transl = [0.0, 0.0, 0.0]
    if has_transl:
        transl = [
            value_map["hybrik_transl_x_m"],
            value_map["hybrik_transl_y_m"],
            value_map["hybrik_transl_z_m"],
        ]

    rotmats = build_rotmats_from_value_map(value_map)
    joint_xyz = compute_smpl_joints(renderer_state, rotmats, device).tolist()

    return {
        "format": "smpl_24_xyz_rotmat",
        "joint_names": list(SMPL_JOINT_NAMES),
        "has_translation": bool(has_transl),
        "transl": transl,
        "joint_xyz": joint_xyz,
        "joint_rotmats": rotmats.tolist(),
        "seq": int(seq),
        "source": "live_gru_udp_infer",
    }, rotmats


def main() -> None:
    args = parse_args()
    send_targets = resolve_send_targets(args)
    checkpoint = torch.load(str(Path(args.checkpoint_path).expanduser().resolve()), map_location="cpu")
    device = torch.device(args.device if not (args.device.startswith("cuda") and not torch.cuda.is_available()) else "cpu")

    model_args = checkpoint["args"]
    input_columns = list(checkpoint["input_columns"])
    target_columns = list(checkpoint["target_columns"])
    seq_len = int(model_args["seq_len"])

    model = GRUPoseRegressor(
        input_size=len(input_columns),
        hidden_size=int(model_args["hidden_size"]),
        num_layers=int(model_args["num_layers"]),
        output_size=len(target_columns),
        dropout=float(model_args["dropout"]),
        bidirectional=bool(model_args["bidirectional"]),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    x_mean = np.asarray(checkpoint["x_mean"], dtype=np.float32)
    x_std = np.asarray(checkpoint["x_std"], dtype=np.float32)
    y_mean = np.asarray(checkpoint["y_mean"], dtype=np.float32)
    y_std = np.asarray(checkpoint["y_std"], dtype=np.float32)

    renderer_state = create_smpl_renderer(device)
    feature_buffer: Deque[np.ndarray] = deque(maxlen=seq_len)

    recv_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    recv_sock.bind((args.listen_host, args.listen_port))
    send_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    recv_sock.settimeout(1.0)

    window_name = "Live Predicted SMPL"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    last_render_time = 0.0
    render_interval = 1.0 / max(args.render_hz, 1e-6)
    sent_count = 0
    skipped_incomplete_count = 0
    last_incomplete_log_time = 0.0

    print(f"Listening UDP on {args.listen_host}:{args.listen_port}")
    target_text = ", ".join(f"{host}:{port}" for host, port in send_targets)
    print(f"Sending predictions to {target_text}")
    print(f"Device: {device}")

    try:
        while True:
            raw_bytes, _ = recv_sock.recvfrom(65535)
            payload = json.loads(raw_bytes.decode("utf-8"))
            try:
                feature_vector = build_feature_vector(input_columns, payload)
            except SensorPayloadIncomplete as exc:
                skipped_incomplete_count += 1
                now = time.time()
                if now - last_incomplete_log_time >= 1.0:
                    print(
                        "Waiting for complete sensor payload; "
                        f"skipped {skipped_incomplete_count} packets. Last issue: {exc}"
                    )
                    last_incomplete_log_time = now
                continue
            feature_buffer.append(feature_vector)

            if len(feature_buffer) < seq_len:
                continue

            window = np.stack(list(feature_buffer), axis=0)
            window = (window - x_mean.reshape(1, -1)) / x_std.reshape(1, -1)
            batch_inputs = torch.from_numpy(window[None]).float().to(device)

            with torch.no_grad():
                prediction = model(batch_inputs).cpu().numpy()[0]
            prediction = prediction * y_std + y_mean
            if args.payload_format == "compact":
                udp_payload, rotmats = prediction_to_udp_payload(target_columns, prediction, renderer_state, device)
                packet = json.dumps({"data": udp_payload}).encode("utf-8")
            else:
                udp_payload, rotmats = prediction_to_direct_smpl_payload(target_columns, prediction, sent_count, renderer_state, device)
                packet = json.dumps(udp_payload).encode("utf-8")
            for send_host, send_port in send_targets:
                send_sock.sendto(packet, (send_host, send_port))
            sent_count += 1

            now = time.time()
            if now - last_render_time >= render_interval:
                last_render_time = now
                smpl_image = render_smpl(renderer_state, rotmats, device)
                cv2.putText(
                    smpl_image,
                    f"Live prediction #{sent_count}",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )
                cv2.imshow(window_name, smpl_image)
                if cv2.waitKey(1) & 0xFF == 27:
                    break
    except KeyboardInterrupt:
        pass
    finally:
        recv_sock.close()
        send_sock.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
