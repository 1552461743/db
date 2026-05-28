#!/usr/bin/env python3
"""Live ROS2 GRU inference from suit sensors to SMPL pose.

Subscribes to topics published by:
- esp32_imu_reader.py
- human_skeleton_tf_publisher.cpp

Loads a checkpoint produced by data/train_sensor_gru.py, reconstructs the GRU,
runs live inference, publishes the result as Float32MultiArray on
`/hybrik/smpl_24`, and shows a standalone SMPL mesh window.
"""

from __future__ import annotations

import sys
import types
from collections import deque
from pathlib import Path
from typing import Deque, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import rclpy
import torch
from geometry_msgs.msg import Vector3
from rclpy.node import Node
from sensor_msgs.msg import Imu, MagneticField
from std_msgs.msg import Float32, Float32MultiArray
from torch import nn


SMPL_layer = None
lbs = None
render_mesh_single_frame = None


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
JOINT_COUNT = len(SMPL_JOINT_NAMES)


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


def ensure_hybrik_imports(repo_root: Path) -> None:
    global SMPL_layer, lbs, render_mesh_single_frame

    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    if "easydict" not in sys.modules:
        easydict_module = types.ModuleType("easydict")

        class EasyDict(dict):
            def __getattr__(self, name):
                try:
                    return self[name]
                except KeyError as exc:
                    raise AttributeError(name) from exc

            def __setattr__(self, name, value):
                self[name] = value

            def __delattr__(self, name):
                try:
                    del self[name]
                except KeyError as exc:
                    raise AttributeError(name) from exc

        easydict_module.EasyDict = EasyDict
        sys.modules["easydict"] = easydict_module

    if SMPL_layer is None or lbs is None or render_mesh_single_frame is None:
        from hybrik.models.layers.smpl.SMPL import SMPL_layer as imported_smpl_layer
        from hybrik.models.layers.smpl.lbs import lbs as imported_lbs
        from hybrik.utils.render_pytorch3d import render_mesh_single_frame as imported_render_mesh_single_frame

        SMPL_layer = imported_smpl_layer
        lbs = imported_lbs
        render_mesh_single_frame = imported_render_mesh_single_frame


def create_smpl_renderer(device: torch.device, repo_root: Path) -> Dict[str, object]:
    ensure_hybrik_imports(repo_root)
    h36m_jregressor = np.load(str(repo_root / "model_files/J_regressor_h36m.npy"))
    smpl_layer = SMPL_layer(
        str(repo_root / "model_files/basicModel_neutral_lbs_10_207_0_v1.0.0.pkl"),
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


def prediction_value_map(target_columns: Sequence[str], values: np.ndarray) -> Dict[str, float]:
    return {column: float(value) for column, value in zip(target_columns, values.tolist())}


def build_rotmats_from_value_map(value_map: Dict[str, float]) -> np.ndarray:
    rotmats = np.zeros((JOINT_COUNT, 3, 3), dtype=np.float32)
    for joint_index, joint_name in enumerate(SMPL_JOINT_NAMES):
        for row in range(3):
            for col in range(3):
                column = f"hybrik_{joint_name}_rotmat_{row}{col}"
                rotmats[joint_index, row, col] = value_map.get(column, 1.0 if row == col else 0.0)
    return rotmats


def build_transl_from_value_map(value_map: Dict[str, float]) -> np.ndarray:
    columns = ["hybrik_transl_x_m", "hybrik_transl_y_m", "hybrik_transl_z_m"]
    if all(column in value_map for column in columns):
        return np.array([value_map[column] for column in columns], dtype=np.float32)
    return np.zeros(3, dtype=np.float32)


@torch.no_grad()
def render_smpl_frame(renderer_state: Dict[str, object], rotmats: np.ndarray, device: torch.device) -> Tuple[np.ndarray, np.ndarray]:
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
    joints = joints - joints[:, [0], :]

    translation = torch.tensor([[0.0, 0.0, 2.5]], dtype=torch.float32, device=device)
    render = render_mesh_single_frame(
        vertices=vertices,
        faces=faces,
        translation=translation,
        focal_length=1200.0,
        height=720,
        width=720,
        device=device,
    )
    render = render.detach().cpu().numpy()
    valid_mask = render[:, :, 3:4] > 0
    color = (render[:, :, :3] * 255.0).astype(np.uint8)
    background = np.full_like(color, 30, dtype=np.uint8)
    image = color * valid_mask.astype(np.uint8) + background * (~valid_mask).astype(np.uint8)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    joints_np = joints.squeeze(0).detach().cpu().numpy().astype(np.float32)
    return image, joints_np


class LiveGruSmplInferNode(Node):
    def __init__(self) -> None:
        super().__init__("live_gru_smpl_infer")

        suit_root = Path(__file__).resolve().parents[4]
        runtime_root = suit_root / "hybrik_runtime"
        legacy_repo_root = Path(__file__).resolve().parents[6]
        default_repo_root = str(
            runtime_root
            if (runtime_root / "hybrik").exists()
            else suit_root
            if (suit_root / "hybrik").exists()
            else legacy_repo_root
        )
        self.declare_parameter("repo_root", default_repo_root)
        self.declare_parameter("checkpoint_path", str(suit_root / "mod/1/best_model.pt"))
        self.declare_parameter("device", "cuda")
        self.declare_parameter("inference_hz", 30.0)
        self.declare_parameter("render_hz", 10.0)
        self.declare_parameter("output_topic", "/hybrik/smpl_24")
        self.declare_parameter("show_window", True)

        self.repo_root = Path(str(self.get_parameter("repo_root").value)).expanduser().resolve()
        if not self.repo_root.exists():
            raise FileNotFoundError(f"repo_root not found: {self.repo_root}")
        ensure_hybrik_imports(self.repo_root)

        checkpoint_path = Path(str(self.get_parameter("checkpoint_path").value)).expanduser().resolve()
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        requested_device = str(self.get_parameter("device").value)
        if requested_device.startswith("cuda") and not torch.cuda.is_available():
            self.device = torch.device("cpu")
        else:
            self.device = torch.device(requested_device)

        self.inference_hz = float(self.get_parameter("inference_hz").value)
        self.render_hz = float(self.get_parameter("render_hz").value)
        self.output_topic = str(self.get_parameter("output_topic").value)
        self.show_window = bool(self.get_parameter("show_window").value)

        self.output_pub = self.create_publisher(Float32MultiArray, self.output_topic, 10)
        self.window_name = "Live Predicted SMPL"
        self.last_render_time = 0.0

        checkpoint = torch.load(str(checkpoint_path), map_location="cpu")
        self.input_columns = list(checkpoint["input_columns"])
        self.target_columns = list(checkpoint["target_columns"])
        checkpoint_args = checkpoint["args"]
        self.seq_len = int(checkpoint_args["seq_len"])
        self.model = GRUPoseRegressor(
            input_size=len(self.input_columns),
            hidden_size=int(checkpoint_args["hidden_size"]),
            num_layers=int(checkpoint_args["num_layers"]),
            output_size=len(self.target_columns),
            dropout=float(checkpoint_args["dropout"]),
            bidirectional=bool(checkpoint_args["bidirectional"]),
        ).to(self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()

        self.x_mean = np.asarray(checkpoint["x_mean"], dtype=np.float32)
        self.x_std = np.asarray(checkpoint["x_std"], dtype=np.float32)
        self.y_mean = np.asarray(checkpoint["y_mean"], dtype=np.float32)
        self.y_std = np.asarray(checkpoint["y_std"], dtype=np.float32)

        self.feature_values: Dict[str, float] = {}
        self.feature_buffer: Deque[np.ndarray] = deque(maxlen=self.seq_len)
        self.required_yaw_channels = sorted(
            {
                int(column.split("_")[2])
                for column in self.input_columns
                if column.startswith("imu_channel_") and column.endswith("_yaw_tilt_deg_data")
            }
        )
        self.required_imu_channels = sorted(
            {
                int(column.split("_")[2])
                for column in self.input_columns
                if column.startswith("imu_channel_")
                and (
                    "linear_acceleration" in column
                    or "angular_velocity" in column
                    or "magnetic_field_magnetic_field" in column
                )
            }
        )

        self.renderer_state = create_smpl_renderer(self.device, self.repo_root)

        self.create_subscription(Float32MultiArray, "/sensor/capacitance", self.on_capacitance, 10)
        self.create_subscription(Float32MultiArray, "/sensor/normalized", self.on_normalized, 10)
        self.create_subscription(Float32MultiArray, "/imu/relative_transform", self.on_relative_transform, 10)

        self.imu_subs = []
        self.mag_subs = []
        for channel in self.required_imu_channels:
            self.imu_subs.append(
                self.create_subscription(Imu, f"/imu/channel_{channel}", lambda msg, ch=channel: self.on_imu(ch, msg), 10)
            )
            self.mag_subs.append(
                self.create_subscription(
                    MagneticField,
                    f"/imu/channel_{channel}/magnetic_field",
                    lambda msg, ch=channel: self.on_magnetic(ch, msg),
                    10,
                )
            )

        self.yaw_subs = []
        for channel in self.required_yaw_channels:
            self.yaw_subs.append(
                self.create_subscription(
                    Float32,
                    f"/imu/channel_{channel}/yaw_tilt_deg",
                    lambda msg, ch=channel: self.on_yaw(ch, msg),
                    10,
                )
            )

        self.timer = self.create_timer(1.0 / self.inference_hz, self.on_timer)

        if self.show_window:
            cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)

        self.get_logger().info(f"Loaded checkpoint: {checkpoint_path}")
        self.get_logger().info(f"repo_root: {self.repo_root}")
        self.get_logger().info(f"Device: {self.device}")
        self.get_logger().info(f"Input features: {len(self.input_columns)}, seq_len={self.seq_len}")
        self.get_logger().info(f"Publishing predictions to: {self.output_topic}")

    def set_array_features(self, prefix: str, values: Sequence[float]) -> None:
        for index, value in enumerate(values):
            key = f"{prefix}_{index}"
            if key in self.input_columns:
                self.feature_values[key] = float(value)

    def on_capacitance(self, msg: Float32MultiArray) -> None:
        self.set_array_features("sensor_capacitance", msg.data)

    def on_normalized(self, msg: Float32MultiArray) -> None:
        self.set_array_features("sensor_normalized", msg.data)

    def on_relative_transform(self, msg: Float32MultiArray) -> None:
        self.set_array_features("imu_relative_transform", msg.data)

    def on_imu(self, channel: int, msg: Imu) -> None:
        prefix = f"imu_channel_{channel}"
        values = {
            f"{prefix}_linear_acceleration_x": msg.linear_acceleration.x,
            f"{prefix}_linear_acceleration_y": msg.linear_acceleration.y,
            f"{prefix}_linear_acceleration_z": msg.linear_acceleration.z,
            f"{prefix}_angular_velocity_x": msg.angular_velocity.x,
            f"{prefix}_angular_velocity_y": msg.angular_velocity.y,
            f"{prefix}_angular_velocity_z": msg.angular_velocity.z,
        }
        for key, value in values.items():
            if key in self.input_columns:
                self.feature_values[key] = float(value)

    def on_magnetic(self, channel: int, msg: MagneticField) -> None:
        prefix = f"imu_channel_{channel}_magnetic_field_magnetic_field"
        values = {
            f"{prefix}_x": msg.magnetic_field.x,
            f"{prefix}_y": msg.magnetic_field.y,
            f"{prefix}_z": msg.magnetic_field.z,
        }
        for key, value in values.items():
            if key in self.input_columns:
                self.feature_values[key] = float(value)

    def on_yaw(self, channel: int, msg: Float32) -> None:
        key = f"imu_channel_{channel}_yaw_tilt_deg_data"
        if key in self.input_columns:
            self.feature_values[key] = float(msg.data)

    def have_complete_feature_vector(self) -> bool:
        return all(column in self.feature_values for column in self.input_columns)

    def build_feature_vector(self) -> np.ndarray:
        return np.asarray([self.feature_values[column] for column in self.input_columns], dtype=np.float32)

    def build_output_payload(self, joints_xyz: np.ndarray, rotmats: np.ndarray, transl: np.ndarray) -> List[float]:
        payload: List[float] = []
        payload.extend(transl.tolist())
        payload.extend(joints_xyz.reshape(-1).tolist())
        payload.extend(rotmats.reshape(-1).tolist())
        return payload

    @torch.no_grad()
    def on_timer(self) -> None:
        if not self.have_complete_feature_vector():
            return

        self.feature_buffer.append(self.build_feature_vector())
        if len(self.feature_buffer) < self.seq_len:
            return

        window = np.stack(list(self.feature_buffer), axis=0)
        window = (window - self.x_mean.reshape(1, -1)) / self.x_std.reshape(1, -1)
        batch_inputs = torch.from_numpy(window[None]).float().to(self.device)
        prediction = self.model(batch_inputs).cpu().numpy()[0]
        prediction = prediction * self.y_std + self.y_mean
        value_map = prediction_value_map(self.target_columns, prediction)

        rotmats = build_rotmats_from_value_map(value_map)
        transl = build_transl_from_value_map(value_map)
        smpl_image, joints_xyz = render_smpl_frame(self.renderer_state, rotmats, self.device)

        output_msg = Float32MultiArray()
        output_msg.data = self.build_output_payload(joints_xyz, rotmats, transl)
        self.output_pub.publish(output_msg)

        now = time.time()
        if self.show_window and now - self.last_render_time >= (1.0 / max(self.render_hz, 1e-6)):
            self.last_render_time = now
            cv2.putText(
                smpl_image,
                "Live GRU SMPL Prediction",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )
            cv2.imshow(self.window_name, smpl_image)
            cv2.waitKey(1)

    def destroy_node(self):
        if self.show_window:
            try:
                cv2.destroyWindow(self.window_name)
            except Exception:
                pass
        return super().destroy_node()


def main() -> None:
    rclpy.init()
    node = LiveGruSmplInferNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
