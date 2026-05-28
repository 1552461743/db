#!/usr/bin/env python3
"""Bridge HybrIK SMPL-24 pose output to Kaipu MuJoCo joint commands.

This node subscribes to `/hybrik/smpl_24` as `Float32MultiArray` and publishes
best-effort Kaipu joint targets as `JointState` on `/mujoco/joint_cmd`.

Supported input payloads:
- 288 floats: 24 * (xyz[3] + rotmat[9])
- 291 floats: transl[3] + 24 * xyz[3] + 24 * rotmat[9]

The mapping is heuristic. It uses SMPL local rotation matrices, combines a few
adjacent body segments, decomposes them into Kaipu joint chains, and clamps to
the actuator ranges defined in `1230-URDF-version-fixed_mode_pos.xml`.
"""

from __future__ import annotations

import warnings
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
from scipy.spatial.transform import Rotation

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float32MultiArray


THIS_DIR = Path(__file__).resolve().parent
DEFAULT_XML_PATH = THIS_DIR / "1230-URDF-version-fixed_mode_pos.xml"

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
SMPL_INDEX = {name: index for index, name in enumerate(SMPL_JOINT_NAMES)}

XYZ_AND_ROTMAT_SIZE = len(SMPL_JOINT_NAMES) * (3 + 9)
TRANSL_XYZ_AND_ROTMAT_SIZE = 3 + len(SMPL_JOINT_NAMES) * 3 + len(SMPL_JOINT_NAMES) * 9

KAIPU_JOINT_ORDER = [
    "waist_roll",
    "waist_yaw",
    "head_yaw",
    "head_pitch",
    "left_hip_roll",
    "left_hip_yaw",
    "left_hip_pitch",
    "left_knee_pitch",
    "left_ankle_pitch",
    "left_ankle_roll",
    "right_hip_roll",
    "right_hip_yaw",
    "right_hip_pitch",
    "right_knee_pitch",
    "right_ankle_pitch",
    "right_ankle_roll",
    "left_shoulder_pitch",
    "left_shoulder_roll",
    "left_shoulder_yaw",
    "left_elbow_pitch",
    "left_elbow_yaw",
    "left_wrist_pitch",
    "left_wrist_yaw",
    "right_shoulder_pitch",
    "right_shoulder_roll",
    "right_shoulder_yaw",
    "right_elbow_pitch",
    "right_elbow_yaw",
    "right_wrist_pitch",
    "right_wrist_yaw",
]


def parse_smpl_payload(data: Sequence[float]) -> np.ndarray:
    array = np.asarray(data, dtype=np.float32)

    if array.size == XYZ_AND_ROTMAT_SIZE:
        rotmats = array[len(SMPL_JOINT_NAMES) * 3 :].reshape(len(SMPL_JOINT_NAMES), 3, 3)
        return rotmats

    if array.size == TRANSL_XYZ_AND_ROTMAT_SIZE:
        rotmats = array[3 + len(SMPL_JOINT_NAMES) * 3 :].reshape(len(SMPL_JOINT_NAMES), 3, 3)
        return rotmats

    raise ValueError(
        f"Unexpected Float32MultiArray length {array.size}. "
        f"Expected {XYZ_AND_ROTMAT_SIZE} or {TRANSL_XYZ_AND_ROTMAT_SIZE}."
    )


def chain_rotation(local_rotations: np.ndarray, joint_names: Iterable[str]) -> np.ndarray:
    result = np.eye(3, dtype=np.float64)
    for joint_name in joint_names:
        result = result @ local_rotations[SMPL_INDEX[joint_name]].astype(np.float64)
    return result


def euler_intrinsic(rotmat: np.ndarray, sequence: str) -> np.ndarray:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return Rotation.from_matrix(rotmat).as_euler(sequence, degrees=False)


def clamp(value: float, limits: Tuple[float, float] | None) -> float:
    if limits is None:
        return float(value)
    lo, hi = limits
    return float(np.clip(value, lo, hi))


def load_actuator_joint_limits(xml_path: Path) -> Dict[str, Tuple[float, float]]:
    root = ET.parse(xml_path).getroot()
    limits: Dict[str, Tuple[float, float]] = {}

    actuator_root = root.find("actuator")
    if actuator_root is None:
        return limits

    for actuator in actuator_root:
        joint_name = actuator.attrib.get("joint")
        ctrlrange = actuator.attrib.get("ctrlrange")
        if not joint_name or not ctrlrange:
            continue

        parts = ctrlrange.split()
        if len(parts) != 2:
            continue

        try:
            limits[joint_name] = (float(parts[0]), float(parts[1]))
        except ValueError:
            continue

    return limits


def build_kaipu_joint_targets(
    local_rotations: np.ndarray,
    joint_limits: Dict[str, Tuple[float, float]],
) -> Dict[str, float]:
    commands: Dict[str, float] = {}

    torso_rot = chain_rotation(local_rotations, ["pelvis", "spine1", "spine2", "spine3"])
    torso_xzy = euler_intrinsic(torso_rot, "XZY")
    commands["waist_roll"] = clamp(torso_xzy[0], joint_limits.get("waist_roll"))
    commands["waist_yaw"] = clamp(torso_xzy[1], joint_limits.get("waist_yaw"))

    head_rot = chain_rotation(local_rotations, ["neck", "jaw"])
    head_zyx = euler_intrinsic(head_rot, "ZYX")
    commands["head_yaw"] = clamp(-head_zyx[0], joint_limits.get("head_yaw"))
    commands["head_pitch"] = clamp(head_zyx[1], joint_limits.get("head_pitch"))

    left_hip = euler_intrinsic(local_rotations[SMPL_INDEX["left_hip"]], "XZY")
    commands["left_hip_roll"] = clamp(left_hip[0], joint_limits.get("left_hip_roll"))
    commands["left_hip_yaw"] = clamp(left_hip[1], joint_limits.get("left_hip_yaw"))
    commands["left_hip_pitch"] = clamp(left_hip[2], joint_limits.get("left_hip_pitch"))

    right_hip = euler_intrinsic(local_rotations[SMPL_INDEX["right_hip"]], "XZY")
    commands["right_hip_roll"] = clamp(-right_hip[0], joint_limits.get("right_hip_roll"))
    commands["right_hip_yaw"] = clamp(-right_hip[1], joint_limits.get("right_hip_yaw"))
    commands["right_hip_pitch"] = clamp(right_hip[2], joint_limits.get("right_hip_pitch"))

    left_knee = euler_intrinsic(local_rotations[SMPL_INDEX["left_knee"]], "YZX")
    commands["left_knee_pitch"] = clamp(left_knee[0], joint_limits.get("left_knee_pitch"))

    right_knee = euler_intrinsic(local_rotations[SMPL_INDEX["right_knee"]], "YZX")
    commands["right_knee_pitch"] = clamp(right_knee[0], joint_limits.get("right_knee_pitch"))

    left_ankle = euler_intrinsic(local_rotations[SMPL_INDEX["left_ankle"]], "YXZ")
    commands["left_ankle_pitch"] = clamp(left_ankle[0], joint_limits.get("left_ankle_pitch"))
    commands["left_ankle_roll"] = clamp(left_ankle[1], joint_limits.get("left_ankle_roll"))

    right_ankle = euler_intrinsic(local_rotations[SMPL_INDEX["right_ankle"]], "YXZ")
    commands["right_ankle_pitch"] = clamp(right_ankle[0], joint_limits.get("right_ankle_pitch"))
    commands["right_ankle_roll"] = clamp(-right_ankle[1], joint_limits.get("right_ankle_roll"))

    left_shoulder_rot = chain_rotation(local_rotations, ["left_collar", "left_shoulder"])
    left_shoulder = euler_intrinsic(left_shoulder_rot, "YXZ")
    commands["left_shoulder_pitch"] = clamp(-left_shoulder[0], joint_limits.get("left_shoulder_pitch"))
    commands["left_shoulder_roll"] = clamp(left_shoulder[1], joint_limits.get("left_shoulder_roll"))
    commands["left_shoulder_yaw"] = clamp(left_shoulder[2], joint_limits.get("left_shoulder_yaw"))

    right_shoulder_rot = chain_rotation(local_rotations, ["right_collar", "right_shoulder"])
    right_shoulder = euler_intrinsic(right_shoulder_rot, "YXZ")
    commands["right_shoulder_pitch"] = clamp(-right_shoulder[0], joint_limits.get("right_shoulder_pitch"))
    commands["right_shoulder_roll"] = clamp(-right_shoulder[1], joint_limits.get("right_shoulder_roll"))
    commands["right_shoulder_yaw"] = clamp(-right_shoulder[2], joint_limits.get("right_shoulder_yaw"))

    left_elbow = euler_intrinsic(local_rotations[SMPL_INDEX["left_elbow"]], "YZX")
    commands["left_elbow_pitch"] = clamp(-left_elbow[0], joint_limits.get("left_elbow_pitch"))
    commands["left_elbow_yaw"] = clamp(left_elbow[1], joint_limits.get("left_elbow_yaw"))

    right_elbow = euler_intrinsic(local_rotations[SMPL_INDEX["right_elbow"]], "YZX")
    commands["right_elbow_pitch"] = clamp(-right_elbow[0], joint_limits.get("right_elbow_pitch"))
    commands["right_elbow_yaw"] = clamp(-right_elbow[1], joint_limits.get("right_elbow_yaw"))

    left_wrist = euler_intrinsic(local_rotations[SMPL_INDEX["left_wrist"]], "XYZ")
    commands["left_wrist_pitch"] = clamp(left_wrist[0], joint_limits.get("left_wrist_pitch"))
    commands["left_wrist_yaw"] = clamp(-left_wrist[1], joint_limits.get("left_wrist_yaw"))

    right_wrist = euler_intrinsic(local_rotations[SMPL_INDEX["right_wrist"]], "XYZ")
    commands["right_wrist_pitch"] = clamp(-right_wrist[0], joint_limits.get("right_wrist_pitch"))
    commands["right_wrist_yaw"] = clamp(-right_wrist[1], joint_limits.get("right_wrist_yaw"))

    return commands


class SmplToKaipuMujocoBridge(Node):
    def __init__(self) -> None:
        super().__init__("smpl_to_kaipu_mujoco_bridge")

        self.declare_parameter("input_topic", "/hybrik/smpl_24")
        self.declare_parameter("output_topic", "/mujoco/joint_cmd")
        self.declare_parameter("xml_path", str(DEFAULT_XML_PATH))
        self.declare_parameter("invert_rotations", True)
        self.declare_parameter("debug_print_period", 2.0)

        self.input_topic = str(self.get_parameter("input_topic").value)
        self.output_topic = str(self.get_parameter("output_topic").value)
        self.xml_path = Path(str(self.get_parameter("xml_path").value)).expanduser().resolve()
        self.invert_rotations = bool(self.get_parameter("invert_rotations").value)
        self.debug_print_period = float(self.get_parameter("debug_print_period").value)
        self.last_debug_time = 0.0

        if not self.xml_path.exists():
            raise FileNotFoundError(f"MuJoCo XML not found: {self.xml_path}")

        self.joint_limits = load_actuator_joint_limits(self.xml_path)
        self.publisher = self.create_publisher(JointState, self.output_topic, 10)
        self.subscription = self.create_subscription(
            Float32MultiArray,
            self.input_topic,
            self.on_message,
            10,
        )

        self.get_logger().info(
            f"Bridging HybrIK SMPL to Kaipu MuJoCo. input='{self.input_topic}', output='{self.output_topic}'"
        )
        self.get_logger().info(f"Using XML: {self.xml_path}")
        self.get_logger().info(f"invert_rotations={self.invert_rotations}, mapped_joints={len(KAIPU_JOINT_ORDER)}")

    def on_message(self, msg: Float32MultiArray) -> None:
        try:
            rotmats = parse_smpl_payload(msg.data)
            local_rotations = np.transpose(rotmats, (0, 2, 1)) if self.invert_rotations else rotmats.copy()
            commands = build_kaipu_joint_targets(local_rotations, self.joint_limits)
        except Exception as exc:
            self.get_logger().warning(f"Failed to map SMPL message: {exc}")
            return

        joint_msg = JointState()
        joint_msg.header.stamp = self.get_clock().now().to_msg()
        joint_msg.name = KAIPU_JOINT_ORDER
        joint_msg.position = [float(commands.get(name, 0.0)) for name in KAIPU_JOINT_ORDER]
        self.publisher.publish(joint_msg)

        if self.debug_print_period > 0.0:
            now_sec = self.get_clock().now().nanoseconds * 1e-9
            if now_sec - self.last_debug_time >= self.debug_print_period:
                self.last_debug_time = now_sec
                debug_names = [
                    "waist_yaw",
                    "left_shoulder_pitch",
                    "right_shoulder_pitch",
                    "left_hip_pitch",
                    "right_hip_pitch",
                ]
                debug_text = ", ".join(f"{name}={commands.get(name, 0.0):+.3f}" for name in debug_names)
                self.get_logger().info(debug_text)


def main() -> None:
    rclpy.init()
    node = SmplToKaipuMujocoBridge()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
