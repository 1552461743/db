#!/usr/bin/env python3
"""Publish a humanoid TF tree from HybrIK SMPL-24 joint outputs.

Expected input topic type:
  std_msgs/msg/Float32MultiArray

Supported payload layouts:
1. 288 floats = 24 * (xyz[3] + rotmat[9])
2. 291 floats = transl[3] + 24 * xyz[3] + 24 * rotmat[9]

TF layout:
- world -> pelvis uses transl + pelvis_xyz when transl is provided.
- child translations are computed from root-relative xyz and rotated into the
  parent frame using the parent's global rotation.
- rotations use the provided per-joint local rotation matrices.
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
import rclpy
from geometry_msgs.msg import TransformStamped
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
from tf2_ros import TransformBroadcaster


JOINT_NAMES = [
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

PARENTS = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19, 20, 21]
JOINT_COUNT = len(JOINT_NAMES)
XYZ_AND_ROTMAT_SIZE = JOINT_COUNT * (3 + 9)
TRANSL_XYZ_AND_ROTMAT_SIZE = 3 + JOINT_COUNT * 3 + JOINT_COUNT * 9


def rotation_matrix_to_quaternion_xyzw(matrix: np.ndarray) -> Tuple[float, float, float, float]:
    trace = float(np.trace(matrix))

    if trace > 0.0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (matrix[2, 1] - matrix[1, 2]) * s
        y = (matrix[0, 2] - matrix[2, 0]) * s
        z = (matrix[1, 0] - matrix[0, 1]) * s
    else:
        if matrix[0, 0] > matrix[1, 1] and matrix[0, 0] > matrix[2, 2]:
            s = 2.0 * np.sqrt(max(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2], 1e-12))
            w = (matrix[2, 1] - matrix[1, 2]) / s
            x = 0.25 * s
            y = (matrix[0, 1] + matrix[1, 0]) / s
            z = (matrix[0, 2] + matrix[2, 0]) / s
        elif matrix[1, 1] > matrix[2, 2]:
            s = 2.0 * np.sqrt(max(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2], 1e-12))
            w = (matrix[0, 2] - matrix[2, 0]) / s
            x = (matrix[0, 1] + matrix[1, 0]) / s
            y = 0.25 * s
            z = (matrix[1, 2] + matrix[2, 1]) / s
        else:
            s = 2.0 * np.sqrt(max(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1], 1e-12))
            w = (matrix[1, 0] - matrix[0, 1]) / s
            x = (matrix[0, 2] + matrix[2, 0]) / s
            y = (matrix[1, 2] + matrix[2, 1]) / s
            z = 0.25 * s

    quat = np.array([x, y, z, w], dtype=np.float64)
    norm = float(np.linalg.norm(quat))
    if norm < 1e-12:
        return 0.0, 0.0, 0.0, 1.0
    quat /= norm
    return float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3])


class HybrikSmplTFPublisher(Node):
    def __init__(self) -> None:
        super().__init__("hybrik_smpl_tf_publisher")

        self.declare_parameter("input_topic", "/hybrik/smpl_24")
        self.declare_parameter("world_frame", "world")
        self.declare_parameter("tf_prefix", "hybrik")
        self.declare_parameter("publish_root_translation", True)
        self.declare_parameter("invert_rotations", True)

        self.input_topic = str(self.get_parameter("input_topic").value)
        self.world_frame = str(self.get_parameter("world_frame").value)
        self.tf_prefix = str(self.get_parameter("tf_prefix").value)
        self.publish_root_translation = bool(self.get_parameter("publish_root_translation").value)
        self.invert_rotations = bool(self.get_parameter("invert_rotations").value)

        self.tf_broadcaster = TransformBroadcaster(self)
        self.subscription = self.create_subscription(
            Float32MultiArray,
            self.input_topic,
            self.on_message,
            10,
        )

        self.get_logger().info(
            "Publishing HybrIK SMPL TF tree. "
            f"input_topic='{self.input_topic}', world_frame='{self.world_frame}', tf_prefix='{self.tf_prefix}'"
        )
        self.get_logger().info(
            f"Supported payload sizes: {XYZ_AND_ROTMAT_SIZE} (xyz+rotmat) or {TRANSL_XYZ_AND_ROTMAT_SIZE} (transl+xyz+rotmat)"
        )
        self.get_logger().info(f"invert_rotations={self.invert_rotations}")

    def frame_name(self, joint_name: str) -> str:
        if not self.tf_prefix:
            return joint_name
        return f"{self.tf_prefix}/{joint_name}"

    def parse_message(self, data: List[float]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, bool]:
        array = np.asarray(data, dtype=np.float32)

        if array.size == XYZ_AND_ROTMAT_SIZE:
            xyz = array[: JOINT_COUNT * 3].reshape(JOINT_COUNT, 3)
            rotmats = array[JOINT_COUNT * 3 :].reshape(JOINT_COUNT, 3, 3)
            transl = xyz[0].copy()
            return transl, xyz, rotmats, False

        if array.size == TRANSL_XYZ_AND_ROTMAT_SIZE:
            transl = array[:3].copy()
            xyz_start = 3
            xyz_end = xyz_start + JOINT_COUNT * 3
            xyz = array[xyz_start:xyz_end].reshape(JOINT_COUNT, 3)
            rotmats = array[xyz_end:].reshape(JOINT_COUNT, 3, 3)
            return transl, xyz, rotmats, True

        raise ValueError(
            f"Unexpected Float32MultiArray length {array.size}. "
            f"Expected {XYZ_AND_ROTMAT_SIZE} or {TRANSL_XYZ_AND_ROTMAT_SIZE}."
        )

    def build_transform(
        self,
        stamp,
        parent_frame: str,
        child_frame: str,
        translation: np.ndarray,
        rotation_matrix: np.ndarray,
    ) -> TransformStamped:
        transform = TransformStamped()
        transform.header.stamp = stamp
        transform.header.frame_id = parent_frame
        transform.child_frame_id = child_frame
        transform.transform.translation.x = float(translation[0])
        transform.transform.translation.y = float(translation[1])
        transform.transform.translation.z = float(translation[2])

        qx, qy, qz, qw = rotation_matrix_to_quaternion_xyzw(rotation_matrix)
        transform.transform.rotation.x = qx
        transform.transform.rotation.y = qy
        transform.transform.rotation.z = qz
        transform.transform.rotation.w = qw
        return transform

    def build_local_rotations(self, rotmats: np.ndarray) -> np.ndarray:
        local_rotations = rotmats.copy()
        if self.invert_rotations:
            local_rotations = np.transpose(local_rotations, (0, 2, 1))
        return local_rotations

    def build_global_rotations(self, local_rotations: np.ndarray) -> np.ndarray:
        global_rotations = np.zeros_like(local_rotations)
        for joint_index in range(JOINT_COUNT):
            parent_index = PARENTS[joint_index]
            if parent_index < 0:
                global_rotations[joint_index] = local_rotations[joint_index]
            else:
                global_rotations[joint_index] = global_rotations[parent_index] @ local_rotations[joint_index]
        return global_rotations

    def on_message(self, msg: Float32MultiArray) -> None:
        try:
            transl, xyz, rotmats, has_transl = self.parse_message(list(msg.data))
        except Exception as exc:
            self.get_logger().warning(str(exc))
            return

        stamp = self.get_clock().now().to_msg()
        transforms: List[TransformStamped] = []
        local_rotations = self.build_local_rotations(rotmats)
        global_rotations = self.build_global_rotations(local_rotations)

        for joint_index, joint_name in enumerate(JOINT_NAMES):
            parent_index = PARENTS[joint_index]
            child_frame = self.frame_name(joint_name)

            if parent_index < 0:
                parent_frame = self.world_frame
                if self.publish_root_translation:
                    translation = transl + xyz[0] if has_transl else xyz[0]
                else:
                    translation = np.zeros(3, dtype=np.float32)
            else:
                parent_frame = self.frame_name(JOINT_NAMES[parent_index])
                translation_global = xyz[joint_index] - xyz[parent_index]
                parent_global_rotation = global_rotations[parent_index]
                translation = parent_global_rotation.T @ translation_global

            rotation_matrix = local_rotations[joint_index]

            transforms.append(
                self.build_transform(
                    stamp=stamp,
                    parent_frame=parent_frame,
                    child_frame=child_frame,
                    translation=translation,
                    rotation_matrix=rotation_matrix,
                )
            )

        self.tf_broadcaster.sendTransform(transforms)


def main() -> None:
    rclpy.init()
    node = HybrikSmplTFPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
