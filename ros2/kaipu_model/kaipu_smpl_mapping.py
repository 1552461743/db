#!/usr/bin/env python3
"""Shared SMPL-to-Kaipu mapping helpers.

Pure Python utilities used by the GUI UDP sender and the MuJoCo receiver.
"""

from __future__ import annotations

import warnings
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, Iterable, Sequence, Tuple

import numpy as np
from scipy.spatial.transform import Rotation


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

# Static command offsets to compensate the rest-pose mismatch between:
# - SMPL neutral pose: arms abducted outward
# - Kaipu robot zero pose: arms hanging down
#
# The largest mismatch is at shoulder roll. A +90 deg offset moves the robot
# arms from the down-hanging zero pose toward the lateral SMPL neutral pose.
COMMAND_OFFSETS = {
    "left_shoulder_roll": float(np.pi * 0.5),
    "right_shoulder_roll": float(np.pi * 0.5),
}

ROTATION_DEADZONE_RAD = 1e-4

XYZ_AND_ROTMAT_SIZE = len(SMPL_JOINT_NAMES) * (3 + 9)
TRANSL_XYZ_AND_ROTMAT_SIZE = 3 + len(SMPL_JOINT_NAMES) * 3 + len(SMPL_JOINT_NAMES) * 9

# SMPL neutral-pose inspection in this repo shows:
# - left-side joints have positive X
# - higher joints have positive Y
# - toes are forward in positive Z
# Therefore SMPL local rotations here are most naturally interpreted as:
#   x = left, y = up, z = forward
#
# Kaipu XML joint axes are defined in a robot frame that is most naturally:
#   x = forward, y = left, z = up
#
# This constant converts vectors/rotations from the Kaipu robot frame into the
# raw SMPL frame before we decompose by the robot's joint-axis order.
ROBOT_FRAME_IN_SMPL = np.array(
    [
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 0.0],
    ],
    dtype=np.float64,
)


def parse_smpl_payload(data: Sequence[float]) -> np.ndarray:
    array = np.asarray(data, dtype=np.float32)

    if array.size == XYZ_AND_ROTMAT_SIZE:
        return array[len(SMPL_JOINT_NAMES) * 3 :].reshape(len(SMPL_JOINT_NAMES), 3, 3)

    if array.size == TRANSL_XYZ_AND_ROTMAT_SIZE:
        return array[3 + len(SMPL_JOINT_NAMES) * 3 :].reshape(len(SMPL_JOINT_NAMES), 3, 3)

    raise ValueError(
        f"Unexpected Float32MultiArray length {array.size}. "
        f"Expected {XYZ_AND_ROTMAT_SIZE} or {TRANSL_XYZ_AND_ROTMAT_SIZE}."
    )


def load_actuator_joint_limits(xml_path: Path = DEFAULT_XML_PATH) -> Dict[str, Tuple[float, float]]:
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


def clamp_with_offset(
    joint_name: str,
    value: float,
    limits: Tuple[float, float] | None,
) -> float:
    return clamp(value + COMMAND_OFFSETS.get(joint_name, 0.0), limits)


def change_rotation_basis(rotmat: np.ndarray, basis: np.ndarray) -> np.ndarray:
    return basis.T @ rotmat.astype(np.float64) @ basis


def smpl_to_robot_frame(local_rotmat: np.ndarray) -> np.ndarray:
    return change_rotation_basis(local_rotmat, ROBOT_FRAME_IN_SMPL)


def rotvec_robot(local_rotmat: np.ndarray) -> np.ndarray:
    robot_rot = smpl_to_robot_frame(local_rotmat)
    rotvec = Rotation.from_matrix(robot_rot).as_rotvec()
    rotvec[np.abs(rotvec) < ROTATION_DEADZONE_RAD] = 0.0
    return rotvec


def rotvec_smpl(local_rotmat: np.ndarray) -> np.ndarray:
    rotvec = Rotation.from_matrix(local_rotmat.astype(np.float64)).as_rotvec()
    rotvec[np.abs(rotvec) < ROTATION_DEADZONE_RAD] = 0.0
    return rotvec


def compose_pose_rotmats_from_euler_degrees(euler_degrees: np.ndarray) -> np.ndarray:
    rotmats = np.zeros((len(SMPL_JOINT_NAMES), 3, 3), dtype=np.float32)
    for joint_index, angles_deg in enumerate(euler_degrees):
        rotmats[joint_index] = Rotation.from_euler("xyz", angles_deg, degrees=True).as_matrix().astype(np.float32)
    return rotmats


def build_kaipu_joint_targets_from_gui_euler(
    euler_degrees: np.ndarray,
    joint_limits: Dict[str, Tuple[float, float]],
) -> Dict[str, float]:
    rotmats = compose_pose_rotmats_from_euler_degrees(euler_degrees)
    commands = build_kaipu_joint_targets(rotmats, joint_limits)

    def radians_for(joint_name: str) -> np.ndarray:
        return np.deg2rad(euler_degrees[SMPL_INDEX[joint_name]].astype(np.float64))

    # Direct slider-to-joint mappings avoid axis coupling caused by converting
    # a combined 3D rotation back into fewer robot DOFs.
    neck = radians_for("neck")
    commands["head_pitch"] = clamp_with_offset("head_pitch", neck[0], joint_limits.get("head_pitch"))
    commands["head_yaw"] = clamp_with_offset("head_yaw", -neck[1], joint_limits.get("head_yaw"))

    left_shoulder = radians_for("left_shoulder")
    commands["left_shoulder_pitch"] = clamp_with_offset(
        "left_shoulder_pitch", -left_shoulder[0], joint_limits.get("left_shoulder_pitch")
    )
    commands["left_shoulder_yaw"] = clamp_with_offset(
        "left_shoulder_yaw", left_shoulder[1], joint_limits.get("left_shoulder_yaw")
    )
    commands["left_shoulder_roll"] = clamp_with_offset(
        "left_shoulder_roll", left_shoulder[2], joint_limits.get("left_shoulder_roll")
    )

    right_shoulder = radians_for("right_shoulder")
    commands["right_shoulder_pitch"] = clamp_with_offset(
        "right_shoulder_pitch", -right_shoulder[0], joint_limits.get("right_shoulder_pitch")
    )
    commands["right_shoulder_yaw"] = clamp_with_offset(
        "right_shoulder_yaw", -right_shoulder[1], joint_limits.get("right_shoulder_yaw")
    )
    commands["right_shoulder_roll"] = clamp_with_offset(
        "right_shoulder_roll", -right_shoulder[2], joint_limits.get("right_shoulder_roll")
    )

    left_elbow = radians_for("left_elbow")
    commands["left_elbow_pitch"] = clamp_with_offset("left_elbow_pitch", left_elbow[1], joint_limits.get("left_elbow_pitch"))
    commands["left_elbow_yaw"] = clamp_with_offset("left_elbow_yaw", left_elbow[0], joint_limits.get("left_elbow_yaw"))

    right_elbow = radians_for("right_elbow")
    commands["right_elbow_pitch"] = clamp_with_offset("right_elbow_pitch", right_elbow[1], joint_limits.get("right_elbow_pitch"))
    commands["right_elbow_yaw"] = clamp_with_offset("right_elbow_yaw", -right_elbow[0], joint_limits.get("right_elbow_yaw"))

    left_wrist = radians_for("left_wrist")
    commands["left_wrist_pitch"] = clamp_with_offset("left_wrist_pitch", left_wrist[2], joint_limits.get("left_wrist_pitch"))
    commands["left_wrist_yaw"] = clamp_with_offset("left_wrist_yaw", left_wrist[1], joint_limits.get("left_wrist_yaw"))

    right_wrist = radians_for("right_wrist")
    commands["right_wrist_pitch"] = clamp_with_offset("right_wrist_pitch", right_wrist[2], joint_limits.get("right_wrist_pitch"))
    commands["right_wrist_yaw"] = clamp_with_offset("right_wrist_yaw", -right_wrist[1], joint_limits.get("right_wrist_yaw"))

    return commands


def summed_rotvec_robot(local_rotations: np.ndarray, joint_names: Sequence[str]) -> np.ndarray:
    total = np.zeros(3, dtype=np.float64)
    for joint_name in joint_names:
        total += rotvec_robot(local_rotations[SMPL_INDEX[joint_name]])
    return total


def build_kaipu_joint_targets(
    local_rotations: np.ndarray,
    joint_limits: Dict[str, Tuple[float, float]],
) -> Dict[str, float]:
    commands: Dict[str, float] = {}

    torso_vec = rotvec_robot(chain_rotation(local_rotations, ["pelvis", "spine1", "spine2", "spine3"]))
    commands["waist_roll"] = clamp_with_offset("waist_roll", torso_vec[0], joint_limits.get("waist_roll"))
    commands["waist_yaw"] = clamp_with_offset("waist_yaw", torso_vec[2], joint_limits.get("waist_yaw"))

    # Kaipu has a 2-DOF head (yaw + pitch) and no separate jaw or head-roll
    # actuator. Use SMPL neck as the sole source for head orientation and ignore
    # jaw so jaw edits do not incorrectly rotate the whole robot head.
    head_vec = rotvec_robot(local_rotations[SMPL_INDEX["neck"]])
    commands["head_yaw"] = clamp_with_offset("head_yaw", -head_vec[2], joint_limits.get("head_yaw"))
    commands["head_pitch"] = clamp_with_offset("head_pitch", head_vec[1], joint_limits.get("head_pitch"))

    left_hip = rotvec_robot(local_rotations[SMPL_INDEX["left_hip"]])
    commands["left_hip_roll"] = clamp_with_offset("left_hip_roll", left_hip[0], joint_limits.get("left_hip_roll"))
    commands["left_hip_yaw"] = clamp_with_offset("left_hip_yaw", left_hip[2], joint_limits.get("left_hip_yaw"))
    commands["left_hip_pitch"] = clamp_with_offset("left_hip_pitch", left_hip[1], joint_limits.get("left_hip_pitch"))

    right_hip = rotvec_robot(local_rotations[SMPL_INDEX["right_hip"]])
    commands["right_hip_roll"] = clamp_with_offset("right_hip_roll", -right_hip[0], joint_limits.get("right_hip_roll"))
    commands["right_hip_yaw"] = clamp_with_offset("right_hip_yaw", -right_hip[2], joint_limits.get("right_hip_yaw"))
    commands["right_hip_pitch"] = clamp_with_offset("right_hip_pitch", right_hip[1], joint_limits.get("right_hip_pitch"))

    left_knee = rotvec_robot(local_rotations[SMPL_INDEX["left_knee"]])
    commands["left_knee_pitch"] = clamp_with_offset("left_knee_pitch", left_knee[1], joint_limits.get("left_knee_pitch"))

    right_knee = rotvec_robot(local_rotations[SMPL_INDEX["right_knee"]])
    commands["right_knee_pitch"] = clamp_with_offset("right_knee_pitch", right_knee[1], joint_limits.get("right_knee_pitch"))

    left_ankle = rotvec_robot(local_rotations[SMPL_INDEX["left_ankle"]])
    commands["left_ankle_pitch"] = clamp_with_offset("left_ankle_pitch", left_ankle[1], joint_limits.get("left_ankle_pitch"))
    commands["left_ankle_roll"] = clamp_with_offset("left_ankle_roll", left_ankle[0], joint_limits.get("left_ankle_roll"))

    right_ankle = rotvec_robot(local_rotations[SMPL_INDEX["right_ankle"]])
    commands["right_ankle_pitch"] = clamp_with_offset("right_ankle_pitch", right_ankle[1], joint_limits.get("right_ankle_pitch"))
    commands["right_ankle_roll"] = clamp_with_offset("right_ankle_roll", -right_ankle[0], joint_limits.get("right_ankle_roll"))

    # Ignore SMPL collar and drive the robot shoulder only from SMPL shoulder.
    # This loses clavicle motion but makes interactive control much cleaner.
    left_shoulder = rotvec_robot(local_rotations[SMPL_INDEX["left_shoulder"]])
    commands["left_shoulder_pitch"] = clamp_with_offset("left_shoulder_pitch", -left_shoulder[1], joint_limits.get("left_shoulder_pitch"))
    commands["left_shoulder_roll"] = clamp_with_offset("left_shoulder_roll", left_shoulder[0], joint_limits.get("left_shoulder_roll"))
    commands["left_shoulder_yaw"] = clamp_with_offset("left_shoulder_yaw", left_shoulder[2], joint_limits.get("left_shoulder_yaw"))

    right_shoulder = rotvec_robot(local_rotations[SMPL_INDEX["right_shoulder"]])
    commands["right_shoulder_pitch"] = clamp_with_offset("right_shoulder_pitch", -right_shoulder[1], joint_limits.get("right_shoulder_pitch"))
    commands["right_shoulder_roll"] = clamp_with_offset("right_shoulder_roll", -right_shoulder[0], joint_limits.get("right_shoulder_roll"))
    commands["right_shoulder_yaw"] = clamp_with_offset("right_shoulder_yaw", -right_shoulder[2], joint_limits.get("right_shoulder_yaw"))

    # Elbows are controlled more intuitively in the raw SMPL local frame:
    # x slider -> elbow yaw, y slider -> elbow pitch, z slider ignored.
    left_elbow = rotvec_smpl(local_rotations[SMPL_INDEX["left_elbow"]])
    commands["left_elbow_pitch"] = clamp_with_offset("left_elbow_pitch", left_elbow[1], joint_limits.get("left_elbow_pitch"))
    commands["left_elbow_yaw"] = clamp_with_offset("left_elbow_yaw", left_elbow[0], joint_limits.get("left_elbow_yaw"))

    right_elbow = rotvec_smpl(local_rotations[SMPL_INDEX["right_elbow"]])
    commands["right_elbow_pitch"] = clamp_with_offset("right_elbow_pitch", right_elbow[1], joint_limits.get("right_elbow_pitch"))
    commands["right_elbow_yaw"] = clamp_with_offset("right_elbow_yaw", -right_elbow[0], joint_limits.get("right_elbow_yaw"))

    # Wrist mapping tuned for interactive control based on user feedback:
    # y slider -> wrist yaw, z slider -> wrist pitch, x ignored.
    left_wrist = rotvec_smpl(local_rotations[SMPL_INDEX["left_wrist"]])
    commands["left_wrist_pitch"] = clamp_with_offset("left_wrist_pitch", left_wrist[2], joint_limits.get("left_wrist_pitch"))
    commands["left_wrist_yaw"] = clamp_with_offset("left_wrist_yaw", left_wrist[1], joint_limits.get("left_wrist_yaw"))

    right_wrist = rotvec_smpl(local_rotations[SMPL_INDEX["right_wrist"]])
    commands["right_wrist_pitch"] = clamp_with_offset("right_wrist_pitch", right_wrist[2], joint_limits.get("right_wrist_pitch"))
    commands["right_wrist_yaw"] = clamp_with_offset("right_wrist_yaw", -right_wrist[1], joint_limits.get("right_wrist_yaw"))

    return commands
