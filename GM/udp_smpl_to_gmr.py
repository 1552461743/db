#!/usr/bin/env python3
from __future__ import annotations

import argparse
import builtins
import importlib.util
import json
import pickle
import socket
import sys
import time
import types
from pathlib import Path
from typing import Dict, List, Mapping, Tuple

import numpy as np
from scipy.spatial.transform import Rotation as R

from smpl_udp_common import GMR_TO_RAW_SMPL_BASIS, RAW_SMPL_TO_GMR_BASIS, SMPL_INDEX, SMPL_JOINT_NAMES


SCRIPT_PATH = Path(__file__).resolve()
DATA_ROOT = SCRIPT_PATH.parent.parent
GMR_ROOT = DATA_ROOT / "GMR"
GMR_PARAMS_PATH = GMR_ROOT / "general_motion_retargeting/params.py"

if str(GMR_ROOT) not in sys.path:
    sys.path.insert(0, str(GMR_ROOT))

try:
    import rich  # noqa: F401
except ImportError:
    rich_stub = types.ModuleType("rich")
    rich_stub.print = builtins.print
    sys.modules["rich"] = rich_stub


def load_module_from_path(module_name: str, module_path: Path):
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load module spec from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


GMR_PARAMS = load_module_from_path("gmr_params_udp_receiver", GMR_PARAMS_PATH)

ROBOT_CHOICES: List[str] = sorted(GMR_PARAMS.IK_CONFIG_DICT["smplx"].keys())

SMPL_PARENTS: Dict[str, str | None] = {
    "pelvis": None,
    "left_hip": "pelvis",
    "right_hip": "pelvis",
    "spine1": "pelvis",
    "left_knee": "left_hip",
    "right_knee": "right_hip",
    "spine2": "spine1",
    "left_ankle": "left_knee",
    "right_ankle": "right_knee",
    "spine3": "spine2",
    "left_foot": "left_ankle",
    "right_foot": "right_ankle",
    "neck": "spine3",
    "left_collar": "spine3",
    "right_collar": "spine3",
    "jaw": "neck",
    "left_shoulder": "left_collar",
    "right_shoulder": "right_collar",
    "left_elbow": "left_shoulder",
    "right_elbow": "right_shoulder",
    "left_wrist": "left_elbow",
    "right_wrist": "right_elbow",
    "left_thumb": "left_wrist",
    "right_thumb": "right_wrist",
}

HEAD_OFFSET_FROM_NECK_RAW = np.array([0.00, 0.14, 0.02], dtype=np.float64)
ROOT_ROTATION_CORRECTION_BY_ROBOT: Dict[str, Tuple[R, bool]] = {
    # G1's viewer/control frame expects an extra world-frame x-axis correction.
    "unitree_g1": (R.from_euler("x", -90.0, degrees=True), True),
    "unitree_g1_with_hands": (R.from_euler("x", -90.0, degrees=True), True),
    "kaipu": (R.from_euler("x", -90.0, degrees=True), True),
    "droid_x2": (R.from_euler("x", -90.0, degrees=True), True),
}
PIPER_SHOULDER_ANCHOR = np.array([0.0, 0.0, 0.123], dtype=np.float64)
# SMPL raw: x=left, y=up, z=forward. Piper/MuJoCo: z=up.
# Use a fixed basis so the robot follows the arm position instead of rotating
# the target frame with the shoulder joint itself.
PIPER_SMPL_TO_ROBOT_BASIS = np.array(
    [
        [0.0, 0.0, 1.0],
        [-1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
    ],
    dtype=np.float64,
)

SMPLX_BODY_ORDER = [
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
    "head",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Receive direct SMPL-24 UDP payloads and retarget them to a GMR robot.",
    )
    parser.add_argument("--udp-host", default="0.0.0.0", help="UDP host/interface to bind")
    parser.add_argument("--udp-port", type=int, default=5007, help="UDP port to listen on")
    parser.add_argument("--robot", choices=ROBOT_CHOICES, default="unitree_g1", help="Target GMR robot")
    parser.add_argument("--fps", type=float, default=30.0, help="Expected incoming packet rate")
    parser.add_argument("--human-height", type=float, default=1.75, help="Approximate human height in meters")
    parser.add_argument("--gender", choices=["male", "female", "neutral"], default="neutral")
    parser.add_argument("--socket-timeout", type=float, default=1.0, help="UDP receive timeout in seconds")
    parser.add_argument("--invert-rotmats", action="store_true", default=False, help="Transpose incoming SMPL rotmats before FK")
    parser.add_argument("--no-invert-rotmats", dest="invert_rotmats", action="store_false", help="Use incoming rotmats as-is")
    parser.add_argument("--use-camera-translation", action="store_true", help="Use HybrIK transl as world root translation")
    parser.add_argument("--ground-offset", type=float, default=0.02, help="Desired robot lowest-point height above ground")
    parser.add_argument("--smoothing-alpha", type=float, default=0.0, help="Low-pass filter weight for qpos")
    parser.add_argument(
        "--retarget-passes",
        type=int,
        default=None,
        help="IK retarget calls per incoming frame. Defaults to 8 for piper and 1 for other robots.",
    )
    parser.add_argument(
        "--disable-outlier-filter",
        action="store_true",
        help="Disable single-frame qpos jump rejection.",
    )
    parser.add_argument(
        "--max-root-step",
        type=float,
        default=0.35,
        help="Drop a frame if root position jumps more than this many meters from the previous output frame.",
    )
    parser.add_argument(
        "--max-root-angle-step",
        type=float,
        default=1.2,
        help="Drop a frame if root rotation jumps more than this many radians from the previous output frame.",
    )
    parser.add_argument(
        "--max-dof-step",
        type=float,
        default=1.2,
        help="Drop a frame if any robot joint jumps more than this many radians from the previous output frame.",
    )
    parser.add_argument("--no-viewer", action="store_true", help="Disable MuJoCo robot viewer")
    parser.add_argument("--record-video", action="store_true", help="Record the robot viewer to video")
    parser.add_argument("--video-path", default="videos/udp_smpl_to_gmr.mp4", help="Recorded video path")
    parser.add_argument("--save-robot-path", default=None, help="Optional path to save retargeted robot motion as pickle")
    parser.add_argument("--save-smpl-path", default=None, help="Optional path to save reconstructed SMPL data as .npz")
    parser.add_argument("--save-smplx-path", default=None, help="Optional path to save reconstructed SMPL-X-like data as .npz")
    parser.add_argument("--twist2-redis-host", default=None, help="Optional Redis host for TWIST2 low-level controller")
    parser.add_argument("--twist2-redis-port", type=int, default=6379, help="Redis port for TWIST2 low-level controller")
    parser.add_argument("--twist2-robot-key", default="unitree_g1_with_hands", help="TWIST2 Redis action key suffix")
    parser.add_argument("--print-every", type=int, default=60, help="Print packet stats every N frames")
    return parser.parse_args()


def ensure_parent_dir(file_path: str | None) -> None:
    if not file_path:
        return
    parent = Path(file_path).expanduser().resolve().parent
    parent.mkdir(parents=True, exist_ok=True)


def load_gmr_runtime():
    from general_motion_retargeting import GeneralMotionRetargeting as GMR
    from general_motion_retargeting import RobotMotionViewer
    import mujoco as mj

    return GMR, RobotMotionViewer, mj


def receive_payload(sock: socket.socket) -> dict | None:
    try:
        packet, _ = sock.recvfrom(65535)
    except socket.timeout:
        return None

    try:
        payload = json.loads(packet.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        print(f"Skipping invalid UDP packet: {exc}")
        return None

    if not isinstance(payload, dict):
        print("Skipping UDP packet because payload is not a JSON object")
        return None
    return payload


def payload_contains_direct_smpl(payload: Mapping[str, object]) -> bool:
    return "joint_rotmats" in payload and "joint_xyz" in payload


def reorder_by_joint_names(values: np.ndarray, joint_names: List[str], expected_rank: int) -> np.ndarray:
    if values.ndim != expected_rank:
        raise ValueError(f"Expected array rank {expected_rank}, got shape {values.shape}")
    if len(joint_names) != len(SMPL_JOINT_NAMES):
        raise ValueError(f"Expected {len(SMPL_JOINT_NAMES)} joint names, got {len(joint_names)}")

    reordered = np.zeros((len(SMPL_JOINT_NAMES),) + values.shape[1:], dtype=np.float64)
    for src_index, joint_name in enumerate(joint_names):
        if joint_name not in SMPL_INDEX:
            raise ValueError(f"Unknown SMPL joint name in payload: {joint_name}")
        reordered[SMPL_INDEX[joint_name]] = values[src_index]
    return reordered


def parse_direct_smpl_payload(
    payload: Mapping[str, object],
    invert_rotmats: bool,
    use_camera_translation: bool,
) -> Tuple[np.ndarray, Dict[str, np.ndarray], np.ndarray]:
    joint_names = payload.get("joint_names")
    if not isinstance(joint_names, list):
        joint_names = list(SMPL_JOINT_NAMES)

    joint_xyz = np.asarray(payload["joint_xyz"], dtype=np.float64)
    joint_rotmats = np.asarray(payload["joint_rotmats"], dtype=np.float64)
    joint_xyz = reorder_by_joint_names(joint_xyz, [str(name) for name in joint_names], expected_rank=2)
    joint_rotmats = reorder_by_joint_names(joint_rotmats, [str(name) for name in joint_names], expected_rank=3)

    if joint_xyz.shape != (len(SMPL_JOINT_NAMES), 3):
        raise ValueError(f"Expected joint_xyz shape {(len(SMPL_JOINT_NAMES), 3)}, got {joint_xyz.shape}")
    if joint_rotmats.shape != (len(SMPL_JOINT_NAMES), 3, 3):
        raise ValueError(
            f"Expected joint_rotmats shape {(len(SMPL_JOINT_NAMES), 3, 3)}, got {joint_rotmats.shape}"
        )

    if invert_rotmats:
        joint_rotmats = np.transpose(joint_rotmats, (0, 2, 1))

    transl_obj = payload.get("transl", [0.0, 0.0, 0.0])
    root_translation_raw = np.asarray(transl_obj, dtype=np.float64).reshape(3)
    if use_camera_translation and bool(payload.get("has_translation", False)):
        joint_xyz = joint_xyz + root_translation_raw[None, :]
    else:
        root_translation_raw = np.zeros(3, dtype=np.float64)

    positions_map = {
        joint_name: joint_xyz[SMPL_INDEX[joint_name]].copy()
        for joint_name in SMPL_JOINT_NAMES
    }
    return joint_rotmats, positions_map, root_translation_raw


def global_rotations_from_local(local_rotmats: np.ndarray) -> Dict[str, np.ndarray]:
    global_rotations: Dict[str, np.ndarray] = {}
    for joint_name in SMPL_JOINT_NAMES:
        joint_idx = SMPL_INDEX[joint_name]
        parent_name = SMPL_PARENTS[joint_name]
        local_rot = local_rotmats[joint_idx]
        if parent_name is None:
            global_rotations[joint_name] = local_rot
        else:
            global_rotations[joint_name] = global_rotations[parent_name] @ local_rot
    return global_rotations


def build_gmr_human_frame(
    global_positions_raw: Mapping[str, np.ndarray],
    global_rotations_raw: Mapping[str, np.ndarray],
    apply_basis_transform: bool,
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    human_frame: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    for joint_name in SMPL_JOINT_NAMES:
        pos_raw = global_positions_raw[joint_name]
        rot_raw = global_rotations_raw[joint_name]
        if apply_basis_transform:
            pos_gmr = RAW_SMPL_TO_GMR_BASIS @ pos_raw
            rot_gmr = RAW_SMPL_TO_GMR_BASIS @ rot_raw @ GMR_TO_RAW_SMPL_BASIS
        else:
            pos_gmr = pos_raw
            rot_gmr = rot_raw
        quat_gmr = R.from_matrix(rot_gmr).as_quat(scalar_first=True)
        human_frame[joint_name] = (pos_gmr.astype(np.float64), quat_gmr.astype(np.float64))

    neck_pos_raw = global_positions_raw["neck"]
    neck_rot_raw = global_rotations_raw["neck"]
    head_pos_raw = neck_pos_raw + neck_rot_raw @ HEAD_OFFSET_FROM_NECK_RAW
    head_rot_raw = global_rotations_raw.get("jaw", neck_rot_raw)
    if apply_basis_transform:
        head_pos_gmr = RAW_SMPL_TO_GMR_BASIS @ head_pos_raw
        head_rot_gmr = RAW_SMPL_TO_GMR_BASIS @ head_rot_raw @ GMR_TO_RAW_SMPL_BASIS
    else:
        head_pos_gmr = head_pos_raw
        head_rot_gmr = head_rot_raw
    human_frame["head"] = (
        head_pos_gmr.astype(np.float64),
        R.from_matrix(head_rot_gmr).as_quat(scalar_first=True).astype(np.float64),
    )
    return human_frame


def build_smplx_like_pose(
    local_rotmats: np.ndarray,
    root_translation_raw: np.ndarray,
    gender: str,
    mocap_frame_rate: float,
) -> Dict[str, np.ndarray]:
    root_orient = R.from_matrix(local_rotmats[SMPL_INDEX["pelvis"]]).as_rotvec().astype(np.float32)

    body_rotvecs: List[np.ndarray] = []
    for joint_name in SMPLX_BODY_ORDER:
        if joint_name == "head":
            source_rot = local_rotmats[SMPL_INDEX["jaw"]]
        else:
            source_rot = local_rotmats[SMPL_INDEX[joint_name]]
        body_rotvecs.append(R.from_matrix(source_rot).as_rotvec().astype(np.float32))

    return {
        "root_orient": root_orient,
        "pose_body": np.concatenate(body_rotvecs, axis=0).astype(np.float32),
        "trans": root_translation_raw.astype(np.float32),
        "betas": np.zeros(16, dtype=np.float32),
        "gender": np.array(gender),
        "mocap_frame_rate": np.array(mocap_frame_rate, dtype=np.float32),
    }


def save_smpl_npz(
    save_path: str,
    joint_positions_raw: List[np.ndarray],
    local_rotmats: List[np.ndarray],
    fps: float,
) -> None:
    np.savez(
        save_path,
        joint_xyz=np.asarray(joint_positions_raw, dtype=np.float32),
        joint_rotmats=np.asarray(local_rotmats, dtype=np.float32),
        joint_names=np.asarray(SMPL_JOINT_NAMES),
        mocap_frame_rate=np.array(fps, dtype=np.float32),
    )


def save_smplx_npz(save_path: str, smplx_frames: List[Dict[str, np.ndarray]]) -> None:
    if not smplx_frames:
        return
    np.savez(
        save_path,
        root_orient=np.stack([frame["root_orient"] for frame in smplx_frames], axis=0),
        pose_body=np.stack([frame["pose_body"] for frame in smplx_frames], axis=0),
        trans=np.stack([frame["trans"] for frame in smplx_frames], axis=0),
        betas=smplx_frames[0]["betas"],
        gender=smplx_frames[0]["gender"],
        mocap_frame_rate=smplx_frames[0]["mocap_frame_rate"],
    )


def save_robot_pickle(save_path: str, qpos_history: List[np.ndarray], fps: float) -> None:
    save_robot_pickle_with_layout(save_path, qpos_history, fps, floating_root=True)


def save_robot_pickle_with_layout(
    save_path: str,
    qpos_history: List[np.ndarray],
    fps: float,
    floating_root: bool,
) -> None:
    if not qpos_history:
        return
    if floating_root:
        root_pos = np.asarray([qpos[:3] for qpos in qpos_history], dtype=np.float32)
        root_rot = np.asarray([qpos[3:7][[1, 2, 3, 0]] for qpos in qpos_history], dtype=np.float32)
        dof_pos = np.asarray([qpos[7:] for qpos in qpos_history], dtype=np.float32)
    else:
        root_pos = np.zeros((len(qpos_history), 3), dtype=np.float32)
        root_rot = np.tile(np.array([[0.0, 0.0, 0.0, 1.0]], dtype=np.float32), (len(qpos_history), 1))
        dof_pos = np.asarray(qpos_history, dtype=np.float32)
    with open(save_path, "wb") as file_obj:
        pickle.dump(
            {
                "fps": fps,
                "root_pos": root_pos,
                "root_rot": root_rot,
                "dof_pos": dof_pos,
                "local_body_pos": None,
                "link_body_list": None,
            },
            file_obj,
        )


def apply_robot_root_rotation_correction(qpos: np.ndarray, robot_name: str) -> np.ndarray:
    correction_entry = ROOT_ROTATION_CORRECTION_BY_ROBOT.get(robot_name)
    if correction_entry is None:
        return qpos.copy()

    correction, rotate_position = correction_entry

    corrected = qpos.copy()
    root_rot = R.from_quat(corrected[3:7], scalar_first=True)

    if rotate_position:
        corrected[:3] = correction.apply(corrected[:3])
        corrected[3:7] = (correction * root_rot).as_quat(scalar_first=True)
    else:
        corrected[3:7] = (root_rot * correction).as_quat(scalar_first=True)
    return corrected


def robot_has_floating_root(model, mj) -> bool:
    return bool(model.njnt > 0 and int(model.jnt_type[0]) == int(mj.mjtJoint.mjJNT_FREE))


def split_qpos_for_robot(qpos: np.ndarray, floating_root: bool) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if floating_root:
        return qpos[:3], qpos[3:7], qpos[7:]
    return (
        np.zeros(3, dtype=np.float64),
        np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64),
        qpos,
    )


def smooth_qpos(
    prev_qpos: np.ndarray | None,
    curr_qpos: np.ndarray,
    alpha: float,
    floating_root: bool,
) -> np.ndarray:
    if prev_qpos is None or alpha <= 0.0:
        return curr_qpos.copy()

    alpha = float(np.clip(alpha, 0.0, 1.0))
    smoothed = curr_qpos.copy()
    if not floating_root:
        smoothed[:] = (1.0 - alpha) * prev_qpos + alpha * curr_qpos
        return smoothed

    smoothed[:3] = (1.0 - alpha) * prev_qpos[:3] + alpha * curr_qpos[:3]

    prev_quat = prev_qpos[3:7].copy()
    curr_quat = curr_qpos[3:7].copy()
    if np.dot(prev_quat, curr_quat) < 0.0:
        curr_quat = -curr_quat
    blended_quat = (1.0 - alpha) * prev_quat + alpha * curr_quat
    quat_norm = np.linalg.norm(blended_quat)
    smoothed[3:7] = blended_quat / quat_norm if quat_norm > 1e-8 else curr_quat

    smoothed[7:] = (1.0 - alpha) * prev_qpos[7:] + alpha * curr_qpos[7:]
    return smoothed


def root_quat_angle(prev_qpos: np.ndarray, curr_qpos: np.ndarray) -> float:
    prev_quat = prev_qpos[3:7].copy()
    curr_quat = curr_qpos[3:7].copy()
    if np.dot(prev_quat, curr_quat) < 0.0:
        curr_quat = -curr_quat
    dot = float(np.clip(abs(np.dot(prev_quat, curr_quat)), 0.0, 1.0))
    return 2.0 * float(np.arccos(dot))


def qpos_outlier_reason(
    prev_qpos: np.ndarray | None,
    curr_qpos: np.ndarray,
    max_root_step: float,
    max_root_angle_step: float,
    max_dof_step: float,
    floating_root: bool,
) -> str | None:
    if prev_qpos is None:
        return None
    if not np.isfinite(curr_qpos).all():
        return "qpos contains non-finite values"

    if not floating_root:
        dof_step = float(np.max(np.abs(curr_qpos - prev_qpos)))
        if dof_step > max_dof_step:
            return f"joint step {dof_step:.3f}rad > {max_dof_step:.3f}rad"
        return None

    root_step = float(np.linalg.norm(curr_qpos[:3] - prev_qpos[:3]))
    if root_step > max_root_step:
        return f"root step {root_step:.3f}m > {max_root_step:.3f}m"

    root_angle_step = root_quat_angle(prev_qpos, curr_qpos)
    if root_angle_step > max_root_angle_step:
        return f"root angle step {root_angle_step:.3f}rad > {max_root_angle_step:.3f}rad"

    if curr_qpos.shape[0] > 7:
        dof_step = float(np.max(np.abs(curr_qpos[7:] - prev_qpos[7:])))
        if dof_step > max_dof_step:
            return f"joint step {dof_step:.3f}rad > {max_dof_step:.3f}rad"
    return None


def create_ground_projector(mj, xml_file: str, ground_offset: float):
    model = mj.MjModel.from_xml_path(str(xml_file))
    data = mj.MjData(model)
    geom_mask = np.asarray(model.geom_bodyid != 0)
    body_mask = np.arange(model.nbody) > 0

    def project(qpos: np.ndarray) -> np.ndarray:
        adjusted = qpos.copy()
        data.qpos[:] = adjusted
        mj.mj_forward(model, data)

        if np.any(geom_mask):
            lowest_z = float(np.min(data.geom_xpos[geom_mask, 2]))
        else:
            lowest_z = float(np.min(data.xpos[body_mask, 2]))

        adjusted[2] += float(ground_offset) - lowest_z
        return adjusted

    return project


def identity_projector(qpos: np.ndarray) -> np.ndarray:
    return qpos


def adapt_human_frame_for_robot(
    human_frame: Dict[str, Tuple[np.ndarray, np.ndarray]],
    robot_name: str,
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    if robot_name != "piper":
        return human_frame

    shoulder_pos, _ = human_frame["right_shoulder"]

    adapted: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    for joint_name, (pos, quat) in human_frame.items():
        pos_local = PIPER_SHOULDER_ANCHOR + PIPER_SMPL_TO_ROBOT_BASIS @ (pos - shoulder_pos)
        adapted[joint_name] = (
            pos_local.astype(np.float64),
            quat,
        )
    return adapted


DEFAULT_TWIST2_MIMIC_OBS_G1 = np.concatenate([
    np.array([0.0, 0.0, 0.8, 0.0, 0.0, 0.0], dtype=np.float64),
    np.array([
        -0.2, 0.0, 0.0, 0.4, -0.2, 0.0,
        -0.2, 0.0, 0.0, 0.4, -0.2, 0.0,
        0.0, 0.0, 0.0,
        0.0, 0.4, 0.0, 1.2, 0.0, 0.0, 0.0,
        0.0, -0.4, 0.0, 1.2, 0.0, 0.0, 0.0,
    ], dtype=np.float64),
])


def wrap_angle(angle: float) -> float:
    return float((angle + np.pi) % (2.0 * np.pi) - np.pi)


def twist2_mimic_obs_from_qpos(qpos: np.ndarray, prev_qpos: np.ndarray | None, dt: float) -> np.ndarray:
    if qpos.shape[0] < 36:
        raise ValueError(f"TWIST2 G1 expects qpos with root(7)+29 dof, got {qpos.shape[0]} dims")

    root_pos = qpos[:3]
    root_rot = R.from_quat(qpos[3:7], scalar_first=True)
    dof_pos = qpos[7:36]
    roll, pitch, yaw = root_rot.as_euler("xyz", degrees=False)

    if prev_qpos is None or dt <= 0.0:
        base_vel_local = np.zeros(3, dtype=np.float64)
        yaw_ang_vel = 0.0
    else:
        base_vel_world = (root_pos - prev_qpos[:3]) / dt
        base_vel_local = root_rot.inv().apply(base_vel_world)
        prev_yaw = R.from_quat(prev_qpos[3:7], scalar_first=True).as_euler("xyz", degrees=False)[2]
        yaw_ang_vel = wrap_angle(float(yaw - prev_yaw)) / dt

    mimic_obs = np.concatenate([
        base_vel_local[:2],
        root_pos[2:3],
        np.array([roll, pitch, yaw_ang_vel], dtype=np.float64),
        dof_pos,
    ])
    if mimic_obs.shape[0] != 35:
        raise ValueError(f"TWIST2 mimic_obs must be 35D, got {mimic_obs.shape[0]}")
    return mimic_obs


class Twist2RedisPublisher:
    def __init__(self, host: str, port: int, robot_key: str):
        try:
            import redis
        except ImportError as exc:
            raise SystemExit("Python package redis is required: pip install redis[hiredis]") from exc

        self.robot_key = robot_key
        self.redis_client = redis.Redis(host=host, port=port, db=0)
        self.redis_client.ping()
        print(f"TWIST2 Redis publisher connected to {host}:{port}, robot_key={robot_key}")

    def publish(self, mimic_obs: np.ndarray) -> None:
        pipeline = self.redis_client.pipeline()
        pipeline.set(f"action_body_{self.robot_key}", json.dumps(mimic_obs.tolist()))
        pipeline.set(f"action_hand_left_{self.robot_key}", json.dumps([0.0] * 7))
        pipeline.set(f"action_hand_right_{self.robot_key}", json.dumps([0.0] * 7))
        pipeline.set(f"action_neck_{self.robot_key}", json.dumps([0.0, 0.0]))
        pipeline.set("t_action", int(time.time() * 1000))
        pipeline.execute()

    def publish_default(self) -> None:
        self.publish(DEFAULT_TWIST2_MIMIC_OBS_G1)


def main() -> None:
    args = parse_args()
    ensure_parent_dir(args.save_robot_path)
    ensure_parent_dir(args.save_smpl_path)
    ensure_parent_dir(args.save_smplx_path)
    ensure_parent_dir(args.video_path if args.record_video else None)

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind((args.udp_host, args.udp_port))
    sock.settimeout(args.socket_timeout)

    try:
        GMR, RobotMotionViewer, mj = load_gmr_runtime()
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "Failed to import GMR runtime dependencies. "
            f"Missing module: {exc.name}. Activate/install the GMR environment first."
        ) from exc

    retargeter = GMR(
        src_human="smplx",
        tgt_robot=args.robot,
        actual_human_height=args.human_height,
        use_velocity_limit=True,
    )
    retarget_passes = args.retarget_passes
    if retarget_passes is None:
        retarget_passes = 8 if args.robot == "piper" else 1
    retarget_passes = max(1, int(retarget_passes))
    floating_root = robot_has_floating_root(retargeter.model, mj)
    if floating_root:
        project_to_ground = create_ground_projector(mj, retargeter.xml_file, args.ground_offset)
    else:
        project_to_ground = identity_projector

    twist2_publisher = None
    if args.twist2_redis_host:
        if args.robot not in {"unitree_g1", "unitree_g1_with_hands"}:
            raise SystemExit("TWIST2 Redis publishing currently expects --robot unitree_g1 or unitree_g1_with_hands")
        twist2_publisher = Twist2RedisPublisher(
            host=args.twist2_redis_host,
            port=args.twist2_redis_port,
            robot_key=args.twist2_robot_key,
        )

    viewer = None
    if not args.no_viewer:
        viewer = RobotMotionViewer(
            robot_type=args.robot,
            motion_fps=args.fps,
            transparent_robot=0,
            record_video=args.record_video,
            video_path=args.video_path,
        )

    qpos_history: List[np.ndarray] = []
    smpl_joint_history: List[np.ndarray] = []
    smpl_rotmat_history: List[np.ndarray] = []
    smplx_history: List[Dict[str, np.ndarray]] = []

    frame_count = 0
    dropped_count = 0
    outlier_count = 0
    start_time = time.time()
    root_translation_raw = np.zeros(3, dtype=np.float64)
    prev_qpos: np.ndarray | None = None
    prev_twist2_qpos: np.ndarray | None = None

    print(f"Listening for SMPL UDP on {args.udp_host}:{args.udp_port}")
    print(f"Retargeting to robot: {args.robot}")
    if args.robot == "piper" and not args.disable_outlier_filter:
        print("Piper: disabling humanoid qpos outlier filter by default")
    print("Press Ctrl+C to stop")

    try:
        while True:
            payload = receive_payload(sock)
            if payload is None:
                continue

            try:
                if not payload_contains_direct_smpl(payload):
                    raise ValueError("UDP payload is not a direct SMPL-24 packet")

                local_rotmats, global_positions_raw, root_translation_raw = parse_direct_smpl_payload(
                    payload,
                    invert_rotmats=args.invert_rotmats,
                    use_camera_translation=args.use_camera_translation,
                )
                smplx_frame = build_smplx_like_pose(local_rotmats, root_translation_raw, args.gender, args.fps)
                global_rotations_raw = global_rotations_from_local(local_rotmats)
                human_frame = build_gmr_human_frame(
                    global_positions_raw,
                    global_rotations_raw,
                    apply_basis_transform=False,
                )
                human_frame = adapt_human_frame_for_robot(human_frame, args.robot)
                qpos = None
                for _ in range(retarget_passes):
                    qpos = retargeter.retarget(human_frame, offset_to_ground=floating_root)
                assert qpos is not None
                if floating_root:
                    qpos = apply_robot_root_rotation_correction(qpos, args.robot)
                qpos = project_to_ground(qpos)

                outlier_reason = None
                use_outlier_filter = not args.disable_outlier_filter and args.robot != "piper"
                if use_outlier_filter:
                    outlier_reason = qpos_outlier_reason(
                        prev_qpos,
                        qpos,
                        args.max_root_step,
                        args.max_root_angle_step,
                        args.max_dof_step,
                        floating_root,
                    )
                if outlier_reason is not None and prev_qpos is not None:
                    outlier_count += 1
                    if outlier_count == 1 or outlier_count % args.print_every == 0:
                        print(f"Rejected qpos outlier: {outlier_reason}")
                    qpos = prev_qpos.copy()
                else:
                    qpos = smooth_qpos(prev_qpos, qpos, args.smoothing_alpha, floating_root)
            except Exception as exc:
                dropped_count += 1
                print(f"Failed to process UDP frame: {exc}")
                continue

            frame_count += 1
            if twist2_publisher is not None:
                mimic_obs = twist2_mimic_obs_from_qpos(qpos, prev_twist2_qpos, 1.0 / args.fps)
                twist2_publisher.publish(mimic_obs)
                prev_twist2_qpos = qpos.copy()

            prev_qpos = qpos.copy()
            qpos_history.append(qpos.copy())
            smpl_joint_history.append(
                np.stack([global_positions_raw[joint_name] for joint_name in SMPL_JOINT_NAMES], axis=0).astype(np.float32)
            )
            smpl_rotmat_history.append(local_rotmats.astype(np.float32))
            smplx_history.append(smplx_frame)

            if viewer is not None:
                root_pos, root_rot, dof_pos = split_qpos_for_robot(qpos, floating_root)
                viewer.step(
                    root_pos=root_pos,
                    root_rot=root_rot,
                    dof_pos=dof_pos,
                    human_motion_data=retargeter.scaled_human_data,
                    rate_limit=False,
                    follow_camera=True,
                )

            if frame_count == 1 or frame_count % args.print_every == 0:
                elapsed = max(time.time() - start_time, 1e-6)
                qpos_text = ""
                if args.robot == "piper":
                    qpos_text = " qpos=" + np.array2string(qpos, precision=3, suppress_small=True)
                print(
                    f"frames={frame_count} dropped={dropped_count} outliers={outlier_count} "
                    f"avg_fps={frame_count / elapsed:.2f} seq={payload.get('seq', 'n/a')} "
                    f"retarget_passes={retarget_passes}{qpos_text}"
                )

    except KeyboardInterrupt:
        print("\nStopping receiver")
    finally:
        sock.close()
        if viewer is not None:
            viewer.close()

        if twist2_publisher is not None:
            twist2_publisher.publish_default()

        if args.save_robot_path:
            save_robot_pickle_with_layout(args.save_robot_path, qpos_history, args.fps, floating_root)
            print(f"Saved robot motion to {args.save_robot_path}")
        if args.save_smpl_path:
            save_smpl_npz(args.save_smpl_path, smpl_joint_history, smpl_rotmat_history, args.fps)
            print(f"Saved SMPL data to {args.save_smpl_path}")
        if args.save_smplx_path:
            save_smplx_npz(args.save_smplx_path, smplx_history)
            print(f"Saved SMPL-X-like data to {args.save_smplx_path}")

        print(f"Total frames: {frame_count} | Dropped: {dropped_count} | Outliers: {outlier_count}")


if __name__ == "__main__":
    main()
