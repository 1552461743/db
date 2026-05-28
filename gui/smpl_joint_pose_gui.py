#!/usr/bin/env python3
"""Interactive SMPL joint pose GUI.

This tool lets you pick any of the 24 SMPL joints, adjust its local XYZ Euler
angles, and preview the resulting mesh deformation in real time.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import pickle
import socket
import sys
import traceback
import tkinter as tk
from pathlib import Path
from tkinter import messagebox, ttk

import cv2
import numpy as np
import torch
from scipy.spatial.transform import Rotation

try:
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    from matplotlib.figure import Figure
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    MATPLOTLIB_AVAILABLE = True
except ImportError:
    FigureCanvasTkAgg = None
    Figure = None
    Poly3DCollection = None
    MATPLOTLIB_AVAILABLE = False

try:
    from PIL import Image, ImageTk

    PIL_AVAILABLE = True
except ImportError:
    Image = None
    ImageTk = None
    PIL_AVAILABLE = False


SUIT_ROOT = Path(__file__).resolve().parents[1]
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
lbs = None
KAIPU_MAPPING_MODULE = None


def load_module_from_path(module_name: str, module_path: Path):
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load module spec from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module



def ensure_render_modules_loaded() -> None:
    global lbs
    if lbs is not None:
        return

    smpl_lbs_module = load_module_from_path(
        "smpl_lbs_direct",
        REPO_ROOT / "hybrik/models/layers/smpl/lbs.py",
    )
    lbs = smpl_lbs_module.lbs


def ensure_kaipu_mapping_loaded():
    global KAIPU_MAPPING_MODULE
    if KAIPU_MAPPING_MODULE is not None:
        return KAIPU_MAPPING_MODULE

    KAIPU_MAPPING_MODULE = load_module_from_path(
        "kaipu_smpl_mapping_direct",
        SUIT_ROOT / "ros2/kaipu_model/kaipu_smpl_mapping.py",
    )
    return KAIPU_MAPPING_MODULE


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

INTERACTIVE_RENDER_DELAY_MS = 30
FULL_RENDER_DELAY_MS = 120
MATPLOTLIB_MESH_FACE_STRIDE = 6


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GUI for editing 24-joint SMPL rotations.")
    parser.add_argument(
        "--device",
        default=None,
        help="Torch device, e.g. cuda:0 or cpu. Default: auto-detect.",
    )
    parser.add_argument(
        "--render-size",
        type=int,
        default=640,
        help="Square preview size in pixels.",
    )
    parser.add_argument(
        "--camera-distance",
        type=float,
        default=2.5,
        help="Camera translation distance along +Z.",
    )
    parser.add_argument(
        "--focal-length",
        type=float,
        default=1200.0,
        help="Perspective camera focal length.",
    )
    parser.add_argument(
        "--udp-host",
        default="127.0.0.1",
        help="UDP host for Kaipu MuJoCo joint commands.",
    )
    parser.add_argument(
        "--udp-port",
        type=int,
        default=5007,
        help="UDP port for Kaipu MuJoCo joint commands.",
    )
    parser.add_argument(
        "--udp-disabled",
        action="store_true",
        help="Disable UDP publishing from the GUI.",
    )
    return parser.parse_args()


def resolve_device(device_arg: str | None) -> torch.device:
    if device_arg:
        return torch.device(device_arg)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def to_numpy(array, dtype=np.float32) -> np.ndarray:
    if hasattr(array, "todense"):
        array = array.todense()
    return np.array(array, dtype=dtype)


def load_smpl_state(device: torch.device) -> dict[str, torch.Tensor]:
    ensure_render_modules_loaded()

    model_dir = REPO_ROOT / "model_files"
    smpl_path = model_dir / "basicModel_neutral_lbs_10_207_0_v1.0.0.pkl"
    h36m_path = model_dir / "J_regressor_h36m.npy"

    if not smpl_path.exists():
        raise FileNotFoundError(f"SMPL model file not found: {smpl_path}")
    if not h36m_path.exists():
        raise FileNotFoundError(f"H36M regressor not found: {h36m_path}")

    try:
        with smpl_path.open("rb") as smpl_file:
            smpl_data = pickle.load(smpl_file, encoding="latin1")
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Loading the SMPL model requires the HybrIK runtime dependencies, including 'chumpy'. "
            "Run this GUI inside your existing 'hybrik' environment."
        ) from exc

    if isinstance(smpl_data, dict):
        smpl_dict = smpl_data
    else:
        smpl_dict = smpl_data.__dict__

    h36m_jregressor = np.load(str(h36m_path))
    posedirs_raw = smpl_dict["posedirs"]
    num_pose_basis = posedirs_raw.shape[-1]
    posedirs = np.reshape(posedirs_raw, [-1, num_pose_basis]).T

    parents = torch.tensor(to_numpy(smpl_dict["kintree_table"][0], dtype=np.int64), dtype=torch.long)
    parents[0] = -1
    parents = parents[:24].to(device)

    faces = torch.tensor(to_numpy(smpl_dict["f"], dtype=np.int64), dtype=torch.long, device=device)
    v_template = torch.tensor(to_numpy(smpl_dict["v_template"]), dtype=torch.float32, device=device)
    shapedirs = torch.tensor(to_numpy(smpl_dict["shapedirs"]), dtype=torch.float32, device=device)
    posedirs_tensor = torch.tensor(to_numpy(posedirs), dtype=torch.float32, device=device)
    j_regressor = torch.tensor(to_numpy(smpl_dict["J_regressor"]), dtype=torch.float32, device=device)
    j_regressor_h36m = torch.tensor(to_numpy(h36m_jregressor), dtype=torch.float32, device=device)
    lbs_weights = torch.tensor(to_numpy(smpl_dict["weights"]), dtype=torch.float32, device=device)
    faces_np = to_numpy(smpl_dict["f"], dtype=np.int32)
    parents_np = to_numpy(smpl_dict["kintree_table"][0], dtype=np.int32)[:24]
    parents_np[0] = -1
    rotation_180_z = Rotation.from_euler("z", 180.0, degrees=True).as_matrix().astype(np.float32)

    return {
        "v_template": v_template,
        "shapedirs": shapedirs,
        "posedirs": posedirs_tensor,
        "j_regressor": j_regressor,
        "j_regressor_h36m": j_regressor_h36m,
        "parents": parents,
        "lbs_weights": lbs_weights,
        "zero_betas": torch.zeros((1, 10), dtype=torch.float32, device=device),
        "faces": faces,
        "faces_np": faces_np,
        "parents_np": parents_np,
        "rotation_180_z": rotation_180_z,
    }


def euler_xyz_degrees_to_rotmat(angles_deg: np.ndarray) -> np.ndarray:
    return Rotation.from_euler("xyz", angles_deg, degrees=True).as_matrix().astype(np.float32)


def compose_pose_rotmats(euler_degrees: np.ndarray) -> np.ndarray:
    rotmats = np.zeros((len(JOINT_NAMES), 3, 3), dtype=np.float32)
    for joint_index, angles_deg in enumerate(euler_degrees):
        rotmats[joint_index] = euler_xyz_degrees_to_rotmat(angles_deg)
    return rotmats


def fit_points_to_canvas(
    vertices_2d: np.ndarray,
    joints_2d: np.ndarray,
    render_size: int,
    margin_ratio: float = 0.08,
) -> tuple[np.ndarray, np.ndarray]:
    valid_mask = np.isfinite(vertices_2d).all(axis=1)
    if not np.any(valid_mask):
        return vertices_2d, joints_2d

    valid_vertices = vertices_2d[valid_mask]
    bbox_min = valid_vertices.min(axis=0)
    bbox_max = valid_vertices.max(axis=0)
    bbox_size = np.maximum(bbox_max - bbox_min, 1e-4)

    target_extent = render_size * (1.0 - 2.0 * margin_ratio)
    scale = min(target_extent / bbox_size[0], target_extent / bbox_size[1])
    bbox_center = 0.5 * (bbox_min + bbox_max)
    canvas_center = np.array([render_size * 0.5, render_size * 0.5], dtype=np.float32)

    vertices_fitted = (vertices_2d - bbox_center) * scale + canvas_center
    joints_fitted = (joints_2d - bbox_center) * scale + canvas_center
    return vertices_fitted, joints_fitted


@torch.no_grad()
def compute_pose_geometry(
    smpl_state: dict[str, torch.Tensor],
    rotmats: np.ndarray,
    device: torch.device,
) -> dict[str, np.ndarray]:
    ensure_render_modules_loaded()

    pose_tensor = torch.from_numpy(rotmats).float().unsqueeze(0).to(device)
    vertices, joints, _, _ = lbs(
        smpl_state["zero_betas"],
        pose_tensor,
        smpl_state["v_template"],
        smpl_state["shapedirs"],
        smpl_state["posedirs"],
        smpl_state["j_regressor"],
        smpl_state["j_regressor_h36m"],
        smpl_state["parents"],
        smpl_state["lbs_weights"],
        pose2rot=False,
        dtype=torch.float32,
    )

    vertices = vertices - joints[:, [0], :]
    joints = joints - joints[:, [0], :]

    return {
        "vertices": vertices[0].detach().cpu().numpy(),
        "joints": joints[0].detach().cpu().numpy(),
        "faces": smpl_state["faces_np"],
        "parents": smpl_state["parents_np"],
        "rotation_180_z": smpl_state["rotation_180_z"],
    }


def set_axes_equal_3d(axis, points: np.ndarray) -> None:
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    center = 0.5 * (mins + maxs)
    radius = 0.5 * max(float((maxs - mins).max()), 1e-3)
    axis.set_xlim(center[0] - radius, center[0] + radius)
    axis.set_ylim(center[1] - radius, center[1] + radius)
    axis.set_zlim(center[2] - radius, center[2] + radius)


def render_pose_matplotlib(
    figure,
    axis,
    geometry: dict[str, np.ndarray],
    display_mode: str,
    highlight_joint_index: int | None,
    fast_mode: bool,
) -> None:
    vertices = geometry["vertices"]
    joints = geometry["joints"]
    faces = geometry["faces"]
    rotation_180_z = geometry["rotation_180_z"]
    parents = geometry["parents"]

    vertices_view = (rotation_180_z @ vertices.T).T
    joints_view = (rotation_180_z @ joints.T).T

    axis.clear()
    axis.set_facecolor((0.12, 0.12, 0.12))
    figure.patch.set_facecolor((0.12, 0.12, 0.12))

    if display_mode == "mesh" and not fast_mode:
        mesh_faces = faces[::MATPLOTLIB_MESH_FACE_STRIDE]
        tri_vertices = vertices_view[mesh_faces]
        mesh = Poly3DCollection(
            tri_vertices,
            facecolors=(0.7, 0.72, 0.78, 0.55),
            edgecolors=(0.2, 0.2, 0.24, 0.12),
            linewidths=0.15,
        )
        axis.add_collection3d(mesh)

    for joint_index, parent_index in enumerate(parents):
        if parent_index < 0:
            continue
        segment = joints_view[[parent_index, joint_index]]
        axis.plot(segment[:, 0], segment[:, 1], segment[:, 2], color="#2ec4ff", linewidth=2.2)

    colors = ["#50fa7b"] * len(joints_view)
    sizes = [24] * len(joints_view)
    if highlight_joint_index is not None and 0 <= highlight_joint_index < len(joints_view):
        colors[highlight_joint_index] = "#ff6b6b"
        sizes[highlight_joint_index] = 54

    axis.scatter(joints_view[:, 0], joints_view[:, 1], joints_view[:, 2], c=colors, s=sizes, depthshade=False)

    combined_points = np.concatenate([vertices_view, joints_view], axis=0)
    set_axes_equal_3d(axis, combined_points)
    axis.view_init(elev=18, azim=-95)
    axis.set_xticks([])
    axis.set_yticks([])
    axis.set_zticks([])
    axis.grid(False)
    axis.set_box_aspect((1.0, 1.9, 0.8))
    axis.set_proj_type("persp")
    figure.tight_layout(pad=0.0)


@torch.no_grad()
def render_pose_preview(
    geometry: dict[str, np.ndarray],
    render_size: int,
    camera_distance: float,
    focal_length: float,
    highlight_joint_index: int | None = None,
    show_mesh: bool = True,
) -> np.ndarray:
    vertices_np = geometry["vertices"]
    joints_np = geometry["joints"]
    faces_np = geometry["faces"]
    parents_np = geometry["parents"]
    rotation_180_z = geometry["rotation_180_z"]
    translation_np = np.array([0.0, 0.0, camera_distance], dtype=np.float32)

    vertices_cam = (rotation_180_z @ (vertices_np + translation_np).T).T
    joints_cam = (rotation_180_z @ (joints_np + translation_np).T).T
    safe_vertex_z = np.clip(vertices_cam[:, 2], 1e-4, None)
    safe_joint_z = np.clip(joints_cam[:, 2], 1e-4, None)

    vertices_2d = np.empty((vertices_cam.shape[0], 2), dtype=np.float32)
    vertices_2d[:, 0] = focal_length * vertices_cam[:, 0] / safe_vertex_z + render_size * 0.5
    vertices_2d[:, 1] = render_size * 0.5 - focal_length * vertices_cam[:, 1] / safe_vertex_z

    joints_2d = np.empty((joints_cam.shape[0], 2), dtype=np.float32)
    joints_2d[:, 0] = focal_length * joints_cam[:, 0] / safe_joint_z + render_size * 0.5
    joints_2d[:, 1] = render_size * 0.5 - focal_length * joints_cam[:, 1] / safe_joint_z
    vertices_2d, joints_2d = fit_points_to_canvas(vertices_2d, joints_2d, render_size)

    image = np.full((render_size, render_size, 3), 30, dtype=np.uint8)
    if show_mesh:
        triangles_3d = vertices_cam[faces_np]
        triangles_2d = vertices_2d[faces_np]
        triangle_depth = triangles_3d[:, :, 2].mean(axis=1)

        for face_index in np.argsort(triangle_depth)[::-1]:
            tri_3d = triangles_3d[face_index]
            if np.any(tri_3d[:, 2] <= 1e-4):
                continue

            tri_2d = triangles_2d[face_index]
            if not np.isfinite(tri_2d).all():
                continue

            normal = np.cross(tri_3d[1] - tri_3d[0], tri_3d[2] - tri_3d[0])
            normal_norm = np.linalg.norm(normal)
            if normal_norm <= 1e-8:
                continue

            normal = normal / normal_norm
            intensity = 0.35 + 0.65 * abs(float(normal[2]))
            shade = int(np.clip(70 + 160 * intensity, 0, 255))
            color = (shade, shade, shade)
            cv2.fillConvexPoly(image, np.round(tri_2d).astype(np.int32), color, lineType=cv2.LINE_AA)

    for joint_index, parent_index in enumerate(parents_np):
        if parent_index < 0 or joint_index >= len(joints_2d) or parent_index >= len(joints_2d):
            continue
        if joints_cam[joint_index, 2] <= 1e-4 or joints_cam[parent_index, 2] <= 1e-4:
            continue

        pt_a = tuple(np.round(joints_2d[parent_index]).astype(np.int32).tolist())
        pt_b = tuple(np.round(joints_2d[joint_index]).astype(np.int32).tolist())
        cv2.line(image, pt_a, pt_b, (40, 180, 255), 2, lineType=cv2.LINE_AA)

    for joint_index, joint_xy in enumerate(joints_2d):
        if joints_cam[joint_index, 2] <= 1e-4:
            continue
        radius = 6 if joint_index == highlight_joint_index else 4
        color = (255, 80, 80) if joint_index == highlight_joint_index else (80, 255, 120)
        cv2.circle(image, tuple(np.round(joint_xy).astype(np.int32).tolist()), radius, color, -1, lineType=cv2.LINE_AA)

    return image


def rgb_to_photoimage(rgb_image: np.ndarray) -> tk.PhotoImage:
    if PIL_AVAILABLE:
        return ImageTk.PhotoImage(Image.fromarray(rgb_image))

    height, width = rgb_image.shape[:2]
    ppm_header = f"P6\n{width} {height}\n255\n".encode("ascii")
    return tk.PhotoImage(data=ppm_header + rgb_image.tobytes(), format="PPM")


class SMPLPoseGUI:
    def __init__(self, root: tk.Tk, args: argparse.Namespace) -> None:
        self.root = root
        self.args = args
        self.device = resolve_device(args.device)
        self.smpl_state = load_smpl_state(self.device)
        self.kaipu_mapping = ensure_kaipu_mapping_loaded()
        self.kaipu_joint_limits = self.kaipu_mapping.load_actuator_joint_limits(self.kaipu_mapping.DEFAULT_XML_PATH)

        self.euler_degrees = np.zeros((len(JOINT_NAMES), 3), dtype=np.float32)
        self.selected_joint_index = 0
        self.pending_render_id: str | None = None
        self.pending_full_render_id: str | None = None
        self.preview_photo: tk.PhotoImage | None = None
        self.preview_rgb = np.full((args.render_size, args.render_size, 3), 30, dtype=np.uint8)
        self.updating_controls = False
        self.backend_var = tk.StringVar(value="software_2d")
        self.display_mode_var = tk.StringVar(value="mesh")
        self.mpl_figure = None
        self.mpl_axis = None
        self.mpl_canvas = None
        self.mpl_canvas_widget = None
        self.udp_enabled = not args.udp_disabled
        self.udp_host = args.udp_host
        self.udp_port = int(args.udp_port)
        self.udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM) if self.udp_enabled else None
        self.udp_seq = 0

        self.root.title("SMPL Joint Pose GUI")
        self.root.geometry("1180x760")
        self.root.minsize(1000, 700)
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

        self.status_var = tk.StringVar(value=f"Device: {self.device}")
        self.joint_label_var = tk.StringVar()
        self.angle_summary_var = tk.StringVar()
        self.x_var = tk.DoubleVar(value=0.0)
        self.y_var = tk.DoubleVar(value=0.0)
        self.z_var = tk.DoubleVar(value=0.0)
        self.transport_status_var = tk.StringVar()

        self.build_layout()
        if not MATPLOTLIB_AVAILABLE:
            self.backend_var.set("software_2d")
        self.sync_controls_from_selection()
        self.render_now(show_mesh=True)
        self.send_udp_joint_targets()


    def build_layout(self) -> None:
        container = ttk.Frame(self.root, padding=12)
        container.pack(fill=tk.BOTH, expand=True)
        container.columnconfigure(0, weight=0)
        container.columnconfigure(1, weight=1)
        container.rowconfigure(0, weight=1)

        control_frame = ttk.Frame(container, padding=(0, 0, 12, 0))
        control_frame.grid(row=0, column=0, sticky="ns")

        preview_frame = ttk.Frame(container)
        preview_frame.grid(row=0, column=1, sticky="nsew")
        preview_frame.columnconfigure(0, weight=1)
        preview_frame.rowconfigure(1, weight=1)

        ttk.Label(control_frame, text="Joints").pack(anchor="w")
        self.joint_listbox = tk.Listbox(control_frame, exportselection=False, height=24)
        for index, joint_name in enumerate(JOINT_NAMES):
            self.joint_listbox.insert(index, f"{index:02d}  {joint_name}")
        self.joint_listbox.selection_set(0)
        self.joint_listbox.bind("<<ListboxSelect>>", self.on_joint_selected)
        self.joint_listbox.pack(fill=tk.X, pady=(4, 10))

        ttk.Label(control_frame, textvariable=self.joint_label_var, font=("TkDefaultFont", 10, "bold")).pack(
            anchor="w", pady=(0, 4)
        )
        ttk.Label(control_frame, text="Local Euler XYZ angles in degrees").pack(anchor="w", pady=(0, 8))

        self.create_angle_slider(control_frame, "X", self.x_var)
        self.create_angle_slider(control_frame, "Y", self.y_var)
        self.create_angle_slider(control_frame, "Z", self.z_var)

        ttk.Label(control_frame, textvariable=self.angle_summary_var).pack(anchor="w", pady=(8, 12))

        button_row = ttk.Frame(control_frame)
        button_row.pack(fill=tk.X, pady=(0, 12))
        ttk.Button(button_row, text="Reset Joint", command=self.reset_selected_joint).pack(side=tk.LEFT, padx=(0, 8))
        ttk.Button(button_row, text="Reset All", command=self.reset_all_joints).pack(side=tk.LEFT)

        ttk.Label(control_frame, text="Display Mode").pack(anchor="w", pady=(6, 4))
        display_frame = ttk.Frame(control_frame)
        display_frame.pack(fill=tk.X, pady=(0, 8))
        ttk.Radiobutton(
            display_frame,
            text="Mesh",
            variable=self.display_mode_var,
            value="mesh",
            command=self.on_render_option_changed,
        ).pack(side=tk.LEFT, padx=(0, 8))
        ttk.Radiobutton(
            display_frame,
            text="Skeleton Only",
            variable=self.display_mode_var,
            value="skeleton",
            command=self.on_render_option_changed,
        ).pack(side=tk.LEFT)

        ttk.Label(control_frame, text="Render Backend").pack(anchor="w", pady=(0, 4))
        backend_frame = ttk.Frame(control_frame)
        backend_frame.pack(fill=tk.X, pady=(0, 12))
        ttk.Radiobutton(
            backend_frame,
            text="Software 2D",
            variable=self.backend_var,
            value="software_2d",
            command=self.on_render_option_changed,
        ).pack(anchor="w")
        self.mpl_backend_radio = ttk.Radiobutton(
            backend_frame,
            text="Matplotlib 3D",
            variable=self.backend_var,
            value="matplotlib_3d",
            command=self.on_render_option_changed,
        )
        self.mpl_backend_radio.pack(anchor="w")
        if not MATPLOTLIB_AVAILABLE:
            self.mpl_backend_radio.state(["disabled"])

        help_text = (
            "Notes:\n"
            "- pelvis controls the whole body root orientation\n"
            "- each slider edits one joint's local rotation\n"
            "- pose order is SMPL's 24-joint order\n"
            "- matplotlib 3D uses a lighter mesh for smoother preview"
        )
        ttk.Label(control_frame, text=help_text, justify=tk.LEFT).pack(anchor="w")
        self.update_transport_status("ready")
        ttk.Label(control_frame, textvariable=self.transport_status_var, justify=tk.LEFT).pack(anchor="w", pady=(8, 0))

        ttk.Label(preview_frame, text="SMPL Preview").grid(row=0, column=0, sticky="w", pady=(0, 6))
        self.preview_container = ttk.Frame(preview_frame)
        self.preview_container.grid(row=1, column=0, sticky="nsew")
        self.preview_container.columnconfigure(0, weight=1)
        self.preview_container.rowconfigure(0, weight=1)
        self.preview_label = ttk.Label(self.preview_container)
        self.preview_label.grid(row=0, column=0, sticky="nsew")

        status_bar = ttk.Label(self.root, textvariable=self.status_var, anchor="w", relief=tk.SUNKEN)
        status_bar.pack(fill=tk.X, side=tk.BOTTOM)


    def create_angle_slider(self, parent: ttk.Frame, axis_label: str, variable: tk.DoubleVar) -> None:
        frame = ttk.Frame(parent)
        frame.pack(fill=tk.X, pady=4)
        ttk.Label(frame, text=f"{axis_label}:", width=3).pack(side=tk.LEFT)
        scale = tk.Scale(
            frame,
            from_=-180.0,
            to=180.0,
            resolution=1.0,
            orient=tk.HORIZONTAL,
            variable=variable,
            length=280,
            command=self.on_slider_changed,
        )
        scale.bind("<ButtonRelease-1>", self.on_slider_released)
        scale.pack(side=tk.LEFT, fill=tk.X, expand=True)


    def on_joint_selected(self, _event: tk.Event | None = None) -> None:
        selection = self.joint_listbox.curselection()
        if not selection:
            return
        self.selected_joint_index = int(selection[0])
        self.sync_controls_from_selection()
        self.schedule_render(show_mesh=True, delay_ms=0)


    def on_render_option_changed(self) -> None:
        if self.backend_var.get() == "matplotlib_3d" and not MATPLOTLIB_AVAILABLE:
            self.backend_var.set("software_2d")
        self.activate_preview_backend()
        self.schedule_render(show_mesh=True, delay_ms=0)


    def activate_preview_backend(self) -> None:
        backend = self.backend_var.get()
        if backend == "matplotlib_3d":
            self.ensure_matplotlib_canvas()
            self.preview_label.grid_remove()
            if self.mpl_canvas_widget is not None:
                self.mpl_canvas_widget.grid(row=0, column=0, sticky="nsew")
            return

        if self.mpl_canvas_widget is not None:
            self.mpl_canvas_widget.grid_remove()
        self.preview_label.grid(row=0, column=0, sticky="nsew")


    def ensure_matplotlib_canvas(self) -> None:
        if self.mpl_canvas is not None or not MATPLOTLIB_AVAILABLE:
            return

        self.mpl_figure = Figure(figsize=(6.5, 6.5), dpi=100)
        self.mpl_axis = self.mpl_figure.add_subplot(111, projection="3d")
        self.mpl_canvas = FigureCanvasTkAgg(self.mpl_figure, master=self.preview_container)
        self.mpl_canvas_widget = self.mpl_canvas.get_tk_widget()


    def sync_controls_from_selection(self) -> None:
        self.updating_controls = True
        try:
            angles = self.euler_degrees[self.selected_joint_index]
            self.x_var.set(float(angles[0]))
            self.y_var.set(float(angles[1]))
            self.z_var.set(float(angles[2]))
            joint_name = JOINT_NAMES[self.selected_joint_index]
            self.joint_label_var.set(f"Selected: {self.selected_joint_index:02d}  {joint_name}")
            self.update_angle_summary()
        finally:
            self.updating_controls = False


    def update_angle_summary(self) -> None:
        x_deg, y_deg, z_deg = self.euler_degrees[self.selected_joint_index]
        self.angle_summary_var.set(f"Current angles: X={x_deg:.0f}  Y={y_deg:.0f}  Z={z_deg:.0f}")


    def on_slider_changed(self, _value: str) -> None:
        if self.updating_controls:
            return

        self.euler_degrees[self.selected_joint_index] = np.array(
            [self.x_var.get(), self.y_var.get(), self.z_var.get()], dtype=np.float32
        )
        self.update_angle_summary()
        self.send_udp_joint_targets()
        self.schedule_render(show_mesh=False, delay_ms=INTERACTIVE_RENDER_DELAY_MS)
        self.schedule_render(show_mesh=True, delay_ms=FULL_RENDER_DELAY_MS)


    def on_slider_released(self, _event: tk.Event | None = None) -> None:
        self.schedule_render(show_mesh=True, delay_ms=0)


    def schedule_render(self, show_mesh: bool, delay_ms: int) -> None:
        if show_mesh:
            if self.pending_full_render_id is not None:
                self.root.after_cancel(self.pending_full_render_id)
            self.pending_full_render_id = self.root.after(delay_ms, lambda: self.render_now(show_mesh=True))
            return

        if self.pending_render_id is not None:
            self.root.after_cancel(self.pending_render_id)
        self.pending_render_id = self.root.after(delay_ms, lambda: self.render_now(show_mesh=False))


    def reset_selected_joint(self) -> None:
        self.euler_degrees[self.selected_joint_index] = 0.0
        self.sync_controls_from_selection()
        self.send_udp_joint_targets()
        self.schedule_render(show_mesh=True, delay_ms=0)


    def reset_all_joints(self) -> None:
        self.euler_degrees.fill(0.0)
        self.sync_controls_from_selection()
        self.send_udp_joint_targets()
        self.schedule_render(show_mesh=True, delay_ms=0)


    def update_transport_status(self, suffix: str) -> None:
        if not self.udp_enabled:
            self.transport_status_var.set("UDP: disabled")
            return
        self.transport_status_var.set(f"UDP: {self.udp_host}:{self.udp_port} | {suffix}")


    def send_udp_joint_targets(self) -> None:
        if not self.udp_enabled or self.udp_socket is None:
            return

        try:
            commands = self.kaipu_mapping.build_kaipu_joint_targets_from_gui_euler(
                self.euler_degrees,
                self.kaipu_joint_limits,
            )
            joint_names = list(self.kaipu_mapping.KAIPU_JOINT_ORDER)
            payload = {
                "seq": self.udp_seq,
                "source": "smpl_gui",
                "joint_names": joint_names,
                "joint_positions": [float(commands.get(name, 0.0)) for name in joint_names],
            }
            self.udp_socket.sendto(json.dumps(payload).encode("utf-8"), (self.udp_host, self.udp_port))
            self.udp_seq += 1
            self.update_transport_status(f"sent seq={self.udp_seq}")
        except Exception as exc:
            self.update_transport_status(f"send failed: {exc}")


    def render_now(self, show_mesh: bool) -> None:
        if show_mesh:
            self.pending_full_render_id = None
        else:
            self.pending_render_id = None

        try:
            rotmats = compose_pose_rotmats(self.euler_degrees)
            geometry = compute_pose_geometry(self.smpl_state, rotmats, self.device)
            backend = self.backend_var.get()
            display_mode = self.display_mode_var.get()

            if backend == "matplotlib_3d":
                self.activate_preview_backend()
                self.ensure_matplotlib_canvas()
                render_pose_matplotlib(
                    self.mpl_figure,
                    self.mpl_axis,
                    geometry,
                    display_mode="mesh" if show_mesh and display_mode == "mesh" else "skeleton",
                    highlight_joint_index=self.selected_joint_index,
                    fast_mode=not show_mesh,
                )
                self.mpl_canvas.draw_idle()
            else:
                rgb_image = render_pose_preview(
                    geometry=geometry,
                    render_size=self.args.render_size,
                    camera_distance=self.args.camera_distance,
                    focal_length=self.args.focal_length,
                    highlight_joint_index=self.selected_joint_index,
                    show_mesh=show_mesh and display_mode == "mesh",
                )
                self.preview_rgb = rgb_image
                self.preview_photo = rgb_to_photoimage(rgb_image)
                self.preview_label.configure(image=self.preview_photo)

            joint_name = JOINT_NAMES[self.selected_joint_index]
            self.status_var.set(
                f"Device: {self.device} | Joint: {joint_name} | Backend: {backend} | Mode: {display_mode} | Render: {'full' if show_mesh else 'fast preview'}"
            )
        except Exception as exc:
            self.status_var.set(f"Render failed: {exc}")
            traceback.print_exc()


    def on_close(self) -> None:
        if self.udp_socket is not None:
            try:
                self.udp_socket.close()
            except Exception:
                pass
        self.root.destroy()


def main() -> None:
    args = parse_args()
    root = tk.Tk()

    try:
        SMPLPoseGUI(root, args)
    except Exception as exc:
        traceback.print_exc()
        messagebox.showerror("SMPL GUI startup failed", str(exc))
        root.destroy()
        raise SystemExit(1) from exc

    root.mainloop()


if __name__ == "__main__":
    main()
