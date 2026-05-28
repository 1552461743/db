#!/usr/bin/env python3
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import mujoco
import mujoco.viewer

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState

from geometry_msgs.msg import TransformStamped
from tf2_ros import TransformBroadcaster

try:
    from ament_index_python.packages import get_package_share_directory
except Exception:
    get_package_share_directory = None


def clamp(x: float, lo: float, hi: float) -> float:
    return float(np.clip(x, lo, hi))


def _safe_name(model: mujoco.MjModel, obj: mujoco.mjtObj, idx: int, fallback: str) -> str:
    try:
        name = mujoco.mj_id2name(model, obj, idx)
        return name if name is not None else fallback
    except Exception:
        return fallback


def _quat_wxyz_to_xyzw(q_wxyz: np.ndarray) -> Tuple[float, float, float, float]:
    # MuJoCo: [w,x,y,z]  -> ROS: [x,y,z,w]
    w, x, y, z = float(q_wxyz[0]), float(q_wxyz[1]), float(q_wxyz[2]), float(q_wxyz[3])
    return x, y, z, w


def _quat_conj_wxyz(q: np.ndarray) -> np.ndarray:
    # q = [w,x,y,z]
    return np.array([q[0], -q[1], -q[2], -q[3]], dtype=float)


def _quat_mul_wxyz(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    # Hamilton product, both [w,x,y,z]
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ], dtype=float)


def _quat_normalize_wxyz(q: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(q))
    if n < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
    return q / n


def set_gravity(model: mujoco.MjModel, gravity: float = -9.81):
    """
    设置重力加速度
    - MuJoCo 默认: [0, 0, -9.81] m/s²
    - 地球标准重力: -9.81 m/s²
    - gravity 参数是 z 方向的值（负数表示向下）
    """
    old_gravity = model.opt.gravity[2]
    model.opt.gravity[2] = gravity
    print(f"[MAIN] 重力加速度: {old_gravity:.2f} -> {gravity:.2f} m/s² (z方向)", flush=True)
    return old_gravity


def add_joint_damping(model: mujoco.MjModel, damping: float = 1.0) -> int:
    """
    为所有关节添加阻尼，防止震荡
    - 阻尼会消耗关节速度的能量
    - 让关节更快稳定下来
    """
    count = 0
    for jid in range(model.njnt):
        jtype = int(model.jnt_type[jid])
        # 只对旋转关节和滑动关节添加阻尼，跳过 freejoint
        if jtype != int(mujoco.mjtJoint.mjJNT_FREE):
            model.dof_damping[model.jnt_dofadr[jid]] = damping
            count += 1

    if count > 0:
        print(f"[MAIN] 为 {count} 个关节添加阻尼: {damping}", flush=True)
    return count


def parse_lock_config(lock_str: str) -> dict:
    """
    解析锁定配置字符串
    格式: "x,y,z,roll,pitch,yaw" 或 "xyz,rpy" 或 "all" 或 "none"

    返回: {
        'lock_x': bool, 'lock_y': bool, 'lock_z': bool,
        'lock_roll': bool, 'lock_pitch': bool, 'lock_yaw': bool
    }

    示例:
        "all" -> 锁定所有 6 个自由度
        "none" -> 不锁定任何自由度
        "xyz" -> 锁定 x,y,z 平移
        "rpy" -> 锁定 roll,pitch,yaw 旋转
        "x,y,roll,pitch,yaw" -> 锁定 x,y 平移和所有旋转
        "z,yaw" -> 只锁定 z 和 yaw
    """
    config = {
        "lock_x": False,
        "lock_y": False,
        "lock_z": False,
        "lock_roll": False,
        "lock_pitch": False,
        "lock_yaw": False,
    }

    lock_str = lock_str.lower().strip()

    if lock_str == "all":
        for key in config:
            config[key] = True
        return config

    if lock_str == "none" or lock_str == "":
        return config

    if "xyz" in lock_str:
        lock_str = lock_str.replace("xyz", "x,y,z")
    elif "xy" in lock_str:
        lock_str = lock_str.replace("xy", "x,y")
    elif "xz" in lock_str:
        lock_str = lock_str.replace("xz", "x,z")
    elif "yz" in lock_str:
        lock_str = lock_str.replace("yz", "y,z")

    if "rpy" in lock_str:
        lock_str = lock_str.replace("rpy", "roll,pitch,yaw")
    elif "rp" in lock_str:
        lock_str = lock_str.replace("rp", "roll,pitch")
    elif "ry" in lock_str:
        lock_str = lock_str.replace("ry", "roll,yaw")
    elif "py" in lock_str:
        lock_str = lock_str.replace("py", "pitch,yaw")

    parts = [p.strip() for p in lock_str.split(",")]

    for part in parts:
        if part == "x":
            config["lock_x"] = True
        elif part == "y":
            config["lock_y"] = True
        elif part == "z":
            config["lock_z"] = True
        elif part == "roll":
            config["lock_roll"] = True
        elif part == "pitch":
            config["lock_pitch"] = True
        elif part == "yaw":
            config["lock_yaw"] = True

    return config


def resolve_default_xml_path() -> str:
    xml_name = "1230-URDF-version-fixed_mode_pos.xml"
    this_file = Path(__file__).resolve()

    candidate_model_dirs = [
        this_file.parent / "kaipu_model",
        this_file.parent.parent / "kaipu_model",
    ]

    if get_package_share_directory is not None:
        try:
            package_share = Path(get_package_share_directory("hybrik_nodes"))
            candidate_model_dirs.append(package_share / "kaipu_model")
        except Exception:
            pass

    seen = set()
    for model_dir in candidate_model_dirs:
        model_dir = model_dir.resolve()
        if str(model_dir) in seen:
            continue
        seen.add(str(model_dir))

        xml_path = model_dir / xml_name
        if xml_path.exists():
            return str(xml_path)

    searched = "\n".join(f"  - {path}" for path in candidate_model_dirs)
    raise FileNotFoundError(
        "Could not locate the default MuJoCo XML. Searched model directories:\n"
        f"{searched}\n"
        "Pass xml_path explicitly or ensure kaipu_model is installed next to the script/package."
    )


class MujocoMocapNode(Node):
    """
    Goals (unchanged):
      - Do NOT modify XML
      - "Half-fixed" base: allow translation (x,y,z), lock rotation (roll/pitch/yaw)
      - Increase ground friction in code
      - Accept ROS2 topic joint targets and drive robot to match (best-effort imitation)
      - Publish current joint states
      - Viewer must visually update (v.sync + v.lock)

    New:
      - Publish MuJoCo body tree as ROS2 TF (/tf)
    """

    def __init__(
        self,
        xml_path: str,
        publish_hz: float = 60.0,
        sim_hz: float = 500.0,
        lock_base_rotation: bool = True,
        max_joint_speed: float = 2.0,
        alpha: float = 0.35,
        ground_friction: float = 3.0,
        publish_tf: bool = True,
        tf_hz: float = 60.0,
        tf_world_frame: str = "world",
        tf_prefix: str = "",
        lock_config: Optional[dict] = None,
        joint_damping: float = 1.0,
        gravity: Optional[float] = None,
    ):
        super().__init__("mujoco_mocap_half_fixed_base")

        self.declare_parameter("xml_path", xml_path)
        self.declare_parameter("publish_hz", publish_hz)
        self.declare_parameter("sim_hz", sim_hz)
        self.declare_parameter("lock_base_rotation", lock_base_rotation)
        self.declare_parameter("max_joint_speed", max_joint_speed)
        self.declare_parameter("alpha", alpha)
        self.declare_parameter("ground_friction", ground_friction)
        self.declare_parameter("joint_damping", joint_damping)
        self.declare_parameter("gravity", gravity if gravity is not None else -9.81)
        self.declare_parameter("publish_tf", publish_tf)
        self.declare_parameter("tf_hz", tf_hz)
        self.declare_parameter("tf_world_frame", tf_world_frame)
        self.declare_parameter("tf_prefix", tf_prefix)

        if lock_config is not None:
            lock_parts = []
            if lock_config.get("lock_x"):
                lock_parts.append("x")
            if lock_config.get("lock_y"):
                lock_parts.append("y")
            if lock_config.get("lock_z"):
                lock_parts.append("z")
            if lock_config.get("lock_roll"):
                lock_parts.append("roll")
            if lock_config.get("lock_pitch"):
                lock_parts.append("pitch")
            if lock_config.get("lock_yaw"):
                lock_parts.append("yaw")
            lock_str = ",".join(lock_parts) if lock_parts else "none"
            self.declare_parameter("lock_freejoint", lock_str)
        else:
            self.declare_parameter("lock_freejoint", "rpy" if lock_base_rotation else "none")

        self.xml_path = xml_path
        self.publish_hz = float(publish_hz)
        self.sim_hz = float(sim_hz)
        self.lock_base_rotation = bool(lock_base_rotation)
        self.max_joint_speed = float(max_joint_speed)
        self.alpha = float(alpha)
        self.ground_friction = float(ground_friction)
        self.joint_damping = float(joint_damping)
        self.publish_tf = bool(publish_tf)
        self.tf_hz = float(tf_hz)
        self.tf_world_frame = str(tf_world_frame)
        self.tf_prefix = str(tf_prefix)

        if lock_config is not None:
            self.lock_config = lock_config
        elif lock_base_rotation:
            self.lock_config = {
                "lock_x": False,
                "lock_y": False,
                "lock_z": False,
                "lock_roll": True,
                "lock_pitch": True,
                "lock_yaw": True,
            }
        else:
            self.lock_config = {
                "lock_x": False,
                "lock_y": False,
                "lock_z": False,
                "lock_roll": False,
                "lock_pitch": False,
                "lock_yaw": False,
            }

        print(f"[MAIN] Loading MuJoCo XML: {xml_path}", flush=True)
        self.m = mujoco.MjModel.from_xml_path(self.xml_path)
        self.d = mujoco.MjData(self.m)

        current_gravity = self.m.opt.gravity[2]
        print(f"[MAIN] XML 中的重力加速度: {current_gravity:.2f} m/s² (z方向)", flush=True)

        if gravity is not None:
            set_gravity(self.m, gravity)

        self._mj_lock = threading.Lock()

        try:
            self.m.opt.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
            print("[MAIN] Using integrator: IMPLICITFAST", flush=True)
        except Exception:
            print("[MAIN] Could not set IMPLICITFAST integrator (ignored).", flush=True)

        self._increase_ground_friction(self.ground_friction)
        add_joint_damping(self.m, self.joint_damping)

        self.actuator_names: List[str] = []
        self.act_id: Dict[str, int] = {}
        self.act_joint: Dict[str, str] = {}

        for i in range(self.m.nu):
            aname = _safe_name(self.m, mujoco.mjtObj.mjOBJ_ACTUATOR, i, f"act_{i}")
            self.actuator_names.append(aname)
            self.act_id[aname] = i

            jid = int(self.m.actuator_trnid[i, 0])
            jname = _safe_name(self.m, mujoco.mjtObj.mjOBJ_JOINT, jid, f"joint_{jid}")
            self.act_joint[aname] = jname

        self.j_qadr: Dict[str, int] = {}
        self.j_dofadr: Dict[str, int] = {}
        for jname in set(self.act_joint.values()):
            jid = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_JOINT, jname)
            if jid < 0:
                continue
            self.j_qadr[jname] = int(self.m.jnt_qposadr[jid])
            self.j_dofadr[jname] = int(self.m.jnt_dofadr[jid])

        self.free_qadr: Optional[int] = None
        self.free_dofadr: Optional[int] = None
        self.free_joint_name: Optional[str] = None

        for jid in range(self.m.njnt):
            if int(self.m.jnt_type[jid]) == int(mujoco.mjtJoint.mjJNT_FREE):
                self.free_qadr = int(self.m.jnt_qposadr[jid])
                self.free_dofadr = int(self.m.jnt_dofadr[jid])
                self.free_joint_name = _safe_name(self.m, mujoco.mjtObj.mjOBJ_JOINT, jid, f"free_{jid}")
                break

        with self._mj_lock:
            mujoco.mj_forward(self.m, self.d)

        self.base_qpos0: Optional[np.ndarray] = None
        self._base_quat0: Optional[np.ndarray] = None
        if self.free_qadr is not None:
            with self._mj_lock:
                self.base_qpos0 = np.array(self.d.qpos[self.free_qadr:self.free_qadr + 7], dtype=float)
            q = self.base_qpos0[3:7].copy()
            q = _quat_normalize_wxyz(q)
            self._base_quat0 = q

        if self.lock_base_rotation:
            if self.free_qadr is None:
                print("[MAIN] lock_base_rotation=True but no FREE joint found.", flush=True)
            else:
                locked_dofs = [k.replace("lock_", "") for k, v in self.lock_config.items() if v]
                if locked_dofs:
                    print(
                        f"[MAIN] Flexible base lock enabled: FREE joint '{self.free_joint_name}' "
                        f"(qadr={self.free_qadr}, dofadr={self.free_dofadr})",
                        flush=True,
                    )
                    print(f"[MAIN] Locked DOFs: {', '.join(locked_dofs)}", flush=True)
                else:
                    print(f"[MAIN] FREE joint '{self.free_joint_name}' is completely free (no locks)", flush=True)

        self.pub_js = self.create_publisher(JointState, "/mujoco/joint_states", 10)
        self.sub_cmd = self.create_subscription(JointState, "/mujoco/joint_cmd", self.on_cmd, 10)
        self.pub_timer = self.create_timer(1.0 / self.publish_hz, self.publish_joint_states)

        self.tf_broadcaster: Optional[TransformBroadcaster] = None
        self.tf_timer = None
        self._tf_body_names: List[str] = []
        self._tf_parent_idx: np.ndarray = np.array([], dtype=int)

        if self.publish_tf:
            self.tf_broadcaster = TransformBroadcaster(self)
            self._tf_body_names = [self._frame_name(self._body_name(i)) for i in range(self.m.nbody)]
            self._tf_parent_idx = np.array(self.m.body_parentid, dtype=int)
            self.tf_timer = self.create_timer(1.0 / self.tf_hz, self.publish_tf_tree)
            self.get_logger().info(
                f"TF enabled: publishing MuJoCo body tree to /tf at {self.tf_hz} Hz, world_frame='{self.tf_world_frame}', prefix='{self.tf_prefix}'"
            )

        self.cmd_lock = threading.Lock()
        self.cmd_targets: Dict[str, float] = {j: 0.0 for j in self.j_qadr.keys()}
        self.prev_targets: Dict[str, float] = {j: 0.0 for j in self.j_qadr.keys()}
        self.last_cmd_time = 0.0
        self.cmd_applied = True

        self._stop = threading.Event()

        print("[MAIN] starting sim_thread ...", flush=True)
        self.sim_thread = threading.Thread(target=self.sim_loop, daemon=True)
        self.sim_thread.start()
        print("[MAIN] sim_thread started.", flush=True)

        self.get_logger().info(f"Loaded: {xml_path}")
        self.get_logger().info(f"nu (actuators) = {self.m.nu}")
        self.get_logger().info(
            f"half_fixed_base(lock rot)={self.lock_base_rotation}, sim_hz={self.sim_hz}, publish_hz={self.publish_hz}, "
            f"ground_friction={self.ground_friction}, joint_damping={self.joint_damping}"
        )
        self.get_logger().info("Subscribe: /mujoco/joint_cmd (JointState), Publish: /mujoco/joint_states (JointState)")

    def _body_name(self, bid: int) -> str:
        return _safe_name(self.m, mujoco.mjtObj.mjOBJ_BODY, bid, f"body_{bid}")

    def _frame_name(self, name: str) -> str:
        if not name:
            name = "unnamed"
        if self.tf_prefix:
            if self.tf_prefix.endswith("/"):
                return self.tf_prefix + name
            return self.tf_prefix + "/" + name
        return name

    def publish_tf_tree(self):
        if self.tf_broadcaster is None:
            return

        stamp = self.get_clock().now().to_msg()

        with self._mj_lock:
            xpos = np.array(self.d.xpos, dtype=float, copy=True)
            xquat = np.array(self.d.xquat, dtype=float, copy=True)

        for i in range(xquat.shape[0]):
            xquat[i] = _quat_normalize_wxyz(xquat[i])

        for bid in range(1, self.m.nbody):
            parent = int(self._tf_parent_idx[bid])

            child_frame = self._tf_body_names[bid]
            if parent <= 0:
                parent_frame = self.tf_world_frame
                p_pos = np.array([0.0, 0.0, 0.0], dtype=float)
                p_quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
            else:
                parent_frame = self._tf_body_names[parent]
                p_pos = xpos[parent]
                p_quat = xquat[parent]

            c_pos = xpos[bid]
            c_quat = xquat[bid]

            q_rel = _quat_mul_wxyz(_quat_conj_wxyz(p_quat), c_quat)
            q_rel = _quat_normalize_wxyz(q_rel)

            R_p = np.zeros(9, dtype=float)
            mujoco.mju_quat2Mat(R_p, p_quat)
            R_p = R_p.reshape(3, 3)
            t_world = (c_pos - p_pos).reshape(3, 1)
            t_rel = (R_p.T @ t_world).flatten()

            x, y, z, w = _quat_wxyz_to_xyzw(q_rel)

            ts = TransformStamped()
            ts.header.stamp = stamp
            ts.header.frame_id = parent_frame
            ts.child_frame_id = child_frame
            ts.transform.translation.x = float(t_rel[0])
            ts.transform.translation.y = float(t_rel[1])
            ts.transform.translation.z = float(t_rel[2])
            ts.transform.rotation.x = float(x)
            ts.transform.rotation.y = float(y)
            ts.transform.rotation.z = float(z)
            ts.transform.rotation.w = float(w)

            self.tf_broadcaster.sendTransform(ts)

    def _increase_ground_friction(self, fric: float):
        slide = float(fric)
        torsion = float(max(0.01, fric * 0.5))
        roll = float(max(0.01, fric * 0.05))

        updated = False

        try:
            gid = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_GEOM, "ground")
            if gid >= 0:
                self.m.geom_friction[gid, 0] = slide
                self.m.geom_friction[gid, 1] = torsion
                self.m.geom_friction[gid, 2] = roll
                updated = True
                print(f"[MAIN] Set friction for geom 'ground' to [{slide},{torsion},{roll}]", flush=True)
        except Exception:
            pass

        if not updated:
            for gid in range(self.m.ngeom):
                if int(self.m.geom_type[gid]) == int(mujoco.mjtGeom.mjGEOM_PLANE):
                    self.m.geom_friction[gid, 0] = slide
                    self.m.geom_friction[gid, 1] = torsion
                    self.m.geom_friction[gid, 2] = roll
                    updated = True
            if updated:
                print(f"[MAIN] Set friction for all PLANE geoms to [{slide},{torsion},{roll}]", flush=True)
            else:
                print("[MAIN] No ground/plane geoms found to update friction (ignored).", flush=True)

    def on_cmd(self, msg: JointState):
        if not msg.name or not msg.position:
            return
        n = min(len(msg.name), len(msg.position))
        now = time.time()
        with self.cmd_lock:
            for i in range(n):
                jname = msg.name[i]
                if jname in self.cmd_targets:
                    self.cmd_targets[jname] = float(msg.position[i])
            self.last_cmd_time = now
            self.cmd_applied = False

    def publish_joint_states(self):
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        names = sorted(self.j_qadr.keys())
        msg.name = names

        with self._mj_lock:
            msg.position = [float(self.d.qpos[self.j_qadr[j]]) for j in names]
            msg.velocity = [float(self.d.qvel[self.j_dofadr[j]]) for j in names]

        self.pub_js.publish(msg)

    def clamp_ctrl(self, aid: int, x: float) -> float:
        if self.m.actuator_ctrllimited[aid]:
            lo, hi = self.m.actuator_ctrlrange[aid]
            return clamp(x, float(lo), float(hi))
        return float(x)

    def apply_half_fixed_base(self):
        if self.free_qadr is None or self.free_dofadr is None or self.base_qpos0 is None:
            return

        if self.lock_config["lock_x"]:
            self.d.qpos[self.free_qadr + 0] = self.base_qpos0[0]
            self.d.qvel[self.free_dofadr + 0] = 0.0
        if self.lock_config["lock_y"]:
            self.d.qpos[self.free_qadr + 1] = self.base_qpos0[1]
            self.d.qvel[self.free_dofadr + 1] = 0.0
        if self.lock_config["lock_z"]:
            self.d.qpos[self.free_qadr + 2] = self.base_qpos0[2]
            self.d.qvel[self.free_dofadr + 2] = 0.0

        if self.lock_config["lock_roll"] or self.lock_config["lock_pitch"] or self.lock_config["lock_yaw"]:
            if self._base_quat0 is not None:
                self.d.qpos[self.free_qadr + 3:self.free_qadr + 7] = self._base_quat0

            if self.lock_config["lock_roll"]:
                self.d.qvel[self.free_dofadr + 3] = 0.0
            if self.lock_config["lock_pitch"]:
                self.d.qvel[self.free_dofadr + 4] = 0.0
            if self.lock_config["lock_yaw"]:
                self.d.qvel[self.free_dofadr + 5] = 0.0

    def sim_loop(self):
        print("[SIM] entered sim_loop()", flush=True)
        with self._mj_lock:
            mujoco.mj_forward(self.m, self.d)

        dt_target = 1.0 / self.sim_hz
        last = time.time()
        last_print = 0.0
        step_count = 0

        internal_target: Dict[str, float] = {j: 0.0 for j in self.j_qadr.keys()}
        debug_joint = "left_knee_pitch" if "left_knee_pitch" in self.j_qadr else (next(iter(self.j_qadr.keys())) if self.j_qadr else None)

        with mujoco.viewer.launch_passive(self.m, self.d) as v:
            print("[SIM] viewer launched", flush=True)
            v.sync()

            while v.is_running() and not self._stop.is_set():
                now = time.time()
                elapsed = now - last
                if elapsed < dt_target:
                    time.sleep(dt_target - elapsed)
                    continue
                dt = elapsed
                last = now

                with self.cmd_lock:
                    has_new_cmd = not self.cmd_applied
                    if has_new_cmd:
                        cmd = dict(self.cmd_targets)
                        self.cmd_applied = True
                    last_cmd_t = self.last_cmd_time

                if has_new_cmd:
                    max_step = self.max_joint_speed * dt
                    a = self.alpha
                    ia = 1.0 - a

                    for j, target in cmd.items():
                        smooth = a * target + ia * self.prev_targets[j]
                        self.prev_targets[j] = smooth

                        cur = internal_target[j]
                        delta = smooth - cur
                        if delta > max_step:
                            cur += max_step
                        elif delta < -max_step:
                            cur -= max_step
                        else:
                            cur = smooth
                        internal_target[j] = cur

                    with v.lock():
                        with self._mj_lock:
                            for aname in self.actuator_names:
                                aid = self.act_id[aname]
                                jname = self.act_joint[aname]
                                self.d.ctrl[aid] = self.clamp_ctrl(aid, internal_target.get(jname, 0.0))

                with v.lock():
                    with self._mj_lock:
                        self.apply_half_fixed_base()
                        mujoco.mj_step(self.m, self.d)
                        self.apply_half_fixed_base()
                        mujoco.mj_forward(self.m, self.d)

                step_count += 1
                if step_count % 5 == 0:
                    v.sync()

                if debug_joint is not None and (now - last_print) > 1.0:
                    last_print = now
                    with self._mj_lock:
                        q = float(self.d.qpos[self.j_qadr[debug_joint]])
                    print(
                        f"[SIM] {debug_joint} qpos={q:.3f} (half_fixed_base={self.lock_base_rotation}, cmd_age={now - last_cmd_t:.2f}s)",
                        flush=True,
                    )

            print("[SIM] viewer closed, sim_loop exit", flush=True)

    def stop(self):
        self._stop.set()
        try:
            self.sim_thread.join(timeout=2.0)
        except Exception:
            pass


def main():
    DEFAULT_XML = resolve_default_xml_path()

    rclpy.init()
    temp_node = rclpy.create_node("temp_param_reader")

    temp_node.declare_parameter("xml_path", DEFAULT_XML)
    temp_node.declare_parameter("publish_hz", 60.0)
    temp_node.declare_parameter("sim_hz", 500.0)
    temp_node.declare_parameter("unlock_rotation", False)
    temp_node.declare_parameter("max_joint_speed", 2.0)
    temp_node.declare_parameter("alpha", 0.35)
    temp_node.declare_parameter("ground_friction", 3.0)
    temp_node.declare_parameter("joint_damping", 1.0)
    temp_node.declare_parameter("gravity", -9.81)
    temp_node.declare_parameter("publish_tf", False)
    temp_node.declare_parameter("tf_hz", 60.0)
    temp_node.declare_parameter("tf_world", "world")
    temp_node.declare_parameter("tf_prefix", "mujoco")
    temp_node.declare_parameter("lock_freejoint", "rpy")

    xml_path = temp_node.get_parameter("xml_path").value
    publish_hz = temp_node.get_parameter("publish_hz").value
    sim_hz = temp_node.get_parameter("sim_hz").value
    unlock_rotation = temp_node.get_parameter("unlock_rotation").value
    max_joint_speed = temp_node.get_parameter("max_joint_speed").value
    alpha = temp_node.get_parameter("alpha").value
    ground_friction = temp_node.get_parameter("ground_friction").value
    joint_damping = temp_node.get_parameter("joint_damping").value
    gravity = temp_node.get_parameter("gravity").value
    publish_tf = temp_node.get_parameter("publish_tf").value
    tf_hz = temp_node.get_parameter("tf_hz").value
    tf_world = temp_node.get_parameter("tf_world").value
    tf_prefix = temp_node.get_parameter("tf_prefix").value
    lock_freejoint_str = temp_node.get_parameter("lock_freejoint").value

    temp_node.destroy_node()

    print(f"[MAIN] Using XML: {xml_path}", flush=True)

    lock_config = None
    if lock_freejoint_str and lock_freejoint_str != "":
        lock_config = parse_lock_config(lock_freejoint_str)
        print(f"[MAIN] Using custom lock config: {lock_freejoint_str}", flush=True)

    node = MujocoMocapNode(
        xml_path=xml_path,
        publish_hz=publish_hz,
        sim_hz=sim_hz,
        lock_base_rotation=(not unlock_rotation),
        max_joint_speed=max_joint_speed,
        alpha=alpha,
        ground_friction=ground_friction,
        joint_damping=joint_damping,
        gravity=gravity,
        publish_tf=publish_tf,
        tf_hz=tf_hz,
        tf_world_frame=tf_world,
        tf_prefix=tf_prefix,
        lock_config=lock_config,
    )

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.stop()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
