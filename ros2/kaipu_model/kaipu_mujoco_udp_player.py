#!/usr/bin/env python3
"""Pure Python MuJoCo UDP player for the Kaipu humanoid model."""

from __future__ import annotations

import argparse
import json
import socket
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import mujoco
import mujoco.viewer
import numpy as np

from kaipu_smpl_mapping import DEFAULT_XML_PATH, KAIPU_JOINT_ORDER


def clamp(x: float, lo: float, hi: float) -> float:
    return float(np.clip(x, lo, hi))


def _safe_name(model: mujoco.MjModel, obj: mujoco.mjtObj, idx: int, fallback: str) -> str:
    try:
        name = mujoco.mj_id2name(model, obj, idx)
        return name if name is not None else fallback
    except Exception:
        return fallback


def _quat_normalize_wxyz(q: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(q))
    if n < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
    return q / n


def set_gravity(model: mujoco.MjModel, gravity: float = -9.81) -> None:
    old_gravity = model.opt.gravity[2]
    model.opt.gravity[2] = gravity
    print(f"[MAIN] gravity: {old_gravity:.2f} -> {gravity:.2f} m/s^2", flush=True)


def add_joint_damping(model: mujoco.MjModel, damping: float = 1.0) -> int:
    count = 0
    for jid in range(model.njnt):
        jtype = int(model.jnt_type[jid])
        if jtype != int(mujoco.mjtJoint.mjJNT_FREE):
            model.dof_damping[model.jnt_dofadr[jid]] = damping
            count += 1
    return count


def parse_lock_config(lock_str: str) -> dict:
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
    if lock_str in ("", "none"):
        return config

    if "xyz" in lock_str:
        lock_str = lock_str.replace("xyz", "x,y,z")
    if "rpy" in lock_str:
        lock_str = lock_str.replace("rpy", "roll,pitch,yaw")

    for part in [p.strip() for p in lock_str.split(",") if p.strip()]:
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Receive UDP joint targets and drive Kaipu MuJoCo.")
    parser.add_argument("--xml-path", default=str(DEFAULT_XML_PATH), help="Path to Kaipu MuJoCo XML")
    parser.add_argument("--udp-host", default="127.0.0.1", help="UDP listen host")
    parser.add_argument("--udp-port", type=int, default=5007, help="UDP listen port")
    parser.add_argument("--sim-hz", type=float, default=500.0, help="Simulation frequency")
    parser.add_argument("--max-joint-speed", type=float, default=2.0, help="Maximum joint speed in rad/s")
    parser.add_argument("--alpha", type=float, default=0.35, help="Command smoothing coefficient")
    parser.add_argument("--ground-friction", type=float, default=3.0, help="Ground friction multiplier")
    parser.add_argument("--joint-damping", type=float, default=1.0, help="Joint damping")
    parser.add_argument("--gravity", type=float, default=-9.81, help="Gravity acceleration")
    parser.add_argument("--lock-freejoint", default="all", help="Lock freejoint DOFs: none/all/x,y,z,roll,pitch,yaw")
    parser.add_argument("--debug-joint", default="left_knee_pitch", help="Joint name to print periodically")
    return parser.parse_args()


class KaipuMujocoUdpPlayer:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.xml_path = Path(args.xml_path).expanduser().resolve()
        if not self.xml_path.exists():
            raise FileNotFoundError(f"MuJoCo XML not found: {self.xml_path}")

        print(f"[MAIN] Loading MuJoCo XML: {self.xml_path}", flush=True)
        self.m = mujoco.MjModel.from_xml_path(str(self.xml_path))
        self.d = mujoco.MjData(self.m)
        self._mj_lock = threading.Lock()

        set_gravity(self.m, args.gravity)
        try:
            self.m.opt.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
        except Exception:
            pass

        self._increase_ground_friction(args.ground_friction)
        damping_count = add_joint_damping(self.m, args.joint_damping)
        print(f"[MAIN] added damping to {damping_count} joints", flush=True)

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
        self.base_qpos0: Optional[np.ndarray] = None
        self.base_quat0: Optional[np.ndarray] = None
        for jid in range(self.m.njnt):
            if int(self.m.jnt_type[jid]) == int(mujoco.mjtJoint.mjJNT_FREE):
                self.free_qadr = int(self.m.jnt_qposadr[jid])
                self.free_dofadr = int(self.m.jnt_dofadr[jid])
                self.base_qpos0 = np.array(self.d.qpos[self.free_qadr : self.free_qadr + 7], dtype=float)
                self.base_quat0 = _quat_normalize_wxyz(self.base_qpos0[3:7].copy())
                break

        self.lock_config = parse_lock_config(args.lock_freejoint)
        self.cmd_lock = threading.Lock()
        self.cmd_targets: Dict[str, float] = {name: 0.0 for name in self.j_qadr.keys()}
        self.prev_targets: Dict[str, float] = {name: 0.0 for name in self.j_qadr.keys()}
        self.last_packet_time = 0.0
        self.packet_count = 0
        self._stop = threading.Event()

        self.udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.udp_socket.bind((args.udp_host, args.udp_port))
        self.udp_socket.settimeout(0.2)
        print(f"[UDP] listening on {args.udp_host}:{args.udp_port}", flush=True)

        self.receiver_thread = threading.Thread(target=self.recv_loop, daemon=True)
        self.receiver_thread.start()

    def _increase_ground_friction(self, friction: float) -> None:
        slide = float(friction)
        torsion = float(max(0.01, friction * 0.5))
        roll = float(max(0.01, friction * 0.05))
        for gid in range(self.m.ngeom):
            if int(self.m.geom_type[gid]) == int(mujoco.mjtGeom.mjGEOM_PLANE):
                self.m.geom_friction[gid, 0] = slide
                self.m.geom_friction[gid, 1] = torsion
                self.m.geom_friction[gid, 2] = roll

    def clamp_ctrl(self, aid: int, value: float) -> float:
        if self.m.actuator_ctrllimited[aid]:
            lo, hi = self.m.actuator_ctrlrange[aid]
            return clamp(value, float(lo), float(hi))
        return float(value)

    def apply_half_fixed_base(self) -> None:
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

        if self.base_quat0 is not None and (
            self.lock_config["lock_roll"] or self.lock_config["lock_pitch"] or self.lock_config["lock_yaw"]
        ):
            self.d.qpos[self.free_qadr + 3 : self.free_qadr + 7] = self.base_quat0
            if self.lock_config["lock_roll"]:
                self.d.qvel[self.free_dofadr + 3] = 0.0
            if self.lock_config["lock_pitch"]:
                self.d.qvel[self.free_dofadr + 4] = 0.0
            if self.lock_config["lock_yaw"]:
                self.d.qvel[self.free_dofadr + 5] = 0.0

    def recv_loop(self) -> None:
        while not self._stop.is_set():
            try:
                packet, _addr = self.udp_socket.recvfrom(65535)
            except socket.timeout:
                continue
            except OSError:
                break

            try:
                payload = json.loads(packet.decode("utf-8"))
                if isinstance(payload, dict) and "joint_names" in payload and "joint_positions" in payload:
                    joint_names = payload["joint_names"]
                    joint_positions = payload["joint_positions"]
                    count = min(len(joint_names), len(joint_positions))
                    with self.cmd_lock:
                        for index in range(count):
                            name = str(joint_names[index])
                            if name in self.cmd_targets:
                                self.cmd_targets[name] = float(joint_positions[index])
                        self.last_packet_time = time.time()
                        self.packet_count += 1
                elif isinstance(payload, dict) and "joint_targets" in payload and isinstance(payload["joint_targets"], dict):
                    with self.cmd_lock:
                        for name, value in payload["joint_targets"].items():
                            if name in self.cmd_targets:
                                self.cmd_targets[name] = float(value)
                        self.last_packet_time = time.time()
                        self.packet_count += 1
            except Exception as exc:
                print(f"[UDP] invalid packet: {exc}", flush=True)

    def run(self) -> None:
        dt_target = 1.0 / self.args.sim_hz
        last = time.time()
        last_print = 0.0
        internal_target: Dict[str, float] = {name: 0.0 for name in self.j_qadr.keys()}
        debug_joint = self.args.debug_joint if self.args.debug_joint in self.j_qadr else (KAIPU_JOINT_ORDER[0] if KAIPU_JOINT_ORDER else None)

        with self._mj_lock:
            mujoco.mj_forward(self.m, self.d)

        with mujoco.viewer.launch_passive(self.m, self.d) as viewer:
            viewer.sync()
            while viewer.is_running() and not self._stop.is_set():
                now = time.time()
                elapsed = now - last
                if elapsed < dt_target:
                    time.sleep(dt_target - elapsed)
                    continue
                dt = elapsed
                last = now

                with self.cmd_lock:
                    cmd = dict(self.cmd_targets)
                    packet_age = now - self.last_packet_time if self.last_packet_time > 0.0 else -1.0
                    packet_count = self.packet_count

                max_step = self.args.max_joint_speed * dt
                alpha = self.args.alpha
                inverse_alpha = 1.0 - alpha

                for joint_name, target in cmd.items():
                    smooth = alpha * target + inverse_alpha * self.prev_targets[joint_name]
                    self.prev_targets[joint_name] = smooth
                    current = internal_target[joint_name]
                    delta = smooth - current
                    if delta > max_step:
                        current += max_step
                    elif delta < -max_step:
                        current -= max_step
                    else:
                        current = smooth
                    internal_target[joint_name] = current

                with viewer.lock():
                    with self._mj_lock:
                        for actuator_name in self.actuator_names:
                            aid = self.act_id[actuator_name]
                            joint_name = self.act_joint[actuator_name]
                            self.d.ctrl[aid] = self.clamp_ctrl(aid, internal_target.get(joint_name, 0.0))

                        self.apply_half_fixed_base()
                        mujoco.mj_step(self.m, self.d)
                        self.apply_half_fixed_base()
                        mujoco.mj_forward(self.m, self.d)

                viewer.sync()

                if debug_joint is not None and (now - last_print) > 1.0:
                    last_print = now
                    with self._mj_lock:
                        qpos = float(self.d.qpos[self.j_qadr[debug_joint]])
                    print(
                        f"[SIM] packets={packet_count} age={packet_age:.2f}s {debug_joint}={qpos:+.3f}",
                        flush=True,
                    )

    def stop(self) -> None:
        self._stop.set()
        try:
            self.udp_socket.close()
        except Exception:
            pass
        try:
            self.receiver_thread.join(timeout=1.0)
        except Exception:
            pass


def main() -> None:
    args = parse_args()
    player = KaipuMujocoUdpPlayer(args)
    try:
        player.run()
    except KeyboardInterrupt:
        pass
    finally:
        player.stop()


if __name__ == "__main__":
    main()
