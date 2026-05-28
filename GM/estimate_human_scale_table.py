#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import sys
from pathlib import Path
from typing import Dict, Mapping

import numpy as np

from smpl_udp_common import SMPL_INDEX, SMPL_JOINT_NAMES


SCRIPT_PATH = Path(__file__).resolve()
DATA_ROOT = SCRIPT_PATH.parent.parent
GMR_ROOT = DATA_ROOT / "GMR"
GMR_PARAMS_PATH = GMR_ROOT / "general_motion_retargeting" / "params.py"

if str(GMR_ROOT) not in sys.path:
    sys.path.insert(0, str(GMR_ROOT))


def load_module_from_path(module_name: str, module_path: Path):
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load module spec from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


GMR_PARAMS = load_module_from_path("gmr_params_scale_estimator", GMR_PARAMS_PATH)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Estimate GMR human_scale_table values from robot XML and optional saved SMPL joint positions.",
    )
    parser.add_argument("--robot", required=True, choices=sorted(GMR_PARAMS.IK_CONFIG_DICT["smplx"].keys()))
    parser.add_argument(
        "--smpl-npz",
        default=None,
        help="Optional .npz saved by udp_smpl_to_gmr.py --save-smpl-path. Without this, only robot distances are printed.",
    )
    parser.add_argument(
        "--smpl-csv",
        action="append",
        default=[],
        help="CSV containing HybrIK columns like hybrik_left_foot_x_m. Can be passed multiple times.",
    )
    parser.add_argument(
        "--smpl-distances-json",
        default=None,
        help="Cached SMPL pelvis-to-joint distances JSON generated from CSV/NPZ.",
    )
    parser.add_argument(
        "--save-smpl-distances-json",
        default=None,
        help="Save the loaded CSV/NPZ SMPL pelvis-to-joint distances to this JSON path.",
    )
    parser.add_argument(
        "--actual-human-height",
        type=float,
        default=None,
        help="Human height used by GMR. If omitted, uses the config human_height_assumption.",
    )
    parser.add_argument("--table", choices=["ik_match_table1", "ik_match_table2"], default="ik_match_table2")
    parser.add_argument(
        "--symmetrize",
        action="store_true",
        help="Average left/right scale pairs to remove measurement noise.",
    )
    parser.add_argument(
        "--round-digits",
        type=int,
        default=None,
        help="Round estimated scale values to this many decimal places.",
    )
    parser.add_argument(
        "--keep-existing",
        action="append",
        default=[],
        help="Keep this human_scale_table entry from the current config. Can be passed multiple times.",
    )
    parser.add_argument("--json", action="store_true", help="Print only the estimated JSON scale table")
    return parser.parse_args()


def load_ik_config(robot_name: str) -> dict:
    config_path = GMR_PARAMS.IK_CONFIG_DICT["smplx"][robot_name]
    with open(config_path, "r", encoding="utf-8") as file_obj:
        return json.load(file_obj)


def robot_body_by_human(ik_config: Mapping[str, object], table_name: str) -> Dict[str, str]:
    table = ik_config[table_name]
    return {entry[0]: robot_body for robot_body, entry in table.items()}


def robot_root_distances(robot_name: str, ik_config: Mapping[str, object], table_name: str) -> Dict[str, float]:
    import mujoco as mj

    model = mj.MjModel.from_xml_path(str(GMR_PARAMS.ROBOT_XML_DICT[robot_name]))
    data = mj.MjData(model)
    data.qpos[:] = model.qpos0
    mj.mj_forward(model, data)

    root_name = str(ik_config["robot_root_name"])
    root_id = model.body(root_name).id
    root_pos = data.xpos[root_id].copy()

    distances: Dict[str, float] = {}
    for human_name, robot_body in robot_body_by_human(ik_config, table_name).items():
        body_id = model.body(robot_body).id
        distances[human_name] = float(np.linalg.norm(data.xpos[body_id] - root_pos))
    return distances


def load_smpl_distances(smpl_npz_path: str) -> Dict[str, float]:
    payload = np.load(smpl_npz_path, allow_pickle=True)
    joint_xyz = np.asarray(payload["joint_xyz"], dtype=np.float64)
    if joint_xyz.ndim == 2:
        joint_xyz = joint_xyz[None]

    joint_names = payload.get("joint_names", np.asarray(SMPL_JOINT_NAMES))
    joint_names = [str(name) for name in joint_names.tolist()]
    source_index = {name: index for index, name in enumerate(joint_names)}

    pelvis = joint_xyz[:, source_index["pelvis"]]
    distances: Dict[str, float] = {}
    for joint_name in SMPL_JOINT_NAMES:
        if joint_name not in source_index:
            continue
        diff = joint_xyz[:, source_index[joint_name]] - pelvis
        distances[joint_name] = float(np.mean(np.linalg.norm(diff, axis=-1)))
    return distances


def load_smpl_distances_from_csv(csv_paths: list[str]) -> Dict[str, float]:
    sums = {joint_name: 0.0 for joint_name in SMPL_JOINT_NAMES}
    counts = {joint_name: 0 for joint_name in SMPL_JOINT_NAMES}

    for csv_path_str in csv_paths:
        csv_path = Path(csv_path_str).expanduser().resolve()
        with csv_path.open("r", newline="", encoding="utf-8") as csv_file:
            reader = csv.DictReader(csv_file)
            required = [
                f"hybrik_{joint_name}_{axis}_m"
                for joint_name in SMPL_JOINT_NAMES
                for axis in ("x", "y", "z")
            ]
            missing = [column for column in required if column not in (reader.fieldnames or [])]
            if missing:
                raise ValueError(f"{csv_path} is missing HybrIK joint columns, first missing: {missing[0]}")

            for row in reader:
                try:
                    pelvis = np.array(
                        [float(row[f"hybrik_pelvis_{axis}_m"]) for axis in ("x", "y", "z")],
                        dtype=np.float64,
                    )
                except (TypeError, ValueError):
                    continue
                if not np.isfinite(pelvis).all():
                    continue

                for joint_name in SMPL_JOINT_NAMES:
                    try:
                        joint_pos = np.array(
                            [float(row[f"hybrik_{joint_name}_{axis}_m"]) for axis in ("x", "y", "z")],
                            dtype=np.float64,
                        )
                    except (TypeError, ValueError):
                        continue
                    if not np.isfinite(joint_pos).all():
                        continue
                    sums[joint_name] += float(np.linalg.norm(joint_pos - pelvis))
                    counts[joint_name] += 1

    return {
        joint_name: sums[joint_name] / counts[joint_name]
        for joint_name in SMPL_JOINT_NAMES
        if counts[joint_name] > 0
    }


def load_smpl_distances_from_json(json_path: str) -> Dict[str, float]:
    cache_path = Path(json_path).expanduser().resolve()
    with cache_path.open("r", encoding="utf-8") as json_file:
        payload = json.load(json_file)
    distances = payload.get("distances", payload)
    return {joint_name: float(distances[joint_name]) for joint_name in distances}


def save_smpl_distances_to_json(json_path: str, smpl_distances: Mapping[str, float], source_paths: list[str]) -> None:
    cache_path = Path(json_path).expanduser().resolve()
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "description": "Mean SMPL/HybrIK pelvis-to-joint distances in meters.",
        "source_paths": [str(Path(path).expanduser().resolve()) for path in source_paths],
        "distances": {joint_name: float(smpl_distances[joint_name]) for joint_name in sorted(smpl_distances)},
    }
    with cache_path.open("w", encoding="utf-8") as json_file:
        json.dump(payload, json_file, indent=4, ensure_ascii=False)
        json_file.write("\n")


def estimate_scale_table(
    robot_distances: Mapping[str, float],
    smpl_distances: Mapping[str, float],
    ik_config: Mapping[str, object],
    actual_human_height: float | None,
) -> Dict[str, float]:
    human_height_assumption = float(ik_config["human_height_assumption"])
    actual_height = human_height_assumption if actual_human_height is None else float(actual_human_height)
    runtime_ratio = actual_height / human_height_assumption

    estimates: Dict[str, float] = {}
    for human_name in ik_config["human_scale_table"].keys():
        robot_distance = robot_distances.get(human_name)
        human_distance = smpl_distances.get(human_name)
        if robot_distance is None or human_distance is None or human_distance < 1e-8:
            estimates[human_name] = float(ik_config["human_scale_table"][human_name])
            continue
        estimates[human_name] = float(robot_distance / human_distance / runtime_ratio)
    return estimates


def symmetrize_left_right(scale_table: Dict[str, float]) -> Dict[str, float]:
    estimates = dict(scale_table)
    for left_name, left_value in scale_table.items():
        if not left_name.startswith("left_"):
            continue
        right_name = "right_" + left_name[len("left_") :]
        if right_name not in scale_table:
            continue
        average = float((left_value + scale_table[right_name]) * 0.5)
        estimates[left_name] = average
        estimates[right_name] = average
    return estimates


def keep_existing_entries(scale_table: Dict[str, float], ik_config: Mapping[str, object], names: list[str]) -> Dict[str, float]:
    estimates = dict(scale_table)
    current_table = ik_config["human_scale_table"]
    for name in names:
        if name not in current_table:
            raise ValueError(f"{name!r} is not in the current human_scale_table")
        estimates[name] = float(current_table[name])
    return estimates


def round_scale_table(scale_table: Dict[str, float], digits: int | None) -> Dict[str, float]:
    if digits is None:
        return scale_table
    return {name: round(value, digits) for name, value in scale_table.items()}


def main() -> None:
    args = parse_args()
    ik_config = load_ik_config(args.robot)
    robot_distances = robot_root_distances(args.robot, ik_config, args.table)

    if args.smpl_npz is None and not args.smpl_csv and args.smpl_distances_json is None:
        if not args.json:
            print("Robot root-to-target distances from XML/MuJoCo zero pose:")
        print(json.dumps(robot_distances, indent=4, ensure_ascii=False))
        if not args.json:
            print(
                "\nProvide --smpl-npz, --smpl-csv, or --smpl-distances-json to convert these distances "
                "into recommended human_scale_table values."
            )
        return

    if args.smpl_distances_json is not None:
        smpl_distances = load_smpl_distances_from_json(args.smpl_distances_json)
    elif args.smpl_npz is not None:
        smpl_distances = load_smpl_distances(args.smpl_npz)
    else:
        smpl_distances = load_smpl_distances_from_csv(args.smpl_csv)
    if args.save_smpl_distances_json is not None:
        source_paths = args.smpl_csv or ([args.smpl_npz] if args.smpl_npz is not None else [args.smpl_distances_json])
        save_smpl_distances_to_json(args.save_smpl_distances_json, smpl_distances, source_paths)
    scale_table = estimate_scale_table(robot_distances, smpl_distances, ik_config, args.actual_human_height)
    if args.symmetrize:
        scale_table = symmetrize_left_right(scale_table)
    scale_table = keep_existing_entries(scale_table, ik_config, args.keep_existing)
    scale_table = round_scale_table(scale_table, args.round_digits)
    if not args.json:
        print("Estimated human_scale_table:")
    print(json.dumps(scale_table, indent=4, ensure_ascii=False))


if __name__ == "__main__":
    main()
