#!/usr/bin/env python3
"""Append HybrIK SMPL joint outputs to a synchronized CSV dataset.

Expected workflow:
1. Use bag_to_csv.py to create a CSV that contains an image path column.
2. Run this script on that CSV.
3. The script loads each image, runs person detection + HybrIK inference, and
   appends SMPL joint rotations and joint positions to a new CSV.
4. Frames whose image is missing/unreadable or whose person cannot be inferred
   are dropped from the output CSV.
5. Every N successful frames, a rendered HybrIK-on-image visualization is saved
   under the output CSV directory for quick inspection.

The appended fields include:
- HybrIK status and selected detection bbox
- Root translation in camera coordinates
- Camera root position
- 24 SMPL joint relative positions in meters
- 24 SMPL joint rotation matrices (3x3)
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
import os
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
from easydict import EasyDict as edict
from torchvision import transforms as T
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from tqdm import tqdm

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

from hybrik.models import builder
from hybrik.models.layers.smpl.SMPL import SMPL_layer
from hybrik.utils.config import update_config
from hybrik.utils.presets.simple_transform_3d_smpl_cam import SimpleTransform3DSMPLCam
from hybrik.utils.render_pytorch3d import render_mesh
from hybrik.utils.vis import get_one_box


DET_TRANSFORM = T.Compose([T.ToTensor()])
SMPL_24_NAMES = SMPL_layer.JOINT_NAMES[:24]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Read a CSV with image paths, run HybrIK on each image, and append SMPL outputs."
    )
    parser.add_argument("csv_path", help="Input CSV path")
    parser.add_argument(
        "--output-csv",
        default=None,
        help="Output CSV path. Default: <input_stem>_with_hybrik.csv",
    )
    parser.add_argument(
        "--image-column",
        default="image_path",
        help="CSV column containing the image path",
    )
    parser.add_argument(
        "--cfg",
        default=str(REPO_ROOT / "configs/256x192_adam_lr1e-3-hrw48_cam_2x_w_pw3d_3dhp.yaml"),
        help="HybrIK config file path",
    )
    parser.add_argument(
        "--ckpt",
        default=str(REPO_ROOT / "pretrained_models/hybrik_hrnet.pth"),
        help="HybrIK checkpoint path",
    )
    parser.add_argument(
        "--gpu",
        type=int,
        default=0,
        help="GPU id to use when CUDA is available",
    )
    parser.add_argument(
        "--det-threshold",
        type=float,
        default=0.9,
        help="Initial detector score threshold for selecting the person bbox",
    )
    parser.add_argument(
        "--save-vis-every",
        type=int,
        default=100,
        help="Save one rendered visualization every N successful frames. Set <=0 to disable.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Number of frames to run through detector/HybrIK per GPU batch. Increase to improve GPU utilization.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=2,
        help="Thread count for image loading when batch-size > 1.",
    )
    parser.add_argument(
        "--vis-dir-name",
        default="hybrik_vis_samples",
        help="Visualization folder name under the output CSV directory",
    )
    return parser.parse_args()


def recover_theta_mats(theta_tensor: torch.Tensor) -> np.ndarray:
    flat = theta_tensor.detach().cpu().numpy().reshape(-1)
    if flat.size % 9 != 0:
        raise ValueError(f"pred_theta_mats size {flat.size} cannot be reshaped to 3x3 matrices")
    return flat.reshape(-1, 3, 3)


def build_detector_model():
    try:
        from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights

        return fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT)
    except Exception:
        return fasterrcnn_resnet50_fpn(pretrained=True)


def extract_model_state_dict(checkpoint: object) -> Dict[str, torch.Tensor]:
    if not isinstance(checkpoint, dict):
        return checkpoint

    if "model" in checkpoint and isinstance(checkpoint["model"], dict):
        state_dict = checkpoint["model"]
    elif "state_dict" in checkpoint and isinstance(checkpoint["state_dict"], dict):
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    cleaned_state_dict = {}
    for key, value in state_dict.items():
        new_key = key[7:] if key.startswith("module.") else key
        cleaned_state_dict[new_key] = value
    return cleaned_state_dict


def build_output_fields() -> List[str]:
    fields = [
        "hybrik_status",
        "hybrik_bbox_x1",
        "hybrik_bbox_y1",
        "hybrik_bbox_x2",
        "hybrik_bbox_y2",
        "hybrik_transl_x_m",
        "hybrik_transl_y_m",
        "hybrik_transl_z_m",
        "hybrik_cam_root_x_m",
        "hybrik_cam_root_y_m",
        "hybrik_cam_root_z_m",
    ]

    for joint_name in SMPL_24_NAMES:
        fields.extend(
            [
                f"hybrik_{joint_name}_x_m",
                f"hybrik_{joint_name}_y_m",
                f"hybrik_{joint_name}_z_m",
            ]
        )
        for row in range(3):
            for col in range(3):
                fields.append(f"hybrik_{joint_name}_rotmat_{row}{col}")

    return fields


def xyxy2xywh(bbox: List[float]) -> List[float]:
    x1, y1, x2, y2 = bbox
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    w = x2 - x1
    h = y2 - y1
    return [cx, cy, w, h]


def resolve_image_path(csv_path: Path, image_value: str) -> Path:
    image_path = Path(image_value).expanduser()
    if not image_path.is_absolute():
        image_path = (csv_path.parent / image_path).resolve()
    return image_path


class HybrIKInferencer:
    def __init__(self, cfg_path: Path, ckpt_path: Path, gpu: int, det_threshold: float) -> None:
        os.chdir(REPO_ROOT)

        self.device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")
        self.det_threshold = det_threshold

        cfg = update_config(str(cfg_path))
        bbox_3d_shape = getattr(cfg.MODEL, "BBOX_3D_SHAPE", (2000, 2000, 2000))
        bbox_3d_shape = [item * 1e-3 for item in bbox_3d_shape]
        self.depth_factor = bbox_3d_shape[2]

        dummy_set = edict(
            {
                "joint_pairs_17": None,
                "joint_pairs_24": None,
                "joint_pairs_29": None,
                "bbox_3d_shape": bbox_3d_shape,
            }
        )
        self.transformation = SimpleTransform3DSMPLCam(
            dummy_set,
            scale_factor=cfg.DATASET.SCALE_FACTOR,
            color_factor=cfg.DATASET.COLOR_FACTOR,
            occlusion=cfg.DATASET.OCCLUSION,
            input_size=cfg.MODEL.IMAGE_SIZE,
            output_size=cfg.MODEL.HEATMAP_SIZE,
            depth_dim=cfg.MODEL.EXTRA.DEPTH_DIM,
            bbox_3d_shape=bbox_3d_shape,
            rot=cfg.DATASET.ROT_FACTOR,
            sigma=cfg.MODEL.EXTRA.SIGMA,
            train=False,
            add_dpg=False,
            loss_type=cfg.LOSS["TYPE"],
        )

        self.det_model = build_detector_model()
        self.hybrik_model = builder.build_sppe(cfg.MODEL)

        save_dict = torch.load(str(ckpt_path), map_location="cpu")
        model_state_dict = extract_model_state_dict(save_dict)
        self.hybrik_model.load_state_dict(model_state_dict)

        self.det_model.to(self.device)
        self.hybrik_model.to(self.device)
        self.det_model.eval()
        self.hybrik_model.eval()
        self.smpl_faces = torch.from_numpy(self.hybrik_model.smpl.faces.astype(np.int32))

    def render_visualization_from_tensors(
        self,
        input_image: np.ndarray,
        bbox: List[float],
        vertices: torch.Tensor,
        transl: torch.Tensor,
    ) -> np.ndarray:
        focal = 1000.0
        bbox_xywh = xyxy2xywh(bbox)
        focal = focal / 256.0 * bbox_xywh[2]

        color_batch = render_mesh(
            vertices=vertices.detach().unsqueeze(0),
            faces=self.smpl_faces,
            translation=transl.detach().unsqueeze(0),
            focal_length=focal,
            height=input_image.shape[0],
            width=input_image.shape[1],
        )

        valid_mask_batch = color_batch[:, :, :, [-1]] > 0
        image_vis_batch = color_batch[:, :, :, :3] * valid_mask_batch
        image_vis_batch = (image_vis_batch * 255).detach().cpu().numpy()

        color = image_vis_batch[0]
        valid_mask = valid_mask_batch[0].detach().cpu().numpy()
        alpha = 0.9
        image_vis = alpha * color[:, :, :3] * valid_mask + (1.0 - alpha) * input_image * valid_mask + (
            1.0 - valid_mask
        ) * input_image

        image_vis = image_vis.astype(np.uint8)
        return cv2.cvtColor(image_vis, cv2.COLOR_RGB2BGR)

    def render_visualization(
        self,
        input_image: np.ndarray,
        bbox: List[float],
        pose_output: object,
    ) -> np.ndarray:
        return self.render_visualization_from_tensors(
            input_image,
            bbox,
            pose_output.pred_vertices.squeeze(0),
            pose_output.transl.squeeze(0),
        )

    def build_row_update(
        self,
        bbox: Sequence[float],
        transl_tensor: torch.Tensor,
        cam_root_tensor: torch.Tensor,
        theta_tensor: torch.Tensor,
        xyz_tensor: torch.Tensor,
    ) -> Dict[str, object]:
        transl = transl_tensor.detach().cpu().numpy()
        cam_root = cam_root_tensor.detach().cpu().numpy()
        theta_mats = recover_theta_mats(theta_tensor)
        joint_xyz_m = xyz_tensor.reshape(24, 3).detach().cpu().numpy() * self.depth_factor

        row_update: Dict[str, object] = {
            "hybrik_status": "ok",
            "hybrik_bbox_x1": f"{bbox[0]:.6f}",
            "hybrik_bbox_y1": f"{bbox[1]:.6f}",
            "hybrik_bbox_x2": f"{bbox[2]:.6f}",
            "hybrik_bbox_y2": f"{bbox[3]:.6f}",
            "hybrik_transl_x_m": f"{transl[0]:.9f}",
            "hybrik_transl_y_m": f"{transl[1]:.9f}",
            "hybrik_transl_z_m": f"{transl[2]:.9f}",
            "hybrik_cam_root_x_m": f"{cam_root[0]:.9f}",
            "hybrik_cam_root_y_m": f"{cam_root[1]:.9f}",
            "hybrik_cam_root_z_m": f"{cam_root[2]:.9f}",
        }

        for joint_index, joint_name in enumerate(SMPL_24_NAMES):
            xyz = joint_xyz_m[joint_index]
            row_update[f"hybrik_{joint_name}_x_m"] = f"{xyz[0]:.9f}"
            row_update[f"hybrik_{joint_name}_y_m"] = f"{xyz[1]:.9f}"
            row_update[f"hybrik_{joint_name}_z_m"] = f"{xyz[2]:.9f}"

            rot = theta_mats[joint_index]
            for row in range(3):
                for col in range(3):
                    row_update[f"hybrik_{joint_name}_rotmat_{row}{col}"] = f"{rot[row, col]:.9f}"

        return row_update

    @torch.no_grad()
    def infer_image(self, image_path: Path, render_visualization: bool = False) -> Tuple[Dict[str, object], Optional[np.ndarray]]:
        image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image_bgr is None:
            return {"hybrik_status": "image_read_failed"}, None

        input_image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        det_input = DET_TRANSFORM(input_image).to(self.device)
        det_output = self.det_model([det_input])[0]
        tight_bbox = get_one_box(det_output, thrd=self.det_threshold)
        if tight_bbox is None:
            return {"hybrik_status": "no_person_detected"}, None

        pose_input, bbox, img_center = self.transformation.test_transform(input_image, tight_bbox)
        pose_input = pose_input.to(self.device)[None, :, :, :]
        pose_output = self.hybrik_model(
            pose_input,
            flip_test=True,
            bboxes=torch.from_numpy(np.array(bbox)).to(self.device).unsqueeze(0).float(),
            img_center=torch.from_numpy(img_center).to(self.device).unsqueeze(0).float(),
        )

        row_update = self.build_row_update(
            bbox,
            pose_output.transl.squeeze(0),
            pose_output.cam_root.squeeze(0),
            pose_output.pred_theta_mats.squeeze(0),
            pose_output.pred_xyz_jts_24_struct.squeeze(0),
        )

        vis_image = None
        if render_visualization:
            try:
                vis_image = self.render_visualization(input_image, bbox, pose_output)
            except Exception:
                vis_image = None

        return row_update, vis_image

    def load_image(self, image_path: Path) -> Tuple[Path, Optional[np.ndarray], Optional[np.ndarray]]:
        image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image_bgr is None:
            return image_path, None, None
        input_image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        return image_path, image_bgr, input_image

    @torch.no_grad()
    def infer_batch(
        self,
        items: Sequence[Tuple[Path, bool]],
        num_workers: int = 2,
    ) -> List[Tuple[Dict[str, object], Optional[np.ndarray]]]:
        if len(items) == 1:
            return [self.infer_image(items[0][0], render_visualization=items[0][1])]

        results: List[Tuple[Dict[str, object], Optional[np.ndarray]]] = [
            ({"hybrik_status": "image_read_failed"}, None) for _ in items
        ]

        if num_workers > 1:
            with ThreadPoolExecutor(max_workers=num_workers) as pool:
                loaded_images = list(pool.map(lambda item: self.load_image(item[0]), items))
        else:
            loaded_images = [self.load_image(image_path) for image_path, _ in items]

        valid_indices: List[int] = []
        det_inputs: List[torch.Tensor] = []
        input_images: List[np.ndarray] = []
        for index, (_, _, input_image) in enumerate(loaded_images):
            if input_image is None:
                continue
            valid_indices.append(index)
            input_images.append(input_image)
            det_inputs.append(DET_TRANSFORM(input_image).to(self.device))

        if not det_inputs:
            return results

        det_outputs = self.det_model(det_inputs)
        pose_entries: List[Tuple[int, np.ndarray, List[float], np.ndarray]] = []
        pose_inputs: List[torch.Tensor] = []
        bboxes: List[List[float]] = []
        img_centers: List[np.ndarray] = []

        for local_index, det_output in enumerate(det_outputs):
            original_index = valid_indices[local_index]
            tight_bbox = get_one_box(det_output, thrd=self.det_threshold)
            if tight_bbox is None:
                results[original_index] = ({"hybrik_status": "no_person_detected"}, None)
                continue

            input_image = input_images[local_index]
            pose_input, bbox, img_center = self.transformation.test_transform(input_image, tight_bbox)
            pose_entries.append((original_index, input_image, bbox, img_center))
            pose_inputs.append(pose_input)
            bboxes.append(bbox)
            img_centers.append(img_center)

        if not pose_inputs:
            return results

        pose_batch = torch.stack(pose_inputs, dim=0).to(self.device)
        bbox_batch = torch.from_numpy(np.asarray(bboxes)).to(self.device).float()
        center_batch = torch.from_numpy(np.asarray(img_centers)).to(self.device).float()
        pose_output = self.hybrik_model(
            pose_batch,
            flip_test=True,
            bboxes=bbox_batch,
            img_center=center_batch,
        )

        for pose_index, (original_index, input_image, bbox, _) in enumerate(pose_entries):
            row_update = self.build_row_update(
                bbox,
                pose_output.transl[pose_index],
                pose_output.cam_root[pose_index],
                pose_output.pred_theta_mats[pose_index],
                pose_output.pred_xyz_jts_24_struct[pose_index],
            )

            vis_image = None
            if items[original_index][1]:
                try:
                    vis_image = self.render_visualization_from_tensors(
                        input_image,
                        bbox,
                        pose_output.pred_vertices[pose_index],
                        pose_output.transl[pose_index],
                    )
                except Exception:
                    vis_image = None
            results[original_index] = (row_update, vis_image)

        return results


def read_csv_rows(csv_path: Path) -> Tuple[List[str], List[Dict[str, str]]]:
    with csv_path.open("r", newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        if reader.fieldnames is None:
            raise RuntimeError(f"CSV has no header: {csv_path}")
        rows = list(reader)
        rows_fieldnames = list(reader.fieldnames)
    return rows_fieldnames, rows


def validate_output_csv_path(output_csv: Path) -> None:
    if output_csv.exists() and output_csv.is_dir():
        raise IsADirectoryError(f"Output CSV path points to an existing directory: {output_csv}")

    for parent in output_csv.parents:
        if parent.exists():
            if parent.is_file():
                raise ValueError(
                    f"Output CSV path is invalid because parent is a file, not a directory: {parent}\n"
                    f"Use a sibling CSV path such as: {parent.with_name(parent.stem + '_with_hybrik.csv')}"
                )
            break


def batched(items: Sequence[Tuple[Dict[str, str], Path, bool]], batch_size: int) -> Iterable[Sequence[Tuple[Dict[str, str], Path, bool]]]:
    for start in range(0, len(items), batch_size):
        yield items[start:start + batch_size]


def main() -> None:
    args = parse_args()
    if args.batch_size < 1:
        raise ValueError("--batch-size must be >= 1")
    if args.num_workers < 0:
        raise ValueError("--num-workers must be >= 0")
    csv_path = Path(args.csv_path).expanduser().resolve()
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    cfg_path = Path(args.cfg).expanduser().resolve()
    ckpt_path = Path(args.ckpt).expanduser().resolve()
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config not found: {cfg_path}")
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    output_csv = (
        Path(args.output_csv).expanduser().resolve()
        if args.output_csv
        else csv_path.with_name(f"{csv_path.stem}_with_hybrik.csv")
    )
    validate_output_csv_path(output_csv)
    vis_dir = output_csv.parent / args.vis_dir_name
    if args.save_vis_every > 0:
        vis_dir.mkdir(parents=True, exist_ok=True)

    fieldnames, rows = read_csv_rows(csv_path)
    if args.image_column not in fieldnames:
        raise KeyError(f"Image column '{args.image_column}' not found in CSV header")

    hybrik_fields = build_output_fields()
    output_fieldnames = list(fieldnames)
    for field in hybrik_fields:
        if field not in output_fieldnames:
            output_fieldnames.append(field)

    inferencer = HybrIKInferencer(
        cfg_path=cfg_path,
        ckpt_path=ckpt_path,
        gpu=args.gpu,
        det_threshold=args.det_threshold,
    )

    processed_rows: List[Dict[str, str]] = []
    kept_count = 0
    dropped_count = 0

    pending_items: List[Tuple[Dict[str, str], Path, bool]] = []
    for row in rows:
        image_value = (row.get(args.image_column) or "").strip()
        if not image_value:
            dropped_count += 1
            continue

        image_path = resolve_image_path(csv_path, image_value)
        if not image_path.exists():
            dropped_count += 1
            continue

        should_save_vis = args.save_vis_every > 0 and (len(pending_items) + kept_count) % args.save_vis_every == 0
        pending_items.append((row, image_path, should_save_vis))

    progress = tqdm(total=len(pending_items), desc="Running HybrIK")
    for batch_items in batched(pending_items, args.batch_size):
        try:
            batch_results = inferencer.infer_batch(
                [(image_path, should_save_vis) for _, image_path, should_save_vis in batch_items],
                num_workers=args.num_workers,
            )
        except Exception:
            dropped_count += len(batch_items)
            progress.update(len(batch_items))
            continue

        for (row, image_path, should_save_vis), (row_update, vis_image) in zip(batch_items, batch_results):
            if row_update.get("hybrik_status") != "ok":
                dropped_count += 1
                progress.update(1)
                continue

            row.update(row_update)
            processed_rows.append(row)

            if should_save_vis and vis_image is not None:
                vis_name = f"frame_{kept_count:06d}_{image_path.stem}.jpg"
                cv2.imwrite(str(vis_dir / vis_name), vis_image)
            kept_count += 1
            progress.update(1)
    progress.close()

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=output_fieldnames)
        writer.writeheader()
        writer.writerows(processed_rows)

    print(f"Input CSV: {csv_path}")
    print(f"Output CSV: {output_csv}")
    print(f"Total rows: {len(rows)}")
    print(f"Kept rows: {kept_count}")
    print(f"Dropped rows: {dropped_count}")
    print(f"Batch size: {args.batch_size}")
    print(f"Image loader workers: {args.num_workers}")
    if args.save_vis_every > 0:
        print(f"Visualization samples: {vis_dir} (every {args.save_vis_every} successful frames)")


if __name__ == "__main__":
    main()
