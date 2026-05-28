#!/usr/bin/env python3
"""GUI for yifu suit dataset post-processing.

The GUI wraps the existing command-line scripts in yifu_quanshen/yifu_banshen:
- bag_to_csv.py
- csv_add_hybrik.py
- merge_csv_datasets.py
- resize_csv_images.py
"""

from __future__ import annotations

import os
import queue
import shlex
import signal
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional

import tkinter as tk
from tkinter import filedialog, messagebox, ttk


DATA_ROOT = Path(__file__).resolve().parent
SUIT_WORKSPACES = {
    "全身款 yifu_quanshen": DATA_ROOT / "yifu_quanshen",
    "上半身款 yifu_banshen": DATA_ROOT / "yifu_banshen",
}


@dataclass
class Task:
    title: str
    command: str
    cwd: Path
    delete_bag_dir: Optional[Path] = None
    skip_message: Optional[str] = None


def quote_path(path: Path | str) -> str:
    return shlex.quote(str(path))


def has_bag(dataset_dir: Path) -> bool:
    return (dataset_dir / "metadata.yaml").exists()


def has_raw_csv(dataset_dir: Path) -> bool:
    return (dataset_dir / "csv_export" / "synced_dataset.csv").exists()


def has_hybrik_csv(dataset_dir: Path) -> bool:
    return (dataset_dir / "csv_export" / "synced_dataset2.csv").exists()


def has_fixed_csv(dataset_dir: Path) -> bool:
    return (dataset_dir / "csv_export" / "synced_dataset3.csv").exists()


def has_merged_csv(dataset_dir: Path) -> bool:
    return (dataset_dir / "merged_dataset.csv").exists()


def has_fixed_merged_csv(dataset_dir: Path) -> bool:
    return (dataset_dir / "merged_dataset3.csv").exists()


class DataProcessingGui:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("动捕服数据处理 GUI")
        self.root.geometry("1180x760")
        self.root.minsize(980, 680)

        self.log_queue: queue.Queue[str] = queue.Queue()
        self.worker_thread: Optional[threading.Thread] = None
        self.current_process: Optional[subprocess.Popen[str]] = None
        self.stop_requested = False
        self.dataset_items: List[Path] = []

        self.suit_var = tk.StringVar(value="全身款 yifu_quanshen")
        self.workspace_var = tk.StringVar()

        self.do_bag_var = tk.BooleanVar(value=True)
        self.do_hybrik_var = tk.BooleanVar(value=True)
        self.do_foot_adjust_var = tk.BooleanVar(value=False)
        self.do_visualize_labels_var = tk.BooleanVar(value=False)
        self.do_merge_var = tk.BooleanVar(value=False)
        self.do_resize_var = tk.BooleanVar(value=False)
        self.do_train_var = tk.BooleanVar(value=False)
        self.do_replay_var = tk.BooleanVar(value=False)
        self.do_delete_bag_var = tk.BooleanVar(value=False)

        self.python_cmd_var = tk.StringVar(value="python3")
        self.hybrik_python_cmd_var = tk.StringVar(value="conda run --no-capture-output -n hybrik python")
        self.source_ros_var = tk.BooleanVar(value=True)
        self.ros_setup_var = tk.StringVar(value="/opt/ros/humble/setup.bash")

        self.max_delta_var = tk.StringVar(value="50")
        self.image_topic_var = tk.StringVar(value="/dongbu")
        self.hybrik_batch_var = tk.StringVar(value="4")
        self.hybrik_workers_var = tk.StringVar(value="2")
        self.foot_adjust_angle_var = tk.StringVar(value="-40")
        self.foot_adjust_axis_var = tk.StringVar(value="x")
        self.foot_adjust_joint_preset_var = tk.StringVar(value="both_ankles")
        self.foot_adjust_space_var = tk.StringVar(value="local")
        self.resize_width_var = tk.StringVar(value="640")
        self.resize_height_var = tk.StringVar(value="480")
        self.resize_workers_var = tk.StringVar(value="4")
        self.resize_source_var = tk.StringVar(value="hybrik_csv")
        self.merge_output_var = tk.StringVar()
        self.train_source_var = tk.StringVar(value="auto")
        self.train_output_dir_var = tk.StringVar()
        self.train_input_groups_var = tk.StringVar(value="capacitance normalized imu yaw")
        self.train_no_mag_var = tk.BooleanVar(value=True)
        self.train_target_joints_var = tk.StringVar(value="upper_body")
        self.train_seq_len_var = tk.StringVar(value="30")
        self.train_epochs_var = tk.StringVar(value="200")
        self.train_batch_var = tk.StringVar(value="64")
        self.train_hidden_size_var = tk.StringVar(value="256")
        self.train_num_layers_var = tk.StringVar(value="2")
        self.train_dropout_var = tk.StringVar(value="0.2")
        self.train_split_mode_var = tk.StringVar(value="random_segment")
        self.train_train_ratio_var = tk.StringVar(value="0.8")
        self.train_val_ratio_var = tk.StringVar(value="0.1")
        self.train_test_ratio_var = tk.StringVar(value="0.1")
        self.train_device_var = tk.StringVar(value="")
        self.replay_model_var = tk.StringVar(value="")
        self.replay_dataset_var = tk.StringVar(value="")
        self.replay_udp_targets_var = tk.StringVar(value="127.0.0.1:5007 127.0.0.1:5008 127.0.0.1:5009")
        self.replay_start_frame_var = tk.StringVar(value="")
        self.replay_frame_count_var = tk.StringVar(value="300")
        self.replay_fps_var = tk.StringVar(value="30")
        self.replay_device_var = tk.StringVar(value="")
        self.replay_hide_window_var = tk.BooleanVar(value=False)
        self.replay_render_mode_var = tk.StringVar(value="skeleton")
        self.replay_render_size_var = tk.StringVar(value="640")
        self.label_vis_csv_var = tk.StringVar(value="")
        self.label_vis_start_frame_var = tk.StringVar(value="0")
        self.label_vis_frame_count_var = tk.StringVar(value="300")
        self.label_vis_fps_var = tk.StringVar(value="10")
        self.label_vis_render_mode_var = tk.StringVar(value="skeleton")
        self.label_vis_render_size_var = tk.StringVar(value="640")
        self.label_vis_device_var = tk.StringVar(value="")
        self.label_vis_send_udp_var = tk.BooleanVar(value=False)
        self.label_vis_udp_targets_var = tk.StringVar(value="127.0.0.1:5007 127.0.0.1:5008 127.0.0.1:5009")

        self.status_var = tk.StringVar(value="空闲")
        self.disk_usage_var = tk.StringVar(value="/home 磁盘: 读取中...")
        self.progress_var = tk.DoubleVar(value=0.0)

        self.build_layout()
        self.on_suit_changed()
        self.root.after(100, self.flush_log_queue)
        self.root.after(100, self.update_disk_usage)
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    def build_layout(self) -> None:
        outer = ttk.Frame(self.root, padding=10)
        outer.pack(fill=tk.BOTH, expand=True)

        top = ttk.Frame(outer)
        top.pack(fill=tk.X)

        ttk.Label(top, text="衣服类型").pack(side=tk.LEFT)
        suit_combo = ttk.Combobox(
            top,
            textvariable=self.suit_var,
            values=list(SUIT_WORKSPACES.keys()),
            state="readonly",
            width=24,
        )
        suit_combo.pack(side=tk.LEFT, padx=(8, 16))
        suit_combo.bind("<<ComboboxSelected>>", lambda _event: self.on_suit_changed())

        ttk.Label(top, text="工作目录").pack(side=tk.LEFT)
        ttk.Entry(top, textvariable=self.workspace_var, state="readonly").pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(8, 8))
        ttk.Button(top, text="刷新数据集", command=self.refresh_datasets).pack(side=tk.LEFT)
        ttk.Button(top, text="预览命令", command=self.preview_commands).pack(side=tk.LEFT, padx=(8, 0))
        ttk.Button(top, text="停止", command=self.stop_tasks).pack(side=tk.LEFT, padx=(8, 0))
        ttk.Button(top, text="开始处理", command=self.start_tasks).pack(side=tk.LEFT, padx=(8, 0))

        body = ttk.PanedWindow(outer, orient=tk.HORIZONTAL)
        body.pack(fill=tk.BOTH, expand=True, pady=(10, 8))

        left = ttk.Frame(body, padding=(0, 0, 8, 0))
        right = ttk.Frame(body)
        body.add(left, weight=1)
        body.add(right, weight=2)

        dataset_frame = ttk.LabelFrame(left, text="选择数据集", padding=8)
        dataset_frame.pack(fill=tk.BOTH, expand=True)

        ttk.Label(dataset_frame, text="可多选。状态: bag=原始bag, csv=synced_dataset.csv, label=synced_dataset2.csv, fixed=synced_dataset3.csv, merged=merged_dataset.csv, merged3=merged_dataset3.csv").pack(anchor=tk.W)
        list_frame = ttk.Frame(dataset_frame)
        list_frame.pack(fill=tk.BOTH, expand=True, pady=(6, 0))
        self.dataset_listbox = tk.Listbox(list_frame, selectmode=tk.EXTENDED, exportselection=False)
        self.dataset_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.dataset_listbox.bind("<<ListboxSelect>>", lambda _event: self.update_merge_output_default())
        scrollbar = ttk.Scrollbar(list_frame, orient=tk.VERTICAL, command=self.dataset_listbox.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.dataset_listbox.configure(yscrollcommand=scrollbar.set)

        dataset_buttons = ttk.Frame(dataset_frame)
        dataset_buttons.pack(fill=tk.X, pady=(8, 0))
        ttk.Button(dataset_buttons, text="全选", command=self.select_all_datasets).pack(side=tk.LEFT)
        ttk.Button(dataset_buttons, text="清空", command=self.clear_dataset_selection).pack(side=tk.LEFT, padx=(8, 0))
        ttk.Button(dataset_buttons, text="打开数据目录", command=self.open_data_dir).pack(side=tk.RIGHT)

        step_frame = ttk.LabelFrame(left, text="处理步骤", padding=8)
        step_frame.pack(fill=tk.X, pady=(10, 0))
        ttk.Checkbutton(step_frame, text="2. Bag 转 CSV", variable=self.do_bag_var).pack(anchor=tk.W)
        ttk.Checkbutton(step_frame, text="3. HybrIK 加 SMPL 标签", variable=self.do_hybrik_var).pack(anchor=tk.W)
        ttk.Checkbutton(step_frame, text="4. 脚腕旋转修正 -> synced_dataset3.csv / merged_dataset3.csv", variable=self.do_foot_adjust_var).pack(anchor=tk.W)
        ttk.Checkbutton(step_frame, text="5. 查看 HybrIK 标签", variable=self.do_visualize_labels_var).pack(anchor=tk.W)
        ttk.Checkbutton(step_frame, text="6. 合并多个 CSV", variable=self.do_merge_var).pack(anchor=tk.W)
        ttk.Checkbutton(step_frame, text="7. 图片降分辨率", variable=self.do_resize_var).pack(anchor=tk.W)
        ttk.Checkbutton(step_frame, text="8. 训练 GRU 模型", variable=self.do_train_var).pack(anchor=tk.W)
        ttk.Checkbutton(step_frame, text="9. GRU 模型测试/UDP 回放", variable=self.do_replay_var).pack(anchor=tk.W)
        ttk.Checkbutton(step_frame, text="10. 删除原始 bag 文件", variable=self.do_delete_bag_var).pack(anchor=tk.W)

        settings = ttk.Notebook(right)
        settings.pack(fill=tk.X)
        self.build_basic_settings(settings)
        self.build_step_settings(settings)
        self.build_label_visualization_settings(settings)
        self.build_train_settings(settings)
        self.build_replay_settings(settings)

        log_frame = ttk.LabelFrame(right, text="运行日志", padding=6)
        log_frame.pack(fill=tk.BOTH, expand=True, pady=(10, 0))
        self.log_text = tk.Text(log_frame, wrap=tk.WORD, height=18)
        self.log_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        log_scrollbar = ttk.Scrollbar(log_frame, orient=tk.VERTICAL, command=self.log_text.yview)
        log_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.log_text.configure(yscrollcommand=log_scrollbar.set)

        bottom = ttk.Frame(outer)
        bottom.pack(fill=tk.X)
        ttk.Label(bottom, textvariable=self.status_var).pack(side=tk.LEFT)
        ttk.Progressbar(bottom, variable=self.progress_var, maximum=100).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=10)
        ttk.Label(bottom, textvariable=self.disk_usage_var).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Button(bottom, text="预览命令", command=self.preview_commands).pack(side=tk.RIGHT, padx=(8, 0))
        ttk.Button(bottom, text="停止", command=self.stop_tasks).pack(side=tk.RIGHT, padx=(8, 0))
        ttk.Button(bottom, text="开始处理", command=self.start_tasks).pack(side=tk.RIGHT)

    def build_basic_settings(self, notebook: ttk.Notebook) -> None:
        frame = ttk.Frame(notebook, padding=10)
        notebook.add(frame, text="环境")

        self.add_labeled_entry(frame, "普通 Python 命令", self.python_cmd_var, 0)
        self.add_labeled_entry(frame, "HybrIK Python 命令", self.hybrik_python_cmd_var, 1)
        self.add_labeled_entry(frame, "ROS setup.bash", self.ros_setup_var, 2)
        ttk.Checkbutton(frame, text="Bag 转 CSV 前 source ROS 环境", variable=self.source_ros_var).grid(
            row=3, column=1, sticky=tk.W, pady=(4, 0)
        )
        frame.columnconfigure(1, weight=1)

    def build_step_settings(self, notebook: ttk.Notebook) -> None:
        frame = ttk.Frame(notebook, padding=10)
        notebook.add(frame, text="参数")

        self.add_labeled_entry(frame, "Bag 同步最大误差 ms", self.max_delta_var, 0, width=12)
        self.add_labeled_entry(frame, "图片话题", self.image_topic_var, 1, width=18)
        self.add_labeled_entry(frame, "HybrIK batch-size", self.hybrik_batch_var, 2, width=12)
        self.add_labeled_entry(frame, "HybrIK num-workers", self.hybrik_workers_var, 3, width=12)
        self.add_labeled_entry(frame, "脚腕修正角度 deg", self.foot_adjust_angle_var, 4, width=12)

        ttk.Label(frame, text="脚腕修正轴").grid(row=5, column=0, sticky=tk.W, pady=4)
        ttk.Combobox(
            frame,
            textvariable=self.foot_adjust_axis_var,
            values=["x", "y", "z"],
            state="readonly",
            width=12,
        ).grid(row=5, column=1, sticky=tk.W, pady=4)
        ttk.Label(frame, text="默认 x，当前推荐 angle=-40").grid(row=5, column=2, sticky=tk.W, padx=(8, 0))

        ttk.Label(frame, text="脚腕修正关节").grid(row=6, column=0, sticky=tk.W, pady=4)
        ttk.Combobox(
            frame,
            textvariable=self.foot_adjust_joint_preset_var,
            values=["both_ankles", "both_feet", "left_ankle", "right_ankle", "left_foot", "right_foot"],
            state="readonly",
            width=18,
        ).grid(row=6, column=1, sticky=tk.W, pady=4)

        ttk.Label(frame, text="脚腕修正空间").grid(row=7, column=0, sticky=tk.W, pady=4)
        ttk.Combobox(
            frame,
            textvariable=self.foot_adjust_space_var,
            values=["local", "global"],
            state="readonly",
            width=12,
        ).grid(row=7, column=1, sticky=tk.W, pady=4)
        ttk.Label(frame, text="默认 local；单数据集输出 synced_dataset3，合并CSV输出 merged_dataset3").grid(row=7, column=2, sticky=tk.W, padx=(8, 0))

        self.add_labeled_entry(frame, "缩放宽度", self.resize_width_var, 8, width=12)
        self.add_labeled_entry(frame, "缩放高度", self.resize_height_var, 9, width=12)
        self.add_labeled_entry(frame, "缩放 workers", self.resize_workers_var, 10, width=12)

        ttk.Label(frame, text="缩放输入").grid(row=11, column=0, sticky=tk.W, pady=4)
        resize_combo = ttk.Combobox(
            frame,
            textvariable=self.resize_source_var,
            values=[
                "hybrik_csv",
                "raw_csv",
                "dataset_dir",
            ],
            state="readonly",
            width=18,
        )
        resize_combo.grid(row=11, column=1, sticky=tk.W, pady=4)
        ttk.Label(frame, text="hybrik_csv=优先 synced_dataset3/2/1, raw_csv=原始同步CSV, dataset_dir=数据集目录").grid(
            row=11, column=2, sticky=tk.W, padx=(8, 0)
        )

        ttk.Label(frame, text="合并输出 CSV").grid(row=12, column=0, sticky=tk.W, pady=4)
        ttk.Entry(frame, textvariable=self.merge_output_var).grid(row=12, column=1, columnspan=2, sticky=tk.EW, pady=4)
        ttk.Button(frame, text="选择", command=self.choose_merge_output).grid(row=12, column=3, sticky=tk.E, padx=(8, 0))
        frame.columnconfigure(1, weight=1)
        frame.columnconfigure(2, weight=1)

    def build_train_settings(self, notebook: ttk.Notebook) -> None:
        frame = ttk.Frame(notebook, padding=10)
        notebook.add(frame, text="训练")

        ttk.Label(frame, text="训练输入").grid(row=0, column=0, sticky=tk.W, pady=4)
        ttk.Combobox(
            frame,
            textvariable=self.train_source_var,
            values=["auto", "merged_csv", "hybrik_csv"],
            state="readonly",
            width=18,
        ).grid(row=0, column=1, sticky=tk.W, pady=4, padx=(8, 0))
        ttk.Label(frame, text="auto=勾选合并时用合并CSV，否则逐个数据集训练").grid(row=0, column=2, sticky=tk.W, padx=(8, 0))

        ttk.Label(frame, text="训练输出目录").grid(row=1, column=0, sticky=tk.W, pady=4)
        ttk.Entry(frame, textvariable=self.train_output_dir_var).grid(row=1, column=1, columnspan=2, sticky=tk.EW, pady=4, padx=(8, 0))
        ttk.Button(frame, text="选择", command=self.choose_train_output_dir).grid(row=1, column=3, sticky=tk.E, padx=(8, 0))

        self.add_labeled_entry(frame, "input-groups", self.train_input_groups_var, 2, width=40)
        ttk.Label(frame, text="半身建议: capacitance normalized imu relative").grid(row=2, column=2, sticky=tk.W, padx=(8, 0))
        self.add_labeled_entry(frame, "半身 target-joints", self.train_target_joints_var, 3, width=18)
        ttk.Label(frame, text="全身款会忽略这个参数").grid(row=3, column=2, sticky=tk.W, padx=(8, 0))
        self.add_labeled_entry(frame, "seq-len", self.train_seq_len_var, 4, width=12)
        self.add_labeled_entry(frame, "epochs", self.train_epochs_var, 5, width=12)
        self.add_labeled_entry(frame, "batch-size", self.train_batch_var, 6, width=12)
        self.add_labeled_entry(frame, "hidden-size", self.train_hidden_size_var, 7, width=12)
        ttk.Label(frame, text="GRU隐藏状态维度，默认256；越大容量越强但更慢").grid(row=7, column=2, sticky=tk.W, padx=(8, 0))
        self.add_labeled_entry(frame, "num-layers", self.train_num_layers_var, 8, width=12)
        ttk.Label(frame, text="GRU堆叠层数，默认2").grid(row=8, column=2, sticky=tk.W, padx=(8, 0))
        self.add_labeled_entry(frame, "dropout", self.train_dropout_var, 9, width=12)
        ttk.Label(frame, text="防过拟合随机丢弃比例，默认0.2").grid(row=9, column=2, sticky=tk.W, padx=(8, 0))

        ttk.Label(frame, text="split-mode").grid(row=10, column=0, sticky=tk.W, pady=4)
        ttk.Combobox(
            frame,
            textvariable=self.train_split_mode_var,
            values=["random_segment", "chronological"],
            state="readonly",
            width=18,
        ).grid(row=10, column=1, sticky=tk.W, pady=4, padx=(8, 0))

        self.add_labeled_entry(frame, "train-ratio", self.train_train_ratio_var, 11, width=12)
        self.add_labeled_entry(frame, "val-ratio", self.train_val_ratio_var, 12, width=12)
        self.add_labeled_entry(frame, "test-ratio", self.train_test_ratio_var, 13, width=12)
        self.add_labeled_entry(frame, "device", self.train_device_var, 14, width=12)
        ttk.Label(frame, text="留空则脚本自动选择；可填 cuda、cuda:0 或 cpu").grid(row=14, column=2, sticky=tk.W, padx=(8, 0))
        ttk.Checkbutton(frame, text="无磁力计训练（不使用 mag 输入组）", variable=self.train_no_mag_var).grid(
            row=15, column=1, sticky=tk.W, pady=(4, 0)
        )
        frame.columnconfigure(1, weight=1)
        frame.columnconfigure(2, weight=1)

    def build_label_visualization_settings(self, notebook: ttk.Notebook) -> None:
        frame = ttk.Frame(notebook, padding=10)
        notebook.add(frame, text="标签查看")

        ttk.Label(frame, text="CSV").grid(row=0, column=0, sticky=tk.W, pady=4)
        ttk.Entry(frame, textvariable=self.label_vis_csv_var).grid(row=0, column=1, columnspan=2, sticky=tk.EW, pady=4, padx=(8, 0))
        ttk.Button(frame, text="选择", command=self.choose_label_vis_csv).grid(row=0, column=3, sticky=tk.E, padx=(8, 0))
        ttk.Label(frame, text="留空时使用所选单个数据集的 synced_dataset3/2/1").grid(row=0, column=4, sticky=tk.W, padx=(8, 0))

        self.add_labeled_entry(frame, "start-frame", self.label_vis_start_frame_var, 1, width=12)
        self.add_labeled_entry(frame, "frame-count", self.label_vis_frame_count_var, 2, width=12)
        self.add_labeled_entry(frame, "fps", self.label_vis_fps_var, 3, width=12)
        ttk.Label(frame, text="render-mode").grid(row=4, column=0, sticky=tk.W, pady=4)
        ttk.Combobox(
            frame,
            textvariable=self.label_vis_render_mode_var,
            values=["skeleton", "mesh"],
            state="readonly",
            width=18,
        ).grid(row=4, column=1, sticky=tk.W, pady=4, padx=(8, 0))
        self.add_labeled_entry(frame, "render-size", self.label_vis_render_size_var, 5, width=12)
        self.add_labeled_entry(frame, "device", self.label_vis_device_var, 6, width=12)
        ttk.Label(frame, text="留空自动选择；mesh 较卡，建议 skeleton 或 render-size=320").grid(row=6, column=2, columnspan=3, sticky=tk.W, padx=(8, 0))
        ttk.Checkbutton(frame, text="同时通过 UDP 发送 SMPL 到 GMR/机器人", variable=self.label_vis_send_udp_var).grid(
            row=7, column=1, sticky=tk.W, pady=(4, 0)
        )
        self.add_labeled_entry(frame, "UDP 目标", self.label_vis_udp_targets_var, 8, width=46)
        ttk.Label(frame, text="空格/逗号分隔；启动对应 udp_smpl_to_gmr.py 后可看机器人").grid(row=8, column=2, columnspan=3, sticky=tk.W, padx=(8, 0))
        frame.columnconfigure(1, weight=1)
        frame.columnconfigure(2, weight=1)

    def build_replay_settings(self, notebook: ttk.Notebook) -> None:
        frame = ttk.Frame(notebook, padding=10)
        notebook.add(frame, text="GRU回放")

        ttk.Label(frame, text="模型").grid(row=0, column=0, sticky=tk.W, pady=4)
        ttk.Entry(frame, textvariable=self.replay_model_var).grid(row=0, column=1, columnspan=2, sticky=tk.EW, pady=4, padx=(8, 0))
        ttk.Button(frame, text="选择", command=self.choose_replay_model).grid(row=0, column=3, sticky=tk.E, padx=(8, 0))
        ttk.Label(frame, text="可填 mod 名称，如 1_15；也可选 best_model.pt 或模型目录").grid(row=0, column=4, sticky=tk.W, padx=(8, 0))

        ttk.Label(frame, text="数据集/CSV").grid(row=1, column=0, sticky=tk.W, pady=4)
        ttk.Entry(frame, textvariable=self.replay_dataset_var).grid(row=1, column=1, columnspan=2, sticky=tk.EW, pady=4, padx=(8, 0))
        ttk.Button(frame, text="选择", command=self.choose_replay_dataset).grid(row=1, column=3, sticky=tk.E, padx=(8, 0))
        ttk.Label(frame, text="可填数据集名、数据集目录或 CSV 路径").grid(row=1, column=4, sticky=tk.W, padx=(8, 0))

        self.add_labeled_entry(frame, "UDP 目标", self.replay_udp_targets_var, 2, width=46)
        ttk.Label(frame, text="空格/逗号分隔，如 127.0.0.1:5007 127.0.0.1:5008").grid(row=2, column=2, columnspan=3, sticky=tk.W, padx=(8, 0))
        self.add_labeled_entry(frame, "start-frame", self.replay_start_frame_var, 3, width=12)
        ttk.Label(frame, text="留空则从 seq_len-1 开始").grid(row=3, column=2, sticky=tk.W, padx=(8, 0))
        self.add_labeled_entry(frame, "frame-count", self.replay_frame_count_var, 4, width=12)
        self.add_labeled_entry(frame, "fps", self.replay_fps_var, 5, width=12)
        self.add_labeled_entry(frame, "device", self.replay_device_var, 6, width=12)
        ttk.Label(frame, text="留空自动选择；可填 cuda、cuda:0 或 cpu").grid(row=6, column=2, sticky=tk.W, padx=(8, 0))

        ttk.Label(frame, text="render-mode").grid(row=7, column=0, sticky=tk.W, pady=4)
        ttk.Combobox(
            frame,
            textvariable=self.replay_render_mode_var,
            values=["skeleton", "mesh"],
            state="readonly",
            width=18,
        ).grid(row=7, column=1, sticky=tk.W, pady=4, padx=(8, 0))
        self.add_labeled_entry(frame, "render-size", self.replay_render_size_var, 8, width=12)
        ttk.Checkbutton(frame, text="隐藏预览窗口，只发 UDP", variable=self.replay_hide_window_var).grid(
            row=9, column=1, sticky=tk.W, pady=(4, 0)
        )
        frame.columnconfigure(1, weight=1)
        frame.columnconfigure(2, weight=1)

    def add_labeled_entry(
        self,
        parent: ttk.Frame,
        label: str,
        variable: tk.StringVar,
        row: int,
        width: int = 60,
    ) -> None:
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky=tk.W, pady=4)
        ttk.Entry(parent, textvariable=variable, width=width).grid(row=row, column=1, sticky=tk.EW, pady=4, padx=(8, 0))

    @property
    def workspace(self) -> Path:
        return SUIT_WORKSPACES[self.suit_var.get()]

    @property
    def data_dir(self) -> Path:
        return self.workspace / "data"

    def on_suit_changed(self) -> None:
        self.workspace_var.set(str(self.workspace))
        if self.workspace.name == "yifu_banshen":
            self.train_input_groups_var.set("capacitance normalized imu relative")
            self.train_no_mag_var.set(False)
            self.train_target_joints_var.set("upper_body")
        else:
            self.train_input_groups_var.set("capacitance normalized imu yaw")
            self.train_no_mag_var.set(True)
            self.train_target_joints_var.set("upper_body")
        self.refresh_datasets()

    def refresh_datasets(self) -> None:
        self.dataset_listbox.delete(0, tk.END)
        self.dataset_items = []
        data_dir = self.data_dir
        data_dir.mkdir(parents=True, exist_ok=True)

        datasets = [path for path in data_dir.iterdir() if path.is_dir()]
        datasets.sort(key=lambda item: self.natural_key(item.name))
        for dataset in datasets:
            flags = []
            if has_bag(dataset):
                flags.append("bag")
            if has_raw_csv(dataset):
                flags.append("csv")
            if has_hybrik_csv(dataset):
                flags.append("label")
            if has_fixed_csv(dataset):
                flags.append("fixed")
            if has_merged_csv(dataset):
                flags.append("merged")
            if has_fixed_merged_csv(dataset):
                flags.append("merged3")
            flag_text = ",".join(flags) if flags else "empty"
            self.dataset_items.append(dataset)
            self.dataset_listbox.insert(tk.END, f"{dataset.name}    [{flag_text}]")
        self.update_merge_output_default()

    @staticmethod
    def natural_key(text: str) -> List[object]:
        parts: List[object] = []
        number = ""
        word = ""
        for char in text:
            if char.isdigit():
                if word:
                    parts.append(word.lower())
                    word = ""
                number += char
            else:
                if number:
                    parts.append(int(number))
                    number = ""
                word += char
        if number:
            parts.append(int(number))
        if word:
            parts.append(word.lower())
        return parts

    def selected_datasets(self) -> List[Path]:
        return [self.dataset_items[index] for index in self.dataset_listbox.curselection()]

    def select_all_datasets(self) -> None:
        self.dataset_listbox.select_set(0, tk.END)
        self.update_merge_output_default()

    def clear_dataset_selection(self) -> None:
        self.dataset_listbox.selection_clear(0, tk.END)
        self.update_merge_output_default()

    def open_data_dir(self) -> None:
        path = self.data_dir
        try:
            subprocess.Popen(["xdg-open", str(path)])
        except Exception as exc:
            messagebox.showerror("打开失败", str(exc))

    def choose_merge_output(self) -> None:
        initial = Path(self.merge_output_var.get()).expanduser() if self.merge_output_var.get() else self.data_dir
        filename = filedialog.asksaveasfilename(
            title="选择合并输出 CSV",
            initialdir=str(initial.parent if initial.suffix else initial),
            initialfile=initial.name if initial.suffix else "merged_dataset.csv",
            defaultextension=".csv",
            filetypes=[("CSV", "*.csv"), ("All files", "*")],
        )
        if filename:
            self.merge_output_var.set(filename)

    def choose_train_output_dir(self) -> None:
        initial = Path(self.train_output_dir_var.get()).expanduser() if self.train_output_dir_var.get() else self.workspace / "mod"
        directory = filedialog.askdirectory(
            title="选择训练输出目录",
            initialdir=str(initial if initial.exists() else self.workspace / "mod"),
        )
        if directory:
            self.train_output_dir_var.set(directory)

    def choose_replay_model(self) -> None:
        initial_dir = self.workspace / "mod"
        filename = filedialog.askopenfilename(
            title="选择 best_model.pt",
            initialdir=str(initial_dir),
            filetypes=[("PyTorch checkpoint", "*.pt"), ("All files", "*")],
        )
        if filename:
            self.replay_model_var.set(filename)

    def choose_replay_dataset(self) -> None:
        initial_dir = self.data_dir
        filename = filedialog.askopenfilename(
            title="选择测试 CSV",
            initialdir=str(initial_dir),
            filetypes=[("CSV", "*.csv"), ("All files", "*")],
        )
        if filename:
            self.replay_dataset_var.set(filename)

    def choose_label_vis_csv(self) -> None:
        initial_dir = self.data_dir
        filename = filedialog.askopenfilename(
            title="选择 HybrIK 标签 CSV",
            initialdir=str(initial_dir),
            filetypes=[("CSV", "*.csv"), ("All files", "*")],
        )
        if filename:
            self.label_vis_csv_var.set(filename)

    def update_merge_output_default(self) -> None:
        selected = self.selected_datasets()
        if not selected:
            self.merge_output_var.set("")
            return
        name = "_".join(dataset.name for dataset in selected[:5])
        if len(selected) > 5:
            name += f"_{len(selected)}sets"
        self.merge_output_var.set(str(self.data_dir / name / "merged_dataset.csv"))

    def build_python_command(self, python_cmd: str, script: Path, args: Iterable[str | Path]) -> str:
        arg_text = " ".join(quote_path(arg) for arg in args)
        return f"{python_cmd} {quote_path(script)} {arg_text}".strip()

    def foot_adjust_script_path(self) -> Path:
        workspace_script = self.workspace / "scripts" / "adjust_csv_foot_rotations.py"
        if workspace_script.exists():
            return workspace_script
        return SUIT_WORKSPACES["全身款 yifu_quanshen"] / "scripts" / "adjust_csv_foot_rotations.py"

    @staticmethod
    def csv_export_path(dataset: Path, filename: str) -> Path:
        return dataset / "csv_export" / filename

    def foot_adjust_paths(self, dataset: Path) -> tuple[Path, Path]:
        csv2_path = self.csv_export_path(dataset, "synced_dataset2.csv")
        if csv2_path.exists() or (self.do_hybrik_var.get() and has_raw_csv(dataset)):
            return csv2_path, self.csv_export_path(dataset, "synced_dataset3.csv")

        merged_path = dataset / "merged_dataset.csv"
        if merged_path.exists():
            return merged_path, dataset / "merged_dataset3.csv"

        fixed_merged_path = dataset / "merged_dataset3.csv"
        if fixed_merged_path.exists():
            return fixed_merged_path, fixed_merged_path

        raise FileNotFoundError(
            f"找不到待修正 CSV: {csv2_path} 或 {merged_path}"
        )

    def preferred_dataset_csv(self, dataset: Path, allow_future: bool = False) -> Path:
        candidates = [
            self.csv_export_path(dataset, "synced_dataset3.csv"),
            dataset / "merged_dataset3.csv",
            self.csv_export_path(dataset, "synced_dataset2.csv"),
            dataset / "merged_dataset.csv",
            self.csv_export_path(dataset, "synced_dataset.csv"),
        ]
        if allow_future and self.do_foot_adjust_var.get():
            return self.foot_adjust_paths(dataset)[1]
        for candidate in candidates:
            if candidate.exists():
                return candidate
        if allow_future:
            if self.do_hybrik_var.get():
                return candidates[2]
            if self.do_bag_var.get():
                return candidates[4]
        raise FileNotFoundError(
            f"找不到可用 CSV: {dataset} 下需要 synced_dataset3.csv / merged_dataset3.csv / synced_dataset2.csv / merged_dataset.csv / synced_dataset.csv"
        )

    def hybrik_label_input_csv(self, dataset: Path) -> Path:
        csv_path = self.csv_export_path(dataset, "synced_dataset2.csv")
        if csv_path.exists() or self.do_hybrik_var.get():
            return csv_path
        raise FileNotFoundError(f"找不到 HybrIK 标签 CSV: {csv_path}")

    def with_env_prefix(self, command: str, source_ros: bool = False) -> str:
        parts = ["export PYTHONUNBUFFERED=1"]
        if source_ros:
            ros_setup = Path(self.ros_setup_var.get()).expanduser()
            parts.append(f"source {quote_path(ros_setup)}")
        parts.append(command)
        return " && ".join(parts)

    def build_tasks(self) -> List[Task]:
        datasets = self.selected_datasets()
        if not datasets:
            raise ValueError("请先选择至少一个数据集")
        if not any([
            self.do_bag_var.get(),
            self.do_hybrik_var.get(),
            self.do_foot_adjust_var.get(),
            self.do_visualize_labels_var.get(),
            self.do_merge_var.get(),
            self.do_resize_var.get(),
            self.do_train_var.get(),
            self.do_replay_var.get(),
            self.do_delete_bag_var.get(),
        ]):
            raise ValueError("请至少勾选一个处理步骤")

        workspace = self.workspace
        scripts_dir = workspace / "scripts"
        python_cmd = self.python_cmd_var.get().strip() or "python3"
        hybrik_python_cmd = self.hybrik_python_cmd_var.get().strip() or python_cmd
        tasks: List[Task] = []

        if self.do_bag_var.get():
            for dataset in datasets:
                if not has_bag(dataset):
                    if has_raw_csv(dataset):
                        tasks.append(Task(
                            f"Bag 转 CSV: {dataset.name}",
                            "<skip bag_to_csv>",
                            workspace,
                            skip_message=f"跳过 {dataset.name}: 未找到 metadata.yaml，但已存在 csv_export/synced_dataset.csv",
                        ))
                        continue
                    raise FileNotFoundError(
                        f"数据集没有原始 bag，也没有 synced_dataset.csv: {dataset}\n"
                        f"如果这个数据集已经处理过，请确认存在: {dataset / 'csv_export' / 'synced_dataset.csv'}"
                    )
                args: List[str | Path] = [dataset]
                max_delta = self.max_delta_var.get().strip()
                if max_delta:
                    args.extend(["--max-delta", max_delta])
                image_topic = self.image_topic_var.get().strip()
                if image_topic:
                    args.extend(["--image-topic", image_topic])
                command = self.build_python_command(python_cmd, scripts_dir / "bag_to_csv.py", args)
                command = self.with_env_prefix(command, source_ros=self.source_ros_var.get())
                tasks.append(Task(f"Bag 转 CSV: {dataset.name}", command, workspace))

        if self.do_hybrik_var.get():
            for dataset in datasets:
                input_csv = dataset / "csv_export" / "synced_dataset.csv"
                output_csv = dataset / "csv_export" / "synced_dataset2.csv"
                if not self.do_bag_var.get() and not input_csv.exists():
                    raise FileNotFoundError(f"找不到原始同步 CSV: {input_csv}")
                args = [
                    input_csv,
                    "--output-csv",
                    output_csv,
                    "--batch-size",
                    self.hybrik_batch_var.get().strip() or "1",
                    "--num-workers",
                    self.hybrik_workers_var.get().strip() or "2",
                ]
                command = self.build_python_command(hybrik_python_cmd, scripts_dir / "csv_add_hybrik.py", args)
                command = self.with_env_prefix(command)
                tasks.append(Task(f"HybrIK 加标签: {dataset.name}", command, workspace))

        if self.do_foot_adjust_var.get():
            script_path = self.foot_adjust_script_path()
            if not script_path.exists():
                raise FileNotFoundError(f"找不到脚腕修正脚本: {script_path}")
            for dataset in datasets:
                input_csv, output_csv = self.foot_adjust_paths(dataset)
                args = [
                    input_csv,
                    "--output-csv",
                    output_csv,
                    "--joint-preset",
                    self.foot_adjust_joint_preset_var.get().strip() or "both_ankles",
                    "--space",
                    self.foot_adjust_space_var.get().strip() or "local",
                    "--axis",
                    self.foot_adjust_axis_var.get().strip() or "x",
                    "--angle-deg",
                    self.foot_adjust_angle_var.get().strip() or "-40",
                    "--overwrite",
                ]
                command = self.build_python_command(python_cmd, script_path, args)
                command = self.with_env_prefix(command)
                tasks.append(Task(f"脚腕旋转修正: {dataset.name} -> {output_csv.name}", command, workspace))

        if self.do_visualize_labels_var.get():
            tasks.append(self.build_label_visualization_task(datasets, hybrik_python_cmd, workspace))

        if self.do_merge_var.get():
            output_csv_text = self.merge_output_var.get().strip()
            if not output_csv_text:
                raise ValueError("请设置合并输出 CSV 路径")
            merge_inputs: List[Path] = []
            for dataset in datasets:
                csv_path = self.preferred_dataset_csv(dataset, allow_future=True)
                merge_inputs.append(csv_path)
            args = [*merge_inputs, "--output-csv", Path(output_csv_text)]
            command = self.build_python_command(python_cmd, scripts_dir / "merge_csv_datasets.py", args)
            command = self.with_env_prefix(command)
            tasks.append(Task("合并 CSV", command, workspace))

        if self.do_resize_var.get():
            resize_inputs: List[Path] = []
            resize_source = self.resize_source_var.get()
            for dataset in datasets:
                if resize_source == "hybrik_csv":
                    csv_path = self.preferred_dataset_csv(dataset, allow_future=True)
                    resize_inputs.append(csv_path)
                elif resize_source == "raw_csv":
                    csv_path = dataset / "csv_export" / "synced_dataset.csv"
                    if not self.do_bag_var.get() and not csv_path.exists():
                        raise FileNotFoundError(f"找不到原始同步 CSV: {csv_path}")
                    resize_inputs.append(csv_path)
                else:
                    resize_inputs.append(dataset)
            args = [
                *resize_inputs,
                "--width",
                self.resize_width_var.get().strip() or "640",
                "--height",
                self.resize_height_var.get().strip() or "480",
                "--num-workers",
                self.resize_workers_var.get().strip() or "1",
            ]
            command = self.build_python_command(python_cmd, scripts_dir / "resize_csv_images.py", args)
            command = self.with_env_prefix(command)
            tasks.append(Task("图片降分辨率", command, workspace))

        if self.do_train_var.get():
            tasks.extend(self.build_train_tasks(datasets, scripts_dir, hybrik_python_cmd, workspace))

        if self.do_replay_var.get():
            tasks.append(self.build_replay_task(datasets, scripts_dir, hybrik_python_cmd, workspace))

        if self.do_delete_bag_var.get():
            for dataset in datasets:
                tasks.append(Task(f"删除原始 bag 文件: {dataset.name}", "<delete original bag files>", workspace, dataset))

        return tasks

    def build_label_visualization_task(
        self,
        datasets: List[Path],
        hybrik_python_cmd: str,
        workspace: Path,
    ) -> Task:
        csv_text = self.label_vis_csv_var.get().strip()
        if csv_text:
            csv_path = Path(csv_text).expanduser()
        elif len(datasets) == 1:
            csv_path = self.preferred_dataset_csv(datasets[0], allow_future=True)
        else:
            raise ValueError("查看 HybrIK 标签选择多个数据集时，请在“标签查看”页指定 CSV")

        if not self.do_hybrik_var.get() and not csv_path.exists():
            raise FileNotFoundError(f"找不到 HybrIK 标签 CSV: {csv_path}")

        args: List[str | Path] = [
            "--csv",
            csv_path,
            "--suit-root",
            workspace,
            "--start-frame",
            self.label_vis_start_frame_var.get().strip() or "0",
            "--frame-count",
            self.label_vis_frame_count_var.get().strip() or "300",
            "--fps",
            self.label_vis_fps_var.get().strip() or "10",
            "--render-mode",
            self.label_vis_render_mode_var.get().strip() or "skeleton",
            "--render-size",
            self.label_vis_render_size_var.get().strip() or "640",
        ]
        device = self.label_vis_device_var.get().strip()
        if device:
            args.extend(["--device", device])
        if self.label_vis_send_udp_var.get():
            targets = self.parse_udp_targets_text(self.label_vis_udp_targets_var.get())
            if not targets:
                raise ValueError("标签查看 UDP 发送至少需要一个目标，例如 127.0.0.1:5007")
            for target in targets:
                args.extend(["--udp-target", target])

        command = self.build_python_command(hybrik_python_cmd, DATA_ROOT / "visualize_hybrik_csv.py", args)
        command = self.with_env_prefix(command)
        return Task("查看 HybrIK 标签", command, workspace)

    def build_train_tasks(
        self,
        datasets: List[Path],
        scripts_dir: Path,
        hybrik_python_cmd: str,
        workspace: Path,
    ) -> List[Task]:
        source = self.train_source_var.get()
        if source == "auto":
            source = "merged_csv" if self.do_merge_var.get() else "hybrik_csv"

        train_jobs: List[tuple[str, Path, Path]] = []
        explicit_output = self.train_output_dir_var.get().strip()
        output_base = Path(explicit_output).expanduser() if explicit_output else None

        if source == "merged_csv":
            merged_csv_text = self.merge_output_var.get().strip()
            if not merged_csv_text:
                raise ValueError("训练输入选择 merged_csv 时，请设置合并输出 CSV 路径")
            merged_csv = Path(merged_csv_text)
            if not self.do_merge_var.get() and not merged_csv.exists():
                raise FileNotFoundError(f"找不到合并 CSV: {merged_csv}")
            default_name = merged_csv.parent.name if merged_csv.parent.name else "merged"
            output_dir = output_base if output_base else workspace / "mod" / default_name
            if self.train_no_mag_var.get() and output_base is None:
                output_dir = output_dir.with_name(f"{output_dir.name}_no_mag")
            train_jobs.append(("合并数据", merged_csv, output_dir))
        else:
            for dataset in datasets:
                csv_path = self.preferred_dataset_csv(dataset, allow_future=True)
                if output_base and len(datasets) == 1:
                    output_dir = output_base
                elif output_base:
                    output_dir = output_base / dataset.name
                else:
                    output_dir = workspace / "mod" / dataset.name
                if self.train_no_mag_var.get() and output_base is None:
                    output_dir = output_dir.with_name(f"{output_dir.name}_no_mag")
                train_jobs.append((dataset.name, csv_path, output_dir))

        tasks: List[Task] = []
        for label, csv_path, output_dir in train_jobs:
            args: List[str | Path] = [
                csv_path,
                "--output-dir",
                output_dir,
                "--input-groups",
            ]
            input_groups = self.train_input_groups_var.get().strip().split()
            if self.train_no_mag_var.get():
                input_groups = [group for group in input_groups if group != "mag"]
                if not input_groups:
                    input_groups = ["capacitance", "normalized", "imu", "yaw"]
            if not input_groups:
                raise ValueError("训练 input-groups 不能为空")
            args.extend(input_groups)
            args.extend([
                "--train-ratio",
                self.train_train_ratio_var.get().strip() or "0.8",
                "--val-ratio",
                self.train_val_ratio_var.get().strip() or "0.1",
                "--test-ratio",
                self.train_test_ratio_var.get().strip() or "0.1",
                "--seq-len",
                self.train_seq_len_var.get().strip() or "30",
                "--epochs",
                self.train_epochs_var.get().strip() or "200",
                "--batch-size",
                self.train_batch_var.get().strip() or "64",
                "--hidden-size",
                self.train_hidden_size_var.get().strip() or "256",
                "--num-layers",
                self.train_num_layers_var.get().strip() or "2",
                "--dropout",
                self.train_dropout_var.get().strip() or "0.2",
                "--split-mode",
                self.train_split_mode_var.get().strip() or "random_segment",
            ])
            if workspace.name == "yifu_banshen":
                args.extend(["--target-joints", self.train_target_joints_var.get().strip() or "upper_body"])
            device = self.train_device_var.get().strip()
            if device:
                args.extend(["--device", device])

            train_script = scripts_dir / ("train_sensor_gru_no_mag.py" if self.train_no_mag_var.get() else "train_sensor_gru.py")
            if self.train_no_mag_var.get() and not train_script.exists():
                raise FileNotFoundError(f"找不到无磁力计训练脚本: {train_script}")
            command = self.build_python_command(hybrik_python_cmd, train_script, args)
            command = self.with_env_prefix(command)
            suffix = "（无磁力计）" if self.train_no_mag_var.get() else ""
            tasks.append(Task(f"训练 GRU 模型{suffix}: {label}", command, workspace))
        return tasks

    def build_replay_task(
        self,
        datasets: List[Path],
        scripts_dir: Path,
        hybrik_python_cmd: str,
        workspace: Path,
    ) -> Task:
        model_arg = self.replay_model_var.get().strip()
        dataset_arg = self.replay_dataset_var.get().strip()

        if not model_arg:
            train_output = self.train_output_dir_var.get().strip()
            if train_output:
                model_arg = str(Path(train_output).expanduser())
            elif self.do_train_var.get() and self.do_merge_var.get() and self.merge_output_var.get().strip():
                merged_csv = Path(self.merge_output_var.get().strip())
                model_arg = str(workspace / "mod" / merged_csv.parent.name)
            else:
                raise ValueError("模型测试需要填写模型名称、模型目录或 best_model.pt 路径")

        if not dataset_arg:
            if self.do_merge_var.get() and self.merge_output_var.get().strip():
                dataset_arg = self.merge_output_var.get().strip()
            elif len(datasets) == 1:
                dataset_arg = str(self.preferred_dataset_csv(datasets[0], allow_future=True))
            else:
                raise ValueError("模型测试选择了多个数据集时，请填写具体测试数据集或 CSV")

        args: List[str | Path] = ["--model", model_arg, "--dataset", dataset_arg]
        start_frame = self.replay_start_frame_var.get().strip()
        if start_frame:
            args.extend(["--start-frame", start_frame])
        frame_count = self.replay_frame_count_var.get().strip()
        if frame_count:
            args.extend(["--frame-count", frame_count])
        fps = self.replay_fps_var.get().strip()
        if fps:
            args.extend(["--fps", fps])

        targets = self.parse_udp_targets_text(self.replay_udp_targets_var.get())
        if not targets:
            raise ValueError("模型测试至少需要一个 UDP 目标，例如 127.0.0.1:5007")
        for target in targets:
            args.extend(["--udp-target", target])

        device = self.replay_device_var.get().strip()
        if device:
            args.extend(["--device", device])
        if self.replay_hide_window_var.get():
            args.append("--hide-window")
        args.extend([
            "--render-mode",
            self.replay_render_mode_var.get().strip() or "skeleton",
            "--render-size",
            self.replay_render_size_var.get().strip() or "640",
        ])

        command = self.build_python_command(hybrik_python_cmd, scripts_dir / "replay_gru_to_smpl_udp.py", args)
        command = self.with_env_prefix(command)
        return Task("模型测试/UDP 回放", command, workspace)

    @staticmethod
    def parse_udp_targets_text(text: str) -> List[str]:
        raw_parts = text.replace(",", " ").split()
        targets: List[str] = []
        for target in raw_parts:
            if ":" not in target:
                raise ValueError(f"UDP 目标格式错误: {target}，应为 HOST:PORT")
            targets.append(target)
        return targets

    def preview_commands(self) -> None:
        try:
            tasks = self.build_tasks()
        except Exception as exc:
            messagebox.showerror("无法生成命令", str(exc))
            return
        self.append_log("\n========== 命令预览 ==========")
        for index, task in enumerate(tasks, start=1):
            self.append_log(f"[{index}] {task.title}")
            self.append_log(f"cwd: {task.cwd}")
            if task.skip_message:
                self.append_log(task.skip_message)
            else:
                self.append_log(task.command)
        self.append_log("========== 预览结束 ==========\n")

    def start_tasks(self) -> None:
        if self.worker_thread and self.worker_thread.is_alive():
            messagebox.showwarning("正在运行", "当前已有任务在运行")
            return
        try:
            tasks = self.build_tasks()
        except Exception as exc:
            messagebox.showerror("参数错误", str(exc))
            return

        if self.do_delete_bag_var.get():
            if not messagebox.askyesno(
                "确认删除原始 bag 文件",
                "将删除所选数据集目录下的 metadata.yaml、*.db3、*.mcap。\n"
                "不会删除 csv_export 或图片。确定继续？",
            ):
                return

        self.stop_requested = False
        self.progress_var.set(0.0)
        self.status_var.set("运行中")
        self.append_log("\n========== 开始处理 ==========")
        self.worker_thread = threading.Thread(target=self.run_tasks, args=(tasks,), daemon=True)
        self.worker_thread.start()

    def run_tasks(self, tasks: List[Task]) -> None:
        total = len(tasks)
        for index, task in enumerate(tasks, start=1):
            if self.stop_requested:
                self.log_queue.put("已停止，剩余任务未执行。\n")
                break

            self.log_queue.put(f"\n[{index}/{total}] {task.title}\n")
            self.log_queue.put(f"cwd: {task.cwd}\n")
            if task.skip_message is not None:
                self.log_queue.put(task.skip_message + "\n")
                returncode = 0
                elapsed = 0.0
            elif task.delete_bag_dir is not None:
                self.log_queue.put(f"delete: {task.delete_bag_dir}/metadata.yaml, *.db3, *.mcap\n")
                returncode = self.delete_original_bag_files(task.delete_bag_dir)
                elapsed = 0.0
            else:
                self.log_queue.put(f"cmd: {task.command}\n")
                started = time.time()
                returncode = self.run_shell_command(task.command, task.cwd)
                elapsed = time.time() - started
            if returncode != 0:
                self.log_queue.put(f"任务失败，退出码 {returncode}，耗时 {elapsed:.1f}s\n")
                self.root.after(0, lambda: self.status_var.set("失败"))
                return
            self.log_queue.put(f"任务完成，耗时 {elapsed:.1f}s\n")
            self.root.after(0, lambda value=index / total * 100.0: self.progress_var.set(value))

        if self.stop_requested:
            self.root.after(0, lambda: self.status_var.set("已停止"))
        else:
            self.log_queue.put("\n========== 全部完成 ==========\n")
            self.root.after(0, lambda: self.status_var.set("完成"))
            self.root.after(0, self.refresh_datasets)

    def run_shell_command(self, command: str, cwd: Path) -> int:
        try:
            self.current_process = subprocess.Popen(
                ["bash", "-lc", command],
                cwd=str(cwd),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                preexec_fn=os.setsid,
            )
            assert self.current_process.stdout is not None
            while True:
                chunk = self.current_process.stdout.read(1)
                if not chunk:
                    break
                self.log_queue.put(chunk)
            return self.current_process.wait()
        except Exception as exc:
            self.log_queue.put(f"启动命令失败: {exc}\n")
            return 1
        finally:
            self.current_process = None

    def delete_original_bag_files(self, dataset_dir: Path) -> int:
        try:
            targets = []
            metadata_path = dataset_dir / "metadata.yaml"
            if metadata_path.exists():
                targets.append(metadata_path)
            targets.extend(sorted(dataset_dir.glob("*.db3")))
            targets.extend(sorted(dataset_dir.glob("*.mcap")))

            if not targets:
                self.log_queue.put("未找到原始 bag 文件，跳过。\n")
                return 0

            for target in targets:
                target.unlink()
                self.log_queue.put(f"已删除: {target}\n")
            return 0
        except Exception as exc:
            self.log_queue.put(f"删除原始 bag 文件失败: {exc}\n")
            return 1

    def stop_tasks(self) -> None:
        self.stop_requested = True
        process = self.current_process
        if process and process.poll() is None:
            try:
                os.killpg(os.getpgid(process.pid), signal.SIGTERM)
                self.append_log("正在停止当前任务...\n")
            except Exception as exc:
                self.append_log(f"停止失败: {exc}\n")

    def append_log(self, text: str) -> None:
        self.log_text.insert(tk.END, text + ("" if text.endswith("\n") else "\n"))
        self.log_text.see(tk.END)

    def append_log_raw(self, text: str) -> None:
        if "\r" in text:
            line_start = self.log_text.index("insert linestart")
            self.log_text.delete(line_start, "insert lineend")
            text = text.replace("\r", "")
        self.log_text.insert(tk.END, text)
        self.log_text.see(tk.END)

    def flush_log_queue(self) -> None:
        try:
            while True:
                self.append_log_raw(self.log_queue.get_nowait())
        except queue.Empty:
            pass
        self.root.after(100, self.flush_log_queue)

    def update_disk_usage(self) -> None:
        try:
            result = subprocess.run(
                ["df", "-h", "/home"],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
            self.disk_usage_var.set(lines[-1] if len(lines) >= 2 else result.stdout.strip())
        except Exception as exc:
            self.disk_usage_var.set(f"/home 磁盘: 读取失败: {exc}")
        self.root.after(1000, self.update_disk_usage)

    def on_close(self) -> None:
        if self.worker_thread and self.worker_thread.is_alive():
            if not messagebox.askyesno("退出", "任务仍在运行，确定停止并退出？"):
                return
            self.stop_tasks()
        self.root.destroy()


def main() -> None:
    root = tk.Tk()
    DataProcessingGui(root)
    root.mainloop()


if __name__ == "__main__":
    main()
