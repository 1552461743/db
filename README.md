# yifu_banshen

这是上半身款动捕服的独立工作区。项目目标是把 10 路柔性电容 + 8 路 IMU 的传感器数据转换成 SMPL 上半身姿态，并继续对接 GMR/机器人重定向。

核心思路：

```text
ESP32 传感器 + 相机
  -> ROS2 topics
  -> ros2 bag
  -> 同步 CSV + 图片
  -> HybrIK 用图片生成 SMPL 伪标签
  -> GRU 学习“传感器序列 -> SMPL 姿态”
  -> 实时传感器输入输出 SMPL
  -> GM/GMR 重定向到机器人
```

当前款式的 ESP32 采集程序来自原来的 `2.py`，已经整理为 ROS2 节点：

```text
/home/nb666/HybrIK/data/yifu_banshen/ros2/src/hybrik_nodes/scripts/esp32_imu_reader.py
```

原始 `2.py` 已保存在：

```text
/home/nb666/HybrIK/data/yifu_banshen/assets/original_esp32_reader_2.py
```

## 目录

- `ros2/`：ROS2 工作区，包含 ESP32 读取、相机发布、校准归一化、实时 GRU 推理、UDP streamer。
- `scripts/`：离线数据处理、HybrIK 标注、CSV 合并、GRU 训练、UDP 实时推理和回放脚本。
- `web_visualizer/`：上半身款传感器网页可视化，只用于看原始传感器状态。
- `GM/`：接收 SMPL UDP，并转给 GMR/机器人 IK 的桥接脚本。
- `GMR/`：General Motion Retargeting 库和机器人 IK 配置。
- `gui/`：SMPL 关节姿态调试 GUI。
- `data/`：上半身款采集数据集目录。当前目录可能为空，训练历史数据路径记录在 `mod/*/metrics.json`。
- `mod/`：上半身款 GRU 模型输出目录，包含 `best_model.pt`、`metrics.json` 和 loss 曲线。
- `hybrik_runtime/`：当前项目所需的 HybrIK 运行依赖本地副本。
- `assets/`：原始程序和压缩包归档。

## 本地 HybrIK 依赖

当前目录已经复制了一份本项目直接用到的 HybrIK 文件，统一放在 `hybrik_runtime/` 下，方便后续按当前工作区维护：

- `hybrik_runtime/hybrik/`：HybrIK Python 代码包。
- `hybrik_runtime/configs/`：HybrIK 配置文件。
- `hybrik_runtime/pretrained_models/hybrik_hrnet.pth`：`csv_add_hybrik.py` 默认使用的 HybrIK 权重。
- `hybrik_runtime/model_files/J_regressor_h36m.npy`
- `hybrik_runtime/model_files/basicModel_neutral_lbs_10_207_0_v1.0.0.pkl`
- `hybrik_runtime/model_files/h36m_mean_beta.npy`

没有复制上层 `model_files/smplx/` 和 `pytorch3d/` 大目录。当前上半身 SMPL/GRU 流程不直接依赖 `model_files/smplx/`；`pytorch3d` 仍需要由当前 Python 环境提供，或继续使用上层环境中已有的安装。

`scripts/csv_add_hybrik.py`、`scripts/train_sensor_gru.py`、`scripts/live_gru_udp_infer.py`、`gui/smpl_joint_pose_gui.py` 和 ROS2 的 `live_gru_smpl_infer.py` 已改为优先使用 `hybrik_runtime/`。如果本地副本不存在，才回退到旧的根目录副本或上层 `/home/nb666/HybrIK`。

## 代码模块逻辑

### 1. ROS2 采集

`ros2/src/hybrik_nodes/scripts/esp32_imu_reader.py` 通过串口读取 ESP32 输出的文本帧，解析：

- `CAP`：10 路电容。
- `ACC`：8 路 IMU 加速度，每路 xyz。
- `GYRO`：8 路 IMU 角速度，每路 xyz。
- `MAG`：8 路磁场，每路 xyz。

节点会等当前轮 `ACC/GYRO/MAG` 都到齐后统一发布一次 IMU 数据，避免不同类型数据明显错帧。

### 2. 校准和归一化

`ros2/src/hybrik_nodes/src/human_skeleton_tf_publisher.cpp` 负责两类校准输出：

- 对 10 路电容做去毛刺、中值滤波、低通滤波，并根据站立基准和最大值归一化为 `/sensor/normalized`。
- 对 8 路 IMU 做 Acc + Gyro 姿态估计，并输出相对初始姿态的四元数 `/imu/relative_transform`。

校准文件保存到：

```text
~/.ros/human_upper_body_calibration.txt
```

### 3. 数据集生成

`scripts/bag_to_csv.py` 使用图片话题作为时间锚点，默认图片话题是 `/dongbu`。每张图片对应 CSV 的一行，其他传感器话题取离图片时间戳最近的一帧。

输出结构通常是：

```text
data/<bag_name>/csv_export/synced_dataset.csv
data/<bag_name>/csv_export/images/
```

### 4. HybrIK 伪标签

`scripts/csv_add_hybrik.py` 读取 CSV 中的 `image_path`，对每张图片运行人体检测 + HybrIK，追加 SMPL 标签：

- `hybrik_status`
- 检测框 `hybrik_bbox_*`
- root translation / camera root
- 24 个 SMPL 关节 xyz
- 24 个 SMPL 关节 `3x3` rotation matrix

训练目标来自这些 HybrIK 伪标签，不是人工手动标注。

### 5. GRU 训练

`scripts/train_sensor_gru.py` 训练一个时序模型，把传感器窗口映射到窗口最后一帧的 SMPL 目标。当前上半身模型常用输入是 100 维：

```text
10 capacitance
10 normalized capacitance
8 * 6 IMU acc/gyro
8 * 4 relative quaternion
= 100
```

默认目标是 `upper_body`，只训练 16 个上半身 SMPL 关节：

```text
pelvis, spine1, spine2, spine3, neck,
left_collar, right_collar, jaw,
left_shoulder, right_shoulder,
left_elbow, right_elbow,
left_wrist, right_wrist,
left_thumb, right_thumb
```

每个关节输出 `xyz 3 + rotmat 9`，所以目标维度通常是 `16 * 12 = 192`。

训练后的 `best_model.pt` 不只保存权重，还保存了：

- `input_columns`
- `target_columns`
- `x_mean/x_std`
- `y_mean/y_std`
- 模型结构参数

实时推理会按 checkpoint 里的列名自动拼输入，不需要手工固定字段顺序。

### 6. 实时推理

项目里有两条实时推理路径。

ROS2 内推理：

```text
ros2/src/hybrik_nodes/scripts/live_gru_smpl_infer.py
```

该节点订阅 ROS2 传感器话题，发布：

```text
/hybrik/smpl_24
```

UDP 推理：

```text
ros2/src/hybrik_nodes/scripts/sensor_udp_streamer.py
scripts/live_gru_udp_infer.py
```

`sensor_udp_streamer.py` 从 ROS2 话题打包最新传感器状态，默认发送到 UDP `127.0.0.1:5010`。`live_gru_udp_infer.py` 接收传感器 JSON，运行 GRU，默认输出 GM 兼容的 SMPL JSON。

### 7. 机器人重定向

`GM/udp_smpl_to_gmr.py` 接收 `live_gru_udp_infer.py` 发出的 SMPL UDP payload，然后转换坐标系、调用 GMR IK，并可视化或保存机器人动作。

## 传感器话题

上半身款 ESP32 节点发布：

- `/sensor/capacitance`：10 路电容。
- `/imu/channel_0` 到 `/imu/channel_7`：8 路 IMU，加速度和角速度。
- `/sensor/acceleration_raw`：8 路原始加速度，24 个 float。
- `/sensor/gyroscope_raw`：8 路原始角速度，24 个 float。
- `/sensor/magnetic_raw`：8 路原始磁场，24 个 float。

上半身款校准节点会从 `/sensor/capacitance` 和 `/imu/channel_*` 生成：

- `/sensor/normalized`：10 路归一化电容。
- `/imu/relative_transform`：8 路相对四元数，32 个 float。

当前上半身 ESP32 节点没有发布全身款的 per-channel magnetic field 和 yaw 话题：

```text
/imu/channel_i/magnetic_field
/imu/channel_i/yaw_tilt_deg
```

训练和实时推理建议使用：

```text
--input-groups capacitance normalized imu relative
```

## 常用命令

完整命令看：

```text
/home/nb666/HybrIK/data/yifu_banshen/how_to_use.md
```

典型流程：

```bash
cd /home/nb666/HybrIK/data/yifu_banshen
python3 scripts/bag_to_csv.py data/1 --max-delta 50
```

图片降分辨率：

```bash
cd /home/nb666/HybrIK/data/yifu_banshen
python3 scripts/resize_csv_images.py \
  /home/nb666/HybrIK/data/yifu_banshen/data/1 \
  /home/nb666/HybrIK/data/yifu_banshen/data/2 \
  --num-workers 4
```

```bash
conda activate hybrik
cd /home/nb666/HybrIK/data/yifu_banshen
python scripts/csv_add_hybrik.py \
  /home/nb666/HybrIK/data/yifu_banshen/data/1/csv_export/synced_dataset.csv \
  --output-csv /home/nb666/HybrIK/data/yifu_banshen/data/1/csv_export/synced_dataset2.csv \
  --batch-size 4 \
  --num-workers 2
```

```bash
conda activate hybrik
cd /home/nb666/HybrIK/data/yifu_banshen
python scripts/train_sensor_gru.py \
  /home/nb666/HybrIK/data/yifu_banshen/data/1/csv_export/synced_dataset2.csv \
  --output-dir /home/nb666/HybrIK/data/yifu_banshen/mod/1 \
  --input-groups capacitance normalized imu relative \
  --target-joints upper_body \
  --train-ratio 0.8 --val-ratio 0.1 --test-ratio 0.1 \
  --seq-len 30 --epochs 200 --batch-size 64 --split-mode random_segment
```

实时 UDP 推理示例：

```bash
cd /home/nb666/HybrIK/data/yifu_banshen/ros2
source /opt/ros/humble/setup.bash
source install/setup.bash
ros2 launch hybrik_nodes sensor_udp_streamer.launch.py
```

另开一个终端：

```bash
conda activate hybrik
cd /home/nb666/HybrIK/data/yifu_banshen
python scripts/live_gru_udp_infer.py \
  --checkpoint-path /home/nb666/HybrIK/data/yifu_banshen/mod/1_9/best_model.pt \
  --listen-port 5010 \
  --send-port 5007 \
  --payload-format direct
```

GMR 接收 SMPL 并重定向：

```bash
cd /home/nb666/HybrIK/data/yifu_banshen/GM
python udp_smpl_to_gmr.py \
  --udp-port 5007 \
  --robot unitree_g1
```

网页查看原始传感器：

```bash
cd /home/nb666/HybrIK/data/yifu_banshen/web_visualizer
pip install -r requirements.txt
python3 web_visualizer.py
```

浏览器打开：

```text
http://localhost:5016
```

## 现有模型

当前 `mod/` 下已有几组训练输出：

- `mod/1_5`
- `mod/1_9`
- `mod/1_11`
- `mod/1_5_7_9`

这些模型的 `metrics.json` 显示它们使用的是 100 维输入和 192 维上半身 SMPL 输出。当前测试集指标里，`mod/1_9` 和 `mod/1_5_7_9` 相对更好。

## 注意

上半身款和全身款传感器数量不同。不要直接使用全身款的模型权重训练输出；上半身款需要使用本目录 `data/` 中采集的数据重新训练模型。

`scripts/train_sensor_gru.py` 在本目录默认使用 `--target-joints upper_body`，只训练上半身 SMPL 关节。实时推理/回放时，下半身关节会用 SMPL 中立姿态补齐，方便继续对接现有 SMPL/GMR 接口。

`ros2/src/hybrik_nodes/launch/live_gru_smpl_infer.launch.py` 里的默认 checkpoint 可能指向 `mod/1/best_model.pt`。如果当前没有 `mod/1`，需要手动改成实际存在的模型路径，例如：

```text
/home/nb666/HybrIK/data/yifu_banshen/mod/1_9/best_model.pt
```

`data/` 当前可能为空。如果要复现实验，需要把原始 bag/CSV 数据放回对应目录，或重新采集。
