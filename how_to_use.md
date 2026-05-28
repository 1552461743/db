# 上半身款动捕服使用命令

工作目录：

```text
/home/nb666/HybrIK/data/yifu_banshen
```

当前工作区已经把本项目直接需要的 HybrIK 运行文件整理到：

```text
/home/nb666/HybrIK/data/yifu_banshen/hybrik_runtime
```

相关脚本会优先使用 `hybrik_runtime/` 下的 `hybrik/`、`configs/`、`model_files/`、`pretrained_models/`；如果本地副本不存在，才回退到上层 `/home/nb666/HybrIK`。

上半身款常用训练/推理输入组是：

```text
--input-groups capacitance normalized imu relative
```

当前上半身 ESP32 节点没有发布全身款的 per-channel magnetic field 和 yaw 话题，不建议在本项目里使用 `mag`、`yaw` 输入组，除非你额外补了对应 ROS2 topic。

## 1. ROS2 编译

第一次使用或修改 ROS2 源码后编译：

```bash
cd /home/nb666/HybrIK/data/yifu_banshen/ros2
source /opt/ros/humble/setup.bash
colcon build
source install/setup.bash
```

## 2. 启动上半身款 ESP32 读取

`2.py` 已整理为 `esp32_imu_reader.py`。

```bash
sudo chmod 777 /dev/*
cd /home/nb666/HybrIK/data/yifu_banshen/ros2
source /opt/ros/humble/setup.bash
source install/setup.bash
ros2 run hybrik_nodes esp32_imu_reader.py
```

也可以用 launch 启动：

```bash
cd /home/nb666/HybrIK/data/yifu_banshen/ros2
source /opt/ros/humble/setup.bash
source install/setup.bash
ros2 launch hybrik_nodes esp32_imu_reader.launch.py
```

默认串口参数：

```text
port=/dev/ttyACM0
baudrate=921600
```

## 3. 启动校准和归一化

先启动 ESP32 读取，再开一个终端启动校准节点：

```bash
cd /home/nb666/HybrIK/data/yifu_banshen/ros2
source /opt/ros/humble/setup.bash
source install/setup.bash
ros2 run hybrik_nodes human_skeleton_tf_publisher
```

也可以用 launch 启动：

```bash
cd /home/nb666/HybrIK/data/yifu_banshen/ros2
source /opt/ros/humble/setup.bash
source install/setup.bash
ros2 launch hybrik_nodes human_skeleton_tf_publisher.launch.py
```

启动后按提示操作：

```text
1 : 开始新的校准
2 : 加载现有校准文件
3 : 使用默认值
```

新校准流程里：

```text
1 : 3 秒站立基准校准
2 : 最大值/IMU 初始姿态校准
p : 打印当前校准结果
s : 保存并结束校准
q : 不保存退出
```

上半身款校准文件保存到：

```text
~/.ros/human_upper_body_calibration.txt
```

校准节点发布：

```text
/sensor/normalized
/imu/relative_transform
```

## 4. 启动相机

```bash
cd /home/nb666/HybrIK/data/yifu_banshen/ros2
source /opt/ros/humble/setup.bash
source install/setup.bash
ros2 run hybrik_nodes latest_camera_publisher.py
```

查看相机：

```bash
cd /home/nb666/HybrIK/data/yifu_banshen/ros2
source /opt/ros/humble/setup.bash
source install/setup.bash
python3 read_image.py
```

## 5. 网页查看上半身款传感器

先启动 ESP32 读取，再启动网页：

```bash
cd /home/nb666/HybrIK/data/yifu_banshen/web_visualizer
pip install -r requirements.txt
python3 web_visualizer.py
```

浏览器打开：

```text
http://localhost:5016
```

## 6. 录制 ROS2 bag

```bash
cd /home/nb666/HybrIK/data/yifu_banshen/data
source /opt/ros/humble/setup.bash
ros2 bag record -a -o 1
```

## 7. Bag 转 CSV

```bash
cd /home/nb666/HybrIK/data/yifu_banshen
source /opt/ros/humble/setup.bash
python3 scripts/bag_to_csv.py /home/nb666/HybrIK/data/yifu_banshen/data/1 --max-delta 50
```

输出：

```text
/home/nb666/HybrIK/data/yifu_banshen/data/1/csv_export/synced_dataset.csv
```

## 8. HybrIK 加 SMPL 标签

```bash
conda activate hybrik
cd /home/nb666/HybrIK/data/yifu_banshen
python scripts/csv_add_hybrik.py \
  /home/nb666/HybrIK/data/yifu_banshen/data/1/csv_export/synced_dataset.csv \
  --output-csv /home/nb666/HybrIK/data/yifu_banshen/data/1/csv_export/synced_dataset2.csv \
  --batch-size 4 \
  --num-workers 2
```

如果显存够，可以把 `--batch-size` 调到 `8`；如果爆显存就降到 `2` 或 `1`。

## 9. 图片降分辨率到 640x480

按数据目录处理图片：

```bash
cd /home/nb666/HybrIK/data/yifu_banshen
python3 scripts/resize_csv_images.py \
  /home/nb666/HybrIK/data/yifu_banshen/data/1 \
  /home/nb666/HybrIK/data/yifu_banshen/data/2 \
  /home/nb666/HybrIK/data/yifu_banshen/data/3 \
  --num-workers 4
```

按 CSV 里的 `image_path` 处理图片：

```bash
cd /home/nb666/HybrIK/data/yifu_banshen
python3 scripts/resize_csv_images.py \
  /home/nb666/HybrIK/data/yifu_banshen/data/1/csv_export/synced_dataset2.csv \
  /home/nb666/HybrIK/data/yifu_banshen/data/2/csv_export/synced_dataset2.csv \
  --num-workers 4
```

默认目标尺寸就是 `640x480`。如果要手动指定：

```bash
python3 scripts/resize_csv_images.py /path/to/data_or_csv --width 640 --height 480 --num-workers 4
```

注意：这个脚本会直接覆盖原图片。如果想保留原图，先复制一份数据目录。

## 10. 查看 CSV 信息

```bash
cd /home/nb666/HybrIK/data/yifu_banshen
python3 scripts/csv_dataset_info.py /home/nb666/HybrIK/data/yifu_banshen/data/1/csv_export/synced_dataset2.csv
```

## 11. 合并 CSV

```bash
cd /home/nb666/HybrIK/data/yifu_banshen
python scripts/merge_csv_datasets.py \
  /home/nb666/HybrIK/data/yifu_banshen/data/1/csv_export/synced_dataset2.csv \
  /home/nb666/HybrIK/data/yifu_banshen/data/2/csv_export/synced_dataset2.csv \
  --output-csv /home/nb666/HybrIK/data/yifu_banshen/data/1_2/merged_dataset.csv
```

## 12. 训练 GRU

上半身款默认只训练上半身 SMPL 关节。建议录 bag 时同时启动校准节点，这样训练输入可以使用 10 路原始电容、10 路归一化电容、8 路 IMU 加速度/角速度、8 路相对四元数。

```bash
conda activate hybrik
cd /home/nb666/HybrIK/data/yifu_banshen
python scripts/train_sensor_gru.py \
  /home/nb666/HybrIK/data/yifu_banshen/data/1/csv_export/synced_dataset2.csv \
  --output-dir /home/nb666/HybrIK/data/yifu_banshen/mod/1 \
  --input-groups capacitance normalized imu relative \
  --target-joints upper_body \
  --train-ratio 0.8 \
  --val-ratio 0.1 \
  --test-ratio 0.1 \
  --seq-len 30 \
  --epochs 200 \
  --batch-size 64 \
  --split-mode random_segment
```

`--target-joints upper_body` 会保留 pelvis、spine、neck、collar、shoulder、elbow、wrist、thumb 等上半身关节，跳过 hip/knee/ankle/foot 等下半身关节。实时推理输出到旧 SMPL/GMR 接口时，下半身会自动用中立姿态补齐。

## 13. 实时推理

实时推理推荐走 UDP 链路：

```text
ROS2 传感器话题 -> sensor_udp_streamer.py -> UDP 5010
UDP 5010 -> live_gru_udp_infer.py -> SMPL UDP 5007
SMPL UDP 5007 -> GM/udp_smpl_to_gmr.py
```

先启动 ESP32 读取和校准节点，确认已有：

```text
/sensor/capacitance
/sensor/normalized
/imu/channel_0 ... /imu/channel_7
/imu/relative_transform
```

然后 ROS2 侧发送传感器 UDP：

```bash
cd /home/nb666/HybrIK/data/yifu_banshen/ros2
source /opt/ros/humble/setup.bash
source install/setup.bash
ros2 launch hybrik_nodes sensor_udp_streamer.launch.py
```

HybrIK 环境中运行 GRU 推理。当前目录下已有模型包括 `mod/1_5`、`mod/1_9`、`mod/1_11`、`mod/1_5_7_9`，可以先用 `mod/1_9/best_model.pt`：

```bash
conda activate hybrik
cd /home/nb666/HybrIK/data/yifu_banshen
python scripts/live_gru_udp_infer.py \
  --checkpoint-path /home/nb666/HybrIK/data/yifu_banshen/mod/1_9/best_model.pt \
  --listen-host 0.0.0.0 \
  --listen-port 5010 \
  --send-host 127.0.0.1 \
  --send-port 5007 \
  --device cuda \
  --render-hz 10 \
  --payload-format direct
```

如果机器没有 CUDA，改成：

```text
--device cpu
```

也可以同时发送到多个 UDP 目标：

```bash
python scripts/live_gru_udp_infer.py \
  --checkpoint-path /home/nb666/HybrIK/data/yifu_banshen/mod/1_9/best_model.pt \
  --listen-port 5010 \
  --send-target 127.0.0.1:5007 \
  --send-target 127.0.0.1:5008 \
  --payload-format direct
```

## 14. GMR 重定向

```bash
conda activate gmr
cd /home/nb666/HybrIK/data/yifu_banshen/GM
python udp_smpl_to_gmr.py --robot unitree_g1 --udp-port 5007
```

常用机器人名称来自 GMR 配置。可以根据实际目标替换 `--robot`，例如 `unitree_g1`、`unitree_g1_with_hands`、`kaipu` 等。

如果只想接收 SMPL 并保存结果，可以加保存参数：

```bash
python udp_smpl_to_gmr.py \
  --robot unitree_g1 \
  --udp-port 5007 \
  --save-robot-path /home/nb666/HybrIK/data/yifu_banshen/data/robot_motion.pkl \
  --save-smpl-path /home/nb666/HybrIK/data/yifu_banshen/data/reconstructed_smpl.npz
```

## 15. ROS2 内直接推理

也可以不用 UDP streamer，直接在 ROS2 节点里跑 GRU：

```bash
cd /home/nb666/HybrIK/data/yifu_banshen/ros2
source /opt/ros/humble/setup.bash
source install/setup.bash
ros2 run hybrik_nodes live_gru_smpl_infer.py \
  --ros-args \
  -p checkpoint_path:=/home/nb666/HybrIK/data/yifu_banshen/mod/1_9/best_model.pt \
  -p repo_root:=/home/nb666/HybrIK/data/yifu_banshen/hybrik_runtime \
  -p device:=cuda \
  -p output_topic:=/hybrik/smpl_24
```

该节点发布的 `/hybrik/smpl_24` 是 `Float32MultiArray`，内容顺序是：

```text
transl 3
joint_xyz 24*3
joint_rotmats 24*9
```

注意：`ros2/src/hybrik_nodes/launch/live_gru_smpl_infer.launch.py` 里的默认 checkpoint 可能还指向 `mod/1/best_model.pt`。如果用 launch 启动，先把里面的 `checkpoint_path` 改成当前实际存在的模型路径，例如：

```text
/home/nb666/HybrIK/data/yifu_banshen/mod/1_9/best_model.pt
```

## 16. 当前目录注意事项

- `data/` 当前可能为空。重新训练需要重新采集 bag，或把旧的 CSV/图片数据放回对应目录。
- `hybrik_runtime/` 没有包含 `model_files/smplx/` 和 `pytorch3d/`。当前上半身 SMPL/GRU 流程不直接需要 SMPL-X 大模型；`pytorch3d` 需要由 conda 环境提供。
- 如果实时推理一直提示传感器 payload 不完整，先检查是否已启动校准节点，并确认 `/sensor/normalized`、`/imu/relative_transform` 有数据。
- 如果 `csv_add_hybrik.py` 找不到模型或配置，确认以下文件存在：

```text
/home/nb666/HybrIK/data/yifu_banshen/hybrik_runtime/configs/256x192_adam_lr1e-3-hrw48_cam_2x_w_pw3d_3dhp.yaml
/home/nb666/HybrIK/data/yifu_banshen/hybrik_runtime/pretrained_models/hybrik_hrnet.pth
/home/nb666/HybrIK/data/yifu_banshen/hybrik_runtime/model_files/basicModel_neutral_lbs_10_207_0_v1.0.0.pkl
/home/nb666/HybrIK/data/yifu_banshen/hybrik_runtime/model_files/J_regressor_h36m.npy
/home/nb666/HybrIK/data/yifu_banshen/hybrik_runtime/model_files/h36m_mean_beta.npy
```
