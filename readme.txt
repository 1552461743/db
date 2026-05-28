上半身款动捕服项目说明
====================

目录：/home/nb666/HybrIK/data/yifu_banshen

这是上半身款动捕服的独立工作区。当前 ESP32 读取程序来自原 `2.py`，已经整理为 ROS2 包中的 `esp32_imu_reader.py`。

主要目录：

- scripts：通用数据处理、HybrIK 标注、GRU 训练和推理脚本。
- ros2：上半身款 ROS2 采集、相机发布和 UDP streamer。
- web_visualizer：查看上半身款 8 路 IMU 和 10 路电容数据的网页。
- data：上半身款采集数据集。
- mod：上半身款训练模型输出。
- GM/GMR：SMPL 到机器人重定向。
- assets：原始 2.py 和 banshen.zip 归档。

数据流：

传感器 + 相机 -> ROS2 bag -> scripts/bag_to_csv.py -> scripts/resize_csv_images.py -> scripts/csv_add_hybrik.py -> scripts/train_sensor_gru.py -> 实时/离线 GRU 推理 -> GM/GMR 重定向。

上半身款传感器：

- 8 路 IMU：/imu/channel_0 到 /imu/channel_7。
- 10 路电容：/sensor/capacitance。
- 原始数组：/sensor/acceleration_raw、/sensor/gyroscope_raw、/sensor/magnetic_raw。
- 校准归一化输出：/sensor/normalized、/imu/relative_transform。

上半身款没有全身款的 per-channel magnetic field 和 yaw。训练时建议录 bag 前启动校准节点，并使用：

python scripts/train_sensor_gru.py ... --input-groups capacitance normalized imu relative --target-joints upper_body

本目录的 train_sensor_gru.py 默认 target-joints=upper_body，只训练上半身 SMPL 关节。实时推理/离线回放为了兼容现有 SMPL/GMR 接口，会把下半身关节用中立姿态补齐。

完整命令见 how_to_use.md。
