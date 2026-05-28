# 上半身款传感器网页可视化

这个目录用来可视化上半身款 ESP32 节点发布的数据。原始 `2.py` 已整理为 `ros2/src/hybrik_nodes/scripts/esp32_imu_reader.py`。

## 对应话题

- `/sensor/capacitance`
- `/sensor/acceleration_raw`
- `/sensor/gyroscope_raw`
- `/sensor/magnetic_raw`

## 启动

先确保 `esp32_imu_reader.py` 已经在发布 ROS2 数据，然后启动网页：

```bash
cd /home/nb666/HybrIK/data/yifu_banshen/web_visualizer
pip install -r requirements.txt
python3 web_visualizer.py
```

浏览器打开：`http://localhost:5016`

## 页面内容

- 8 路 IMU 通道切换
- 原始加速度 / 角速度 / 磁场曲线
- 8 路 IMU 最新值总览表
- 10 路电容折线图和最新值表
