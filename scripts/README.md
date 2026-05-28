# scripts

上半身款动捕服的数据处理、训练和推理脚本集中放在这里。

- `bag_to_csv.py`：ROS2 bag 转同步 CSV 和图片。
- `csv_add_hybrik.py`：用 HybrIK 给 CSV 图片追加 SMPL 标签。
- `csv_dataset_info.py`：查看 CSV 帧数和时长。
- `merge_csv_datasets.py`：合并多个 CSV 数据集。
- `resize_csv_images.py`：按目录或 CSV 批量缩放图片。
- `train_sensor_gru.py`：训练传感器序列到 SMPL 姿态的 GRU；本目录默认只训练上半身关节。
- `live_gru_udp_infer.py`：实时接收 UDP 传感器并输出 SMPL。
- `replay_gru_to_smpl_udp.py`：离线回放模型输出到一个或多个 UDP 目标。

推荐从衣服根目录运行，例如：

```bash
cd /home/nb666/HybrIK/data/yifu_banshen
python scripts/bag_to_csv.py data/11 --max-delta 50
```
