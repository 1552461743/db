# 上半身款 GMR 使用命令

## 1. 回放到 Piper 机械臂

启动 Piper GMR 接收端：

```bash
conda activate gmr
cd /home/nb666/HybrIK/data/yifu_banshen/GM
python udp_smpl_to_gmr.py --robot piper --udp-port 5007
```

Piper 默认每帧做 8 次 IK 收敛。如果机械臂仍然反应慢，可以加大：

```bash
python udp_smpl_to_gmr.py --robot piper --udp-port 5007 --retarget-passes 16
```

另开终端回放上半身 GRU 到 Piper：

```bash
conda activate hybrik
cd /home/nb666/HybrIK/data/yifu_banshen
python scripts/replay_gru_to_smpl_udp.py --udp-target 127.0.0.1:5007
```

如果要指定模型和 CSV：

```bash
conda activate hybrik
cd /home/nb666/HybrIK/data/yifu_banshen
python scripts/replay_gru_to_smpl_udp.py \
  --model 1 \
  --csv /home/nb666/HybrIK/data/yifu_banshen/data/1/csv_export/synced_dataset2.csv \
  --udp-target 127.0.0.1:5007
```

Piper 当前只做末端执行器映射：`right_shoulder -> link1` 作为底座参考，`right_wrist -> link6` 作为末端目标；不约束手肘。人体肩到腕长度到机械臂底座到末端长度的比例在 `GMR/general_motion_retargeting/ik_configs/smplx_to_piper.json` 的 `human_scale_table.right_wrist` 里调。当前 `right_wrist=0.9`，如果机械臂仍然显得长，继续降到 `0.8` 或 `0.75`。

## 2. 回放上半身 GRU 输出到多个 UDP 端口

从 `yifu_banshen` 根目录运行：

```bash
conda activate hybrik
cd /home/nb666/HybrIK/data/yifu_banshen
python scripts/replay_gru_to_smpl_udp.py \
  --udp-target 127.0.0.1:5007 \
  --udp-target 127.0.0.1:5008 \
  --udp-target 127.0.0.1:5009
```

如果要指定模型和 CSV：

```bash
conda activate hybrik
cd /home/nb666/HybrIK/data/yifu_banshen
python scripts/replay_gru_to_smpl_udp.py \
  --model 1 \
  --csv /home/nb666/HybrIK/data/yifu_banshen/data/1/csv_export/synced_dataset2.csv \
  --udp-target 127.0.0.1:5007 \
  --udp-target 127.0.0.1:5008 \
  --udp-target 127.0.0.1:5009
```

注意：不要写 `../scripts/replay_gru_to_smpl_udp.py`。从 `yifu_banshen` 根目录运行时，正确路径是 `scripts/replay_gru_to_smpl_udp.py`。

## 3. 启动其他 GMR 接收端

Unitree G1：

```bash
conda activate gmr
cd /home/nb666/HybrIK/data/yifu_banshen/GM
python udp_smpl_to_gmr.py --robot unitree_g1 --udp-port 5007
```

Kaipu：

```bash
conda activate gmr
cd /home/nb666/HybrIK/data/yifu_banshen/GM
python udp_smpl_to_gmr.py --robot kaipu --udp-port 5008
```

Droid X2：

```bash
conda activate gmr
cd /home/nb666/HybrIK/data/yifu_banshen/GM
python udp_smpl_to_gmr.py --robot droid_x2 --udp-port 5009
```
