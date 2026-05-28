# 新机器人 GMR 重定向移植说明

本文档说明如何把一个新的机器人接入当前 `SMPL-24 -> GMR -> Robot qpos` 流程，并总结这次 Kaipu 调试中遇到的问题和解决方式。

## 核心思路

不要手工把 SMPL 关节角直接映射到机器人关节角。

推荐流程是：

```text
SMPL 24 关节位置/旋转
-> 构造 human_frame
-> GMR 根据机器人 XML 和 ik_config 做全身 IK
-> 输出机器人 qpos
```

入口脚本是：

```bash
/home/nb666/HybrIK/data/yifu_banshen/GM/udp_smpl_to_gmr.py
```

其中 `qpos` 结构是：

```text
qpos[:3]   = 机器人 root position
qpos[3:7] = 机器人 root quaternion, wxyz
qpos[7:]  = 机器人各关节角
```

## 需要新增或修改的文件

### 1. 机器人 MuJoCo XML

准备一个可以被 MuJoCo 正常加载的机器人 XML。

建议放在：

```text
/home/nb666/HybrIK/data/yifu_banshen/GMR/assets/<robot_name>/<robot>.xml
```

也可以放在别的目录，但需要在 `params.py` 中注册正确路径。

### 2. 注册机器人

修改：

```text
/home/nb666/HybrIK/data/yifu_banshen/GMR/general_motion_retargeting/params.py
```

需要添加：

```python
ROBOT_XML_DICT["new_robot"] = ASSET_ROOT / "new_robot" / "new_robot.xml"
IK_CONFIG_DICT["smplx"]["new_robot"] = IK_CONFIG_ROOT / "smplx_to_new_robot.json"
ROBOT_BASE_DICT["new_robot"] = "root_body_name"
VIEWER_CAM_DISTANCE_DICT["new_robot"] = 2.0
```

其中 `root_body_name` 是机器人 XML 里主躯干/root body 的名字。

### 3. 新增 IK 配置

新增文件：

```text
/home/nb666/HybrIK/data/yifu_banshen/GMR/general_motion_retargeting/ik_configs/smplx_to_new_robot.json
```

不要只把文件放到 `/home/nb666/HybrIK/data/yifu_banshen/GM`。`data/GM` 主要放 UDP 接收和调试脚本；真正的机器人 IK 配置应该放在 GMR 的 `ik_configs` 目录。

## IK 配置格式

每个映射项格式是：

```json
"robot_body_name": [
    "human_body_name",
    position_weight,
    rotation_weight,
    position_offset,
    rotation_offset
]
```

含义：

```text
robot_body_name   = 机器人 XML 中的 body 名
human_body_name   = SMPL/SMPL-X 目标人体关节名
position_weight   = 位置约束权重
rotation_weight   = 朝向约束权重
position_offset   = 目标点局部位置偏移
rotation_offset   = 人体朝向到机器人 link 朝向的四元数修正, wxyz
```

常用人体目标点：

```text
pelvis
spine3
left_hip / right_hip
left_knee / right_knee
left_foot / right_foot
left_shoulder / right_shoulder
left_elbow / right_elbow
left_wrist / right_wrist
```

## 推荐选机器人 body 的方式

从机器人 XML 中选择代表 link：

```text
pelvis/root  -> 机器人主躯干 root body
spine3       -> torso / waist / trunk 相关 body
hip          -> hip_roll 或 hip_yaw 这类髋部上游 link
knee         -> knee link
foot         -> ankle_roll / foot / toe / sole link
shoulder     -> shoulder_roll / shoulder_yaw 这类肩部中后段 link
elbow        -> elbow link
wrist        -> wrist / hand link
```

经验：不要优先选太下游或太末端的 hip link。比如 Kaipu 一开始选 `left_hip_pitch/right_hip_pitch` 效果很差，改成 `left_hip_roll/right_hip_roll` 后明显更合理。

## human_scale_table 怎么调

`human_scale_table` 决定人体目标点距离如何缩放到机器人尺寸。

如果 scale 太小：

```text
脚目标离 pelvis 太近
-> IK 为了够到脚目标
-> 机器人膝盖大幅弯曲
```

如果 scale 太大：

```text
脚目标离 pelvis 太远
-> 机器人腿伸直也够不到
-> root/髋/踝出现异常补偿
```

调试建议：

```text
腿总是弯       -> 增大 pelvis/hip/knee/foot scale
腿伸太直够不到 -> 减小 pelvis/hip/knee/foot scale
上肢太短       -> 增大 shoulder/elbow/wrist scale
上肢太长       -> 减小 shoulder/elbow/wrist scale
```

Kaipu 最终下半身 scale 从 `0.62` 提高到接近机器人真实腿长后，膝盖大幅弯曲问题明显改善。

### 用真实 HybrIK CSV 估计 scale

可以用下面脚本从机器人 XML 零姿态和真实 HybrIK CSV 计算初始 `human_scale_table`：

```text
/home/nb666/HybrIK/data/yifu_banshen/GM/estimate_human_scale_table.py
```

第一次从 CSV 生成缓存：

```bash
python /home/nb666/HybrIK/data/yifu_banshen/GM/estimate_human_scale_table.py \
  --robot kaipu \
  --actual-human-height 1.75 \
  --symmetrize \
  --round-digits 2 \
  --keep-existing spine3 \
  --save-smpl-distances-json /home/nb666/HybrIK/data/yifu_banshen/GM/hybrik_smpl_distances.json \
  --smpl-csv /home/nb666/HybrIK/data/yifu_banshen/data/1/csv_export/synced_dataset.csv \
  --smpl-csv /home/nb666/HybrIK/data/yifu_banshen/data/2/csv_export/synced_dataset.csv \
  --json
```

后续复用缓存，不需要再扫 CSV：

```bash
python /home/nb666/HybrIK/data/yifu_banshen/GM/estimate_human_scale_table.py \
  --robot droid_x2 \
  --actual-human-height 1.75 \
  --symmetrize \
  --round-digits 2 \
  --keep-existing spine3 \
  --smpl-distances-json /home/nb666/HybrIK/data/yifu_banshen/GM/hybrik_smpl_distances.json \
  --json
```

参数含义：

```text
--smpl-csv              读取真实 HybrIK joint xyz 列，如 hybrik_left_foot_x_m
--smpl-distances-json   读取已经缓存好的 SMPL pelvis 到各关节距离
--save-smpl-distances-json  把 CSV/NPZ 算出的 SMPL 距离保存成缓存 JSON
--actual-human-height   运行 GMR 时使用的真实人体身高
--symmetrize            左右同名关节取平均，避免机器人左右不对称
--round-digits 2        输出两位小数，方便写进 JSON
--keep-existing spine3  保留当前配置中的 spine3 scale
```

注意：自动估计只适合作为初始值。`spine3` 这类 torso 目标很依赖机器人 body 选择，比如 Kaipu 的 `waist_yaw` 离 root 太近，Droid X2 的 `Spine2_link` 离 root 较远，直接按距离比计算可能明显失真。

## root 姿态修正

如果机器人整体躺下、头朝地、侧躺，不要先乱改每个关节。

先检查 root 姿态修正：

```text
/home/nb666/HybrIK/data/yifu_banshen/GM/udp_smpl_to_gmr.py
```

在 `ROOT_ROTATION_CORRECTION_BY_ROBOT` 中加入机器人特定修正：

```python
"new_robot": (R.from_euler("x", -90.0, degrees=True), True)
```

第二个参数含义：

```text
True  = 同时旋转 root position 和 root rotation，世界系修正
False = 只把修正右乘到 root rotation，机器人局部系修正
```

大多数情况下先试 `True` 的世界系修正。

## 坐标系注意事项

当前 UDP 输入来自 HybrIK/SMPL，代码中约定：

```text
SMPL raw frame: x = left, y = up, z = forward
```

`smpl_udp_common.py` 中有 `RAW_SMPL_TO_GMR_BASIS`，但这次 Kaipu 调试中发现，不能随便给某一个机器人单独在 IK 前启用 basis transform，否则会和现有 G1 成功路径不一致，导致目标姿态和机器人 XML 坐标混乱。

经验：

```text
先复用 G1 当前成功路径
不要轻易加机器人单独 basis transform
如果整体姿态不对，优先调 root correction
如果局部关节不对，再调 ik_config 的 body 选择、scale、rot_offset
```

## 调试顺序

建议按下面顺序调，不要一开始就调所有关节：

```text
1. 确认机器人 XML 能被 MuJoCo 加载
2. 确认 root body 名正确
3. 只看 root 是否站正
4. 看双脚是否大致在地面
5. 看膝盖是否自然
6. 看 torso 是否跟随
7. 看肩、肘、腕是否大致跟随
8. 最后再调每个 link 的 rotation_offset
```

如果机器人完全不动：

```text
检查终端是否有 Failed to process UDP frame
检查 ik_match_table1/table2 是否覆盖 human_scale_table 中所有 human body
检查 robot_body_name 是否真的存在于 XML
检查 JSON 格式是否正确
```

如果机器人抽搐：

```text
检查是否有重复 human_body_name 绑定
检查某些 position_weight 是否过大
检查 rot_offset 是否明显错误
检查 root correction 是否和 IK 坐标冲突
```

如果膝盖一直弯：

```text
优先检查腿部 scale 是否太小
再检查 foot body 是否选错
再检查 knee joint range 是否只允许正向屈膝
最后再看 knee rot_offset 或 joint axis
```

如果机器人头朝地：

```text
优先检查 root correction
其次检查 pelvis/root 的 rotation_offset
不要先改 knee/hip/ankle
```

## Kaipu 这次具体问题和解决

### 问题 1：旧方法完全不对

旧方法在 `kaipu_model` 中手工拆 SMPL 欧拉角，再映射到机器人关节角。

问题：

```text
强依赖坐标轴定义
强依赖欧拉角顺序
左右镜像容易反
没有利用机器人整机运动链
```

解决：

```text
放弃手工关节角映射
改用 GMR 全身 IK
```

### 问题 2：机器人头朝地

原因：root 坐标系和 viewer/control frame 不一致。

解决：在 `udp_smpl_to_gmr.py` 中给 Kaipu 加 root 修正：

```python
"kaipu": (R.from_euler("x", -90.0, degrees=True), True)
```

### 问题 3：只用位置约束后机器人抽搐

原因：GMR 其它可用配置并不是 position-only，而是依赖合理的 `rotation_weight + rotation_offset` 来稳定 link 朝向。

解决：恢复和其它 `smplx_to_xxx.json` 一致的结构，让主要 link 同时使用位置和朝向约束。

### 问题 4：hip body 选错

之前选了：

```text
left_hip_pitch / right_hip_pitch
```

更合理的是：

```text
left_hip_roll / right_hip_roll
```

原因：其它机器人配置通常选择髋链更上游的 roll/yaw link 作为 hip 目标，能让整条腿链更稳定。

### 问题 5：膝盖一直深弯

原因：Kaipu 腿很长，但下半身 scale 太小。脚目标被缩到离 pelvis 太近，IK 只能弯膝盖。

解决：把下半身 scale 从 `0.62` 提到 `0.95`。

## 新机器人移植最小清单

```text
1. 准备 MuJoCo XML
2. 在 params.py 注册 ROBOT_XML_DICT
3. 在 params.py 注册 IK_CONFIG_DICT["smplx"]
4. 在 params.py 注册 ROBOT_BASE_DICT
5. 在 params.py 注册 VIEWER_CAM_DISTANCE_DICT
6. 新建 smplx_to_<robot>.json
7. 先参考 G1/PM01/N1/T1 的结构选 body
8. 调 root correction
9. 调 human_scale_table
10. 调 rotation_offset
```

## 推荐参考文件

优先参考：

```text
/home/nb666/HybrIK/data/yifu_banshen/GMR/general_motion_retargeting/ik_configs/smplx_to_g1.json
/home/nb666/HybrIK/data/yifu_banshen/GMR/general_motion_retargeting/ik_configs/smplx_to_pm01.json
/home/nb666/HybrIK/data/yifu_banshen/GMR/general_motion_retargeting/ik_configs/smplx_to_n1.json
/home/nb666/HybrIK/data/yifu_banshen/GMR/general_motion_retargeting/ik_configs/smplx_to_t1_29dof.json
```

当前接收脚本：

```text
/home/nb666/HybrIK/data/yifu_banshen/GM/udp_smpl_to_gmr.py
```

当前 Kaipu 配置：

```text
/home/nb666/HybrIK/data/yifu_banshen/GMR/general_motion_retargeting/ik_configs/smplx_to_kaipu.json
```
