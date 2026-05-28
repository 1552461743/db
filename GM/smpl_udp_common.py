from __future__ import annotations

import numpy as np

SMPL_UDP_FORMAT = "smpl_24_xyz_rotmat"

SMPL_JOINT_NAMES = [
    "pelvis",
    "left_hip",
    "right_hip",
    "spine1",
    "left_knee",
    "right_knee",
    "spine2",
    "left_ankle",
    "right_ankle",
    "spine3",
    "left_foot",
    "right_foot",
    "neck",
    "left_collar",
    "right_collar",
    "jaw",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_thumb",
    "right_thumb",
]

SMPL_INDEX = {name: index for index, name in enumerate(SMPL_JOINT_NAMES)}

# HybrIK/SMPL payload frame is treated as:
# x = left, y = up, z = forward
#
# GMR's existing real-time integrations convert y-up data into a right-handed
# z-up frame using:
#   x' = right = -left
#   y' = back  = -forward
#   z' = up    = up
#
# This matches the transform used in `general_motion_retargeting/xrobot_utils.py`
# except for the extra X sign flip required because SMPL uses +X to the left.
RAW_SMPL_TO_GMR_BASIS = np.array(
    [
        [-1.0, 0.0, 0.0],
        [0.0, 0.0, -1.0],
        [0.0, 1.0, 0.0],
    ],
    dtype=np.float64,
)
GMR_TO_RAW_SMPL_BASIS = RAW_SMPL_TO_GMR_BASIS.T
