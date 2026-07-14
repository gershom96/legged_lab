from __future__ import annotations

import numpy as np


ISAAC_JOINT_NAMES = (
    "left_hip_pitch_joint",
    "right_hip_pitch_joint",
    "waist_yaw_joint",
    "left_hip_roll_joint",
    "right_hip_roll_joint",
    "waist_roll_joint",
    "left_hip_yaw_joint",
    "right_hip_yaw_joint",
    "waist_pitch_joint",
    "left_knee_joint",
    "right_knee_joint",
    "left_shoulder_pitch_joint",
    "right_shoulder_pitch_joint",
    "left_ankle_pitch_joint",
    "right_ankle_pitch_joint",
    "left_shoulder_roll_joint",
    "right_shoulder_roll_joint",
    "left_ankle_roll_joint",
    "right_ankle_roll_joint",
    "left_shoulder_yaw_joint",
    "right_shoulder_yaw_joint",
    "left_elbow_joint",
    "right_elbow_joint",
    "left_wrist_roll_joint",
    "right_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "right_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_wrist_yaw_joint",
)

MUJOCO_JOINT_NAMES = (
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
)

# Gather indices are named for the order of the resulting vector. For example,
# ``action_isaac[ISAAC_INDICES_IN_MUJOCO_ORDER]`` is in MuJoCo actuator order.
MUJOCO_INDICES_IN_ISAAC_ORDER = np.asarray(
    [MUJOCO_JOINT_NAMES.index(name) for name in ISAAC_JOINT_NAMES], dtype=np.int64
)
ISAAC_INDICES_IN_MUJOCO_ORDER = np.asarray(
    [ISAAC_JOINT_NAMES.index(name) for name in MUJOCO_JOINT_NAMES], dtype=np.int64
)

# Compatibility aliases for checkpoints/tools written before the explicit names.
MUJOCO_TO_ISAAC = MUJOCO_INDICES_IN_ISAAC_ORDER
ISAAC_TO_MUJOCO = ISAAC_INDICES_IN_MUJOCO_ORDER

KEY_BODY_NAMES = (
    "left_ankle_roll_link",
    "right_ankle_roll_link",
    "left_wrist_yaw_link",
    "right_wrist_yaw_link",
    "left_shoulder_roll_link",
    "right_shoulder_roll_link",
)
UPPER_KEY_BODY_INDICES = (2, 3, 4, 5)

LOWER_ACTION_INDICES = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 13, 14, 17, 18)
UPPER_ACTION_INDICES = (11, 12, 15, 16, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28)
ANKLE_ACTION_INDICES = (13, 14, 17, 18)

DEFAULT_JOINT_POS_ISAAC = np.zeros(29, dtype=np.float32)
DEFAULT_JOINT_POS_ISAAC[[0, 1]] = -0.1
DEFAULT_JOINT_POS_ISAAC[[9, 10]] = 0.3
DEFAULT_JOINT_POS_ISAAC[[13, 14]] = -0.2
DEFAULT_JOINT_POS_ISAAC[[11, 12]] = 0.3
DEFAULT_JOINT_POS_ISAAC[15] = 0.25
DEFAULT_JOINT_POS_ISAAC[16] = -0.25
DEFAULT_JOINT_POS_ISAAC[[21, 22]] = 0.97
DEFAULT_JOINT_POS_ISAAC[23] = 0.15
DEFAULT_JOINT_POS_ISAAC[24] = -0.15


def _joint_parameters() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    stiffness = np.zeros(29, dtype=np.float32)
    damping = np.zeros(29, dtype=np.float32)
    effort = np.zeros(29, dtype=np.float32)
    for index, name in enumerate(ISAAC_JOINT_NAMES):
        if "hip_pitch" in name or "hip_yaw" in name:
            stiffness[index], damping[index], effort[index] = 100.0, 2.0, 88.0
        elif "hip_roll" in name:
            stiffness[index], damping[index], effort[index] = 100.0, 2.0, 139.0
        elif "knee" in name:
            stiffness[index], damping[index], effort[index] = 150.0, 4.0, 139.0
        elif name == "waist_yaw_joint":
            stiffness[index], damping[index], effort[index] = 200.0, 5.0, 88.0
        elif name in {"waist_roll_joint", "waist_pitch_joint"}:
            stiffness[index], damping[index], effort[index] = 40.0, 5.0, 25.0
        elif "wrist_pitch" in name or "wrist_yaw" in name:
            stiffness[index], damping[index], effort[index] = 40.0, 1.0, 5.0
        elif "ankle" in name:
            stiffness[index], damping[index], effort[index] = 40.0, 2.0, 25.0
        else:
            stiffness[index], damping[index], effort[index] = 40.0, 1.0, 25.0
    return stiffness, damping, effort


STIFFNESS_ISAAC, DAMPING_ISAAC, EFFORT_LIMIT_ISAAC = _joint_parameters()
ACTION_SCALE = 0.25
PACKED_ACTOR_FRAME_DIM = 114
CRITIC_STATE_DIM = 117
