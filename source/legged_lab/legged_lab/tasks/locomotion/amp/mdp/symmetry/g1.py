
"""Functions to specify the symmetry in the observation and action space for Unitree G1 29dof."""

from __future__ import annotations

import torch
from tensordict import TensorDict
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from omni.isaac.lab.envs import ManagerBasedRLEnv

# specify the functions that are available for import
__all__ = ["compute_symmetric_states"]

PACKED_ACTOR_FRAME_DIM = 114
BASE_ANG_VEL_DIM = 3
PROJECTED_GRAVITY_DIM = 3
VELOCITY_COMMAND_DIM = 3
G1_29DOF_DIM = 29
KEY_BODY_POS_DIM = 18


@torch.no_grad()
def compute_symmetric_states(
    env: ManagerBasedRLEnv,
    obs: TensorDict | None = None,
    actions: torch.Tensor | None = None,
):
    """Augments the given observations and actions by applying symmetry transformations.

    This function creates augmented versions of the provided observations and actions by applying
    four symmetrical transformations: original, left-right, front-back, and diagonal. The symmetry
    transformations are beneficial for reinforcement learning tasks by providing additional
    diverse data without requiring additional data collection.

    Args:
        env: The environment instance.
        obs: The original observation tensor dictionary. Defaults to None.
        actions: The original actions tensor. Defaults to None.

    Returns:
        Augmented observations and actions tensors, or None if the respective input was None.
    """
    if obs is not None:
        batch_size = obs.batch_size[0]
        # since we have 2 different symmetries, we need to augment the batch size by 2
        obs_aug = obs.repeat(2)
        
        # policy observation group
        # -- original
        obs_aug["policy"][:batch_size] = obs["policy"][:]
        # -- left-right
        obs_aug["policy"][batch_size:2*batch_size] = _transform_policy_obs_left_right(
            env.unwrapped, obs["policy"][:]
        )
    else:
        obs_aug = None 
        
    if actions is not None:
        batch_size = actions.shape[0]
        # since we have 2 different symmetries, we need to augment the batch size by 2
        actions_aug = torch.zeros(batch_size * 2, actions.shape[1], device=actions.device)
        # -- original
        actions_aug[:batch_size] = actions[:]
        # -- left-right
        actions_aug[batch_size : 2 * batch_size] = _transform_actions_left_right(actions)
    else:
        actions_aug = None
        
    return obs_aug, actions_aug


"""
Symmetry functions for observations.
"""


def _transform_policy_obs_left_right(env: ManagerBasedRLEnv, obs: torch.Tensor) -> torch.Tensor:
    """Apply a left-right symmetry transformation to the observation tensor.

    This function modifies the given observation tensor by applying transformations
    that represent a symmetry with respect to the left-right axis. This includes
    negating selected components of angular velocity, projected gravity, velocity
    commands, and swapping the G1 joint positions, joint velocities, last actions,
    and key body positions. If height-scan data is present, it is flipped along the
    left-right axis.

    Args:
        env: The environment instance from which the observation is obtained.
        obs: The observation tensor to be transformed.

    Returns:
        The transformed observation tensor with left-right symmetry applied.
    """
    obs = obs.clone()
    device = obs.device

    term_names = env.observation_manager.active_terms["policy"]
    term_dims = env.observation_manager.group_obs_term_dim["policy"]

    start_idx = 0
    for term_name, term_dim in zip(term_names, term_dims):
        end_idx = start_idx + int(torch.tensor(term_dim).prod().item())
        term_obs = obs[:, start_idx:end_idx]

        if term_name == "base_ang_vel":
            term_obs = _apply_repeated_sign_flip(term_obs, torch.tensor([-1, 1, -1], device=device))
        elif term_name == "projected_gravity":
            term_obs = _apply_repeated_sign_flip(term_obs, torch.tensor([1, -1, 1], device=device))
        elif term_name == "velocity_commands":
            term_obs = _apply_repeated_sign_flip(term_obs, torch.tensor([1, -1, -1], device=device))
        elif term_name in {"joint_pos", "joint_vel", "actions"}:
            term_obs = _switch_g1_29dof_joints_left_right(term_obs)
        elif term_name == "key_body_pos_b":
            term_obs = _switch_g1_29dof_key_body_pos_left_right(term_obs)
        elif term_name == "height_scan":
            term_obs = _flip_height_scan_left_right(env, term_obs)
        elif term_name == "packed_actor_obs":
            term_obs = _transform_packed_actor_obs_left_right(term_obs)

        obs[:, start_idx:end_idx] = term_obs
        start_idx = end_idx

    return obs


def _transform_packed_actor_obs_left_right(obs: torch.Tensor) -> torch.Tensor:
    """Mirror packed [full_obs_t-k, ..., full_obs_t] actor observations."""
    if obs.shape[-1] % PACKED_ACTOR_FRAME_DIM != 0:
        raise ValueError(
            f"Packed actor obs dim {obs.shape[-1]} is not divisible by "
            f"{PACKED_ACTOR_FRAME_DIM}."
        )

    original_shape = obs.shape
    obs = obs.reshape(-1, PACKED_ACTOR_FRAME_DIM)

    start = 0
    end = start + BASE_ANG_VEL_DIM
    obs[:, start:end] = obs[:, start:end] * torch.tensor([-1, 1, -1], device=obs.device)

    start = end
    end = start + PROJECTED_GRAVITY_DIM
    obs[:, start:end] = obs[:, start:end] * torch.tensor([1, -1, 1], device=obs.device)

    start = end
    end = start + VELOCITY_COMMAND_DIM
    obs[:, start:end] = obs[:, start:end] * torch.tensor([1, -1, -1], device=obs.device)

    start = end
    end = start + G1_29DOF_DIM
    obs[:, start:end] = _switch_g1_29dof_joints_left_right(obs[:, start:end])

    start = end
    end = start + G1_29DOF_DIM
    obs[:, start:end] = _switch_g1_29dof_joints_left_right(obs[:, start:end])

    start = end
    end = start + G1_29DOF_DIM
    obs[:, start:end] = _switch_g1_29dof_joints_left_right(obs[:, start:end])

    start = end
    end = start + KEY_BODY_POS_DIM
    obs[:, start:end] = _switch_g1_29dof_key_body_pos_left_right(obs[:, start:end])

    if end != PACKED_ACTOR_FRAME_DIM:
        raise RuntimeError(f"Internal packed actor obs layout error: consumed {end} dims.")

    return obs.reshape(original_shape)


def _apply_repeated_sign_flip(obs: torch.Tensor, signs: torch.Tensor) -> torch.Tensor:
    """Apply a sign pattern to current or flattened-history observations."""
    feature_dim = signs.numel()
    if obs.shape[-1] % feature_dim != 0:
        raise ValueError(f"Observation dim {obs.shape[-1]} is not divisible by feature dim {feature_dim}.")
    return (obs.reshape(obs.shape[0], -1, feature_dim) * signs.reshape(1, 1, feature_dim)).reshape_as(obs)


def _flip_height_scan_left_right(env: ManagerBasedRLEnv, height_scan: torch.Tensor) -> torch.Tensor:
    """Flip flattened height-scan observations across the robot's left-right axis."""
    sensors = getattr(env.scene, "sensors", {})
    sensor = sensors.get("height_scanner") if hasattr(sensors, "get") else None
    if sensor is None or not hasattr(sensor.cfg, "shape"):
        return height_scan

    x_len, y_len = sensor.cfg.shape
    map_dim = x_len * y_len
    if height_scan.shape[-1] % map_dim != 0:
        raise ValueError(f"Height-scan dim {height_scan.shape[-1]} is not divisible by scanner map dim {map_dim}.")

    ordering = getattr(sensor.cfg.pattern_cfg, "ordering", "xy")
    if ordering == "yx":
        return height_scan.reshape(height_scan.shape[0], -1, x_len, y_len).flip(dims=[3]).reshape_as(height_scan)
    if ordering == "xy":
        height_scan_xy = height_scan.reshape(height_scan.shape[0], -1, y_len, x_len).transpose(-1, -2)
        height_scan_xy = height_scan_xy.flip(dims=[3])
        return height_scan_xy.transpose(-1, -2).reshape_as(height_scan)
    raise ValueError(f"Unsupported height-scan ordering: {ordering}.")


"""
Symmetry functions for actions.
"""


def _transform_actions_left_right(actions: torch.Tensor) -> torch.Tensor:
    """Applies a left-right symmetry transformation to the actions tensor.

    This function modifies the given actions tensor by swapping the G1 left and
    right joint action components and negating mirrored axes where needed.

    Args:
        actions: The actions tensor to be transformed.

    Returns:
        The transformed actions tensor with left-right symmetry applied.
    """
    actions = actions.clone()
    actions[:] = _switch_g1_29dof_joints_left_right(actions[:])
    return actions



"""
Lab joint names:
 0 - left_hip_pitch_joint
 1 - right_hip_pitch_joint
 2 - waist_yaw_joint
 3 - left_hip_roll_joint
 4 - right_hip_roll_joint
 5 - waist_roll_joint
 6 - left_hip_yaw_joint
 7 - right_hip_yaw_joint
 8 - waist_pitch_joint
 9 - left_knee_joint
10 - right_knee_joint
11 - left_shoulder_pitch_joint
12 - right_shoulder_pitch_joint
13 - left_ankle_pitch_joint
14 - right_ankle_pitch_joint
15 - left_shoulder_roll_joint
16 - right_shoulder_roll_joint
17 - left_ankle_roll_joint
18 - right_ankle_roll_joint
19 - left_shoulder_yaw_joint
20 - right_shoulder_yaw_joint
21 - left_elbow_joint
22 - right_elbow_joint
23 - left_wrist_roll_joint
24 - right_wrist_roll_joint
25 - left_wrist_pitch_joint
26 - right_wrist_pitch_joint
27 - left_wrist_yaw_joint
28 - right_wrist_yaw_joint
"""

def _switch_g1_29dof_joints_left_right(joint_data: torch.Tensor) -> torch.Tensor:
    """Applies a left-right symmetry transformation to the joint data tensor."""
    feature_dim = 29
    if joint_data.shape[-1] % feature_dim != 0:
        raise ValueError(f"Joint data dim {joint_data.shape[-1]} is not divisible by {feature_dim}.")
    original_shape = joint_data.shape
    joint_data = joint_data.reshape(-1, feature_dim)
    joint_data_switched = _switch_g1_29dof_joints_left_right_single(joint_data)
    return joint_data_switched.reshape(original_shape)


def _switch_g1_29dof_joints_left_right_single(joint_data: torch.Tensor) -> torch.Tensor:
    """Apply left-right symmetry to one 29-DoF joint/action vector per row."""
    joint_data_switched = torch.zeros_like(joint_data)
    
    # Indices for left and right joints
    left_indices = [0, 3, 6, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27]
    right_indices = [1, 4, 7, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28]
    
    # Indices for roll and yaw joints that need sign flipping
    roll_indices = [3, 4, 15, 16, 17, 18, 23, 24]
    yaw_indices = [6, 7, 19, 20, 27, 28]

    # Copy non-symmetric joints first (waist joints)
    joint_data_switched[..., [2, 5, 8]] = joint_data[..., [2, 5, 8]]

    # Swap left and right joints
    joint_data_switched[..., left_indices] = joint_data[..., right_indices]
    joint_data_switched[..., right_indices] = joint_data[..., left_indices]

    # Flip the sign of roll and yaw joints
    joint_data_switched[..., roll_indices] *= -1.0
    joint_data_switched[..., yaw_indices] *= -1.0
    
    # Flip the sign of waist_yaw, waist_roll
    joint_data_switched[..., [2, 5]] *= -1.0
    
    return joint_data_switched


def _switch_g1_29dof_key_body_pos_left_right(key_body_pos: torch.Tensor) -> torch.Tensor:
    """Applies a left-right symmetry transformation to the key body positions tensor."""
    
    # We assume that the key body are in pair, for example:
    # "left_ankle_roll_link", 
    # "right_ankle_roll_link",
    # "left_wrist_yaw_link",
    # "right_wrist_yaw_link",
    # "left_shoulder_roll_link",
    # "right_shoulder_roll_link",
    
    feature_dim = 18
    if key_body_pos.shape[-1] % feature_dim != 0:
        raise ValueError(f"Key-body position dim {key_body_pos.shape[-1]} is not divisible by {feature_dim}.")
    original_shape = key_body_pos.shape
    key_body_pos = key_body_pos.reshape(-1, feature_dim)
    key_body_pos_switched = key_body_pos.clone()
    num_key_bodies = feature_dim // 3
    
    for i in range(num_key_bodies // 2):
        left_idx = i * 2
        right_idx = i * 2 + 1
        
        # Swap left and right key body positions
        key_body_pos_switched[..., left_idx * 3 : left_idx * 3 + 3] = key_body_pos[..., right_idx * 3 : right_idx * 3 + 3]
        key_body_pos_switched[..., right_idx * 3 : right_idx * 3 + 3] = key_body_pos[..., left_idx * 3 : left_idx * 3 + 3]
        
        # Flip the y-coordinate to reflect left-right symmetry
        key_body_pos_switched[..., left_idx * 3 + 1] *= -1.0
        key_body_pos_switched[..., right_idx * 3 + 1] *= -1.0
    
    return key_body_pos_switched.reshape(original_shape)
    
    
    
