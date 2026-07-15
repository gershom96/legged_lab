from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import RayCaster

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def terrain_out_of_tile(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    allowed_half_extent: float = 20.0,
) -> torch.Tensor:
    """Terminate when the robot leaves its assigned terrain tile region."""
    if env.scene.cfg.terrain.terrain_type != "generator":
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    asset: RigidObject = env.scene[asset_cfg.name]
    local_xy = asset.data.root_pos_w[:, :2] - env.scene.env_origins[:, :2]
    return torch.any(torch.abs(local_xy) > allowed_half_extent, dim=1)


def root_height_below_terrain(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("height_scanner"),
    minimum_clearance: float = 0.2,
) -> torch.Tensor:
    """Terminate when root height is too low relative to the local terrain surface."""
    asset: RigidObject = env.scene[asset_cfg.name]
    sensor: RayCaster = env.scene.sensors[sensor_cfg.name]
    center_ray_id = sensor.data.ray_hits_w.shape[1] // 2
    terrain_z = sensor.data.ray_hits_w[:, center_ray_id, 2]
    clearance = asset.data.root_pos_w[:, 2] - terrain_z
    return clearance < minimum_clearance
