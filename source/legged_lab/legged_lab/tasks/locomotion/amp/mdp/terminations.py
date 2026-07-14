from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg

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
