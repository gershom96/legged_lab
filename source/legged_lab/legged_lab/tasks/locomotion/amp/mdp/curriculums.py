from __future__ import annotations

from collections.abc import Sequence

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.terrains import TerrainImporter


def terrain_levels_amp(
    env,
    env_ids: Sequence[int],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    command_name: str = "base_velocity",
    min_command_speed: float = 0.2,
    distance_success_ratio: float = 0.65,
    distance_failure_ratio: float = 0.25,
    terrain_success_ratio: float = 0.45,
    min_lin_vel_tracking: float = 0.6,
    min_ang_vel_tracking: float = 0.2,
    lin_vel_reward_term_name: str = "track_lin_vel_xy_exp",
    ang_vel_reward_term_name: str = "track_ang_vel_z_exp",
    replay_probability: float = 0.15,
    flat_replay_probability: float = 0.05,
    stage_rows: tuple[tuple[int, ...], ...] | None = None,
    current_stage_probability: float = 0.70,
    previous_stage_probability: float = 0.20,
    easy_stage_probability: float = 0.10,
) -> torch.Tensor | dict[str, torch.Tensor]:
    """Move envs up/down terrain rows using survival, distance progress, and tracking quality."""
    terrain: TerrainImporter = env.scene.terrain
    if getattr(terrain, "terrain_origins", None) is None:
        if stage_rows:
            return _terrain_stage_log(torch.zeros(env.num_envs, device=env.device, dtype=torch.long), len(stage_rows))
        return torch.tensor(0.0, device=env.device)

    asset: Articulation = env.scene[asset_cfg.name]
    env_ids = torch.as_tensor(env_ids, device=env.device, dtype=torch.long)
    if env_ids.numel() == 0:
        if stage_rows:
            return _terrain_stage_log(_get_or_create_terrain_stages(env, terrain, stage_rows), len(stage_rows))
        return torch.mean(terrain.terrain_levels.float())

    command = env.command_manager.get_command(command_name)
    command_speed = torch.linalg.norm(command[env_ids, :2], dim=1)
    moving_command = command_speed > min_command_speed

    distance = torch.linalg.norm(asset.data.root_pos_w[env_ids, :2] - env.scene.env_origins[env_ids, :2], dim=1)
    expected_distance = command_speed * env.max_episode_length_s
    terrain_distance = float(terrain.cfg.terrain_generator.size[0]) * terrain_success_ratio
    target_distance = torch.minimum(
        expected_distance * distance_success_ratio,
        torch.full_like(expected_distance, terrain_distance),
    )
    failed_distance = torch.minimum(expected_distance * distance_failure_ratio, target_distance * 0.75)

    lin_tracking = _episode_reward_mean(env, lin_vel_reward_term_name, env_ids)
    ang_tracking = _episode_reward_mean(env, ang_vel_reward_term_name, env_ids)
    tracking_ok = torch.logical_and(lin_tracking >= min_lin_vel_tracking, ang_tracking >= min_ang_vel_tracking)

    terminated = getattr(env, "reset_terminated", None)
    if terminated is None:
        alive = torch.ones_like(moving_command, dtype=torch.bool)
    else:
        alive = ~terminated[env_ids].to(torch.bool)

    move_up = moving_command & alive & tracking_ok & (distance >= target_distance)
    move_down = (~alive) | (moving_command & (distance < failed_distance))
    move_down &= ~move_up

    if stage_rows:
        stages = _get_or_create_terrain_stages(env, terrain, stage_rows)
        stages[env_ids] += move_up.to(stages.dtype) - move_down.to(stages.dtype)
        stages[env_ids] = torch.clamp(stages[env_ids], 0, len(stage_rows) - 1)
        levels = _sample_rows_for_stages(
            stages[env_ids],
            stage_rows,
            current_stage_probability,
            previous_stage_probability,
            easy_stage_probability,
        )
        terrain.terrain_levels[env_ids] = levels
        terrain.env_origins[env_ids] = terrain.terrain_origins[levels, terrain.terrain_types[env_ids]]
        env.scene.env_origins[env_ids] = terrain.env_origins[env_ids]
        return _terrain_stage_log(stages, len(stage_rows))

    terrain.update_env_origins(env_ids, move_up, move_down)
    _apply_lower_level_replay(terrain, env_ids, replay_probability, flat_replay_probability)
    return torch.mean(terrain.terrain_levels.float())


def _terrain_stage_log(stages: torch.Tensor, num_stages: int) -> dict[str, torch.Tensor]:
    stages = stages.to(dtype=torch.long)
    counts = torch.bincount(stages, minlength=num_stages)
    total = max(int(stages.numel()), 1)
    result: dict[str, torch.Tensor] = {"mean_stage": torch.mean(stages.float())}
    for stage in range(num_stages):
        count = counts[stage].float()
        result[f"stage_{stage}_count"] = count
        result[f"stage_{stage}_fraction"] = count / float(total)
    return result


def _episode_reward_mean(env, term_name: str, env_ids: torch.Tensor) -> torch.Tensor:
    reward_manager = getattr(env, "reward_manager", None)
    episode_sums = getattr(reward_manager, "_episode_sums", {})
    if term_name not in episode_sums:
        return torch.zeros(env_ids.numel(), device=env.device)
    return episode_sums[term_name][env_ids] / env.max_episode_length_s


def _apply_lower_level_replay(
    terrain: TerrainImporter,
    env_ids: torch.Tensor,
    replay_probability: float,
    flat_replay_probability: float,
) -> None:
    """Occasionally reset promoted envs onto lower rows to retain easier-terrain behavior."""
    if replay_probability <= 0.0 and flat_replay_probability <= 0.0:
        return

    levels = terrain.terrain_levels[env_ids].clone()
    can_replay = levels > 0
    if replay_probability > 0.0:
        replay_mask = (torch.rand(levels.shape, device=levels.device) < replay_probability) & can_replay
        if torch.any(replay_mask):
            levels[replay_mask] = torch.floor(
                torch.rand_like(levels[replay_mask].float()) * levels[replay_mask].float()
            ).to(levels.dtype)

    if flat_replay_probability > 0.0:
        flat_mask = (torch.rand(levels.shape, device=levels.device) < flat_replay_probability) & can_replay
        levels[flat_mask] = 0

    if not torch.equal(levels, terrain.terrain_levels[env_ids]):
        terrain.terrain_levels[env_ids] = levels
        terrain.env_origins[env_ids] = terrain.terrain_origins[levels, terrain.terrain_types[env_ids]]


def _get_or_create_terrain_stages(
    env,
    terrain: TerrainImporter,
    stage_rows: tuple[tuple[int, ...], ...],
) -> torch.Tensor:
    stages = getattr(env, "_terrain_curriculum_stages", None)
    if stages is not None and stages.shape == terrain.terrain_levels.shape:
        return stages

    row_to_stage = torch.zeros(terrain.max_terrain_level, device=terrain.terrain_levels.device, dtype=torch.long)
    for stage_idx, rows in enumerate(stage_rows):
        for row in rows:
            if 0 <= row < terrain.max_terrain_level:
                row_to_stage[row] = stage_idx

    stages = row_to_stage[terrain.terrain_levels]
    env._terrain_curriculum_stages = stages
    return stages


def _sample_rows_for_stages(
    stages: torch.Tensor,
    stage_rows: tuple[tuple[int, ...], ...],
    current_stage_probability: float,
    previous_stage_probability: float,
    easy_stage_probability: float,
) -> torch.Tensor:
    device = stages.device
    sampled_rows = torch.empty_like(stages)
    roll = torch.rand(stages.shape, device=device)
    probability_sum = max(current_stage_probability + previous_stage_probability + easy_stage_probability, 1e-6)
    current_probability = current_stage_probability / probability_sum
    previous_probability = previous_stage_probability / probability_sum
    current_cutoff = current_probability
    previous_cutoff = current_probability + previous_probability

    for stage_idx in torch.unique(stages).detach().cpu().tolist():
        stage = int(stage_idx)
        stage_mask = stages == stage
        if not torch.any(stage_mask):
            continue

        current_rows = _rows_tensor(stage_rows[stage], device)
        previous_rows = _rows_tensor(stage_rows[max(stage - 1, 0)], device)
        easier_rows = _rows_tensor(
            tuple(row for rows in stage_rows[: max(stage - 1, 1)] for row in rows),
            device,
        )

        current_mask = stage_mask & (roll < current_cutoff)
        previous_mask = stage_mask & (roll >= current_cutoff) & (roll < previous_cutoff)
        easy_mask = stage_mask & (roll >= previous_cutoff)

        sampled_rows[current_mask] = _sample_from_rows(current_rows, int(current_mask.sum().item()))
        sampled_rows[previous_mask] = _sample_from_rows(previous_rows, int(previous_mask.sum().item()))
        sampled_rows[easy_mask] = _sample_from_rows(easier_rows, int(easy_mask.sum().item()))

    return sampled_rows


def _rows_tensor(rows: tuple[int, ...], device: torch.device) -> torch.Tensor:
    if not rows:
        rows = (0,)
    return torch.tensor(rows, device=device, dtype=torch.long)


def _sample_from_rows(rows: torch.Tensor, count: int) -> torch.Tensor:
    if count <= 0:
        return torch.empty(0, device=rows.device, dtype=torch.long)
    indices = torch.randint(rows.numel(), (count,), device=rows.device)
    return rows[indices]
