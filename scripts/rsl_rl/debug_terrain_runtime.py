# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Runtime timing harness for terrain, height-scan, and render debugging."""

from __future__ import annotations

import argparse
import sys
import time

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Time IsaacLab env stepping for terrain/perception isolation.")
parser.add_argument("--task", type=str, required=True, help="Task to instantiate.")
parser.add_argument("--agent", type=str, default="rsl_rl_cfg_entry_point", help="Hydra agent config entry point.")
parser.add_argument("--num_envs", type=int, default=512, help="Number of envs.")
parser.add_argument("--steps", type=int, default=1000, help="Number of env steps.")
parser.add_argument("--heartbeat", type=int, default=100, help="Print before/after timing every N steps.")
parser.add_argument("--disable_height_scan", action="store_true", help="Remove height scanner and height_scan obs terms.")
parser.add_argument("--force_terrain_level", type=int, default=None, help="Force all envs onto one terrain row.")
parser.add_argument("--round_robin_terrain_levels", action="store_true", help="Spread envs over all terrain rows.")
parser.add_argument("--disable_curriculum", action="store_true", help="Disable terrain curriculum updates.")
parser.add_argument("--skip_amp_motion", action="store_true", help="Skip AMP motion/animation/disc obs where possible.")
parser.add_argument("--random_actions", action="store_true", help="Use random actions instead of zeros.")
parser.add_argument("--render_every", type=int, default=0, help="Call sim.render() every N steps. 0 disables render.")
parser.add_argument("--read_height_scan", action="store_true", help="Read height scanner tensor after each env.step.")
parser.add_argument("--curriculum_video", action="store_true", help="Use the in-process curriculum video recorder.")
parser.add_argument("--curriculum_video_interval", type=int, default=4800, help="Recorder interval in env steps.")
parser.add_argument("--curriculum_video_length", type=int, default=300, help="Recorder video length in env steps.")

AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

if args_cli.render_every > 0 or args_cli.curriculum_video:
    args_cli.enable_cameras = True

sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


import gymnasium as gym
import torch

from isaaclab.envs import DirectMARLEnv, DirectMARLEnvCfg, DirectRLEnvCfg, ManagerBasedRLEnvCfg, multi_agent_to_single_agent
from isaaclab_tasks.utils.hydra import hydra_task_config

import isaaclab_tasks  # noqa: F401
import legged_lab.tasks  # noqa: F401

from curriculum_video import InProcessCurriculumVideoRecorder


def _remove_height_scan(env_cfg) -> None:
    if hasattr(env_cfg.scene, "height_scanner"):
        env_cfg.scene.height_scanner = None
    observations = getattr(env_cfg, "observations", None)
    for group_name in ("policy", "critic"):
        group_cfg = getattr(observations, group_name, None)
        if group_cfg is not None and hasattr(group_cfg, "height_scan"):
            group_cfg.height_scan = None


def _skip_amp_motion(env_cfg) -> None:
    if hasattr(env_cfg, "motion_data") and hasattr(env_cfg.motion_data, "motion_dataset"):
        env_cfg.motion_data.motion_dataset = None
    if hasattr(env_cfg, "animation") and hasattr(env_cfg.animation, "animation"):
        env_cfg.animation.animation = None
    if hasattr(env_cfg, "observations"):
        if hasattr(env_cfg.observations, "disc"):
            env_cfg.observations.disc = None
        if hasattr(env_cfg.observations, "disc_demo"):
            env_cfg.observations.disc_demo = None


def _disable_curriculum(env_cfg) -> None:
    curriculum = getattr(env_cfg, "curriculum", None)
    if curriculum is not None and hasattr(curriculum, "terrain_levels"):
        curriculum.terrain_levels = None


def _force_terrain_origins(base_env, level: int | None, round_robin: bool) -> None:
    terrain = getattr(base_env.scene, "terrain", None)
    terrain_origins = getattr(terrain, "terrain_origins", None)
    if terrain_origins is None:
        print("[DIAG] no terrain_origins available; terrain forcing skipped", flush=True)
        return

    max_level = int(getattr(terrain, "max_terrain_level", terrain_origins.shape[0]))
    if round_robin:
        levels = torch.arange(base_env.num_envs, device=base_env.device, dtype=terrain.terrain_levels.dtype) % max_level
    elif level is not None:
        if level < 0 or level >= max_level:
            raise ValueError(f"force_terrain_level={level} outside valid range [0, {max_level - 1}]")
        levels = torch.full_like(terrain.terrain_levels, level)
    else:
        return

    terrain.terrain_levels[:] = levels
    terrain.env_origins[:] = terrain.terrain_origins[terrain.terrain_levels, terrain.terrain_types]
    base_env.scene.env_origins[:] = terrain.env_origins
    unique, counts = torch.unique(terrain.terrain_levels, return_counts=True)
    summary = {int(k.item()): int(v.item()) for k, v in zip(unique, counts)}
    print(f"[DIAG] forced terrain levels: {summary}", flush=True)


def _make_actions(base_env):
    action_dim = base_env.action_manager.total_action_dim
    if args_cli.random_actions:
        return 0.2 * (2.0 * torch.rand((base_env.num_envs, action_dim), device=base_env.device) - 1.0)
    return torch.zeros((base_env.num_envs, action_dim), device=base_env.device)


def _maybe_sync_height_scan(base_env) -> None:
    if not args_cli.read_height_scan:
        return
    sensor = base_env.scene.sensors.get("height_scanner")
    if sensor is not None:
        _ = float(sensor.data.ray_hits_w[..., 2].mean().detach().cpu().item())


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, _agent_cfg):
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    env_cfg.seed = 0

    if args_cli.disable_height_scan:
        _remove_height_scan(env_cfg)
    if args_cli.skip_amp_motion:
        _skip_amp_motion(env_cfg)
    if args_cli.disable_curriculum or args_cli.force_terrain_level is not None or args_cli.round_robin_terrain_levels:
        _disable_curriculum(env_cfg)

    print(
        "[DIAG] config: "
        f"task={args_cli.task} num_envs={args_cli.num_envs} steps={args_cli.steps} "
        f"disable_height_scan={args_cli.disable_height_scan} render_every={args_cli.render_every} "
        f"force_level={args_cli.force_terrain_level} round_robin={args_cli.round_robin_terrain_levels}",
        flush=True,
    )

    t0 = time.perf_counter()
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.curriculum_video else None)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)
    base_env = env.unwrapped
    print(f"[DIAG] make_env_s={time.perf_counter() - t0:.3f}", flush=True)

    _force_terrain_origins(base_env, args_cli.force_terrain_level, args_cli.round_robin_terrain_levels)

    t0 = time.perf_counter()
    print("[DIAG] before reset", flush=True)
    env.reset()
    print(f"[DIAG] after reset reset_s={time.perf_counter() - t0:.3f}", flush=True)

    if args_cli.force_terrain_level is not None or args_cli.round_robin_terrain_levels:
        _force_terrain_origins(base_env, args_cli.force_terrain_level, args_cli.round_robin_terrain_levels)
        print("[DIAG] before forced-origin reset", flush=True)
        t0 = time.perf_counter()
        env.reset()
        print(f"[DIAG] after forced-origin reset reset_s={time.perf_counter() - t0:.3f}", flush=True)

    sensor = base_env.scene.sensors.get("height_scanner")
    if sensor is None:
        print("[DIAG] height_scanner: none", flush=True)
    else:
        print(
            "[DIAG] height_scanner: "
            f"rays={sensor.data.ray_hits_w.shape[1]} update_period={sensor.cfg.update_period} "
            f"mesh={sensor.cfg.mesh_prim_paths}",
            flush=True,
        )

    step_times: list[float] = []
    render_times: list[float] = []
    height_times: list[float] = []
    video_times: list[float] = []
    video_recorder = None
    if args_cli.curriculum_video:
        video_recorder = InProcessCurriculumVideoRecorder(
            env,
            log_dir="/tmp/legged_lab_debug_terrain_runtime",
            interval_steps=args_cli.curriculum_video_interval,
            video_length=args_cli.curriculum_video_length,
        )
        print(
            "[DIAG] curriculum_video: "
            f"interval={args_cli.curriculum_video_interval} length={args_cli.curriculum_video_length}",
            flush=True,
        )

    for step in range(args_cli.steps):
        actions = _make_actions(base_env)
        heartbeat = args_cli.heartbeat > 0 and step % args_cli.heartbeat == 0
        if heartbeat:
            print(f"[DIAG] before env.step {step}", flush=True)

        t0 = time.perf_counter()
        env.step(actions)
        step_s = time.perf_counter() - t0
        step_times.append(step_s)

        t0 = time.perf_counter()
        _maybe_sync_height_scan(base_env)
        height_s = time.perf_counter() - t0
        height_times.append(height_s)

        render_s = 0.0
        if args_cli.render_every > 0 and step % args_cli.render_every == 0:
            t0 = time.perf_counter()
            base_env.sim.render()
            render_s = time.perf_counter() - t0
            render_times.append(render_s)

        video_s = 0.0
        if video_recorder is not None:
            t0 = time.perf_counter()
            video_recorder.on_step(step)
            video_s = time.perf_counter() - t0
            video_times.append(video_s)

        if heartbeat:
            print(
                f"[DIAG] after env.step {step} "
                f"step_s={step_s:.6f} height_sync_s={height_s:.6f} "
                f"render_s={render_s:.6f} video_s={video_s:.6f}",
                flush=True,
            )

    def _summary(name: str, values: list[float]) -> None:
        if not values:
            return
        tensor = torch.tensor(values)
        print(
            f"[DIAG] {name}: mean={tensor.mean().item():.6f}s "
            f"p95={tensor.quantile(0.95).item():.6f}s max={tensor.max().item():.6f}s n={len(values)}",
            flush=True,
        )

    _summary("env_step", step_times)
    _summary("height_sync", height_times)
    _summary("render", render_times)
    _summary("curriculum_video", video_times)
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
