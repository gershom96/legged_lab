# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Visualize terrain height-scan observations for a few environments."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Dump and visualize height-scan tensors for an IsaacLab task.")
parser.add_argument("--task", type=str, required=True, help="Task name with a height_scanner sensor.")
parser.add_argument("--num_envs", type=int, default=4, help="Number of environments to create.")
parser.add_argument("--env_ids", type=str, default="0,1,2,3", help="Comma-separated env ids to visualize.")
parser.add_argument("--steps", type=int, default=2, help="Number of zero-action steps before dumping scans.")
parser.add_argument("--output_dir", type=str, default="logs/height_scan_debug", help="Directory for PNG/CSV outputs.")
parser.add_argument("--offset", type=float, default=0.5, help="Height-scan offset used by mdp.height_scan.")
parser.add_argument(
    "--env_spacing",
    type=float,
    default=None,
    help="Override scene env spacing. Use a small value to put neighboring robots inside each other's scan footprint.",
)
parser.add_argument(
    "--deterministic_reset",
    action="store_true",
    help="Reset bases and joints to deterministic defaults for sensor debugging.",
)
parser.add_argument("--term_stats", action="store_true", help="Print per-term policy and critic observation stats.")
parser.add_argument("--debug_vis", action="store_true", help="Enable Isaac ray-caster debug visualization.")
parser.add_argument("--pause", action="store_true", help="Keep the simulator open after writing outputs.")
parser.add_argument(
    "--load_amp_motion",
    action="store_true",
    help="Load AMP motion/animation managers. Off by default because height-scan visualization does not need them.",
)
parser.add_argument("--agent", type=str, default="rsl_rl_cfg_entry_point", help="Hydra agent config entry point.")
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


import csv
import math
import time

import gymnasium as gym
import numpy as np
import torch

from isaaclab.envs import DirectMARLEnv, DirectMARLEnvCfg, DirectRLEnvCfg, ManagerBasedRLEnvCfg, multi_agent_to_single_agent
from isaaclab_tasks.utils.hydra import hydra_task_config

import isaaclab_tasks  # noqa: F401
import legged_lab.tasks  # noqa: F401


def _parse_env_ids(env_ids: str, num_envs: int) -> list[int]:
    parsed = [int(item.strip()) for item in env_ids.split(",") if item.strip()]
    return [env_id for env_id in parsed if 0 <= env_id < num_envs]


def _height_scan_shape(base_env) -> tuple[int, int]:
    sensor = base_env.scene.sensors["height_scanner"]
    if hasattr(sensor.cfg, "shape") and sensor.cfg.shape[0] > 0 and sensor.cfg.shape[1] > 0:
        return int(sensor.cfg.shape[0]), int(sensor.cfg.shape[1])
    map_dim = int(sensor.data.ray_hits_w.shape[1])
    side = int(math.sqrt(map_dim))
    if side * side == map_dim:
        return side, side
    return 1, map_dim


def _raw_height_scan(base_env, offset: float) -> torch.Tensor:
    sensor = base_env.scene.sensors["height_scanner"]
    return sensor.data.pos_w[:, 2].unsqueeze(1) - sensor.data.ray_hits_w[..., 2] - offset


def _reshape_scan_for_image(base_env, values: torch.Tensor) -> torch.Tensor:
    sensor = base_env.scene.sensors["height_scanner"]
    x_len, y_len = _height_scan_shape(base_env)
    ordering = getattr(sensor.cfg.pattern_cfg, "ordering", "xy")
    if ordering == "yx":
        return values.reshape(x_len, y_len)
    if ordering == "xy":
        return values.reshape(y_len, x_len).transpose(0, 1)
    raise ValueError(f"Unsupported height-scan ordering: {ordering}.")


def _policy_height_scan(base_env) -> torch.Tensor | None:
    obs_manager = base_env.observation_manager
    if "policy" not in obs_manager.active_terms:
        return None
    if "height_scan" not in obs_manager.active_terms["policy"]:
        return None
    obs = obs_manager.compute_group("policy", update_history=False)
    term_names = obs_manager.active_terms["policy"]
    term_dims = obs_manager.group_obs_term_dim["policy"]
    start = 0
    for term_name, term_dim in zip(term_names, term_dims):
        width = int(np.prod(term_dim))
        if term_name == "height_scan":
            return obs[:, start : start + width]
        start += width
    return None


def _save_heatmap(values: np.ndarray, path: Path, title: str) -> bool:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        data_range = float(values.max() - values.min())
        vmin = vmax = None
        if data_range < 1.0e-4:
            center = float(values.mean())
            vmin = center - 0.1
            vmax = center + 0.1

        fig, ax = plt.subplots(figsize=(6, 4), constrained_layout=True)
        im = ax.imshow(values.T, origin="lower", cmap="viridis", aspect="equal", vmin=vmin, vmax=vmax)
        ax.set_title(f"{title}\nmin={values.min():.6f}, max={values.max():.6f}, range={data_range:.2e}")
        ax.set_xlabel("x index")
        ax.set_ylabel("y index")
        fig.colorbar(im, ax=ax, label="height scan value")
        fig.savefig(path, dpi=180)
        plt.close(fig)
        return True
    except Exception as exc:
        print(f"[WARN] Could not write PNG {path}: {type(exc).__name__}: {exc}")
        return False


def _write_outputs(base_env, output_dir: Path, env_ids: list[int], offset: float) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    shape = _height_scan_shape(base_env)
    raw = _raw_height_scan(base_env, offset).detach().cpu()
    policy = _policy_height_scan(base_env)
    policy = policy.detach().cpu() if policy is not None else None

    print(f"[INFO] Height-scan shape: {shape[0]} x {shape[1]} ({shape[0] * shape[1]} rays)")
    print(f"[INFO] Raycast mesh paths: {base_env.scene.sensors['height_scanner'].cfg.mesh_prim_paths}")

    summary_path = output_dir / "height_scan_summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "env_id",
                "kind",
                "min",
                "mean",
                "max",
                "std",
                "center",
                "file_csv",
                "file_png",
            ],
        )
        writer.writeheader()
        for env_id in env_ids:
            for kind, tensor in (("raw", raw), ("policy", policy)):
                if tensor is None:
                    continue
                flat = tensor[env_id]
                if flat.numel() != shape[0] * shape[1]:
                    print(f"[WARN] Skipping {kind} env {env_id}: dim {flat.numel()} does not match {shape}.")
                    continue
                values = _reshape_scan_for_image(base_env, flat).numpy()
                csv_path = output_dir / f"env_{env_id:03d}_{kind}.csv"
                png_path = output_dir / f"env_{env_id:03d}_{kind}.png"
                np.savetxt(csv_path, values, delimiter=",", fmt="%.8f")
                wrote_png = _save_heatmap(values, png_path, f"env {env_id} {kind} height scan")
                center = float(values[shape[0] // 2, shape[1] // 2])
                writer.writerow(
                    {
                        "env_id": env_id,
                        "kind": kind,
                        "min": float(values.min()),
                        "mean": float(values.mean()),
                        "max": float(values.max()),
                        "std": float(values.std()),
                        "center": center,
                        "file_csv": str(csv_path),
                        "file_png": str(png_path) if wrote_png else "",
                    }
                )
                print(
                    f"[INFO] env={env_id:03d} {kind:6s} "
                    f"min={values.min(): .4f} mean={values.mean(): .4f} "
                    f"max={values.max(): .4f} std={values.std(): .4f} center={center: .4f}"
                )
    print(f"[INFO] Wrote summary: {summary_path}")


def _print_group_term_stats(base_env, group_name: str) -> None:
    obs_manager = base_env.observation_manager
    if group_name not in obs_manager.active_terms:
        return
    obs = obs_manager.compute_group(group_name, update_history=False)
    term_names = obs_manager.active_terms[group_name]
    term_dims = obs_manager.group_obs_term_dim[group_name]
    start = 0
    print(f"[INFO] {group_name} observation term stats:")
    for term_name, term_dim in zip(term_names, term_dims):
        width = int(np.prod(term_dim))
        values = obs[:, start : start + width].detach().cpu()
        start += width
        print(
            f"[INFO]   {term_name:22s} shape={tuple(term_dim)!s:12s} "
            f"min={values.min().item(): .5f} mean={values.mean().item(): .5f} "
            f"max={values.max().item(): .5f} std={values.std(unbiased=False).item(): .5f} "
            f"abs_p95={values.abs().quantile(0.95).item(): .5f}"
        )


def _disable_amp_motion_for_sensor_debug(env_cfg) -> None:
    """Avoid loading expert motion data when only the terrain sensor is being inspected."""
    if hasattr(env_cfg, "motion_data") and hasattr(env_cfg.motion_data, "motion_dataset"):
        env_cfg.motion_data.motion_dataset = None
    if hasattr(env_cfg, "animation") and hasattr(env_cfg.animation, "animation"):
        env_cfg.animation.animation = None
    if hasattr(env_cfg, "events") and hasattr(env_cfg.events, "reset_from_ref"):
        env_cfg.events.reset_from_ref = None
    if hasattr(env_cfg, "observations"):
        if hasattr(env_cfg.observations, "disc"):
            env_cfg.observations.disc = None
        if hasattr(env_cfg.observations, "disc_demo"):
            env_cfg.observations.disc_demo = None


def _make_reset_deterministic(env_cfg) -> None:
    """Remove reset noise so close-packed envs are easy to inspect."""
    events = getattr(env_cfg, "events", None)
    reset_base = getattr(events, "reset_base", None)
    if reset_base is not None:
        reset_base.params["pose_range"] = {"x": (0.0, 0.0), "y": (0.0, 0.0), "yaw": (0.0, 0.0)}
        reset_base.params["velocity_range"] = {
            "x": (0.0, 0.0),
            "y": (0.0, 0.0),
            "z": (0.0, 0.0),
            "roll": (0.0, 0.0),
            "pitch": (0.0, 0.0),
            "yaw": (0.0, 0.0),
        }
    reset_joints = getattr(events, "reset_robot_joints", None)
    if reset_joints is not None:
        reset_joints.params["position_range"] = (1.0, 1.0)
        reset_joints.params["velocity_range"] = (0.0, 0.0)


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, _agent_cfg):
    env_cfg.scene.num_envs = args_cli.num_envs
    if args_cli.env_spacing is not None:
        env_cfg.scene.env_spacing = args_cli.env_spacing
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    if hasattr(env_cfg.scene, "height_scanner") and env_cfg.scene.height_scanner is not None:
        env_cfg.scene.height_scanner.debug_vis = args_cli.debug_vis
    if not args_cli.load_amp_motion:
        _disable_amp_motion_for_sensor_debug(env_cfg)
    if args_cli.deterministic_reset:
        _make_reset_deterministic(env_cfg)

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)
    base_env = env.unwrapped

    if "height_scanner" not in base_env.scene.sensors:
        raise RuntimeError(f"Task {args_cli.task} does not have a 'height_scanner' sensor.")

    env.reset()
    action_dim = base_env.action_manager.total_action_dim
    zero_actions = torch.zeros((base_env.num_envs, action_dim), device=base_env.device)
    for _ in range(args_cli.steps):
        env.step(zero_actions)

    env_ids = _parse_env_ids(args_cli.env_ids, base_env.num_envs)
    if not env_ids:
        raise ValueError(f"No valid env ids from '{args_cli.env_ids}' for num_envs={base_env.num_envs}.")

    _write_outputs(base_env, Path(args_cli.output_dir).expanduser().resolve(), env_ids, args_cli.offset)
    if args_cli.term_stats:
        _print_group_term_stats(base_env, "policy")
        _print_group_term_stats(base_env, "critic")

    if args_cli.pause:
        print("[INFO] Keeping simulator open. Close the window or press Ctrl+C to exit.")
        try:
            while simulation_app.is_running():
                env.step(zero_actions)
                time.sleep(base_env.step_dt)
        except KeyboardInterrupt:
            pass
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
