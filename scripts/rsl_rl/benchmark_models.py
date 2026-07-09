# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Benchmark a folder of RSL-RL model checkpoints on one IsaacLab task.

The script evaluates every compatible checkpoint in a folder and writes CSV/JSON
summaries with velocity tracking, stability, posture, and effort metrics.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip


parser = argparse.ArgumentParser(description="Benchmark RSL-RL checkpoints on an IsaacLab task.")
parser.add_argument("--task", type=str, default=None, help="Name of the task to evaluate.")
parser.add_argument(
    "--agent",
    type=str,
    default="rsl_rl_cfg_entry_point",
    help="Name of the RL agent configuration entry point.",
)
parser.add_argument("--checkpoint_dir", type=str, default=None, help="Folder containing model checkpoint files.")
parser.add_argument("--pattern", type=str, default="model_*.pt", help="Glob pattern for checkpoints.")
parser.add_argument("--recursive", action="store_true", help="Search for checkpoints recursively.")
parser.add_argument("--max_models", type=int, default=None, help="Optional maximum number of checkpoints to evaluate.")
parser.add_argument("--num_envs", type=int, default=256, help="Number of parallel environments.")
parser.add_argument("--num_steps", type=int, default=1000, help="Number of measured environment steps per checkpoint.")
parser.add_argument("--warmup_steps", type=int, default=0, help="Warmup steps not included in metrics.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment.")
parser.add_argument("--output", type=str, default=None, help="Output CSV path. JSON is written next to it.")
parser.add_argument("--asset_name", type=str, default="robot", help="Scene asset name used for robot metrics.")
parser.add_argument("--command_name", type=str, default="base_velocity", help="Velocity command term name.")
parser.add_argument(
    "--height_body_name",
    type=str,
    default="torso_link",
    help="Body name used for torso/body height metrics. Falls back to root height if unavailable.",
)
parser.add_argument("--height_target", type=float, default=0.75, help="Target torso/root height for height score.")
parser.add_argument("--lin_vel_scale", type=float, default=0.5, help="Tracking score scale for xy velocity error.")
parser.add_argument("--yaw_vel_scale", type=float, default=0.5, help="Tracking score scale for yaw velocity error.")
parser.add_argument("--tilt_scale", type=float, default=0.5, help="Stability score scale for body tilt in radians.")
parser.add_argument("--height_scale", type=float, default=0.25, help="Stability score scale for height error.")
parser.add_argument("--torque_scale", type=float, default=100.0, help="Effort score scale for mean absolute torque.")
parser.add_argument("--action_rate_scale", type=float, default=1.0, help="Effort score scale for action-rate L2.")
parser.add_argument("--joint_acc_scale", type=float, default=50.0, help="Effort score scale for mean absolute joint acc.")
parser.add_argument(
    "--print_every",
    type=int,
    default=1,
    help="Print progress every N evaluated checkpoints.",
)

cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
if args_cli.task is None:
    parser.error("--task is required.")
if args_cli.checkpoint_dir is None:
    parser.error("--checkpoint_dir is required.")

sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


import csv
import json
import math
import os
import re
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

import gymnasium as gym
import torch

from isaaclab.envs import DirectMARLEnv, DirectMARLEnvCfg, DirectRLEnvCfg, ManagerBasedRLEnvCfg, multi_agent_to_single_agent
from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper
from isaaclab_tasks.utils.hydra import hydra_task_config
from rsl_rl.runners import DistillationRunner, OnPolicyRunner

import isaaclab_tasks  # noqa: F401
import legged_lab.tasks  # noqa: F401


def _checkpoint_sort_key(path: Path) -> tuple[int, str]:
    match = re.search(r"model_(\d+)\.pt$", path.name)
    if match:
        return int(match.group(1)), path.name
    return 10**18, path.name


def _find_checkpoints(checkpoint_dir: str, pattern: str, recursive: bool, max_models: int | None) -> list[Path]:
    root = Path(checkpoint_dir).expanduser().resolve()
    globber = root.rglob if recursive else root.glob
    paths = sorted((p for p in globber(pattern) if p.is_file()), key=_checkpoint_sort_key)
    if max_models is not None:
        paths = paths[:max_models]
    if not paths:
        raise FileNotFoundError(f"No checkpoints matching '{pattern}' found in {root}.")
    return paths


def _construct_runner(env: RslRlVecEnvWrapper, agent_cfg: RslRlBaseRunnerCfg):
    if agent_cfg.class_name == "OnPolicyRunner":
        return OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    if agent_cfg.class_name == "AMPRunner":
        from rsl_rl.runners import AMPRunner

        return AMPRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    if agent_cfg.class_name == "DistillationRunner":
        return DistillationRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")


def _load_policy_checkpoint(runner, checkpoint: Path, device: str) -> tuple[bool, str]:
    try:
        runner.load(str(checkpoint), load_optimizer=False, map_location=device)
        return True, "runner.load"
    except Exception as runner_error:
        try:
            loaded = torch.load(checkpoint, weights_only=False, map_location=device)
            state_dict = loaded["model_state_dict"] if isinstance(loaded, dict) and "model_state_dict" in loaded else loaded
            runner.alg.policy.load_state_dict(state_dict)
            return True, f"model_state_dict_only_after:{type(runner_error).__name__}"
        except Exception as fallback_error:
            return False, f"{type(runner_error).__name__}: {runner_error}; fallback {type(fallback_error).__name__}: {fallback_error}"


def _get_policy_module(runner):
    try:
        return runner.alg.policy
    except AttributeError:
        return runner.alg.actor_critic


def _safe_mean(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else float("nan")


def _safe_min(values: list[float]) -> float:
    return float(min(values)) if values else float("nan")


def _safe_max(values: list[float]) -> float:
    return float(max(values)) if values else float("nan")


def _exp_score(error: float, scale: float) -> float:
    if not math.isfinite(error):
        return 0.0
    return 100.0 * math.exp(-max(error, 0.0) / max(scale, 1.0e-8))


def _command(base_env, command_name: str) -> torch.Tensor | None:
    command_manager = getattr(base_env, "command_manager", None)
    if command_manager is None:
        return None
    try:
        return command_manager.get_command(command_name)
    except Exception:
        return None


def _robot(base_env, asset_name: str):
    try:
        return base_env.scene[asset_name]
    except Exception:
        return None


def _termination_counts(base_env) -> dict[str, float]:
    termination_manager = getattr(base_env, "termination_manager", None)
    if termination_manager is None:
        return {}
    term_dones = getattr(termination_manager, "_term_dones", None)
    term_names = getattr(termination_manager, "active_terms", [])
    if term_dones is None:
        return {}
    return {name: float(term_dones[:, idx].float().sum().detach().cpu().item()) for idx, name in enumerate(term_names)}


@dataclass
class MetricAccumulator:
    num_envs: int
    device: torch.device
    measured_steps: int = 0
    env_steps: int = 0
    done_count: int = 0
    timeout_count: int = 0
    episode_lengths: list[float] = field(default_factory=list)
    sums: dict[str, float] = field(default_factory=lambda: defaultdict(float))
    mins: dict[str, float] = field(default_factory=dict)
    maxs: dict[str, float] = field(default_factory=dict)
    termination_counts: dict[str, float] = field(default_factory=lambda: defaultdict(float))
    current_measured_lengths: torch.Tensor = field(init=False)
    height_body_index: int | None = None
    height_body_lookup_done: bool = False

    def __post_init__(self):
        self.current_measured_lengths = torch.zeros(self.num_envs, device=self.device)

    def _add_mean(self, name: str, value: torch.Tensor):
        value = value.detach()
        mean_value = float(value.mean().cpu().item())
        self.sums[name] += mean_value
        self.mins[name] = min(self.mins.get(name, float("inf")), float(value.min().cpu().item()))
        self.maxs[name] = max(self.maxs.get(name, float("-inf")), float(value.max().cpu().item()))

    def _add_body_height(self, robot, height_body_name: str | None):
        if robot is None or not height_body_name:
            return
        data = robot.data
        body_pos_w = getattr(data, "body_pos_w", None)
        if body_pos_w is None:
            return
        if not self.height_body_lookup_done:
            self.height_body_lookup_done = True
            try:
                body_ids, _ = robot.find_bodies(height_body_name)
                if body_ids:
                    self.height_body_index = int(body_ids[0])
            except Exception:
                self.height_body_index = None
        if self.height_body_index is not None:
            self._add_mean("body_height", body_pos_w[:, self.height_body_index, 2])

    def update(
        self,
        base_env,
        asset_name: str,
        command_name: str,
        height_body_name: str | None,
        actions: torch.Tensor,
        prev_actions: torch.Tensor | None,
        dones: torch.Tensor,
        extras: dict[str, Any],
        measure: bool,
    ):
        self.env_steps += 1
        if measure:
            self.current_measured_lengths += 1

        done_mask = dones.bool()
        if done_mask.any():
            if measure:
                measured_lengths = self.current_measured_lengths[done_mask]
                measured_lengths = measured_lengths[measured_lengths > 0]
                self.episode_lengths.extend(measured_lengths.detach().cpu().tolist())
            self.current_measured_lengths[done_mask] = 0.0

        if not measure:
            return

        self.measured_steps += 1
        self.done_count += int(done_mask.sum().detach().cpu().item())
        timeouts = extras.get("time_outs")
        if timeouts is not None:
            self.timeout_count += int(timeouts.bool().sum().detach().cpu().item())

        for name, count in _termination_counts(base_env).items():
            self.termination_counts[name] += count

        robot = _robot(base_env, asset_name)
        command = _command(base_env, command_name)

        if robot is not None:
            data = robot.data
            root_pos_w = getattr(data, "root_pos_w", None)
            root_lin_vel_b = getattr(data, "root_lin_vel_b", None)
            root_ang_vel_b = getattr(data, "root_ang_vel_b", None)
            projected_gravity_b = getattr(data, "projected_gravity_b", None)
            joint_torque = getattr(data, "applied_torque", None)
            if joint_torque is None:
                joint_torque = getattr(data, "computed_torque", None)
            joint_vel = getattr(data, "joint_vel", None)
            joint_acc = getattr(data, "joint_acc", None)

            if root_pos_w is not None:
                self._add_mean("root_height", root_pos_w[:, 2])
            self._add_body_height(robot, height_body_name)
            if root_lin_vel_b is not None:
                self._add_mean("root_z_vel_abs", root_lin_vel_b[:, 2].abs())
                self._add_mean("actual_lin_speed_xy", torch.linalg.norm(root_lin_vel_b[:, :2], dim=-1))
            if root_ang_vel_b is not None:
                self._add_mean("actual_yaw_speed_abs", root_ang_vel_b[:, 2].abs())
            if projected_gravity_b is not None:
                tilt = torch.acos(torch.clamp(-projected_gravity_b[:, 2], -1.0, 1.0))
                self._add_mean("tilt_rad", tilt)
            if joint_torque is not None:
                self._add_mean("joint_torque_abs", joint_torque.abs())
                self._add_mean("joint_torque_l2", torch.sum(torch.square(joint_torque), dim=-1))
            if joint_vel is not None:
                self._add_mean("joint_vel_abs", joint_vel.abs())
            if joint_acc is not None:
                self._add_mean("joint_acc_abs", joint_acc.abs())
            if joint_torque is not None and joint_vel is not None:
                self._add_mean("joint_power_abs", torch.sum(torch.abs(joint_torque * joint_vel), dim=-1))
            if command is not None and root_lin_vel_b is not None and root_ang_vel_b is not None:
                self._add_mean("cmd_lin_speed_xy", torch.linalg.norm(command[:, :2], dim=-1))
                self._add_mean("cmd_yaw_speed_abs", command[:, 2].abs())
                self._add_mean("lin_vel_error_xy", torch.linalg.norm(command[:, :2] - root_lin_vel_b[:, :2], dim=-1))
                self._add_mean("yaw_vel_error", torch.abs(command[:, 2] - root_ang_vel_b[:, 2]))

        self._add_mean("action_abs", actions.abs())
        if prev_actions is not None:
            self._add_mean("action_rate_l2", torch.sum(torch.square(actions - prev_actions), dim=-1))

    def summarize(self, checkpoint: Path, load_status: str, args) -> dict[str, Any]:
        denom = max(self.measured_steps, 1)
        total_env_steps = max(self.measured_steps * self.num_envs, 1)
        means = {name: value / denom for name, value in self.sums.items()}

        lin_err = means.get("lin_vel_error_xy", float("nan"))
        yaw_err = means.get("yaw_vel_error", float("nan"))
        tilt = means.get("tilt_rad", float("nan"))
        height = means.get("body_height", means.get("root_height", float("nan")))
        torque = means.get("joint_torque_abs", float("nan"))
        action_rate = means.get("action_rate_l2", float("nan"))
        joint_acc = means.get("joint_acc_abs", float("nan"))

        tracking_score = 0.7 * _exp_score(lin_err, args.lin_vel_scale) + 0.3 * _exp_score(yaw_err, args.yaw_vel_scale)
        survival_rate = 1.0 - self.done_count / total_env_steps
        timeout_rate = self.timeout_count / total_env_steps
        height_error = abs(height - args.height_target) if math.isfinite(height) else float("inf")
        stability_score = 100.0 * (
            0.45 * max(survival_rate, 0.0)
            + 0.20 * timeout_rate
            + 0.20 * math.exp(-max(tilt, 0.0) / max(args.tilt_scale, 1.0e-8))
            + 0.15 * math.exp(-height_error / max(args.height_scale, 1.0e-8))
        )
        effort_cost = 0.0
        if math.isfinite(torque):
            effort_cost += torque / max(args.torque_scale, 1.0e-8)
        if math.isfinite(action_rate):
            effort_cost += action_rate / max(args.action_rate_scale, 1.0e-8)
        if math.isfinite(joint_acc):
            effort_cost += joint_acc / max(args.joint_acc_scale, 1.0e-8)
        effort_score = 100.0 / (1.0 + effort_cost)
        overall_score = 0.45 * tracking_score + 0.35 * stability_score + 0.20 * effort_score

        result = {
            "checkpoint": str(checkpoint),
            "checkpoint_name": checkpoint.name,
            "load_status": load_status,
            "measured_steps": self.measured_steps,
            "num_envs": self.num_envs,
            "total_env_steps": total_env_steps,
            "done_count": self.done_count,
            "timeout_count": self.timeout_count,
            "termination_rate": self.done_count / total_env_steps,
            "timeout_rate": timeout_rate,
            "survival_rate": survival_rate,
            "completed_episode_length_mean": _safe_mean(self.episode_lengths),
            "completed_episode_length_min": _safe_min(self.episode_lengths),
            "completed_episode_length_max": _safe_max(self.episode_lengths),
            "tracking_score": tracking_score,
            "stability_score": stability_score,
            "effort_score": effort_score,
            "overall_score": overall_score,
        }

        for name, value in sorted(means.items()):
            result[f"{name}_mean"] = value
            if name in self.mins:
                result[f"{name}_min"] = self.mins[name]
            if name in self.maxs:
                result[f"{name}_max"] = self.maxs[name]
        for name, value in sorted(self.termination_counts.items()):
            result[f"termination_{name}_count"] = value
            result[f"termination_{name}_rate"] = value / total_env_steps
        return result


def _evaluate_checkpoint(env, runner, checkpoint: Path, args, device: str) -> dict[str, Any]:
    loaded, load_status = _load_policy_checkpoint(runner, checkpoint, device)
    if not loaded:
        return {
            "checkpoint": str(checkpoint),
            "checkpoint_name": checkpoint.name,
            "load_status": f"FAILED: {load_status}",
            "overall_score": float("nan"),
        }

    policy = runner.get_inference_policy(device=env.unwrapped.device)
    policy_module = _get_policy_module(runner)
    obs, _ = env.reset()
    policy_module.reset(torch.ones(env.num_envs, dtype=torch.bool, device=env.device))

    accumulator = MetricAccumulator(num_envs=env.num_envs, device=torch.device(env.device))
    prev_actions = None
    total_steps = args.warmup_steps + args.num_steps

    for step in range(total_steps):
        with torch.inference_mode():
            actions = policy(obs)
            obs, _, dones, extras = env.step(actions)
            policy_module.reset(dones)

        accumulator.update(
            base_env=env.unwrapped,
            asset_name=args.asset_name,
            command_name=args.command_name,
            height_body_name=args.height_body_name,
            actions=actions,
            prev_actions=prev_actions,
            dones=dones,
            extras=extras,
            measure=step >= args.warmup_steps,
        )
        prev_actions = actions.detach().clone()

    return accumulator.summarize(checkpoint=checkpoint, load_status=load_status, args=args)


def _write_results(results: list[dict[str, Any]], output_csv: Path):
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    seen = set()
    preferred = [
        "checkpoint_name",
        "overall_score",
        "tracking_score",
        "stability_score",
        "effort_score",
        "lin_vel_error_xy_mean",
        "yaw_vel_error_mean",
        "survival_rate",
        "termination_rate",
        "completed_episode_length_mean",
        "body_height_mean",
        "body_height_min",
        "root_height_mean",
        "root_height_min",
        "tilt_rad_mean",
        "joint_torque_abs_mean",
        "joint_power_abs_mean",
        "action_rate_l2_mean",
        "load_status",
        "checkpoint",
    ]
    for key in preferred:
        if any(key in row for row in results):
            keys.append(key)
            seen.add(key)
    for row in results:
        for key in row:
            if key not in seen:
                keys.append(key)
                seen.add(key)

    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(results)

    output_json = output_csv.with_suffix(".json")
    with output_json.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"[INFO] Wrote CSV:  {output_csv}")
    print(f"[INFO] Wrote JSON: {output_json}")


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs
    if args_cli.seed is not None:
        env_cfg.seed = args_cli.seed
        agent_cfg.seed = args_cli.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    checkpoints = _find_checkpoints(args_cli.checkpoint_dir, args_cli.pattern, args_cli.recursive, args_cli.max_models)
    output_csv = (
        Path(args_cli.output).expanduser().resolve()
        if args_cli.output is not None
        else Path(args_cli.checkpoint_dir).expanduser().resolve() / "benchmark_results.csv"
    )
    print(f"[INFO] Benchmarking {len(checkpoints)} checkpoint(s).")
    print(f"[INFO] Task: {args_cli.task}")
    print(f"[INFO] Output: {output_csv}")

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    runner = _construct_runner(env, agent_cfg)

    results = []
    start_time = time.time()
    for idx, checkpoint in enumerate(checkpoints, start=1):
        print(f"[INFO] [{idx}/{len(checkpoints)}] Evaluating {checkpoint.name}")
        try:
            result = _evaluate_checkpoint(env, runner, checkpoint, args_cli, agent_cfg.device)
        except Exception as exc:
            result = {
                "checkpoint": str(checkpoint),
                "checkpoint_name": checkpoint.name,
                "load_status": f"FAILED_EVAL: {type(exc).__name__}: {exc}",
                "overall_score": float("nan"),
            }
        results.append(result)
        if idx % max(args_cli.print_every, 1) == 0:
            score = result.get("overall_score", float("nan"))
            tracking = result.get("tracking_score", float("nan"))
            stability = result.get("stability_score", float("nan"))
            print(
                f"[INFO] {checkpoint.name}: overall={score:.2f}, "
                f"tracking={tracking:.2f}, stability={stability:.2f}, "
                f"load={result.get('load_status')}"
            )

    results.sort(key=lambda row: (math.isnan(row.get("overall_score", float("nan"))), -row.get("overall_score", -math.inf)))
    _write_results(results, output_csv)
    print(f"[INFO] Benchmark time: {time.time() - start_time:.1f}s")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
