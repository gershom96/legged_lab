#!/usr/bin/env python3
"""Train the perception-enabled split-policy G1 AMP task entirely in JAX/MJX."""

from __future__ import annotations

import argparse
import copy
from collections import deque
from dataclasses import asdict, replace
from datetime import datetime
import gc
import json
import os
from pathlib import Path
import shutil
import sys
import time
from typing import Any

import numpy as np
import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "mujoco_mjx/configs/g1_rsl_rl_mjx_amp.yaml")
    checkpoint = parser.add_mutually_exclusive_group(required=True)
    checkpoint.add_argument("--checkpoint", type=Path, help="Converted JAX .npz warm start")
    checkpoint.add_argument("--resume", type=Path, help="Native-JAX training .npz with optimizer state")
    parser.add_argument(
        "--repartition-curriculum-on-resume",
        action="store_true",
        help="After --resume, reinitialize env slots with their saved per-environment terrain targets.",
    )
    parser.add_argument("--num-envs", type=int)
    parser.add_argument("--device")
    parser.add_argument("--mjx-impl", choices=("warp", "jax"))
    parser.add_argument("--terrain")
    parser.add_argument("--terrain-difficulty", type=float)
    parser.add_argument(
        "--domain-randomization",
        choices=("on", "off"),
        help="Override domain_randomization.enabled from the config.",
    )
    parser.add_argument(
        "--observation-layout",
        choices=("isaac_scan_first_v2", "legacy_native_history_first_v1"),
        help="Override the checkpoint/default observation contract for a diagnostic replay.",
    )
    parser.add_argument("--ean-root", type=Path)
    parser.add_argument("--ean-scene-config", type=Path)
    parser.add_argument("--motion-max-files", type=int)
    parser.add_argument("--max-iterations", type=int)
    parser.add_argument("--num-mini-batches", type=int)
    parser.add_argument("--micro-batch-size", type=int)
    parser.add_argument("--logger", choices=("none", "wandb"), default="wandb")
    parser.add_argument("--run-name", default="")
    parser.add_argument("--video", action="store_true")
    parser.add_argument("--video-interval", type=int, default=200)
    parser.add_argument("--video-length", type=int, default=300)
    parser.add_argument("--video-env-index", type=int, default=0)
    parser.add_argument("--video-width", type=int, default=640)
    parser.add_argument("--video-height", type=int, default=480)
    parser.add_argument("--eval-only", action="store_true", help="Record one frozen-policy evaluation and exit.")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--smoke-steps", type=int, default=2)
    parser.add_argument("--log-root", type=Path, default=REPO_ROOT / "logs/rsl_rl")
    parser.add_argument("--build-root", type=Path, default=REPO_ROOT / "mujoco_mjx/outputs/build_native")
    return parser.parse_args()


def serializable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: serializable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [serializable(item) for item in value]
    return value


class RslRlConsolePrinter:
    """Console formatter matching the RSL-RL AMP logger layout."""

    def __init__(self, train_cfg: dict[str, Any], num_envs: int, start_iteration: int, total_iterations: int) -> None:
        self.train_cfg = train_cfg
        self.num_envs = num_envs
        self.start_iteration = start_iteration
        self.total_iterations = total_iterations
        self.collection_size = int(train_cfg["num_steps_per_env"]) * num_envs
        self.total_time = 0.0
        self.episode_length_buffer: deque[float] = deque(maxlen=100)
        self.reward_buffer: deque[float] = deque(maxlen=100)
        self.total_reward_buffer: deque[float] = deque(maxlen=100)
        self.style_reward_buffer: deque[float] = deque(maxlen=100)
        self.lower_task_reward_buffer: deque[float] = deque(maxlen=100)
        self.upper_task_reward_buffer: deque[float] = deque(maxlen=100)
        self.cur_episode_length = np.zeros(num_envs, dtype=np.float64)
        self.cur_reward_sum = np.zeros(num_envs, dtype=np.float64)
        self.cur_total_reward_sum = np.zeros(num_envs, dtype=np.float64)
        self.cur_style_reward_sum = np.zeros(num_envs, dtype=np.float64)
        self.cur_lower_task_reward_sum = np.zeros(num_envs, dtype=np.float64)
        self.cur_upper_task_reward_sum = np.zeros(num_envs, dtype=np.float64)

    @staticmethod
    def _fmt(value: Any, precision: int = 4) -> str:
        if value is None:
            return "N/A"
        return f"{float(value):.{precision}f}"

    def process_rollout(self, rollout: Any) -> None:
        rewards = np.asarray(rollout.rewards, dtype=np.float64)
        task_rewards = np.asarray(rollout.task_rewards, dtype=np.float64)
        style_rewards = np.asarray(rollout.style_rewards, dtype=np.float64)
        dones = np.asarray(rollout.dones, dtype=bool)
        completed_lengths = np.asarray(rollout.completed_lengths, dtype=np.float64)

        for step in range(rewards.shape[0]):
            total_reward = np.sum(rewards[step], axis=-1)
            self.cur_reward_sum += total_reward
            self.cur_total_reward_sum += total_reward
            self.cur_style_reward_sum += style_rewards[step]
            self.cur_lower_task_reward_sum += task_rewards[step, :, 0]
            self.cur_upper_task_reward_sum += task_rewards[step, :, 1]
            self.cur_episode_length += 1.0

            done_ids = np.flatnonzero(dones[step])
            if done_ids.size == 0:
                continue

            self.reward_buffer.extend(self.cur_reward_sum[done_ids].tolist())
            self.total_reward_buffer.extend(self.cur_total_reward_sum[done_ids].tolist())
            self.style_reward_buffer.extend(self.cur_style_reward_sum[done_ids].tolist())
            self.lower_task_reward_buffer.extend(self.cur_lower_task_reward_sum[done_ids].tolist())
            self.upper_task_reward_buffer.extend(self.cur_upper_task_reward_sum[done_ids].tolist())
            self.episode_length_buffer.extend(completed_lengths[step, done_ids].tolist())

            self.cur_episode_length[done_ids] = 0.0
            self.cur_reward_sum[done_ids] = 0.0
            self.cur_total_reward_sum[done_ids] = 0.0
            self.cur_style_reward_sum[done_ids] = 0.0
            self.cur_lower_task_reward_sum[done_ids] = 0.0
            self.cur_upper_task_reward_sum[done_ids] = 0.0

    @staticmethod
    def _mean_buffer(values: deque[float]) -> float | None:
        if not values:
            return None
        return float(sum(values) / len(values))

    def add_buffered_metrics(self, metrics: dict[str, float]) -> None:
        buffered = {
            "AMP/mean_total_reward": self._mean_buffer(self.total_reward_buffer),
            "AMP/mean_style_reward": self._mean_buffer(self.style_reward_buffer),
            "Train/mean_reward": self._mean_buffer(self.reward_buffer),
            "Train/mean_episode_length": self._mean_buffer(self.episode_length_buffer),
            "Task/mean_lower_reward": self._mean_buffer(self.lower_task_reward_buffer),
            "Task/mean_upper_reward": self._mean_buffer(self.upper_task_reward_buffer),
        }
        for key, value in buffered.items():
            if value is not None:
                metrics[key] = value
        mean_episode_length = self._mean_buffer(self.episode_length_buffer)
        metrics["Loss/amp/active"] = 1.0
        metrics["Loss/amp/mean_episode_length"] = 0.0 if mean_episode_length is None else mean_episode_length

    def log(self, iteration: int, metrics: dict[str, float], width: int = 80, pad: int = 40) -> None:
        collect_time = float(metrics.get("Perf/collection_time", metrics.get("Perf/collect_seconds", 0.0)))
        learn_time = float(metrics.get("Perf/learning_time", metrics.get("Perf/learn_seconds", 0.0)))
        iteration_time = collect_time + learn_time
        self.total_time += iteration_time
        fps = float(metrics.get("Perf/total_fps", metrics.get("Perf/steps_per_second", 0.0)))
        total_steps = iteration * self.collection_size

        loss_items = [
            (key.removeprefix("Loss/"), value)
            for key, value in metrics.items()
            if key.startswith("Loss/") and key != "Loss/learning_rate"
        ]
        loss_items.sort(key=lambda item: item[0])
        episode_items = [
            (key.removeprefix("Episode_Reward/"), value)
            for key, value in metrics.items()
            if key.startswith("Episode_Reward/")
        ]
        episode_items.sort(key=lambda item: item[0])

        total_it = self.total_iterations
        log_string = f"{'#' * width}\n"
        log_string += f"\033[1m{f' Learning iteration {iteration}/{total_it} '.center(width)}\033[0m \n\n"
        run_name = self.train_cfg.get("run_name")
        if run_name:
            log_string += f"{'Run name:':>{pad}} {run_name}\n"
        log_string += (
            f"{'Total steps:':>{pad}} {total_steps} \n"
            f"{'Steps per second:':>{pad}} {fps:.0f} \n"
            f"{'Collection time:':>{pad}} {collect_time:.3f}s \n"
            f"{'Learning time:':>{pad}} {learn_time:.3f}s \n"
        )
        for key, value in loss_items:
            log_string += f"{f'Mean {key} loss:':>{pad}} {float(value):.4f}\n"

        if self.reward_buffer:
            log_string += f"{'Mean AMP total reward:':>{pad}} {self._fmt(metrics.get('AMP/mean_total_reward'), 2)}\n"
            log_string += f"{'Mean AMP style reward:':>{pad}} {self._fmt(metrics.get('AMP/mean_style_reward'), 2)}\n"
            log_string += f"{'Mean reward:':>{pad}} {self._fmt(metrics.get('Train/mean_reward'), 2)}\n"
            log_string += f"{'Mean lower task reward:':>{pad}} {self._fmt(metrics.get('Task/mean_lower_reward'), 2)}\n"
            log_string += f"{'Mean upper task reward:':>{pad}} {self._fmt(metrics.get('Task/mean_upper_reward'), 2)}\n"
            log_string += f"{'Mean episode length:':>{pad}} {self._fmt(metrics.get('Train/mean_episode_length'), 2)}\n"
        log_string += f"{'Mean action noise std:':>{pad}} {self._fmt(metrics.get('Policy/mean_noise_std'), 2)}\n"
        stage_counts = [
            metrics.get(f"Curriculum/stage_env_count_{stage}")
            for stage in range(6)
        ]
        if any(count is not None for count in stage_counts):
            stage_text = " ".join(
                f"s{stage}={int(count or 0)}" for stage, count in enumerate(stage_counts)
            )
            log_string += f"{'Curriculum target stages:':>{pad}} {stage_text}\n"
            length_parts = []
            for stage in range(6):
                value = metrics.get(f"Train/mean_episode_length_stage_{stage}")
                if value is not None:
                    length_parts.append(f"s{stage}={float(value):.1f}")
            if length_parts:
                log_string += f"{'Episode length by stage:':>{pad}} {' '.join(length_parts)}\n"
            termination_parts = []
            for stage in range(6):
                value = metrics.get(f"Train/termination_rate_stage_{stage}")
                if value is not None:
                    termination_parts.append(f"s{stage}={float(value):.3f}")
            if termination_parts:
                log_string += f"{'Termination rate by stage:':>{pad}} {' '.join(termination_parts)}\n"
        else:
            log_string += f"{'Curriculum stage:':>{pad}} {int(metrics.get('Curriculum/stage', 0.0))}\n"

        for key, value in episode_items:
            log_string += f"{f'Episode_Reward/{key}:':>{pad}} {float(value):.4f}\n"

        done_it = max(iteration - self.start_iteration, 1)
        remaining_it = max(total_it - iteration, 0)
        eta = self.total_time / done_it * remaining_it
        log_string += (
            f"{'-' * width}\n"
            f"{'Iteration time:':>{pad}} {iteration_time:.2f}s\n"
            f"{'Time elapsed:':>{pad}} {time.strftime('%H:%M:%S', time.gmtime(self.total_time))}\n"
            f"{'ETA:':>{pad}} {time.strftime('%H:%M:%S', time.gmtime(eta))}\n"
        )
        print(log_string, flush=True)


def main() -> None:
    args = parse_args()
    if args.video:
        os.environ.setdefault("MUJOCO_GL", "egl")
    with args.config.expanduser().resolve().open("r", encoding="utf-8") as stream:
        payload = yaml.safe_load(stream)
    if not isinstance(payload, dict) or "rsl_rl" not in payload:
        raise ValueError(f"Invalid native-JAX configuration: {args.config}")

    from mujoco_mjx.rsl_rl_mujoco.spec import MujocoRslRlEnvSpec
    from mujoco_mjx.rsl_rl_mujoco.native_physics import configure_native_jax

    spec = MujocoRslRlEnvSpec.from_mapping(payload, REPO_ROOT)
    checkpoint = (args.resume or args.checkpoint).expanduser().resolve()
    # Native checkpoints written before the source-layout correction had no
    # layout metadata and were trained with [history, scan].  Keep that
    # contract on resume; converted Isaac checkpoints remain scan-first.
    if args.resume is not None:
        with np.load(checkpoint, allow_pickle=False) as checkpoint_payload:
            checkpoint_metadata = json.loads(str(checkpoint_payload["__metadata_json__"].item()))
        saved_layout = checkpoint_metadata.get("observation_layout")
        if saved_layout is None:
            spec = replace(spec, observation_layout="legacy_native_history_first_v1")
        elif saved_layout != spec.observation_layout:
            spec = replace(spec, observation_layout=str(saved_layout))
    if args.observation_layout is not None:
        spec = replace(spec, observation_layout=args.observation_layout)
    if args.num_envs is not None:
        spec = replace(spec, num_envs=args.num_envs)
    if args.device is not None:
        spec = replace(spec, device=args.device)
    if args.mjx_impl is not None:
        spec = replace(spec, mjx_impl=args.mjx_impl)
    if args.motion_max_files is not None:
        spec = replace(spec, motion=replace(spec.motion, max_files=args.motion_max_files))
    if args.domain_randomization is not None:
        spec = replace(
            spec,
            domain_randomization=replace(
                spec.domain_randomization,
                enabled=args.domain_randomization == "on",
            ),
        )
    terrain = spec.terrain
    if args.terrain is not None:
        curriculum = replace(terrain.curriculum, enabled=args.terrain == "curriculum")
        terrain = replace(terrain, type=args.terrain, curriculum=curriculum)
    if args.terrain_difficulty is not None:
        terrain = replace(terrain, difficulty=args.terrain_difficulty)
    if args.ean_root is not None:
        terrain = replace(terrain, ean_root=args.ean_root.expanduser().resolve())
    if args.ean_scene_config is not None:
        terrain = replace(terrain, ean_scene_config=args.ean_scene_config.expanduser().resolve())
    spec = replace(spec, terrain=terrain)
    configure_native_jax(spec.device)

    if checkpoint.suffix != ".npz":
        raise ValueError("Native training accepts .npz only. Convert .pt with convert_pt_to_jax.py first.")
    train_cfg = copy.deepcopy(payload["rsl_rl"])
    if args.num_mini_batches is not None:
        if args.num_mini_batches <= 0:
            raise ValueError("--num-mini-batches must be positive")
        train_cfg["algorithm"]["num_mini_batches"] = args.num_mini_batches
    if args.micro_batch_size is not None:
        if args.micro_batch_size <= 0:
            raise ValueError("--micro-batch-size must be positive")
        train_cfg["algorithm"]["micro_batch_size"] = args.micro_batch_size
    if args.smoke_test:
        train_cfg["num_steps_per_env"] = args.smoke_steps
        train_cfg["algorithm"]["num_learning_epochs"] = 1
        train_cfg["algorithm"]["num_mini_batches"] = 1
    max_iterations = int(args.max_iterations or train_cfg["max_iterations"])
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    suffix = f"_{args.run_name}" if args.run_name else ""
    experiment = f"{train_cfg['experiment_name']}_native_jax"
    log_dir = args.log_root.expanduser().resolve() / experiment / f"{timestamp}{suffix}"
    build_dir = args.build_root.expanduser().resolve() / f"{timestamp}{suffix}"
    log_dir.mkdir(parents=True, exist_ok=False)
    build_dir.mkdir(parents=True, exist_ok=False)
    shutil.copy2(args.config.expanduser().resolve(), log_dir / "source_config.yaml")
    with (log_dir / "resolved_env.yaml").open("w", encoding="utf-8") as stream:
        yaml.safe_dump(serializable(asdict(spec)), stream, sort_keys=False)

    from mujoco_mjx.rsl_rl_mujoco.native_env import NativeJaxAmpEnv
    from mujoco_mjx.rsl_rl_mujoco.native_trainer import NativeJaxAmpTrainer

    print(f"[INFO] Native JAX/MJX implementation: {spec.mjx_impl} on {spec.device}")
    print(f"[INFO] Terrain: {spec.terrain.type}, difficulty={spec.terrain.difficulty}")
    if spec.terrain.curriculum.enabled:
        print(
            f"[INFO] Curriculum: start stage={spec.terrain.curriculum.start_stage}, "
            f"patches=15, patch size={spec.terrain.size}, "
            f"safe half-extent={spec.terrain.curriculum.safe_half_extent} m"
        )
    print(f"[INFO] Checkpoint: {checkpoint}")
    print(f"[INFO] Logs: {log_dir}")
    env = NativeJaxAmpEnv(spec, train_cfg["algorithm"]["split_reward_cfg"], build_dir)
    print(
        f"[INFO] AMP motion dataset: {env.num_motion_files} files from {spec.motion.directory} "
        f"(default_weight={spec.motion.default_weight}, motionbricks_weight={spec.motion.motionbricks_weight})"
    )
    trainer = NativeJaxAmpTrainer(env, train_cfg, checkpoint, resume=args.resume is not None)
    if args.repartition_curriculum_on_resume:
        if args.resume is None:
            raise ValueError("--repartition-curriculum-on-resume requires --resume")
        reset_seed = spec.seed + int(trainer.jax.device_get(trainer.state.iteration))
        trainer.env_state, trainer.observation = env.initialize(
            seed=reset_seed,
            terrain_stages=trainer.env_state.terrain_stage,
            randomize_episode_lengths=True,
        )
        print(
            "[INFO] Reinitialized curriculum env slots using their per-environment terrain targets"
        )
    if "torch" in sys.modules:
        raise RuntimeError("PyTorch was imported into the native-JAX training process")
    print("[PASS] Native runtime imported no PyTorch modules.")

    wandb_run = None
    if args.logger == "wandb":
        import wandb

        wandb_run = wandb.init(
            project=train_cfg["wandb_project"],
            name=log_dir.name,
            config={
                "backend": "native_jax_mjx",
                "env": serializable(asdict(spec)),
                "train": serializable(train_cfg),
                "checkpoint": str(checkpoint),
            },
        )
    video = None
    if args.video:
        from mujoco_mjx.rsl_rl_mujoco.native_video import NativeMujocoVideoRecorder

        video = NativeMujocoVideoRecorder(
            env,
            log_dir,
            args.video_interval,
            args.video_length,
            env_index=args.video_env_index,
            width=args.video_width,
            height=args.video_height,
            log_to_wandb=wandb_run is not None,
        )
        video.maybe_record(trainer, force=True)

    if args.eval_only:
        if video is None:
            raise ValueError("--eval-only requires --video")
        video.close()
        if wandb_run is not None:
            wandb_run.finish()
        return

    try:
        start_iteration = int(trainer.jax.device_get(trainer.state.iteration))
        end_iteration = start_iteration + (1 if args.smoke_test else max_iterations)
        console = RslRlConsolePrinter(train_cfg, spec.num_envs, start_iteration, end_iteration)
        for _ in range(start_iteration, end_iteration):
            metrics, rollout = trainer.train_iteration(video_recorder=video)
            iteration = int(trainer.jax.device_get(trainer.state.iteration))
            metrics["Train/iteration"] = iteration
            console.process_rollout(rollout)
            console.add_buffered_metrics(metrics)
            if wandb_run is not None:
                wandb_run.log(metrics, step=iteration)
            console.log(iteration, metrics)
            if video is not None:
                video.maybe_record(trainer)
            save_interval = int(train_cfg["save_interval"])
            if iteration % save_interval == 0 or args.smoke_test:
                saved = trainer.save(log_dir / f"model_{iteration}.npz")
                print(f"[INFO] Saved native-JAX checkpoint: {saved}")
            if args.smoke_test:
                values = np.asarray(trainer.jax.device_get(rollout.rewards))
                if not np.isfinite(values).all():
                    raise FloatingPointError("Non-finite native-JAX rollout rewards")
                print(
                    "[PASS] Native JAX full-update smoke test: "
                    f"policy={rollout.policy_obs.shape}, critic={rollout.critic_obs.shape}, "
                    f"disc={rollout.disc_obs.shape}, iteration={iteration}"
                )
            del rollout
            trainer.jax.effects_barrier()
            env.physics.release_backend_transients()
            gc.collect()
        if not args.smoke_test:
            trainer.save(log_dir / f"model_{int(trainer.jax.device_get(trainer.state.iteration))}.npz")
    finally:
        if video is not None:
            video.close()
        if wandb_run is not None:
            wandb_run.finish()


if __name__ == "__main__":
    main()
