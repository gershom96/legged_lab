from __future__ import annotations

import json
from pathlib import Path
import time
from typing import Any, NamedTuple

import numpy as np

from .native_env import NativeEnvState, NativeJaxAmpEnv, REWARD_TERM_NAMES
from .native_model import (
    CHECKPOINT_FORMAT,
    actor_apply,
    critic_apply,
    discriminator_apply,
    discriminator_score,
    gaussian_entropy,
    gaussian_log_prob_parts,
    load_checkpoint,
    normalize_discriminator_obs,
    style_reward,
    update_normalizer,
)
from .native_optim import adam_init, adam_update, tree_l2_norm
from .native_symmetry import augment_policy_batch, switch_joints


class NativeTrainState(NamedTuple):
    policy: Any
    discriminator: Any
    normalizer: Any
    policy_optimizer: Any
    discriminator_optimizer: Any
    rng: Any
    learning_rate: Any
    iteration: Any
    curriculum_stage: Any
    curriculum_stage_start_iteration: Any


class ReplayState(NamedTuple):
    policy: Any
    demonstration: Any
    pointer: Any
    length: Any


class RolloutBatch(NamedTuple):
    policy_obs: Any
    critic_obs: Any
    actions: Any
    old_values: Any
    old_mean: Any
    old_std: Any
    rewards: Any
    dones: Any
    returns: Any
    advantages: Any
    disc_obs: Any
    disc_demo_obs: Any
    weighted_terms: Any
    task_rewards: Any
    style_rewards: Any
    disc_scores: Any
    completed_term_sums: Any
    completed_lengths: Any
    completed_patch_index: Any
    completed_mask: Any
    terminated: Any
    video_qpos: Any
    video_qvel: Any
    video_patch_index: Any


class RolloutStep(NamedTuple):
    policy_obs: Any
    critic_obs: Any
    actions: Any
    old_values: Any
    old_mean: Any
    old_std: Any
    rewards: Any
    dones: Any
    disc_obs: Any
    disc_demo_obs: Any
    weighted_terms: Any
    task_rewards: Any
    style_rewards: Any
    disc_scores: Any
    completed_term_sums: Any
    completed_lengths: Any
    completed_patch_index: Any
    terminated: Any
    video_qpos: Any
    video_qvel: Any
    video_patch_index: Any


class NativeJaxAmpTrainer:
    """Native-JAX equivalent of the split-policy RSL-RL PPO+AMP update."""

    def __init__(
        self,
        env: NativeJaxAmpEnv,
        train_cfg: dict[str, Any],
        checkpoint: Path,
        *,
        resume: bool = False,
        seed: int | None = None,
    ) -> None:
        import jax
        import jax.numpy as jnp

        self.jax, self.jnp = jax, jnp
        self.env = env
        self.cfg = train_cfg
        self.algorithm = train_cfg["algorithm"]
        self.policy_cfg = train_cfg["policy"]
        self.steps_per_env = int(train_cfg["num_steps_per_env"])
        self.num_epochs = int(self.algorithm["num_learning_epochs"])
        self.num_mini_batches = int(self.algorithm["num_mini_batches"])
        rollout_sample_count = self.steps_per_env * env.num_envs
        if rollout_sample_count % self.num_mini_batches:
            raise ValueError(
                f"Rollout batch {rollout_sample_count} is not divisible by "
                f"{self.num_mini_batches} mini-batches"
            )
        logical_batch_size = rollout_sample_count // self.num_mini_batches
        requested_micro_batch_size = int(self.algorithm.get("micro_batch_size", logical_batch_size))
        self.micro_batch_size = min(requested_micro_batch_size, logical_batch_size)
        if self.micro_batch_size <= 0 or logical_batch_size % self.micro_batch_size:
            raise ValueError(
                f"Logical mini-batch {logical_batch_size} is not divisible by "
                f"micro-batch {self.micro_batch_size}"
            )
        self.num_micro_batches = logical_batch_size // self.micro_batch_size
        print(
            f"[INFO] PPO rollout samples={rollout_sample_count}, logical mini-batches="
            f"{self.num_mini_batches} x {logical_batch_size}, gradient micro-batches="
            f"{self.num_micro_batches} x {self.micro_batch_size}"
        )
        self.gamma = float(self.algorithm["gamma"])
        self.lam = float(self.algorithm["lam"])
        self.clip_param = float(self.algorithm["clip_param"])
        self.value_loss_coef = float(self.algorithm["value_loss_coef"])
        self.entropy_coef = float(self.algorithm["entropy_coef"])
        self.max_grad_norm = float(self.algorithm["max_grad_norm"])
        self.desired_kl = float(self.algorithm["desired_kl"])
        self.symmetry_coeff = float(self.algorithm["symmetry_cfg"]["mirror_loss_coeff"])
        self.amp_cfg = self.algorithm["amp_cfg"]
        self.curriculum_cfg = env.spec.terrain.curriculum
        self.observation_layout = env.spec.observation_layout
        self.curriculum_history: list[dict[str, float]] = []
        self._resume_terrain_stages: np.ndarray | None = None
        if self.amp_cfg["loss_type"] != "LSGAN":
            raise ValueError("The native trainer currently mirrors the configured LSGAN AMP path only")
        if self.policy_cfg.get("actor_obs_normalization") or self.policy_cfg.get("critic_obs_normalization"):
            raise ValueError("Native checkpoint parity currently requires disabled actor/critic observation normalization")
        if self.policy_cfg.get("noise_std_type", "scalar") != "scalar":
            raise ValueError("Native checkpoint parity currently requires scalar action standard deviations")

        policy, discriminator, normalizer, metadata = load_checkpoint(checkpoint)
        rng = jax.random.PRNGKey(int(train_cfg.get("seed", 42) if seed is None else seed))
        state = NativeTrainState(
            policy=policy,
            discriminator=discriminator,
            normalizer=normalizer,
            policy_optimizer=adam_init(policy),
            discriminator_optimizer=adam_init(discriminator),
            rng=rng,
            learning_rate=jnp.asarray(self.algorithm["learning_rate"], dtype=jnp.float32),
            iteration=jnp.asarray(0, dtype=jnp.int32),
            curriculum_stage=jnp.asarray(self.curriculum_cfg.start_stage, dtype=jnp.int32),
            curriculum_stage_start_iteration=jnp.asarray(0, dtype=jnp.int32),
        )
        if resume:
            state = self._load_resume(checkpoint, state)
        self.state = state
        self.source_metadata = metadata
        capacity = int(self.amp_cfg["disc_obs_buffer_size"])
        if self.steps_per_env > capacity:
            raise ValueError(
                f"Rollout length {self.steps_per_env} exceeds AMP replay capacity {capacity}"
            )
        self.replay = ReplayState(
            policy=jnp.zeros((capacity, env.num_envs, 4, 49), dtype=jnp.float32),
            demonstration=jnp.zeros((capacity, env.num_envs, 4, 49), dtype=jnp.float32),
            pointer=jnp.asarray(0, dtype=jnp.int32),
            length=jnp.asarray(0, dtype=jnp.int32),
        )
        self.env_state, self.observation = env.initialize(
            seed=seed,
            terrain_stages=self._resume_terrain_stages,
            randomize_episode_lengths=True,
        )
        self._collect_rollout_jit = jax.jit(self._collect_rollout_scan)
        self._append_replay_batch_jit = jax.jit(self._append_replay_batch)
        self._policy_update_jit = jax.jit(self._policy_update)
        self._disc_update_jit = jax.jit(self._discriminator_update)
        self._evaluate_policy_jit = jax.jit(
            self._evaluate_policy_scan, static_argnames=("video_length",)
        )

    def collect_rollout(self, video_recorder: Any | None = None) -> RolloutBatch:
        jax, jnp = self.jax, self.jnp
        demonstrations = jnp.asarray(
            self.env.sample_demonstration_schedule(self.steps_per_env),
            dtype=jnp.float32,
        )
        video_env_index = jnp.asarray(
            0 if video_recorder is None else video_recorder.env_index,
            dtype=jnp.int32,
        )
        env_state, observation, rng, rollout = self._collect_rollout_jit(
            self.state.policy,
            self.state.discriminator,
            self.state.normalizer,
            self.env_state,
            self.observation,
            self.state.rng,
            self.state.iteration,
            self.state.curriculum_stage,
            demonstrations,
            video_env_index,
        )
        jax.block_until_ready(rollout.rewards)
        self.replay = self._append_replay_batch_jit(
            self.replay,
            rollout.disc_obs,
            rollout.disc_demo_obs,
        )
        self.state = self.state._replace(rng=rng)
        self.env_state = env_state
        self.observation = observation
        return rollout

    def evaluate_policy(
        self,
        video_length: int,
        env_index: int = 0,
        curriculum_stage: int | None = None,
        forced_patch_index: int | None = None,
    ) -> tuple[Any, Any, Any]:
        """Run a continuous deterministic trajectory with one frozen policy."""
        iteration = int(np.asarray(self.jax.device_get(self.state.iteration)))
        stage = (
            int(np.asarray(self.jax.device_get(self.state.curriculum_stage)))
            if curriculum_stage is None
            else int(curriculum_stage)
        )
        eval_state, eval_observation = self.env.initialize(
            seed=self.env.spec.seed + 1000003 + iteration,
            curriculum_stage=stage,
            forced_patch_index=forced_patch_index,
            sample_demonstration=False,
        )
        forced_reset_patch = -1 if forced_patch_index is None else int(forced_patch_index)
        qpos, qvel, patch_index = self._evaluate_policy_jit(
            self.state.policy,
            eval_state,
            eval_observation,
            self.state.iteration,
            self.jnp.asarray(stage, dtype=self.jnp.int32),
            self.jnp.asarray(env_index, dtype=self.jnp.int32),
            self.jnp.asarray(forced_reset_patch, dtype=self.jnp.int32),
            video_length=video_length,
        )
        self.jax.block_until_ready(qpos)
        return qpos, qvel, patch_index

    def _evaluate_policy_scan(
        self,
        policy: Any,
        env_state: NativeEnvState,
        observation: Any,
        learning_iteration: Any,
        curriculum_stage: Any,
        env_index: Any,
        forced_reset_patch_index: Any,
        *,
        video_length: int,
    ) -> tuple[Any, Any, Any]:
        def scan_step(carry: tuple[Any, Any], _: Any) -> tuple[tuple[Any, Any], tuple[Any, Any, Any]]:
            current_state, current_observation = carry
            actions = actor_apply(policy, current_observation.policy, self.observation_layout)
            next_state, output = self.env._step(
                current_state,
                actions,
                learning_iteration,
                current_state.demo_obs,
                curriculum_stage,
                forced_reset_patch_index,
            )
            frame = (
                next_state.physics.qpos[env_index],
                next_state.physics.qvel[env_index],
                next_state.patch_index[env_index],
            )
            return (next_state, output.observation), frame

        _, frames = self.jax.lax.scan(
            scan_step,
            (env_state, observation),
            None,
            length=video_length,
        )
        return frames

    def _collect_rollout_scan(
        self,
        policy: Any,
        discriminator: Any,
        normalizer: Any,
        env_state: NativeEnvState,
        observation: Any,
        rng: Any,
        learning_iteration: Any,
        curriculum_stage: Any,
        demonstrations: Any,
        video_env_index: Any,
    ) -> tuple[NativeEnvState, Any, Any, RolloutBatch]:
        jax, jnp = self.jax, self.jnp
        std = policy["std"]
        lerp = float(self.amp_cfg["amp_discriminator"]["task_style_lerp"])
        style_scale = float(self.amp_cfg["amp_discriminator"]["style_reward_scale"])

        def scan_step(carry: tuple[Any, Any, Any], demonstration: Any) -> tuple[Any, RolloutStep]:
            current_env_state, current_observation, current_rng = carry
            current_rng, action_key = jax.random.split(current_rng)
            mean = actor_apply(policy, current_observation.policy, self.observation_layout)
            values = critic_apply(policy, current_observation.critic, self.observation_layout)
            actions = mean + std * jax.random.normal(action_key, mean.shape, dtype=mean.dtype)
            next_env_state, output = self.env._step(
                current_env_state,
                actions,
                learning_iteration,
                demonstration,
                curriculum_stage,
            )
            score = discriminator_score(discriminator, normalizer, output.transition_disc)
            amp_reward = style_reward(score, self.env.step_dt, style_scale)
            upper_reward = lerp * output.task_rewards[:, 1] + (1.0 - lerp) * amp_reward
            rewards = jnp.stack((output.task_rewards[:, 0], upper_reward), axis=-1)
            rewards += self.gamma * values * output.time_outs[:, None].astype(jnp.float32)
            step = RolloutStep(
                policy_obs=current_observation.policy,
                critic_obs=current_observation.critic,
                actions=actions,
                old_values=values,
                old_mean=mean,
                old_std=jnp.broadcast_to(std, mean.shape),
                rewards=rewards,
                dones=output.dones,
                disc_obs=output.transition_disc,
                disc_demo_obs=output.observation.disc_demo,
                weighted_terms=output.weighted_terms,
                task_rewards=output.task_rewards,
                style_rewards=amp_reward,
                disc_scores=score,
                completed_term_sums=output.completed_term_sums,
                completed_lengths=output.completed_lengths,
                completed_patch_index=current_env_state.patch_index,
                terminated=output.terminated,
                video_qpos=next_env_state.physics.qpos[video_env_index],
                video_qvel=next_env_state.physics.qvel[video_env_index],
                video_patch_index=next_env_state.patch_index[video_env_index],
            )
            return (next_env_state, output.observation, current_rng), step

        (env_state, observation, rng), steps = jax.lax.scan(
            scan_step,
            (env_state, observation, rng),
            demonstrations,
        )
        last_values = critic_apply(policy, observation.critic, self.observation_layout)
        returns, advantages = self._compute_gae(
            steps.rewards,
            steps.dones,
            steps.old_values,
            last_values,
        )
        rollout = RolloutBatch(
            policy_obs=steps.policy_obs,
            critic_obs=steps.critic_obs,
            actions=steps.actions,
            old_values=steps.old_values,
            old_mean=steps.old_mean,
            old_std=steps.old_std,
            rewards=steps.rewards,
            dones=steps.dones,
            returns=returns,
            advantages=advantages,
            disc_obs=steps.disc_obs,
            disc_demo_obs=steps.disc_demo_obs,
            weighted_terms=steps.weighted_terms,
            task_rewards=steps.task_rewards,
            style_rewards=steps.style_rewards,
            disc_scores=steps.disc_scores,
            completed_term_sums=steps.completed_term_sums,
            completed_lengths=steps.completed_lengths,
            completed_patch_index=steps.completed_patch_index,
            completed_mask=steps.dones,
            terminated=steps.terminated,
            video_qpos=steps.video_qpos,
            video_qvel=steps.video_qvel,
            video_patch_index=steps.video_patch_index,
        )
        return env_state, observation, rng, rollout

    def update(self, rollout: RolloutBatch) -> dict[str, float]:
        jax, jnp = self.jax, self.jnp
        count = self.steps_per_env * self.env.num_envs
        mini_batch_size = count // self.num_mini_batches
        if count % self.num_mini_batches:
            raise ValueError(f"Rollout batch {count} is not divisible by {self.num_mini_batches} mini-batches")
        state = self.state
        rng, policy_permutation_key, disc_policy_key, disc_demo_key = jax.random.split(state.rng, 4)
        policy_indices = jax.random.permutation(policy_permutation_key, count)
        flattened = {
            "policy_obs": rollout.policy_obs.reshape(count, -1),
            "critic_obs": rollout.critic_obs.reshape(count, -1),
            "actions": rollout.actions.reshape(count, -1),
            "old_values": rollout.old_values.reshape(count, 2),
            "old_mean": rollout.old_mean.reshape(count, -1),
            "old_std": rollout.old_std.reshape(count, -1),
            "returns": rollout.returns.reshape(count, 2),
            "advantages": rollout.advantages.reshape(count, 2),
        }
        replay_length = int(np.asarray(jax.device_get(self.replay.length)))
        total_replay = replay_length * self.env.num_envs
        if total_replay < count:
            raise RuntimeError(f"AMP replay has {total_replay} samples but PPO requires {count}")
        policy_selected = jax.random.permutation(disc_policy_key, total_replay)[:count]
        demo_selected = jax.random.permutation(disc_demo_key, total_replay)[:count]
        policy_time = policy_selected // self.env.num_envs
        policy_env = policy_selected % self.env.num_envs
        demo_time = demo_selected // self.env.num_envs
        demo_env = demo_selected % self.env.num_envs
        disc_policy = self.replay.policy[policy_time, policy_env]
        disc_demo = self.replay.demonstration[demo_time, demo_env]

        policy_metrics: list[Any] = []
        disc_metrics: list[Any] = []
        for epoch in range(self.num_epochs):
            rng, disc_epoch_key = jax.random.split(rng)
            disc_indices = jax.random.permutation(disc_epoch_key, count)
            for mini_batch in range(self.num_mini_batches):
                start = mini_batch * mini_batch_size
                stop = start + mini_batch_size
                indices = policy_indices[start:stop]
                batch = {key: value[indices] for key, value in flattened.items()}
                policy, policy_optimizer, learning_rate, metrics = self._policy_update_jit(
                    state.policy,
                    state.policy_optimizer,
                    state.learning_rate,
                    batch,
                )
                disc_batch_indices = disc_indices[start:stop]
                discriminator, disc_optimizer, normalizer, disc_result = self._disc_update_jit(
                    state.discriminator,
                    state.discriminator_optimizer,
                    state.normalizer,
                    disc_policy[disc_batch_indices],
                    disc_demo[disc_batch_indices],
                )
                state = state._replace(
                    policy=policy,
                    policy_optimizer=policy_optimizer,
                    learning_rate=learning_rate,
                    discriminator=discriminator,
                    discriminator_optimizer=disc_optimizer,
                    normalizer=normalizer,
                )
                policy_metrics.append(metrics)
                disc_metrics.append(disc_result)
        state = state._replace(rng=rng, iteration=state.iteration + 1)
        self.state = state
        policy_values = jax.device_get(jax.tree_util.tree_map(lambda *values: jnp.mean(jnp.stack(values)), *policy_metrics))
        disc_values = jax.device_get(jax.tree_util.tree_map(lambda *values: jnp.mean(jnp.stack(values)), *disc_metrics))
        result: dict[str, float] = {}
        for key, value in policy_values.items():
            if key.startswith("grad_layer/"):
                result[f"Gradient/policy/{key.removeprefix('grad_layer/')}"] = float(value)
            elif key.startswith("grad/"):
                result[f"Gradient/policy/{key.removeprefix('grad/')}"] = float(value)
            else:
                result[f"Loss/{key}"] = float(value)
        for key, value in disc_values.items():
            if key.startswith("grad_layer/"):
                result[f"Gradient/discriminator/{key.removeprefix('grad_layer/')}"] = float(value)
            elif key.startswith("grad/"):
                result[f"Gradient/discriminator/{key.removeprefix('grad/')}"] = float(value)
            else:
                result[f"Loss/amp/{key}"] = float(value)
        result["Loss/learning_rate"] = float(np.asarray(jax.device_get(state.learning_rate)))
        result["Policy/mean_noise_std"] = float(np.asarray(jax.device_get(jnp.mean(state.policy["std"]))))
        return result

    def iteration_metrics(self, rollout: RolloutBatch, collect_time: float, learn_time: float) -> dict[str, float]:
        jax, jnp = self.jax, self.jnp
        steps_per_second = self.steps_per_env * self.env.num_envs / max(collect_time + learn_time, 1.0e-9)
        metrics = {
            "Perf/collection_time": collect_time,
            "Perf/learning_time": learn_time,
            "Perf/total_fps": float(int(steps_per_second)),
            "Perf/collect_seconds": collect_time,
            "Perf/learn_seconds": learn_time,
            "Perf/steps_per_second": steps_per_second,
            "Reward/lower_task": float(np.asarray(jax.device_get(jnp.mean(rollout.task_rewards[..., 0])))),
            "Reward/upper_task": float(np.asarray(jax.device_get(jnp.mean(rollout.task_rewards[..., 1])))),
            "Reward/style": float(np.asarray(jax.device_get(jnp.mean(rollout.style_rewards)))),
            "Reward/total": float(np.asarray(jax.device_get(jnp.mean(jnp.sum(rollout.rewards, axis=-1))))),
            "AMP/disc_score": float(np.asarray(jax.device_get(jnp.mean(rollout.disc_scores)))),
        }
        mask = np.asarray(jax.device_get(rollout.completed_mask), dtype=bool)
        if mask.any():
            completed_lengths = np.asarray(jax.device_get(rollout.completed_lengths))[mask]
            completed_terms = np.asarray(jax.device_get(rollout.completed_term_sums))[mask]
            completed_terminated = np.asarray(jax.device_get(rollout.terminated))[mask]
            completed_patch_index = np.asarray(jax.device_get(rollout.completed_patch_index), dtype=np.int32)[mask]
            metrics["Train/mean_episode_length"] = float(completed_lengths.mean())
            metrics["Train/termination_rate"] = float(completed_terminated.mean())
            for index, name in enumerate(REWARD_TERM_NAMES):
                metrics[f"Episode_Reward/{name}"] = float(completed_terms[:, index].mean() / self.env.spec.episode_length_s)
            patch_stages = np.asarray(jax.device_get(self.env.patch_stages), dtype=np.int32)
            completed_patch_index = np.clip(completed_patch_index, 0, patch_stages.shape[0] - 1)
            completed_stages = patch_stages[completed_patch_index]
            for stage_index in range(6):
                stage_mask = completed_stages == stage_index
                if not stage_mask.any():
                    continue
                metrics[f"Train/mean_episode_length_stage_{stage_index}"] = float(
                    completed_lengths[stage_mask].mean()
                )
                metrics[f"Train/termination_rate_stage_{stage_index}"] = float(
                    completed_terminated[stage_mask].mean()
                )
                metrics[f"Train/completed_episodes_stage_{stage_index}"] = float(stage_mask.sum())
        return metrics

    def train_iteration(self, video_recorder: Any | None = None) -> tuple[dict[str, float], RolloutBatch]:
        start = time.perf_counter()
        rollout = self.collect_rollout(video_recorder=video_recorder)
        self.jax.block_until_ready(rollout.rewards)
        collect_time = time.perf_counter() - start
        start = time.perf_counter()
        metrics = self.update(rollout)
        self.jax.block_until_ready(self.state.learning_rate)
        learn_time = time.perf_counter() - start
        metrics.update(self.iteration_metrics(rollout, collect_time, learn_time))
        self._update_curriculum(metrics)
        self._add_curriculum_stage_metrics(metrics)
        return metrics, rollout

    def _update_curriculum(self, metrics: dict[str, float]) -> None:
        """Keep compatibility metrics without globally reassigning environment stages.

        Terrain promotion/demotion is performed inside ``NativeJaxAmpEnv._step``
        from each environment's completed episode.  In particular, this function
        must never reinitialize all slots at one harder stage.
        """
        stage = int(self.curriculum_cfg.start_stage)
        metrics["Curriculum/stage"] = float(stage)
        metrics["Curriculum/global_stage"] = float(stage)
        metrics["Curriculum/promoted"] = 0.0

    def _add_curriculum_stage_metrics(self, metrics: dict[str, float]) -> None:
        jax = self.jax
        target_stages = np.asarray(jax.device_get(self.env_state.terrain_stage), dtype=np.int32)
        if target_stages.size == 0:
            return

        max_stage = max(5, int(target_stages.max()))
        counts = np.bincount(target_stages, minlength=max_stage + 1)
        denominator = max(int(self.env.num_envs), 1)
        for stage_index, count in enumerate(counts):
            metrics[f"Curriculum/stage_env_count_{stage_index}"] = float(count)
            metrics[f"Curriculum/stage_env_fraction_{stage_index}"] = float(count / denominator)

    def save(self, path: Path) -> Path:
        jax = self.jax
        output = path.expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        state = jax.device_get(self.state)
        flat: dict[str, np.ndarray] = {
            "__metadata_json__": np.asarray(
                json.dumps(
                    {
                        "format": CHECKPOINT_FORMAT,
                        "native_training_checkpoint": True,
                        "iteration": int(np.asarray(state.iteration)),
                        "optimizer_state": "native_jax_adam",
                        "observation_layout": self.observation_layout,
                        "height_scan_reference": "torso_link_frame_z_minus_terrain_hit_z_minus_offset",
                        "height_scan_offset": self.env.spec.height_scan.offset,
                        "source_checkpoint": self.source_metadata.get("source_checkpoint"),
                        "curriculum_history": self.curriculum_history,
                    }
                ),
                dtype=np.str_,
            ),
            "normalizer::mean": np.asarray(state.normalizer["mean"]),
            "normalizer::var": np.asarray(state.normalizer["var"]),
            "normalizer::std": np.asarray(state.normalizer["std"]),
            "normalizer::count": np.asarray(state.normalizer["count"]),
            "trainer::learning_rate": np.asarray(state.learning_rate),
            "trainer::iteration": np.asarray(state.iteration),
            "trainer::curriculum_stage": np.asarray(state.curriculum_stage),
            "trainer::curriculum_stage_start_iteration": np.asarray(
                state.curriculum_stage_start_iteration
            ),
            "env::terrain_stage": np.asarray(jax.device_get(self.env_state.terrain_stage)),
        }
        for prefix, tree in (("policy", state.policy), ("discriminator", state.discriminator)):
            for key, value in tree.items():
                flat[f"{prefix}::{key}"] = np.asarray(value)
        for prefix, optimizer in (
            ("policy_optimizer", state.policy_optimizer),
            ("discriminator_optimizer", state.discriminator_optimizer),
        ):
            flat[f"{prefix}::t"] = np.asarray(optimizer["t"])
            for moment in ("m", "v"):
                for key, value in optimizer[moment].items():
                    flat[f"{prefix}::{moment}::{key}"] = np.asarray(value)
        np.savez_compressed(output, **flat)
        return output

    def _load_resume(self, path: Path, state: NativeTrainState) -> NativeTrainState:
        import jax.numpy as jnp

        with np.load(path.expanduser().resolve(), allow_pickle=False) as payload:
            metadata = json.loads(str(payload["__metadata_json__"].item()))
            if not metadata.get("native_training_checkpoint"):
                raise ValueError("--resume requires a native training checkpoint, not a converted warm start")
            self.curriculum_history = []
            if "env::terrain_stage" in payload.files:
                terrain_stages = np.asarray(payload["env::terrain_stage"], dtype=np.int32).reshape(-1)
                if terrain_stages.shape != (self.env.num_envs,):
                    raise ValueError(
                        "Checkpoint terrain-stage count does not match --num-envs: "
                        f"{terrain_stages.shape[0]} != {self.env.num_envs}."
                    )
                self._resume_terrain_stages = terrain_stages

            def optimizer(prefix: str, template: Any) -> dict[str, Any]:
                return {
                    "t": jnp.asarray(np.asarray(payload[f"{prefix}::t"], dtype=np.int32)),
                    "m": {
                        key: jnp.asarray(np.asarray(payload[f"{prefix}::m::{key}"], dtype=np.float32))
                        for key in template["m"]
                    },
                    "v": {
                        key: jnp.asarray(np.asarray(payload[f"{prefix}::v::{key}"], dtype=np.float32))
                        for key in template["v"]
                    },
                }

            return state._replace(
                policy_optimizer=optimizer("policy_optimizer", state.policy_optimizer),
                discriminator_optimizer=optimizer("discriminator_optimizer", state.discriminator_optimizer),
                learning_rate=jnp.asarray(np.asarray(payload["trainer::learning_rate"], dtype=np.float32)),
                iteration=jnp.asarray(np.asarray(payload["trainer::iteration"], dtype=np.int32)),
                curriculum_stage=jnp.asarray(self.curriculum_cfg.start_stage, dtype=np.int32),
                curriculum_stage_start_iteration=jnp.asarray(0, dtype=np.int32),
            )

    def _append_replay_batch(self, replay: ReplayState, policy: Any, demonstration: Any) -> ReplayState:
        capacity = replay.policy.shape[0]
        count = policy.shape[0]
        indices = (replay.pointer + self.jnp.arange(count, dtype=self.jnp.int32)) % capacity
        return ReplayState(
            policy=replay.policy.at[indices].set(policy),
            demonstration=replay.demonstration.at[indices].set(demonstration),
            pointer=(replay.pointer + count) % capacity,
            length=self.jnp.minimum(replay.length + count, capacity),
        )

    def _compute_gae(self, rewards: Any, dones: Any, values: Any, last_values: Any) -> tuple[Any, Any]:
        jax, jnp = self.jax, self.jnp
        next_values = jnp.concatenate((values[1:], last_values[None]), axis=0)

        def step(advantage: Any, inputs: tuple[Any, Any, Any, Any]) -> tuple[Any, Any]:
            reward, done, value, next_value = inputs
            non_terminal = 1.0 - done[:, None].astype(jnp.float32)
            delta = reward + non_terminal * self.gamma * next_value - value
            advantage = delta + non_terminal * self.gamma * self.lam * advantage
            return advantage, advantage + value

        _, reversed_returns = jax.lax.scan(
            step,
            jnp.zeros_like(last_values),
            tuple(value[::-1] for value in (rewards, dones, values, next_values)),
        )
        returns = reversed_returns[::-1]
        advantages = returns - values
        sample_count = advantages.shape[0] * advantages.shape[1]
        mean = jnp.mean(advantages, axis=(0, 1), keepdims=True)
        variance = jnp.sum((advantages - mean) ** 2, axis=(0, 1), keepdims=True) / max(sample_count - 1, 1)
        return returns, (advantages - mean) / (jnp.sqrt(variance) + 1.0e-8)

    def _policy_update(
        self,
        params: Any,
        optimizer: Any,
        learning_rate: Any,
        batch: dict[str, Any],
    ) -> tuple[Any, Any, Any, dict[str, Any]]:
        jax, jnp = self.jax, self.jnp
        micro_batches = jax.tree_util.tree_map(
            lambda value: value.reshape(
                (self.num_micro_batches, self.micro_batch_size) + value.shape[1:]
            ),
            batch,
        )

        def gradients_for(micro_batch: dict[str, Any]) -> tuple[Any, dict[str, Any]]:
            batch_size = micro_batch["actions"].shape[0]
            policy_obs, actions = augment_policy_batch(
                micro_batch["policy_obs"],
                micro_batch["actions"],
                self.observation_layout,
            )
            critic_obs = jnp.concatenate((micro_batch["critic_obs"], micro_batch["critic_obs"]), axis=0)
            old_values = jnp.concatenate((micro_batch["old_values"], micro_batch["old_values"]), axis=0)
            returns = jnp.concatenate((micro_batch["returns"], micro_batch["returns"]), axis=0)
            advantages = jnp.concatenate((micro_batch["advantages"], micro_batch["advantages"]), axis=0)
            old_lower, old_upper = gaussian_log_prob_parts(
                micro_batch["old_mean"],
                micro_batch["old_std"],
                micro_batch["actions"],
            )
            old_lower = jnp.concatenate((old_lower, old_lower), axis=0)
            old_upper = jnp.concatenate((old_upper, old_upper), axis=0)

            def loss_function(current: Any) -> tuple[Any, dict[str, Any]]:
                mean = actor_apply(current, policy_obs, self.observation_layout)
                value = critic_apply(current, critic_obs, self.observation_layout)
                std = current["std"]
                current_std = jnp.broadcast_to(std, mean.shape)
                lower_log, upper_log = gaussian_log_prob_parts(mean, current_std, actions)
                lower_ratio, upper_ratio = jnp.exp(lower_log - old_lower), jnp.exp(upper_log - old_upper)
                lower_advantage, upper_advantage = advantages[:, 0], advantages[:, 1]
                lower_loss = jnp.mean(
                    jnp.maximum(
                        -lower_advantage * lower_ratio,
                        -lower_advantage
                        * jnp.clip(lower_ratio, 1.0 - self.clip_param, 1.0 + self.clip_param),
                    )
                )
                upper_loss = jnp.mean(
                    jnp.maximum(
                        -upper_advantage * upper_ratio,
                        -upper_advantage
                        * jnp.clip(upper_ratio, 1.0 - self.clip_param, 1.0 + self.clip_param),
                    )
                )
                surrogate = 0.5 * (lower_loss + upper_loss)
                value_clipped = old_values + jnp.clip(value - old_values, -self.clip_param, self.clip_param)
                value_loss = jnp.mean(jnp.maximum((value - returns) ** 2, (value_clipped - returns) ** 2))
                entropy = jnp.mean(gaussian_entropy(jnp.broadcast_to(std, (batch_size, 29))))
                mirrored_target = jax.lax.stop_gradient(switch_joints(mean[:batch_size]))
                symmetry = jnp.mean((mean[batch_size:] - mirrored_target) ** 2)
                total = surrogate + self.value_loss_coef * value_loss - self.entropy_coef * entropy
                total += self.symmetry_coeff * symmetry
                current_std_original = jnp.broadcast_to(std, micro_batch["old_std"].shape)
                kl = jnp.mean(
                    jnp.sum(
                        jnp.log(current_std_original / micro_batch["old_std"] + 1.0e-5)
                        + (
                            micro_batch["old_std"] ** 2
                            + (micro_batch["old_mean"] - mean[:batch_size]) ** 2
                        )
                        / (2.0 * current_std_original**2)
                        - 0.5,
                        axis=-1,
                    )
                )
                return total, {
                    "value": value_loss,
                    "surrogate": surrogate,
                    "surrogate_lower": lower_loss,
                    "surrogate_upper": upper_loss,
                    "entropy": entropy,
                    "symmetry": symmetry,
                    "kl": kl,
                }

            (_, metrics), gradients = jax.value_and_grad(loss_function, has_aux=True)(params)
            return gradients, metrics

        first_batch = jax.tree_util.tree_map(lambda value: value[0], micro_batches)
        gradients, metrics = gradients_for(first_batch)
        if self.num_micro_batches > 1:
            remaining = jax.tree_util.tree_map(lambda value: value[1:], micro_batches)

            def accumulate(carry: tuple[Any, Any], micro_batch: Any) -> tuple[Any, None]:
                gradient_sum, metric_sum = carry
                next_gradients, next_metrics = gradients_for(micro_batch)
                gradient_sum = jax.tree_util.tree_map(
                    lambda total, value: total + value,
                    gradient_sum,
                    next_gradients,
                )
                metric_sum = jax.tree_util.tree_map(
                    lambda total, value: total + value,
                    metric_sum,
                    next_metrics,
                )
                return (gradient_sum, metric_sum), None

            (gradients, metrics), _ = jax.lax.scan(accumulate, (gradients, metrics), remaining)
            divisor = float(self.num_micro_batches)
            gradients = jax.tree_util.tree_map(lambda value: value / divisor, gradients)
            metrics = jax.tree_util.tree_map(lambda value: value / divisor, metrics)

        new_learning_rate = jnp.where(
            metrics["kl"] > self.desired_kl * 2.0,
            jnp.maximum(1.0e-5, learning_rate / 1.5),
            jnp.where(
                (metrics["kl"] < self.desired_kl / 2.0) & (metrics["kl"] > 0.0),
                jnp.minimum(1.0e-2, learning_rate * 1.5),
                learning_rate,
            ),
        )
        raw_grad_norm = tree_l2_norm(gradients)
        grad_scale = jnp.minimum(1.0, self.max_grad_norm / (raw_grad_norm + 1.0e-8))
        metrics["grad/total_raw_norm"] = raw_grad_norm
        metrics["grad/total_clipped_norm"] = raw_grad_norm * grad_scale
        for key, value in gradients.items():
            metrics[f"grad_layer/{key}"] = jnp.linalg.norm(value)
        params, optimizer = adam_update(
            params,
            gradients,
            optimizer,
            new_learning_rate,
            self.max_grad_norm,
        )
        return params, optimizer, new_learning_rate, metrics

    def _discriminator_update(
        self,
        params: Any,
        optimizer: Any,
        normalizer: Any,
        policy_obs: Any,
        demonstration_obs: Any,
    ) -> tuple[Any, Any, Any, dict[str, Any]]:
        jax, jnp = self.jax, self.jnp
        policy_batches = policy_obs.reshape(
            (self.num_micro_batches, self.micro_batch_size) + policy_obs.shape[1:]
        )
        demo_batches = demonstration_obs.reshape(
            (self.num_micro_batches, self.micro_batch_size) + demonstration_obs.shape[1:]
        )

        def gradients_for(inputs: tuple[Any, Any]) -> tuple[Any, dict[str, Any]]:
            policy_batch, demo_batch = inputs
            policy_normalized = jax.lax.stop_gradient(
                normalize_discriminator_obs(policy_batch, normalizer)
            ).reshape(self.micro_batch_size, -1)
            demo_normalized = jax.lax.stop_gradient(
                normalize_discriminator_obs(demo_batch, normalizer)
            ).reshape(self.micro_batch_size, -1)

            def loss_function(current: Any) -> tuple[Any, dict[str, Any]]:
                policy_score = discriminator_apply(current, policy_normalized)
                demo_score = discriminator_apply(current, demo_normalized)
                policy_loss = jnp.mean((policy_score + 1.0) ** 2)
                demo_loss = jnp.mean((demo_score - 1.0) ** 2)
                disc_loss = 0.5 * (policy_loss + demo_loss)

                def score_one(input_obs: Any) -> Any:
                    return discriminator_apply(current, input_obs[None])[0]

                input_gradient = jax.vmap(jax.grad(score_one))(demo_normalized)
                penalty = float(self.amp_cfg["grad_penalty_scale"]) * jnp.mean(
                    jnp.linalg.norm(input_gradient, axis=1) ** 2
                )
                return disc_loss + penalty, {
                    "disc_loss": disc_loss,
                    "disc_grad_penalty": penalty,
                    "disc_score": jnp.mean(policy_score),
                    "disc_demo_score": jnp.mean(demo_score),
                }

            (_, metrics), gradients = jax.value_and_grad(loss_function, has_aux=True)(params)
            return gradients, metrics

        gradients, metrics = gradients_for((policy_batches[0], demo_batches[0]))
        if self.num_micro_batches > 1:

            def accumulate(carry: tuple[Any, Any], inputs: Any) -> tuple[Any, None]:
                gradient_sum, metric_sum = carry
                next_gradients, next_metrics = gradients_for(inputs)
                gradient_sum = jax.tree_util.tree_map(
                    lambda total, value: total + value,
                    gradient_sum,
                    next_gradients,
                )
                metric_sum = jax.tree_util.tree_map(
                    lambda total, value: total + value,
                    metric_sum,
                    next_metrics,
                )
                return (gradient_sum, metric_sum), None

            (gradients, metrics), _ = jax.lax.scan(
                accumulate,
                (gradients, metrics),
                (policy_batches[1:], demo_batches[1:]),
            )
            divisor = float(self.num_micro_batches)
            gradients = jax.tree_util.tree_map(lambda value: value / divisor, gradients)
            metrics = jax.tree_util.tree_map(lambda value: value / divisor, metrics)
        metrics["grad/total_raw_norm"] = tree_l2_norm(gradients)
        for key, value in gradients.items():
            metrics[f"grad_layer/{key}"] = jnp.linalg.norm(value)
        decay = {
            key: jnp.asarray(
                self.amp_cfg["disc_linear_weight_decay"]
                if key.startswith("disc_linear.")
                else self.amp_cfg["disc_trunk_weight_decay"],
                dtype=value.dtype,
            )
            for key, value in params.items()
        }
        # The source PPOAMP stores disc_max_grad_norm but does not apply it.
        params, optimizer = adam_update(
            params,
            gradients,
            optimizer,
            jnp.asarray(self.amp_cfg["disc_learning_rate"], dtype=jnp.float32),
            max_grad_norm=1.0e30,
            weight_decay=decay,
        )
        normalizer = update_normalizer(normalizer, policy_obs)
        return params, optimizer, normalizer, metrics
