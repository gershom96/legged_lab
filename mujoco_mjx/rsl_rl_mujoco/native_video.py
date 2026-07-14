from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

import cv2
import mujoco
import numpy as np

from .native_env import NativeJaxAmpEnv
from .native_model import actor_apply


class NativeMujocoVideoRecorder:
    """Capture small frozen-policy eval rollouts and render them after the GPU rollout completes."""

    def __init__(
        self,
        env: Any,
        log_dir: Path,
        interval_iterations: int,
        video_length: int,
        env_index: int = 0,
        width: int = 640,
        height: int = 480,
        log_to_wandb: bool = False,
    ) -> None:
        if interval_iterations <= 0 or video_length <= 0:
            raise ValueError("Video interval and length must be positive")
        if not 0 <= env_index < env.num_envs:
            raise ValueError(f"Video environment {env_index} is outside [0, {env.num_envs})")
        self.env = env
        self.log_dir = log_dir
        self.interval_iterations = interval_iterations
        self.video_length = video_length
        self.env_index = env_index
        self.width = width
        self.height = height
        self.log_to_wandb = log_to_wandb
        self.fps = max(1, int(round(1.0 / env.step_dt)))
        self.output_dir = log_dir / "videos" / "train"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.last_trigger: int | None = None
        self._video_env_cache: dict[int, NativeJaxAmpEnv] = {}
        self._rollout_cache: dict[int, Any] = {}

    def maybe_record(self, trainer: Any, force: bool = False) -> None:
        iteration = int(np.asarray(trainer.jax.device_get(trainer.state.iteration)))
        if not force and iteration % self.interval_iterations != 0:
            return
        if iteration == self.last_trigger:
            return
        self.last_trigger = iteration
        active_stages = np.unique(
            np.asarray(trainer.jax.device_get(trainer.env_state.terrain_stage), dtype=np.int32)
        )
        records = self._video_records_for_stage(trainer, active_stages)
        if not records:
            return
        iteration_dir = self.output_dir / f"iteration_{iteration:08d}"
        iteration_dir.mkdir(parents=True, exist_ok=True)
        video_env = self._get_video_env(len(records))
        print(
            "[INFO] Running frozen-policy video evaluation for "
            f"{len(records)} curriculum stage(s) at iteration {iteration}"
        )
        qpos, qvel, patch_index = self._evaluate_records(trainer, video_env, records, iteration)
        for slot, (stage, _, patch_name) in enumerate(records):
            path = iteration_dir / f"stage_{stage}_{patch_name}.mp4"
            frames = (qpos[:, slot], qvel[:, slot], patch_index[:, slot])
            self._render(
                path,
                iteration,
                stage,
                frames,
                wandb_key=f"Video/stage_{stage}_{patch_name}",
                render_env=video_env,
            )

    def close(self) -> None:
        pass

    def _get_video_env(self, num_envs: int) -> NativeJaxAmpEnv:
        cached = self._video_env_cache.get(num_envs)
        if cached is not None:
            return cached
        video_spec = replace(self.env.spec, num_envs=num_envs)
        build_dir = self.log_dir / "video_eval_build" / f"envs_{num_envs}"
        video_env = NativeJaxAmpEnv(
            video_spec,
            self.env.split_reward_cfg,
            build_dir,
            motion_cache=(self.env.motion_data, self.env.num_motion_files),
        )
        self._video_env_cache[num_envs] = video_env
        return video_env

    def _evaluate_records(
        self,
        trainer: Any,
        video_env: NativeJaxAmpEnv,
        records: list[tuple[int, int | None, str]],
        iteration: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if video_env.terrain.patches:
            forced_patch = np.asarray(
                [0 if patch_index is None else patch_index for _, patch_index, _ in records],
                dtype=np.int32,
            )
            forced_patch_jax = video_env.jnp.asarray(forced_patch, dtype=video_env.jnp.int32)
            init_forced_patch: Any = forced_patch_jax
        else:
            forced_patch_jax = video_env.jnp.full((len(records),), -1, dtype=video_env.jnp.int32)
            init_forced_patch = None
        eval_state, eval_observation = video_env.initialize(
            seed=video_env.spec.seed + 1000003 + iteration,
            terrain_stages=video_env.jnp.asarray(
                [stage for stage, _, _ in records], dtype=video_env.jnp.int32
            ),
            forced_patch_index=init_forced_patch,
            sample_demonstration=False,
        )
        rollout = self._get_rollout(video_env)
        qpos, qvel, patch_index = rollout(
            trainer.state.policy,
            eval_state,
            eval_observation,
            trainer.state.iteration,
            video_env.jnp.asarray(0, dtype=video_env.jnp.int32),
            forced_patch_jax,
        )
        video_env.jax.block_until_ready(qpos)
        return video_env.jax.device_get((qpos, qvel, patch_index))

    def _get_rollout(self, video_env: NativeJaxAmpEnv) -> Any:
        num_envs = video_env.num_envs
        cached = self._rollout_cache.get(num_envs)
        if cached is not None:
            return cached

        def rollout(
            policy: Any,
            env_state: Any,
            observation: Any,
            learning_iteration: Any,
            curriculum_stage: Any,
            forced_reset_patch_index: Any,
        ) -> tuple[Any, Any, Any]:
            def scan_step(carry: tuple[Any, Any], _: Any) -> tuple[tuple[Any, Any], tuple[Any, Any, Any]]:
                current_state, current_observation = carry
                actions = actor_apply(policy, current_observation.policy, video_env.spec.observation_layout)
                next_state, output = video_env._step(
                    current_state,
                    actions,
                    learning_iteration,
                    current_state.demo_obs,
                    curriculum_stage,
                    forced_reset_patch_index,
                )
                frame = (
                    next_state.physics.qpos,
                    next_state.physics.qvel,
                    next_state.patch_index,
                )
                return (next_state, output.observation), frame

            _, frames = video_env.jax.lax.scan(
                scan_step,
                (env_state, observation),
                None,
                length=self.video_length,
            )
            return frames

        compiled = video_env.jax.jit(rollout)
        self._rollout_cache[num_envs] = compiled
        return compiled

    def _video_records_for_stage(self, trainer: Any, active_stages: np.ndarray) -> list[tuple[int, int | None, str]]:
        if not self.env.terrain.patches:
            return [(0, None, self.env.terrain.terrain_type)]
        counts = np.asarray(trainer.jax.device_get(self.env.stage_patch_counts))
        indices = np.asarray(trainer.jax.device_get(self.env.stage_patch_indices))
        records: list[tuple[int, int | None, str]] = []
        for stage in sorted(set(int(value) for value in active_stages)):
            if stage >= len(counts) or counts[stage] <= 0:
                continue
            patch_index = int(indices[stage, 0])
            patch = self.env.terrain.patches[patch_index]
            if patch.stage != stage:
                continue
            patch_name = patch.name
            records.append((stage, patch_index, patch_name))
        return records

    def _render(
        self,
        path: Path,
        iteration: int,
        curriculum_stage: int,
        frames: Any,
        wandb_key: str,
        render_env: Any | None = None,
    ) -> None:
        render_env = self.env if render_env is None else render_env
        qpos_frames, qvel_frames, patch_frames = frames

        writer: cv2.VideoWriter | None = None
        renderer: mujoco.Renderer | None = None
        try:
            renderer = mujoco.Renderer(render_env.model, height=self.height, width=self.width)
            render_data = mujoco.MjData(render_env.model)
            camera = mujoco.MjvCamera()
            mujoco.mjv_defaultCamera(camera)
            writer = cv2.VideoWriter(
                str(path),
                cv2.VideoWriter_fourcc(*"mp4v"),
                self.fps,
                (self.width, self.height),
            )
            if not writer.isOpened():
                raise RuntimeError(f"Could not open MP4 writer for {path}")
            for qpos, qvel, patch_index in zip(qpos_frames, qvel_frames, patch_frames):
                patch_index = int(patch_index)
                render_data.qpos[:] = qpos
                render_data.qvel[:] = qvel
                mujoco.mj_forward(render_env.model, render_data)
                root = qpos[:3]
                camera.type = mujoco.mjtCamera.mjCAMERA_FREE
                camera.lookat[:] = root + np.asarray((0.0, 0.0, 0.45))
                camera.distance = 3.5
                camera.azimuth = 135.0
                camera.elevation = -20.0
                renderer.update_scene(render_data, camera=camera)
                frame = cv2.cvtColor(renderer.render(), cv2.COLOR_RGB2BGR)
                patch_name = (
                    render_env.terrain.patches[patch_index].name
                    if render_env.terrain.patches
                    else render_env.terrain.terrain_type
                )
                cv2.putText(
                    frame,
                    f"iteration {iteration} | stage {curriculum_stage} | {patch_name}",
                    (16, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (245, 245, 245),
                    2,
                    cv2.LINE_AA,
                )
                writer.write(frame)
        finally:
            if writer is not None:
                writer.release()
            if renderer is not None:
                renderer.close()

        print(f"[INFO] Saved native-JAX video: {path}")
        if self.log_to_wandb:
            try:
                import wandb

                if wandb.run is not None:
                    wandb.log({wandb_key: wandb.Video(str(path), format="mp4")}, step=iteration)
            except Exception as exc:
                print(f"[WARN] Could not upload native-JAX video to W&B: {exc}")
