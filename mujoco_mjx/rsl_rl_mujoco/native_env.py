from __future__ import annotations

from pathlib import Path
from typing import Any, NamedTuple

import numpy as np

from .constants import (
    ANKLE_ACTION_INDICES,
    CRITIC_STATE_DIM,
    DEFAULT_JOINT_POS_ISAAC,
    ISAAC_INDICES_IN_MUJOCO_ORDER,
    LOWER_ACTION_INDICES,
    MUJOCO_INDICES_IN_ISAAC_ORDER,
    PACKED_ACTOR_FRAME_DIM,
    UPPER_ACTION_INDICES,
    UPPER_KEY_BODY_INDICES,
)
from .model import build_mujoco_model
from .native_math import (
    projected_gravity,
    quat_apply_inverse,
    root_local_rot_tan_norm,
    wrap_to_pi,
    yaw_from_quat,
)
from .native_motion import load_motion_data, sample_discriminator_observations_numpy
from .native_physics import PhysicsRandomization, NativeMjxPhysics, validate_native_device
from .spec import MujocoRslRlEnvSpec
from .terrain import TerrainData, generate_terrain


REWARD_TERM_NAMES = (
    "track_lin_vel_xy_exp",
    "track_ang_vel_z_exp",
    "feet_air_time",
    "feet_slide",
    "flat_orientation_l2",
    "ang_vel_xy_l2",
    "lin_vel_z_l2",
    "root_height_below_target",
    "dof_torques_l2_lower",
    "dof_torques_l2_upper",
    "dof_acc_l2_lower",
    "dof_acc_l2_upper",
    "action_rate_l2_lower",
    "action_rate_l2_upper",
    "dof_pos_limits",
    "dof_pos_limits_upper",
    "joint_deviation_lower_body",
    "joint_deviation_arms",
    "termination_penalty",
)


class RobotState(NamedTuple):
    root_pos: Any
    root_quat: Any
    root_lin_vel_w: Any
    root_lin_vel_b: Any
    root_ang_vel_b: Any
    gravity_b: Any
    height_scan_pos_w: Any
    height_scan_quat_w: Any
    joint_pos: Any
    joint_vel: Any
    key_body_pos_w: Any
    key_body_pos_b: Any
    foot_vel_w: Any
    joint_acc: Any
    torque: Any
    foot_contacts: Any


class NativeEnvState(NamedTuple):
    physics: Any
    physics_randomization: PhysicsRandomization
    rng: Any
    episode_length: Any
    command: Any
    heading_target: Any
    command_age: Any
    last_actions: Any
    air_time: Any
    contact_time: Any
    episode_term_sums: Any
    previous_key_body_pos: Any
    previous_joint_vel: Any
    actor_history: Any
    height_history: Any
    disc_history: Any
    actor_scan: Any
    critic_state: Any
    demo_obs: Any
    patch_index: Any
    terrain_stage: Any
    episode_start_xy: Any
    push_countdown: Any


class NativeObservation(NamedTuple):
    policy: Any
    critic: Any
    disc: Any
    disc_demo: Any


class NativeStepOutput(NamedTuple):
    observation: NativeObservation
    transition_disc: Any
    task_rewards: Any
    weighted_terms: Any
    dones: Any
    terminated: Any
    time_outs: Any
    out_of_bounds: Any
    completed_term_sums: Any
    completed_lengths: Any


class NativeJaxAmpEnv:
    """Perception-enabled G1 environment with observations, rewards, resets, and AMP sampling in JAX."""

    def __init__(
        self,
        spec: MujocoRslRlEnvSpec,
        split_reward_cfg: dict[str, Any],
        build_dir: Path,
        motion_cache: tuple[Any, int] | None = None,
    ) -> None:
        import jax
        import jax.numpy as jnp

        validate_native_device(spec)
        self.jax = jax
        self.jnp = jnp
        self.spec = spec
        self.split_reward_cfg = split_reward_cfg
        self.num_envs = spec.num_envs
        self.num_actions = spec.action_dim
        self.step_dt = spec.step_dt
        self.max_episode_length = spec.max_episode_length
        self.height_scan_shape = spec.height_scan.shape
        self.terrain: TerrainData = generate_terrain(spec.terrain)
        self.model, self.metadata, self.scene_xml = build_mujoco_model(spec, self.terrain, build_dir)
        self.physics = NativeMjxPhysics(self.model, self.metadata, spec)
        if motion_cache is None:
            self.motion_data, self.num_motion_files = load_motion_data(spec.motion)
        else:
            self.motion_data, self.num_motion_files = motion_cache
        self.motion_rng = np.random.default_rng(spec.motion.seed)

        self.qpos_indices = jnp.asarray(self.metadata.joint_qpos_indices, dtype=jnp.int32)
        self.qvel_indices = jnp.asarray(self.metadata.joint_qvel_indices, dtype=jnp.int32)
        self.mujoco_indices_in_isaac_order = jnp.asarray(MUJOCO_INDICES_IN_ISAAC_ORDER, dtype=jnp.int32)
        self.isaac_indices_in_mujoco_order = jnp.asarray(ISAAC_INDICES_IN_MUJOCO_ORDER, dtype=jnp.int32)
        self.key_body_ids = jnp.asarray(self.metadata.key_body_ids, dtype=jnp.int32)
        self.height_scan_body_id = self.metadata.height_scan_body_id
        self.default_joint_pos = jnp.asarray(DEFAULT_JOINT_POS_ISAAC)
        limits_mujoco = jnp.asarray(self.metadata.joint_limits_mujoco)
        self.joint_limits = limits_mujoco[self.mujoco_indices_in_isaac_order]
        centers = jnp.mean(self.joint_limits, axis=-1)
        half_ranges = 0.45 * (self.joint_limits[:, 1] - self.joint_limits[:, 0])
        self.soft_joint_limits = jnp.stack((centers - half_ranges, centers + half_ranges), axis=-1)
        self.terrain_heights = jnp.asarray(self.terrain.scan_surface_heights())
        if self.terrain.patches:
            patch_origins = np.asarray([patch.origin for patch in self.terrain.patches], dtype=np.float32)
            patch_extents = np.asarray([patch.safe_half_extent for patch in self.terrain.patches], dtype=np.float32)
            patch_stages = np.asarray([patch.stage for patch in self.terrain.patches], dtype=np.int32)
        else:
            margin = np.minimum(np.asarray(self.terrain.size, dtype=np.float32) * 0.05, 2.0)
            patch_origins = np.asarray([self.terrain.spawn_xy], dtype=np.float32)
            patch_extents = np.asarray([np.asarray(self.terrain.size) / 2.0 - margin], dtype=np.float32)
            patch_stages = np.zeros((1,), dtype=np.int32)
        self.patch_origins = jnp.asarray(patch_origins)
        self.patch_extents = jnp.asarray(patch_extents)
        self.patch_stages = jnp.asarray(patch_stages)
        stage_lists = [np.flatnonzero(patch_stages == stage) for stage in range(6)]
        max_stage_patches = max((len(indices) for indices in stage_lists), default=1)
        stage_patch_indices = np.zeros((6, max_stage_patches), dtype=np.int32)
        stage_patch_counts = np.ones((6,), dtype=np.int32)
        for stage, indices in enumerate(stage_lists):
            if len(indices):
                stage_patch_indices[stage, : len(indices)] = indices
                stage_patch_counts[stage] = len(indices)
        self.stage_patch_indices = jnp.asarray(stage_patch_indices)
        self.stage_patch_counts = jnp.asarray(stage_patch_counts)
        self.lower_indices = jnp.asarray(LOWER_ACTION_INDICES)
        self.upper_indices = jnp.asarray(UPPER_ACTION_INDICES)
        self.ankle_indices = jnp.asarray(ANKLE_ACTION_INDICES)
        self.upper_key_indices = jnp.asarray(UPPER_KEY_BODY_INDICES)
        self.reward_weights = jnp.asarray(
            (
                1.0,
                1.0,
                0.5,
                -0.1,
                -1.0,
                -0.05,
                -0.2,
                -20.0,
                -2.0e-6,
                -2.0e-6,
                -1.0e-7,
                -1.0e-7,
                -0.005,
                -0.005,
                -1.0,
                -0.25,
                -0.1,
                -0.05,
                -200.0,
            ),
            dtype=jnp.float32,
        )
        self.lower_reward_scales, self.upper_reward_scales = self._split_reward_scales(split_reward_cfg)
        x_points, y_points = self.height_scan_shape
        self.scan_x_local = jnp.tile(
            jnp.linspace(-spec.height_scan.size[0] / 2.0, spec.height_scan.size[0] / 2.0, x_points),
            y_points,
        )
        self.scan_y_local = jnp.repeat(
            jnp.linspace(-spec.height_scan.size[1] / 2.0, spec.height_scan.size[1] / 2.0, y_points),
            x_points,
        )
        self._step_jit = jax.jit(self._step)

    def initialize(
        self,
        seed: int | None = None,
        curriculum_stage: int | None = None,
        terrain_stages: Any | None = None,
        forced_patch_index: int | Any | None = None,
        sample_demonstration: bool = True,
        randomize_episode_lengths: bool = False,
    ) -> tuple[NativeEnvState, NativeObservation]:
        jax, jnp = self.jax, self.jnp
        key = jax.random.PRNGKey(self.spec.seed if seed is None else int(seed))
        (
            key,
            patch_key,
            reset_key,
            command_key,
            noise_key,
            episode_key,
            randomization_key,
            push_key,
        ) = jax.random.split(key, 8)
        stage = self.spec.terrain.curriculum.start_stage if curriculum_stage is None else curriculum_stage
        terrain_stage = (
            jnp.full((self.num_envs,), stage, dtype=jnp.int32)
            if terrain_stages is None
            else jnp.asarray(terrain_stages, dtype=jnp.int32).reshape((self.num_envs,))
        )
        if forced_patch_index is None:
            patch_index = self._sample_patch_indices(patch_key, terrain_stage)
        else:
            forced_patch = jnp.asarray(forced_patch_index, dtype=jnp.int32)
            patch_index = (
                jnp.full((self.num_envs,), forced_patch, dtype=jnp.int32)
                if forced_patch.ndim == 0
                else forced_patch.reshape((self.num_envs,))
            )
        qpos, qvel = self._sample_reset_values(reset_key, patch_index)
        physics_randomization = self._sample_physics_randomization(randomization_key)
        physics = self.physics.initial_data(qpos, qvel, physics_randomization)
        key_positions = physics.xpos[:, self.key_body_ids]
        initial_joint_vel = physics.qvel[:, self.qvel_indices][:, self.mujoco_indices_in_isaac_order]
        robot = self._extract_state(physics, jnp.zeros((self.num_envs, 29)), key_positions, initial_joint_vel)
        command = jnp.zeros((self.num_envs, 3), dtype=jnp.float32)
        heading = jnp.zeros((self.num_envs,), dtype=jnp.float32)
        age = jnp.zeros((self.num_envs,), dtype=jnp.int32)
        command, heading, age = self._sample_commands(
            command,
            heading,
            age,
            robot.root_quat,
            jnp.ones((self.num_envs,), dtype=bool),
            command_key,
        )
        last_actions = jnp.zeros((self.num_envs, 29), dtype=jnp.float32)
        noise_keys = jax.random.split(noise_key, 5)
        actor_frame = self._packed_actor_frame(robot, command, last_actions, noise_keys[0], corrupt=True)
        scan = self._height_scan(robot, noise_keys[1], corrupt=False)
        disc = self._disc_frame(robot)
        actor_scan = self._height_scan(robot, noise_keys[2], corrupt=True)
        critic_state = self._critic_state(robot, command, last_actions)
        demo = (
            self.sample_demonstrations()
            if sample_demonstration
            else jnp.zeros(
                (self.num_envs, self.spec.discriminator_history_length, 49), dtype=jnp.float32
            )
        )
        episode_length = jnp.zeros((self.num_envs,), dtype=jnp.int32)
        if randomize_episode_lengths:
            episode_length = jax.random.randint(
                episode_key,
                (self.num_envs,),
                minval=0,
                maxval=self.max_episode_length,
                dtype=jnp.int32,
            )
        state = NativeEnvState(
            physics=physics,
            physics_randomization=physics_randomization,
            rng=key,
            episode_length=episode_length,
            command=command,
            heading_target=heading,
            command_age=age,
            last_actions=last_actions,
            air_time=jnp.zeros((self.num_envs, 2), dtype=jnp.float32),
            contact_time=jnp.zeros((self.num_envs, 2), dtype=jnp.float32),
            episode_term_sums=jnp.zeros((self.num_envs, len(REWARD_TERM_NAMES)), dtype=jnp.float32),
            previous_key_body_pos=robot.key_body_pos_w,
            previous_joint_vel=robot.joint_vel,
            actor_history=jnp.repeat(actor_frame[:, None, :], self.spec.actor_history_length, axis=1),
            height_history=jnp.repeat(scan[:, None, :], self.spec.height_scan.critic_history_length, axis=1),
            disc_history=jnp.repeat(disc[:, None, :], self.spec.discriminator_history_length, axis=1),
            actor_scan=actor_scan,
            critic_state=critic_state,
            demo_obs=demo,
            patch_index=patch_index,
            terrain_stage=terrain_stage,
            episode_start_xy=robot.root_pos[:, :2],
            push_countdown=self._sample_push_countdown(push_key),
        )
        return state, self.observation(state)

    def sample_demonstrations(self) -> Any:
        return self.jnp.asarray(self.sample_demonstration_schedule(1)[0])

    def sample_demonstration_schedule(self, steps: int) -> np.ndarray:
        """Sample one host-side expert batch for every policy step in a rollout."""
        if steps <= 0:
            raise ValueError("Demonstration schedule length must be positive")
        return np.stack(
            tuple(
                sample_discriminator_observations_numpy(
                    self.motion_data,
                    self.motion_rng,
                    self.num_envs,
                    self.spec.discriminator_history_length,
                    self.step_dt,
                )
                for _ in range(steps)
            ),
            axis=0,
        )

    def step(
        self,
        state: NativeEnvState,
        actions: Any,
        learning_iteration: Any,
        demonstration: Any,
        curriculum_stage: Any | None = None,
    ) -> tuple[NativeEnvState, NativeStepOutput]:
        stage = self.spec.terrain.curriculum.start_stage if curriculum_stage is None else curriculum_stage
        return self._step_jit(state, actions, learning_iteration, demonstration, stage)

    def observation(self, state: NativeEnvState) -> NativeObservation:
        jnp = self.jnp
        if self.spec.observation_layout == "legacy_native_history_first_v1":
            return NativeObservation(
                policy=jnp.concatenate((state.actor_history.reshape(self.num_envs, -1), state.actor_scan), axis=-1),
                critic=jnp.concatenate((state.critic_state, state.height_history.reshape(self.num_envs, -1)), axis=-1),
                disc=state.disc_history,
                disc_demo=state.demo_obs,
            )
        return NativeObservation(
            # Preserve the Isaac observation-manager term order.  ``height_scan``
            # is a declared PolicyCfg field and ``packed_actor_obs`` is appended
            # dynamically, so the saved source checkpoints receive
            # [height_scan, packed_history].  OnPolicyRunner discovers that first
            # height-scan term and passes actor_height_scan_slice=(0, 187).
            policy=jnp.concatenate((state.actor_scan, state.actor_history.reshape(self.num_envs, -1)), axis=-1),
            # The RSL-RL runner discovers the first height_scan term and passes
            # its slice to the source policy.  The saved split-policy checkpoint
            # therefore consumes [height_scan_history, critic_state].
            critic=jnp.concatenate((state.height_history.reshape(self.num_envs, -1), state.critic_state), axis=-1),
            disc=state.disc_history,
            disc_demo=state.demo_obs,
        )

    def _step(
        self,
        state: NativeEnvState,
        actions: Any,
        learning_iteration: Any,
        demonstration: Any,
        curriculum_stage: Any,
        forced_reset_patch_index: Any = None,
    ) -> tuple[NativeEnvState, NativeStepOutput]:
        jax, jnp = self.jax, self.jnp
        (
            key,
            command_key,
            history_key,
            reset_key,
            reset_command_key,
            reset_history_key,
            push_key,
        ) = jax.random.split(state.rng, 7)
        push_due = state.push_countdown <= 0
        push_velocity = self._sample_push_velocity(push_key)
        physics = self.physics.apply_push(
            state.physics, push_velocity, push_due, state.physics_randomization
        )
        push_countdown = jnp.where(
            push_due,
            self._sample_push_countdown(push_key),
            state.push_countdown - 1,
        )
        physics, torque = self.physics.step(physics, actions, state.physics_randomization)
        episode_length = state.episode_length + 1
        command_age = state.command_age + 1
        robot = self._extract_state(physics, torque, state.previous_key_body_pos, state.previous_joint_vel)
        interval = max(1, int(round(self.spec.command_resampling_time_s / self.step_dt)))
        command, heading, command_age = self._sample_commands(
            state.command,
            state.heading_target,
            command_age,
            robot.root_quat,
            command_age >= interval,
            command_key,
        )
        command = self._update_heading_commands(command, heading, robot.root_quat)
        terminated, timed_out, out_of_bounds = self._termination_flags(
            robot, episode_length, state.patch_index
        )
        dones = terminated | timed_out | out_of_bounds
        contact_time = jnp.where(
            robot.foot_contacts,
            state.contact_time + self.step_dt,
            jnp.zeros_like(state.contact_time),
        )
        air_time = jnp.where(
            robot.foot_contacts,
            jnp.zeros_like(state.air_time),
            state.air_time + self.step_dt,
        )
        weighted_terms = self._compute_reward_terms(
            robot,
            command,
            actions,
            state.last_actions,
            contact_time,
            air_time,
            terminated,
            learning_iteration,
        )
        episode_term_sums = state.episode_term_sums + weighted_terms * self.step_dt
        lower_reward = jnp.sum(weighted_terms * self.lower_reward_scales, axis=-1) * self.step_dt
        upper_reward = jnp.sum(weighted_terms * self.upper_reward_scales, axis=-1) * self.step_dt
        task_rewards = jnp.stack((lower_reward, upper_reward), axis=-1)

        history_keys = jax.random.split(history_key, 3)
        actor_frame = self._packed_actor_frame(robot, command, actions, history_keys[0], corrupt=True)
        scan = self._height_scan(robot, history_keys[1], corrupt=False)
        disc = self._disc_frame(robot)
        actor_history = jnp.concatenate((state.actor_history[:, 1:], actor_frame[:, None, :]), axis=1)
        height_history = jnp.concatenate((state.height_history[:, 1:], scan[:, None, :]), axis=1)
        terminal_disc = jnp.concatenate((state.disc_history[:, 1:], disc[:, None, :]), axis=1)
        actor_scan = self._height_scan(robot, history_keys[2], corrupt=True)
        critic_state = self._critic_state(robot, command, actions)

        reset_patch_key, reset_pose_key = jax.random.split(reset_key)
        terrain_stage = self._update_terrain_stage(
            state.terrain_stage,
            state.episode_start_xy,
            state.command,
            robot.root_pos,
            episode_length,
            episode_term_sums,
            terminated,
            dones,
        )
        sampled_patch_index = self._sample_patch_indices(reset_patch_key, terrain_stage)
        if forced_reset_patch_index is not None:
            forced_patch = jnp.asarray(forced_reset_patch_index, dtype=jnp.int32)
            forced = (
                jnp.full((self.num_envs,), forced_patch, dtype=jnp.int32)
                if forced_patch.ndim == 0
                else forced_patch.reshape((self.num_envs,))
            )
            sampled_patch_index = jnp.where(forced >= 0, forced, sampled_patch_index)
        patch_index = jnp.where(dones, sampled_patch_index, state.patch_index)
        qpos, qvel = self._sample_reset_values(reset_pose_key, patch_index)
        physics = self.physics.reset(physics, qpos, qvel, dones, state.physics_randomization)
        reset_key_positions = physics.xpos[:, self.key_body_ids]
        reset_joint_vel = physics.qvel[:, self.qvel_indices][:, self.mujoco_indices_in_isaac_order]
        reset_robot = self._extract_state(physics, jnp.zeros_like(torque), reset_key_positions, reset_joint_vel)
        last_actions = jnp.where(dones[:, None], 0.0, actions)
        episode_length_next = jnp.where(dones, 0, episode_length)
        command_age = jnp.where(dones, 0, command_age)
        command, heading, command_age = self._sample_commands(
            command,
            heading,
            command_age,
            reset_robot.root_quat,
            dones,
            reset_command_key,
        )
        reset_keys = jax.random.split(reset_history_key, 3)
        reset_actor = self._packed_actor_frame(reset_robot, command, last_actions, reset_keys[0], corrupt=True)
        reset_scan = self._height_scan(reset_robot, reset_keys[1], corrupt=False)
        reset_disc = self._disc_frame(reset_robot)
        reset_actor_history = jnp.repeat(reset_actor[:, None, :], self.spec.actor_history_length, axis=1)
        reset_height_history = jnp.repeat(reset_scan[:, None, :], self.spec.height_scan.critic_history_length, axis=1)
        reset_disc_history = jnp.repeat(reset_disc[:, None, :], self.spec.discriminator_history_length, axis=1)
        actor_history = jnp.where(dones[:, None, None], reset_actor_history, actor_history)
        height_history = jnp.where(dones[:, None, None], reset_height_history, height_history)
        disc_history = jnp.where(dones[:, None, None], reset_disc_history, terminal_disc)
        reset_actor_scan = self._height_scan(reset_robot, reset_keys[2], corrupt=True)
        actor_scan = jnp.where(dones[:, None], reset_actor_scan, actor_scan)
        reset_critic = self._critic_state(reset_robot, command, last_actions)
        critic_state = jnp.where(dones[:, None], reset_critic, critic_state)
        previous_key_body_pos = jnp.where(
            dones[:, None, None], reset_robot.key_body_pos_w, robot.key_body_pos_w
        )
        previous_joint_vel = jnp.where(dones[:, None], reset_robot.joint_vel, robot.joint_vel)
        episode_start_xy = jnp.where(dones[:, None], reset_robot.root_pos[:, :2], state.episode_start_xy)
        next_state = NativeEnvState(
            physics=physics,
            physics_randomization=state.physics_randomization,
            rng=key,
            episode_length=episode_length_next,
            command=command,
            heading_target=heading,
            command_age=command_age,
            last_actions=last_actions,
            air_time=jnp.where(dones[:, None], 0.0, air_time),
            contact_time=jnp.where(dones[:, None], 0.0, contact_time),
            episode_term_sums=jnp.where(dones[:, None], 0.0, episode_term_sums),
            previous_key_body_pos=previous_key_body_pos,
            previous_joint_vel=previous_joint_vel,
            actor_history=actor_history,
            height_history=height_history,
            disc_history=disc_history,
            actor_scan=actor_scan,
            critic_state=critic_state,
            demo_obs=demonstration,
            patch_index=patch_index,
            terrain_stage=terrain_stage,
            episode_start_xy=episode_start_xy,
            push_countdown=push_countdown,
        )
        observation = self.observation(next_state)
        output = NativeStepOutput(
            observation=observation,
            transition_disc=terminal_disc,
            task_rewards=task_rewards,
            weighted_terms=weighted_terms,
            dones=dones,
            terminated=terminated,
            time_outs=timed_out | out_of_bounds,
            out_of_bounds=out_of_bounds,
            completed_term_sums=episode_term_sums,
            completed_lengths=episode_length,
        )
        return next_state, output

    def _update_terrain_stage(
        self,
        current_stage: Any,
        episode_start_xy: Any,
        command: Any,
        root_pos: Any,
        episode_length: Any,
        episode_term_sums: Any,
        terminated: Any,
        dones: Any,
    ) -> Any:
        """Promote or demote only environments that have just completed an episode.

        This mirrors the Isaac curriculum rule: moving commands advance after a
        surviving, well-tracked traversal and move down after a fall or clearly
        insufficient progress.  The persistent target stage is separate from the
        tile sampled for replay, so an easier replay episode cannot erase progress.
        """
        jnp = self.jnp
        if not self.spec.terrain.curriculum.enabled or not self.terrain.patches:
            return current_stage

        command_speed = jnp.linalg.norm(command[:, :2], axis=1)
        moving_command = command_speed > 0.2
        distance = jnp.linalg.norm(root_pos[:, :2] - episode_start_xy, axis=1)
        expected_distance = command_speed * self.spec.episode_length_s
        target_distance = jnp.minimum(
            expected_distance * 0.65,
            jnp.full_like(expected_distance, self.spec.terrain.size[0] * 0.45),
        )
        failed_distance = jnp.minimum(expected_distance * 0.25, target_distance * 0.75)
        lin_tracking = episode_term_sums[:, 0] / self.spec.episode_length_s
        ang_tracking = episode_term_sums[:, 1] / self.spec.episode_length_s
        tracking_ok = (lin_tracking >= self.spec.terrain.curriculum.min_lin_tracking) & (
            ang_tracking >= self.spec.terrain.curriculum.min_ang_tracking
        )
        move_up = moving_command & ~terminated & tracking_ok & (distance >= target_distance)
        move_down = terminated | (moving_command & (distance < failed_distance))
        move_down &= ~move_up
        candidate = current_stage + move_up.astype(jnp.int32) - move_down.astype(jnp.int32)
        candidate = jnp.clip(candidate, 0, 5)
        return jnp.where(dones, candidate, current_stage)

    def _extract_state(
        self,
        data: Any,
        torque: Any,
        previous_key_body_pos: Any,
        previous_joint_vel: Any,
    ) -> RobotState:
        jnp = self.jnp
        qpos, qvel = data.qpos, data.qvel
        root_quat = qpos[:, 3:7]
        joint_pos = qpos[:, self.qpos_indices][:, self.mujoco_indices_in_isaac_order]
        joint_vel = qvel[:, self.qvel_indices][:, self.mujoco_indices_in_isaac_order]
        joint_acc = (joint_vel - previous_joint_vel) / self.step_dt
        root_lin_vel_w = qvel[:, :3]
        # MuJoCo free-joint angular velocity is expressed in world coordinates.
        # Isaac's ``root_ang_vel_b`` observation/reward contract is body-frame.
        root_ang_vel_b = quat_apply_inverse(root_quat, qvel[:, 3:6])
        key_body_pos_w = data.xpos[:, self.key_body_ids]
        height_scan_pos_w = data.xpos[:, self.height_scan_body_id]
        height_scan_quat_w = data.xquat[:, self.height_scan_body_id]
        root_quat_expanded = jnp.broadcast_to(root_quat[:, None, :], (self.num_envs, 6, 4))
        key_body_pos_b = quat_apply_inverse(root_quat_expanded, key_body_pos_w - qpos[:, None, :3])
        key_velocity_w = (key_body_pos_w - previous_key_body_pos) / self.step_dt
        return RobotState(
            root_pos=qpos[:, :3],
            root_quat=root_quat,
            root_lin_vel_w=root_lin_vel_w,
            root_lin_vel_b=quat_apply_inverse(root_quat, root_lin_vel_w),
            root_ang_vel_b=root_ang_vel_b,
            gravity_b=projected_gravity(root_quat),
            height_scan_pos_w=height_scan_pos_w,
            height_scan_quat_w=height_scan_quat_w,
            joint_pos=joint_pos,
            joint_vel=joint_vel,
            key_body_pos_w=key_body_pos_w,
            key_body_pos_b=key_body_pos_b,
            foot_vel_w=key_velocity_w[:, :2],
            joint_acc=joint_acc,
            torque=torque,
            foot_contacts=self.physics.foot_contacts(data),
        )

    def _packed_actor_frame(
        self,
        robot: RobotState,
        command: Any,
        last_actions: Any,
        key: Any,
        corrupt: bool,
    ) -> Any:
        jax, jnp = self.jax, self.jnp
        ang_vel = robot.root_ang_vel_b
        gravity = robot.gravity_b
        joint_pos = robot.joint_pos - self.default_joint_pos
        joint_vel = robot.joint_vel
        if corrupt:
            keys = jax.random.split(key, 4)
            ang_vel = ang_vel + jax.random.uniform(keys[0], ang_vel.shape, minval=-0.35, maxval=0.35)
            gravity = gravity + jax.random.uniform(keys[1], gravity.shape, minval=-0.05, maxval=0.05)
            joint_pos = joint_pos + jax.random.uniform(keys[2], joint_pos.shape, minval=-0.03, maxval=0.03)
            joint_vel = joint_vel + jax.random.uniform(keys[3], joint_vel.shape, minval=-1.75, maxval=1.75)
        result = jnp.concatenate(
            (
                ang_vel,
                gravity,
                command,
                joint_pos,
                joint_vel,
                last_actions,
                robot.key_body_pos_b.reshape(self.num_envs, -1),
            ),
            axis=-1,
        )
        if result.shape[-1] != PACKED_ACTOR_FRAME_DIM:
            raise RuntimeError(f"Packed actor frame has width {result.shape[-1]}")
        return result

    def _critic_state(self, robot: RobotState, command: Any, last_actions: Any) -> Any:
        result = self.jnp.concatenate(
            (
                robot.root_lin_vel_b,
                robot.root_ang_vel_b,
                robot.gravity_b,
                command,
                robot.joint_pos,
                robot.joint_vel,
                last_actions,
                robot.key_body_pos_b.reshape(self.num_envs, -1),
            ),
            axis=-1,
        )
        if result.shape[-1] != CRITIC_STATE_DIM:
            raise RuntimeError(f"Critic state has width {result.shape[-1]}")
        return result

    def _disc_frame(self, robot: RobotState) -> Any:
        return self.jnp.concatenate(
            (
                root_local_rot_tan_norm(robot.root_quat),
                robot.root_ang_vel_b,
                robot.joint_pos[:, self.upper_indices],
                robot.joint_vel[:, self.upper_indices],
                robot.key_body_pos_b[:, self.upper_key_indices].reshape(self.num_envs, -1),
            ),
            axis=-1,
        )

    def _height_scan(self, robot: RobotState, key: Any, corrupt: bool) -> Any:
        jax, jnp = self.jax, self.jnp
        yaw = yaw_from_quat(robot.height_scan_quat_w)
        cosine, sine = jnp.cos(yaw)[:, None], jnp.sin(yaw)[:, None]
        world_x = robot.height_scan_pos_w[:, 0:1] + cosine * self.scan_x_local - sine * self.scan_y_local
        world_y = robot.height_scan_pos_w[:, 1:2] + sine * self.scan_x_local + cosine * self.scan_y_local
        terrain_height = self._sample_terrain_height(jnp.stack((world_x, world_y), axis=-1))
        # Isaac's sensor.data.pos_w is the attached torso frame position.  Its
        # ray-start offset affects ray casting but is not added to this tensor.
        scanner_z = robot.height_scan_pos_w[:, 2:3]
        scan = scanner_z - terrain_height - self.spec.height_scan.offset
        if corrupt and self.spec.height_scan.noise > 0.0:
            scan += jax.random.uniform(
                key,
                scan.shape,
                minval=-self.spec.height_scan.noise,
                maxval=self.spec.height_scan.noise,
            )
        return scan

    def _sample_terrain_height(self, xy: Any) -> Any:
        jnp = self.jnp
        nx, ny = self.terrain_heights.shape
        fx = (xy[..., 0] + self.terrain.size[0] / 2.0) * (nx - 1) / self.terrain.size[0]
        fy = (xy[..., 1] + self.terrain.size[1] / 2.0) * (ny - 1) / self.terrain.size[1]
        fx, fy = jnp.clip(fx, 0.0, nx - 1.0), jnp.clip(fy, 0.0, ny - 1.0)
        x0, y0 = jnp.floor(fx).astype(jnp.int32), jnp.floor(fy).astype(jnp.int32)
        x1, y1 = jnp.minimum(x0 + 1, nx - 1), jnp.minimum(y0 + 1, ny - 1)
        wx, wy = fx - x0.astype(jnp.float32), fy - y0.astype(jnp.float32)
        h00 = self.terrain_heights[x0, y0]
        h10 = self.terrain_heights[x1, y0]
        h01 = self.terrain_heights[x0, y1]
        h11 = self.terrain_heights[x1, y1]
        return (1 - wx) * (1 - wy) * h00 + wx * (1 - wy) * h10 + (1 - wx) * wy * h01 + wx * wy * h11

    def _compute_reward_terms(
        self,
        robot: RobotState,
        command: Any,
        actions: Any,
        previous_actions: Any,
        contact_time: Any,
        air_time: Any,
        terminated: Any,
        learning_iteration: Any,
    ) -> Any:
        jnp = self.jnp
        invalid_state = self._invalid_state(robot)
        upright = jnp.clip(-robot.gravity_b[:, 2], 0.0, 0.7) / 0.7
        lin_error = jnp.sum((command[:, :2] - robot.root_lin_vel_b[:, :2]) ** 2, axis=1)
        ang_error = (command[:, 2] - robot.root_ang_vel_b[:, 2]) ** 2
        in_mode_time = jnp.where(robot.foot_contacts, contact_time, air_time)
        single_stance = jnp.sum(robot.foot_contacts, axis=1) == 1
        feet_air = jnp.minimum(jnp.min(jnp.where(single_stance[:, None], in_mode_time, 0.0), axis=1), 0.4)
        feet_air *= jnp.linalg.norm(command[:, :2], axis=1) > 0.1
        feet_slide = self._feet_slide_reward(robot)
        action_delta = actions - previous_actions
        low = jnp.maximum(self.soft_joint_limits[:, 0] - robot.joint_pos, 0.0)
        high = jnp.maximum(robot.joint_pos - self.soft_joint_limits[:, 1], 0.0)
        violation = low + high
        activity = jnp.sqrt(jnp.sum(command[:, :2] ** 2, axis=1) + (0.5 * command[:, 2]) ** 2)
        arm_scale = 0.5 + 0.5 * jnp.exp(-((activity / 0.5) ** 2))
        root_height = jnp.maximum(0.75 - robot.root_pos[:, 2], 0.0) ** 2
        root_height = jnp.where(
            learning_iteration < self.spec.root_height_reward_start_iteration,
            0.0,
            root_height,
        )
        raw = jnp.stack(
            (
                jnp.exp(-lin_error / 0.25) * upright,
                jnp.exp(-ang_error / 0.25) * upright,
                feet_air,
                feet_slide,
                jnp.sum(robot.gravity_b[:, :2] ** 2, axis=1),
                jnp.sum(robot.root_ang_vel_b[:, :2] ** 2, axis=1),
                robot.root_lin_vel_b[:, 2] ** 2,
                root_height,
                jnp.sum(robot.torque[:, self.lower_indices] ** 2, axis=1),
                jnp.sum(robot.torque[:, self.upper_indices] ** 2, axis=1),
                jnp.sum(robot.joint_acc[:, self.lower_indices] ** 2, axis=1),
                jnp.sum(robot.joint_acc[:, self.upper_indices] ** 2, axis=1),
                jnp.sum(action_delta[:, self.lower_indices] ** 2, axis=1),
                jnp.sum(action_delta[:, self.upper_indices] ** 2, axis=1),
                jnp.sum(violation[:, self.ankle_indices], axis=1),
                jnp.sum(violation[:, self.upper_indices], axis=1),
                jnp.sum(jnp.abs(robot.joint_pos[:, self.lower_indices] - self.default_joint_pos[self.lower_indices]), axis=1),
                arm_scale
                * jnp.sum(jnp.abs(robot.joint_pos[:, self.upper_indices] - self.default_joint_pos[self.upper_indices]), axis=1),
                terminated.astype(jnp.float32),
            ),
            axis=-1,
        )
        raw = jnp.where(invalid_state[:, None], jnp.zeros_like(raw), raw)
        raw = raw.at[:, -1].set(terminated.astype(jnp.float32))
        return raw * self.reward_weights

    def _feet_slide_reward(self, robot: RobotState) -> Any:
        jnp = self.jnp
        # Match Isaac's feet_slide reward: subtract root velocity, rotate into
        # the base frame, then measure lateral foot speed while in contact.
        foot_relative_vel_w = robot.foot_vel_w - robot.root_lin_vel_w[:, None, :]
        foot_quat = jnp.broadcast_to(robot.root_quat[:, None, :], (self.num_envs, 2, 4))
        foot_vel_b = quat_apply_inverse(foot_quat, foot_relative_vel_w)
        return jnp.sum(jnp.linalg.norm(foot_vel_b[:, :, :2], axis=-1) * robot.foot_contacts, axis=1)

    def _invalid_state(self, robot: RobotState) -> Any:
        jnp = self.jnp
        finite = (
            jnp.all(jnp.isfinite(robot.root_pos), axis=1)
            & jnp.all(jnp.isfinite(robot.root_quat), axis=1)
            & jnp.all(jnp.isfinite(robot.root_lin_vel_w), axis=1)
            & jnp.all(jnp.isfinite(robot.root_ang_vel_b), axis=1)
            & jnp.all(jnp.isfinite(robot.joint_pos), axis=1)
            & jnp.all(jnp.isfinite(robot.joint_vel), axis=1)
        )
        lower = self.soft_joint_limits[:, 0] - 2.0
        upper = self.soft_joint_limits[:, 1] + 2.0
        joint_outside_physical_range = jnp.any((robot.joint_pos < lower) | (robot.joint_pos > upper), axis=1)
        root_velocity_exploded = (
            jnp.any(jnp.abs(robot.root_lin_vel_w) > 50.0, axis=1)
            | jnp.any(jnp.abs(robot.root_ang_vel_b) > 50.0, axis=1)
        )
        joint_velocity_exploded = jnp.any(jnp.abs(robot.joint_vel) > 200.0, axis=1)
        return (~finite) | joint_outside_physical_range | root_velocity_exploded | joint_velocity_exploded

    def _termination_flags(
        self, robot: RobotState, episode_length: Any, patch_index: Any
    ) -> tuple[Any, Any, Any]:
        jnp = self.jnp
        base_height = robot.root_pos[:, 2] < self.spec.min_root_height
        orientation = -robot.gravity_b[:, 2] < np.cos(self.spec.max_tilt_radians)
        invalid = self._invalid_state(robot)
        timed_out = episode_length >= self.max_episode_length
        origin = self.patch_origins[patch_index]
        extent = self.patch_extents[patch_index]
        out = jnp.any(jnp.abs(robot.root_pos[:, :2] - origin) > extent, axis=1)
        return base_height | orientation | invalid, timed_out, out

    def _sample_patch_indices(self, key: Any, curriculum_stage: Any) -> Any:
        jax, jnp = self.jax, self.jnp
        if not self.terrain.patches:
            return jnp.zeros((self.num_envs,), dtype=jnp.int32)
        bucket_key, easy_stage_key, patch_key = jax.random.split(key, 3)
        cfg = self.spec.terrain.curriculum
        stage = jnp.clip(curriculum_stage, 0, 5).astype(jnp.int32)
        bucket = jax.random.uniform(bucket_key, (self.num_envs,))
        previous_stage = jnp.maximum(stage - 1, 0)
        easier_max = jnp.maximum(stage - 2, 0)
        easier_stage = jax.random.randint(
            easy_stage_key, (self.num_envs,), minval=0, maxval=easier_max + 1
        )
        sampled_stage = jnp.where(
            bucket < cfg.current_probability,
            stage,
            jnp.where(
                bucket < cfg.current_probability + cfg.previous_probability,
                previous_stage,
                easier_stage,
            ),
        )
        return self._sample_patch_indices_for_stages(patch_key, sampled_stage)

    def _sample_patch_indices_for_stages(self, key: Any, sampled_stage: Any) -> Any:
        jax, jnp = self.jax, self.jnp
        if not self.terrain.patches:
            return jnp.zeros((self.num_envs,), dtype=jnp.int32)
        sampled_stage = jnp.clip(sampled_stage, 0, self.stage_patch_counts.shape[0] - 1).astype(jnp.int32)
        count = self.stage_patch_counts[sampled_stage]
        slot = jnp.floor(jax.random.uniform(key, (self.num_envs,)) * count).astype(jnp.int32)
        return self.stage_patch_indices[sampled_stage, slot]

    def _sample_reset_values(self, key: Any, patch_index: Any) -> tuple[Any, Any]:
        jax, jnp = self.jax, self.jnp
        xy_key, yaw_key, joint_key, velocity_key = jax.random.split(key, 4)
        qpos = jnp.zeros((self.num_envs, self.model.nq), dtype=jnp.float32)
        qvel = jnp.zeros((self.num_envs, self.model.nv), dtype=jnp.float32)
        spawn_jitter = 0.15 if self.terrain.terrain_type.startswith("ean:") else 0.5
        xy = jax.random.uniform(xy_key, (self.num_envs, 2), minval=-spawn_jitter, maxval=spawn_jitter)
        xy += self.patch_origins[patch_index]
        yaw = jax.random.uniform(yaw_key, (self.num_envs,), minval=-jnp.pi, maxval=jnp.pi)
        qpos = qpos.at[:, :2].set(xy)
        qpos = qpos.at[:, 2].set(0.8 + self._sample_terrain_height(xy))
        qpos = qpos.at[:, 3].set(jnp.cos(0.5 * yaw))
        qpos = qpos.at[:, 6].set(jnp.sin(0.5 * yaw))
        joint = self.default_joint_pos[None, :] * jax.random.uniform(
            joint_key, (self.num_envs, 29), minval=0.8, maxval=1.2
        )
        qpos = qpos.at[:, self.qpos_indices].set(joint[:, self.isaac_indices_in_mujoco_order])
        qvel = qvel.at[:, :6].set(
            jax.random.uniform(velocity_key, (self.num_envs, 6), minval=-0.2, maxval=0.2)
        )
        return qpos, qvel

    def _sample_physics_randomization(self, key: Any) -> PhysicsRandomization:
        """Sample Isaac's startup randomization once for each native world."""
        jax, jnp = self.jax, self.jnp
        cfg = self.spec.domain_randomization
        body_mass = jnp.broadcast_to(jnp.asarray(self.model.body_mass), (self.num_envs, self.model.nbody))
        body_inertia = jnp.broadcast_to(jnp.asarray(self.model.body_inertia), (self.num_envs, self.model.nbody, 3))
        body_ipos = jnp.broadcast_to(jnp.asarray(self.model.body_ipos), (self.num_envs, self.model.nbody, 3))
        dof_armature = jnp.broadcast_to(jnp.asarray(self.model.dof_armature), (self.num_envs, self.model.nv))
        geom_friction = jnp.broadcast_to(
            jnp.asarray(self.model.geom_friction), (self.num_envs, self.model.ngeom, 3)
        )
        ones = jnp.ones((self.num_envs, 29), dtype=jnp.float32)
        if not cfg.enabled:
            return PhysicsRandomization(body_mass, body_inertia, body_ipos, dof_armature, geom_friction, ones, ones)

        (
            base_mass_key,
            com_key,
            limb_mass_key,
            stiffness_key,
            damping_key,
            armature_key,
            friction_key,
        ) = jax.random.split(key, 7)
        body_mass = body_mass.at[:, self.metadata.base_mass_body_id].add(
            jax.random.uniform(
                base_mass_key,
                (self.num_envs,),
                minval=cfg.base_mass_add_range[0],
                maxval=cfg.base_mass_add_range[1],
            )
        )
        if self.metadata.com_body_ids.size:
            com_ids = jnp.asarray(self.metadata.com_body_ids)
            com_delta = jax.random.uniform(
                com_key,
                (self.num_envs, com_ids.size, 3),
                minval=cfg.com_range[0],
                maxval=cfg.com_range[1],
            )
            body_ipos = body_ipos.at[:, com_ids, :].add(com_delta)
        if self.metadata.limb_body_ids.size:
            limb_ids = jnp.asarray(self.metadata.limb_body_ids)
            limb_scale = jax.random.uniform(
                limb_mass_key,
                (self.num_envs, limb_ids.size),
                minval=cfg.limb_mass_scale_range[0],
                maxval=cfg.limb_mass_scale_range[1],
            )
            body_mass = body_mass.at[:, limb_ids].multiply(limb_scale)
        base_mass = jnp.asarray(self.model.body_mass)
        mass_ratio = body_mass / jnp.maximum(base_mass[None, :], 1.0e-8)
        body_inertia = body_inertia * mass_ratio[:, :, None]
        stiffness_scale_isaac = jax.random.uniform(
            stiffness_key,
            (self.num_envs, 29),
            minval=cfg.actuator_gain_scale_range[0],
            maxval=cfg.actuator_gain_scale_range[1],
        )
        damping_scale_isaac = jax.random.uniform(
            damping_key,
            (self.num_envs, 29),
            minval=cfg.actuator_gain_scale_range[0],
            maxval=cfg.actuator_gain_scale_range[1],
        )
        armature_scale_isaac = jax.random.uniform(
            armature_key,
            (self.num_envs, 29),
            minval=cfg.armature_scale_range[0],
            maxval=cfg.armature_scale_range[1],
        )
        dof_armature = dof_armature.at[:, self.qvel_indices].multiply(
            armature_scale_isaac[:, self.isaac_indices_in_mujoco_order]
        )
        if self.metadata.robot_geom_ids.size:
            robot_geom_ids = jnp.asarray(self.metadata.robot_geom_ids)
            static_key, dynamic_key, assignment_key = jax.random.split(friction_key, 3)
            static_friction = jax.random.uniform(
                static_key,
                (cfg.material_num_buckets,),
                minval=cfg.static_friction_range[0],
                maxval=cfg.static_friction_range[1],
            )
            dynamic_friction = jax.random.uniform(
                dynamic_key,
                (cfg.material_num_buckets,),
                minval=cfg.dynamic_friction_range[0],
                maxval=cfg.dynamic_friction_range[1],
            )
            material_dynamic_friction = jnp.minimum(static_friction, dynamic_friction)
            material_ids = jax.random.randint(
                assignment_key,
                (self.num_envs, robot_geom_ids.size),
                0,
                cfg.material_num_buckets,
                dtype=jnp.int32,
            )
            # MuJoCo has one sliding-friction scalar; use the consistent
            # dynamic coefficient sampled from Isaac's 64 material buckets.
            sliding_friction = material_dynamic_friction[material_ids]
            geom_friction = geom_friction.at[:, robot_geom_ids, 0].set(sliding_friction)
        stiffness_scale_mujoco = stiffness_scale_isaac[:, self.isaac_indices_in_mujoco_order]
        damping_scale_mujoco = damping_scale_isaac[:, self.isaac_indices_in_mujoco_order]
        return PhysicsRandomization(
            body_mass,
            body_inertia,
            body_ipos,
            dof_armature,
            geom_friction,
            stiffness_scale_mujoco,
            damping_scale_mujoco,
        )

    def _sample_push_countdown(self, key: Any) -> Any:
        jax, jnp = self.jax, self.jnp
        cfg = self.spec.domain_randomization
        if not cfg.enabled:
            return jnp.full((self.num_envs,), jnp.iinfo(jnp.int32).max, dtype=jnp.int32)
        lower = max(1, int(round(cfg.push_interval_s[0] / self.step_dt)))
        upper = max(lower, int(round(cfg.push_interval_s[1] / self.step_dt)))
        return jax.random.randint(key, (self.num_envs,), lower, upper + 1, dtype=jnp.int32)

    def _sample_push_velocity(self, key: Any) -> Any:
        jax, jnp = self.jax, self.jnp
        cfg = self.spec.domain_randomization
        xy_key, yaw_key = jax.random.split(key)
        velocity = jnp.zeros((self.num_envs, 3), dtype=jnp.float32)
        velocity = velocity.at[:, :2].set(
            jax.random.uniform(
                xy_key,
                (self.num_envs, 2),
                minval=cfg.push_linear_velocity_range[0],
                maxval=cfg.push_linear_velocity_range[1],
            )
        )
        return velocity.at[:, 2].set(
            jax.random.uniform(
                yaw_key,
                (self.num_envs,),
                minval=cfg.push_yaw_velocity_range[0],
                maxval=cfg.push_yaw_velocity_range[1],
            )
        )

    def _sample_commands(
        self,
        command: Any,
        heading: Any,
        age: Any,
        root_quat: Any,
        mask: Any,
        key: Any,
    ) -> tuple[Any, Any, Any]:
        jax, jnp = self.jax, self.jnp
        x_key, y_key, heading_key, standing_key = jax.random.split(key, 4)
        sampled = jnp.zeros_like(command)
        sampled = sampled.at[:, 0].set(
            jax.random.uniform(x_key, (self.num_envs,), minval=self.spec.command_lin_vel_x[0], maxval=self.spec.command_lin_vel_x[1])
        )
        sampled = sampled.at[:, 1].set(
            jax.random.uniform(y_key, (self.num_envs,), minval=self.spec.command_lin_vel_y[0], maxval=self.spec.command_lin_vel_y[1])
        )
        sampled_heading = jax.random.uniform(heading_key, (self.num_envs,), minval=-jnp.pi, maxval=jnp.pi)
        standing = jax.random.uniform(standing_key, (self.num_envs,)) < self.spec.standing_env_ratio
        sampled = jnp.where(standing[:, None], 0.0, sampled)
        sampled_heading = jnp.where(standing, yaw_from_quat(root_quat), sampled_heading)
        command = jnp.where(mask[:, None], sampled, command)
        heading = jnp.where(mask, sampled_heading, heading)
        age = jnp.where(mask, 0, age)
        return self._update_heading_commands(command, heading, root_quat), heading, age

    def _update_heading_commands(self, command: Any, heading: Any, root_quat: Any) -> Any:
        jnp = self.jnp
        yaw_error = wrap_to_pi(heading - yaw_from_quat(root_quat))
        angular = jnp.clip(0.5 * yaw_error, *self.spec.command_ang_vel_z)
        angular = jnp.where(jnp.linalg.norm(command[:, :2], axis=1) == 0.0, 0.0, angular)
        return command.at[:, 2].set(angular)

    def _split_reward_scales(self, config: dict[str, Any]) -> tuple[Any, Any]:
        jnp = self.jnp
        index = {name: position for position, name in enumerate(REWARD_TERM_NAMES)}
        lower = np.zeros(len(REWARD_TERM_NAMES), dtype=np.float32)
        upper = np.zeros(len(REWARD_TERM_NAMES), dtype=np.float32)
        for target, values in ((lower, config.get("lower", {})), (upper, config.get("upper", {}))):
            missing = sorted(set(values) - set(index))
            if missing and config.get("strict", True):
                raise KeyError(f"Unknown split reward terms: {missing}")
            for name, scale in values.items():
                if name in index:
                    target[index[name]] = float(scale)
        return jnp.asarray(lower), jnp.asarray(upper)
