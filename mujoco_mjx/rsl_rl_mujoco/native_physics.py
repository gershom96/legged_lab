from __future__ import annotations

import os
from typing import Any, NamedTuple

import numpy as np

from .constants import (
    ACTION_SCALE,
    DAMPING_ISAAC,
    DEFAULT_JOINT_POS_ISAAC,
    EFFORT_LIMIT_ISAAC,
    ISAAC_INDICES_IN_MUJOCO_ORDER,
    MUJOCO_INDICES_IN_ISAAC_ORDER,
    STIFFNESS_ISAAC,
)
from .model import ModelMetadata
from .spec import MujocoRslRlEnvSpec


os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
# XLA's caching allocator retains blocks that MJX-Warp needs for convex
# collision scratch. The platform allocator returns completed allocations to
# CUDA, allowing both runtimes to share one GPU across long rollouts.
os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "platform")


class PhysicsRandomization(NamedTuple):
    """Per-world MuJoCo model fields and actuator gains sampled at startup."""

    body_mass: Any
    body_inertia: Any
    body_ipos: Any
    dof_armature: Any
    geom_friction: Any
    stiffness_scale_mujoco: Any
    damping_scale_mujoco: Any


class NativeMjxPhysics:
    """Pure-JAX batched MJX physics; no Torch tensors cross this boundary."""

    def __init__(self, model: Any, metadata: ModelMetadata, spec: MujocoRslRlEnvSpec) -> None:
        import jax
        import jax.numpy as jnp
        from mujoco import mjx

        self.jax = jax
        self.jnp = jnp
        self.mjx = mjx
        self.model = model
        self.metadata = metadata
        self.spec = spec
        self.num_envs = spec.num_envs
        self.impl = spec.mjx_impl
        self.mx = mjx.put_model(model, impl=self.impl)
        if self.impl == "warp":
            base_data = mjx.make_data(
                model,
                impl=self.impl,
                naconmax=spec.warp_naconmax,
                njmax=spec.njmax,
            )
            print(
                f"[INFO] MJX-Warp capacities: naconmax={spec.warp_naconmax} global "
                f"({spec.contacts_per_env}/env), njmax={spec.njmax}/env"
            )
        else:
            base_data = mjx.make_data(self.mx)
        self.base_data = base_data
        self.batched_base_data = jax.vmap(lambda _: base_data)(jnp.arange(self.num_envs))
        self.qpos_indices = jnp.asarray(metadata.joint_qpos_indices, dtype=jnp.int32)
        self.qvel_indices = jnp.asarray(metadata.joint_qvel_indices, dtype=jnp.int32)
        self.actuator_indices = jnp.asarray(metadata.actuator_indices, dtype=jnp.int32)
        self.isaac_indices_in_mujoco_order = jnp.asarray(ISAAC_INDICES_IN_MUJOCO_ORDER, dtype=jnp.int32)
        self.mujoco_indices_in_isaac_order = jnp.asarray(MUJOCO_INDICES_IN_ISAAC_ORDER, dtype=jnp.int32)
        self.default_mujoco = jnp.asarray(DEFAULT_JOINT_POS_ISAAC)[self.isaac_indices_in_mujoco_order]
        self.kp_mujoco = jnp.asarray(STIFFNESS_ISAAC)[self.isaac_indices_in_mujoco_order]
        self.kd_mujoco = jnp.asarray(DAMPING_ISAAC)[self.isaac_indices_in_mujoco_order]
        self.effort_mujoco = jnp.asarray(EFFORT_LIMIT_ISAAC)[self.isaac_indices_in_mujoco_order]
        self.terrain_geom_ids = jnp.asarray(metadata.terrain_geom_ids, dtype=jnp.int32)
        self.foot_geom_ids = tuple(jnp.asarray(value, dtype=jnp.int32) for value in metadata.foot_geom_ids)
        self._compile()

    def _compile(self) -> None:
        jax, jnp, mjx, mx = self.jax, self.jnp, self.mjx, self.mx

        def choose(mask: Any, replacement: Any, current: Any) -> Any:
            shape = (mask.shape[0],) + (1,) * (current.ndim - 1)
            return jnp.where(mask.reshape(shape), replacement, current)

        def randomized_model(randomization: PhysicsRandomization) -> Any:
            return mx.replace(
                body_mass=randomization.body_mass,
                body_inertia=randomization.body_inertia,
                body_ipos=randomization.body_ipos,
                dof_armature=randomization.dof_armature,
                geom_friction=randomization.geom_friction,
            )

        def reset(data: Any, qpos: Any, qvel: Any, mask: Any, randomization: PhysicsRandomization) -> Any:
            data = data.replace(
                qpos=choose(mask, qpos, data.qpos),
                qvel=choose(mask, qvel, data.qvel),
                ctrl=choose(mask, jnp.zeros_like(data.ctrl), data.ctrl),
                act=choose(mask, jnp.zeros_like(data.act), data.act),
                time=choose(mask, jnp.zeros_like(data.time), data.time),
                qfrc_applied=choose(mask, jnp.zeros_like(data.qfrc_applied), data.qfrc_applied),
                xfrc_applied=choose(mask, jnp.zeros_like(data.xfrc_applied), data.xfrc_applied),
            )
            return jax.vmap(lambda model_randomization, world: mjx.forward(randomized_model(model_randomization), world))(
                randomization, data
            )

        def step_one(
            data: Any, action_isaac: Any, randomization: PhysicsRandomization
        ) -> tuple[Any, Any]:
            model = randomized_model(randomization)
            action_mujoco = action_isaac[self.isaac_indices_in_mujoco_order]
            target = self.default_mujoco + ACTION_SCALE * action_mujoco

            def substep(carry: tuple[Any, Any], _: Any) -> tuple[tuple[Any, Any], None]:
                current, torque_sum = carry
                joint_pos = current.qpos[self.qpos_indices]
                joint_vel = current.qvel[self.qvel_indices]
                torque = (
                    self.kp_mujoco * randomization.stiffness_scale_mujoco * (target - joint_pos)
                    - self.kd_mujoco * randomization.damping_scale_mujoco * joint_vel
                )
                torque = jnp.clip(torque, -self.effort_mujoco, self.effort_mujoco)
                ctrl = jnp.zeros_like(current.ctrl).at[self.actuator_indices].set(torque)
                return (mjx.step(model, current.replace(ctrl=ctrl)), torque_sum + torque), None

            (result, torque_sum), _ = jax.lax.scan(
                substep,
                (data, jnp.zeros_like(self.default_mujoco)),
                None,
                length=int(self.spec.decimation),
            )
            torque_isaac = (torque_sum / float(self.spec.decimation))[self.mujoco_indices_in_isaac_order]
            return result, torque_isaac

        self.reset = jax.jit(reset)
        def apply_push(data: Any, velocity: Any, mask: Any, randomization: PhysicsRandomization) -> Any:
            qvel = data.qvel
            qvel = qvel.at[:, 0].set(jnp.where(mask, qvel[:, 0] + velocity[:, 0], qvel[:, 0]))
            qvel = qvel.at[:, 1].set(jnp.where(mask, qvel[:, 1] + velocity[:, 1], qvel[:, 1]))
            qvel = qvel.at[:, 5].set(jnp.where(mask, qvel[:, 5] + velocity[:, 2], qvel[:, 5]))
            data = data.replace(qvel=qvel)
            return jax.vmap(lambda model_randomization, world: mjx.forward(randomized_model(model_randomization), world))(
                randomization, data
            )

        self.step = jax.jit(jax.vmap(step_one))
        self.apply_push = jax.jit(apply_push)

    def initial_data(self, qpos: Any, qvel: Any, randomization: PhysicsRandomization) -> Any:
        return self.reset(
            self.batched_base_data,
            qpos,
            qvel,
            self.jnp.ones((self.num_envs,), dtype=bool),
            randomization,
        )

    def foot_contacts(self, data: Any) -> Any:
        """Return terrain contacts using active MuJoCo contact pairs.

        Isaac's corresponding term thresholds net force at 1 N. MJX-Warp does
        not expose per-contact forces here, so use non-positive contact distance
        rather than the previous 1 mm near-contact proxy.
        """
        if self.impl == "warp":
            implementation = data._impl
            pairs = implementation.contact__geom.astype(self.jnp.int32)
            distance = implementation.contact__dist.reshape(-1)
            world_ids = implementation.contact__worldid.reshape(-1).astype(self.jnp.int32)
            count = implementation.nacon.reshape(-1)[0]
            active = self.jnp.arange(pairs.shape[0]) < count
            active &= distance <= 0.0
            active &= (world_ids >= 0) & (world_ids < self.num_envs)
            safe_world_ids = self.jnp.clip(world_ids, 0, self.num_envs - 1)
            is_terrain_0 = self.jnp.any(pairs[:, 0:1] == self.terrain_geom_ids[None, :], axis=-1)
            is_terrain_1 = self.jnp.any(pairs[:, 1:2] == self.terrain_geom_ids[None, :], axis=-1)
            outputs = []
            for foot_ids in self.foot_geom_ids:
                is_foot_0 = self.jnp.any(pairs[:, 0:1] == foot_ids[None, :], axis=-1)
                is_foot_1 = self.jnp.any(pairs[:, 1:2] == foot_ids[None, :], axis=-1)
                hit = active & ((is_foot_0 & is_terrain_1) | (is_foot_1 & is_terrain_0))
                counts = self.jnp.zeros((self.num_envs,), dtype=self.jnp.int32).at[safe_world_ids].add(hit.astype(self.jnp.int32))
                outputs.append(counts > 0)
            return self.jnp.stack(outputs, axis=1)

        pairs = data.contact.geom.astype(self.jnp.int32)
        distance = data.contact.dist.reshape(self.num_envs, -1)
        active = distance <= 0.0
        is_terrain_0 = self.jnp.any(pairs[..., 0, None] == self.terrain_geom_ids, axis=-1)
        is_terrain_1 = self.jnp.any(pairs[..., 1, None] == self.terrain_geom_ids, axis=-1)
        outputs = []
        for foot_ids in self.foot_geom_ids:
            is_foot_0 = self.jnp.any(pairs[..., 0, None] == foot_ids, axis=-1)
            is_foot_1 = self.jnp.any(pairs[..., 1, None] == foot_ids, axis=-1)
            outputs.append(self.jnp.any(active & ((is_foot_0 & is_terrain_1) | (is_foot_1 & is_terrain_0)), axis=1))
        return self.jnp.stack(outputs, axis=1)

    def release_backend_transients(self) -> None:
        """Release address-specific MJX-Warp graphs between training updates."""
        if self.impl != "warp":
            return

        import warp as wp
        from mujoco.mjx.third_party.warp._src.jax_experimental import ffi

        wp.synchronize_device(self.spec.device)
        ffi.clear_jax_callable_graph_cache()


def configure_native_jax(device: str) -> None:
    if device == "cpu":
        os.environ.setdefault("JAX_PLATFORMS", "cpu")
    elif device.startswith("cuda"):
        os.environ.setdefault("JAX_PLATFORMS", "cuda")


def validate_native_device(spec: MujocoRslRlEnvSpec) -> None:
    import jax

    if spec.device == "cpu" and spec.mjx_impl == "warp":
        raise ValueError("MJX-Warp requires CUDA; use --mjx-impl jax for CPU tests")
    if spec.mjx_impl not in {"warp", "jax"}:
        raise ValueError(f"Unsupported MJX implementation {spec.mjx_impl!r}")
    if spec.device.startswith("cuda") and not any(device.platform == "gpu" for device in jax.devices()):
        raise RuntimeError("CUDA was requested but JAX has no GPU device")
