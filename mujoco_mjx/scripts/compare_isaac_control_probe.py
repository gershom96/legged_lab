#!/usr/bin/env python3
"""Replay an Isaac control probe through MJX and report torque/state deltas.

The probe produced by ``scripts/rsl_rl/diagnose_g1_control.py`` starts both
simulators from the *recorded* Isaac state, applies the same 29-D Isaac-order
action for one policy step (four 5 ms substeps), and compares the controller
torques and resulting state.  Domain randomization is intentionally disabled.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("/tmp/mujoco_g1_control_probe.npz"))
    parser.add_argument(
        "--config", type=Path, default=REPO_ROOT / "mujoco_mjx/configs/g1_rsl_rl_mjx_amp.yaml"
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--mjx-impl", choices=("warp", "jax"), default="warp")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with np.load(args.probe.expanduser().resolve(), allow_pickle=False) as probe:
        required = {
            "action",
            "joint_names",
            "torques",
            "initial_root_pos",
            "initial_root_quat",
            "initial_root_lin_vel_w",
            "initial_root_ang_vel_w",
            "initial_joint_pos",
            "initial_joint_vel",
            "root_pos",
            "root_quat",
            "root_lin_vel_w",
            "root_ang_vel_w",
            "joint_pos",
            "joint_vel",
        }
        missing = required.difference(probe.files)
        if missing:
            raise ValueError(f"Probe is missing required arrays: {sorted(missing)}")
        source = {key: np.asarray(probe[key]).copy() for key in probe.files}

    from mujoco_mjx.rsl_rl_mujoco.constants import (
        ACTION_SCALE,
        DEFAULT_JOINT_POS_ISAAC,
        EFFORT_LIMIT_ISAAC,
        ISAAC_INDICES_IN_MUJOCO_ORDER,
        ISAAC_JOINT_NAMES,
        MUJOCO_INDICES_IN_ISAAC_ORDER,
        STIFFNESS_ISAAC,
        DAMPING_ISAAC,
    )
    from mujoco_mjx.rsl_rl_mujoco.model import build_mujoco_model
    from mujoco_mjx.rsl_rl_mujoco.native_physics import configure_native_jax
    from mujoco_mjx.rsl_rl_mujoco.spec import (
        DomainRandomizationSpec,
        MujocoRslRlEnvSpec,
        TerrainCurriculumSpec,
        TerrainSpec,
    )
    from mujoco_mjx.rsl_rl_mujoco.terrain import generate_terrain

    if tuple(source["joint_names"].tolist()) != tuple(ISAAC_JOINT_NAMES):
        raise ValueError("The Isaac probe joint names do not match the required 29-D Isaac action order")

    configure_native_jax(args.device)
    import jax
    import jax.numpy as jnp
    from mujoco import mjx

    with args.config.expanduser().resolve().open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    spec = MujocoRslRlEnvSpec.from_mapping(config, REPO_ROOT)
    spec = replace(
        spec,
        num_envs=1,
        device=args.device,
        mjx_impl=args.mjx_impl,
        terrain=TerrainSpec(type="flat", size=(45.0, 45.0), curriculum=TerrainCurriculumSpec(enabled=False)),
        domain_randomization=DomainRandomizationSpec(enabled=False),
    )
    terrain = generate_terrain(spec.terrain)
    model, metadata, _ = build_mujoco_model(spec, terrain, REPO_ROOT / "mujoco_mjx/outputs/control_probe")
    mx = mjx.put_model(model, impl=spec.mjx_impl)
    data = mjx.make_data(model, impl=spec.mjx_impl) if spec.mjx_impl == "warp" else mjx.make_data(mx)

    qpos = np.zeros((model.nq,), dtype=np.float32)
    qvel = np.zeros((model.nv,), dtype=np.float32)
    qpos[:3] = source["initial_root_pos"]
    qpos[3:7] = source["initial_root_quat"]
    qvel[:3] = source["initial_root_lin_vel_w"]
    qvel[3:6] = source["initial_root_ang_vel_w"]
    qpos[np.asarray(metadata.joint_qpos_indices)] = source["initial_joint_pos"][
        np.asarray(ISAAC_INDICES_IN_MUJOCO_ORDER)
    ]
    qvel[np.asarray(metadata.joint_qvel_indices)] = source["initial_joint_vel"][
        np.asarray(ISAAC_INDICES_IN_MUJOCO_ORDER)
    ]
    data = mjx.forward(mx, data.replace(qpos=jnp.asarray(qpos), qvel=jnp.asarray(qvel)))

    qpos_indices = jnp.asarray(metadata.joint_qpos_indices, dtype=jnp.int32)
    qvel_indices = jnp.asarray(metadata.joint_qvel_indices, dtype=jnp.int32)
    actuator_indices = jnp.asarray(metadata.actuator_indices, dtype=jnp.int32)
    isaac_to_mujoco = jnp.asarray(ISAAC_INDICES_IN_MUJOCO_ORDER, dtype=jnp.int32)
    mujoco_to_isaac = jnp.asarray(MUJOCO_INDICES_IN_ISAAC_ORDER, dtype=jnp.int32)
    default_mujoco = jnp.asarray(DEFAULT_JOINT_POS_ISAAC)[isaac_to_mujoco]
    kp_mujoco = jnp.asarray(STIFFNESS_ISAAC)[isaac_to_mujoco]
    kd_mujoco = jnp.asarray(DAMPING_ISAAC)[isaac_to_mujoco]
    limit_mujoco = jnp.asarray(EFFORT_LIMIT_ISAAC)[isaac_to_mujoco]
    action_isaac = jnp.asarray(source["action"], dtype=jnp.float32)
    target = default_mujoco + ACTION_SCALE * action_isaac[isaac_to_mujoco]

    def step_policy(data):
        def substep(current, _):
            joint_pos = current.qpos[qpos_indices]
            joint_vel = current.qvel[qvel_indices]
            torque_mujoco = jnp.clip(
                kp_mujoco * (target - joint_pos) - kd_mujoco * joint_vel,
                -limit_mujoco,
                limit_mujoco,
            )
            ctrl = jnp.zeros_like(current.ctrl).at[actuator_indices].set(torque_mujoco)
            return mjx.step(mx, current.replace(ctrl=ctrl)), torque_mujoco[mujoco_to_isaac]

        def traced_substep(current, _):
            next_data, torque = substep(current, None)
            return next_data, (torque, next_data.qpos, next_data.qvel)

        return jax.lax.scan(traced_substep, data, xs=None, length=4)

    final_data, (torque_trace, qpos_trace, qvel_trace) = jax.jit(step_policy)(data)
    final_data = jax.block_until_ready(final_data)
    torque_trace = np.asarray(jax.block_until_ready(torque_trace))
    qpos_trace = np.asarray(jax.block_until_ready(qpos_trace))
    qvel_trace = np.asarray(jax.block_until_ready(qvel_trace))

    final_joint_pos = np.asarray(final_data.qpos)[np.asarray(metadata.joint_qpos_indices)][
        np.asarray(MUJOCO_INDICES_IN_ISAAC_ORDER)
    ]
    final_joint_vel = np.asarray(final_data.qvel)[np.asarray(metadata.joint_qvel_indices)][
        np.asarray(MUJOCO_INDICES_IN_ISAAC_ORDER)
    ]
    final_root_pos = np.asarray(final_data.qpos)[:3]
    final_root_quat = np.asarray(final_data.qpos)[3:7]
    final_root_lin_vel = np.asarray(final_data.qvel)[:3]
    final_root_ang_vel = np.asarray(final_data.qvel)[3:6]
    root_pos_trace = qpos_trace[:, :3]
    root_quat_trace = qpos_trace[:, 3:7]
    root_lin_vel_trace = qvel_trace[:, :3]
    root_ang_vel_trace = qvel_trace[:, 3:6]
    joint_pos_trace = qpos_trace[:, np.asarray(metadata.joint_qpos_indices)][:, np.asarray(MUJOCO_INDICES_IN_ISAAC_ORDER)]
    joint_vel_trace = qvel_trace[:, np.asarray(metadata.joint_qvel_indices)][:, np.asarray(MUJOCO_INDICES_IN_ISAAC_ORDER)]

    def report(name: str, mujoco_value: np.ndarray, isaac_value: np.ndarray) -> None:
        delta = mujoco_value - isaac_value
        print(
            f"{name:24s} max_abs={np.max(np.abs(delta)):.7g} "
            f"rms={np.sqrt(np.mean(delta**2)):.7g}"
        )

    print("[CONTROL PARITY] Isaac vs MJX, same recorded state/action, 4 x 0.005 s")
    report("applied_torque", torque_trace, source["torques"])
    for step, (native, isaac) in enumerate(zip(torque_trace, source["torques"], strict=True)):
        report(f"torque_substep_{step}", native, isaac)
    report("root_position", final_root_pos, source["root_pos"])
    report("root_quaternion", final_root_quat, source["root_quat"])
    report("root_linear_velocity", final_root_lin_vel, source["root_lin_vel_w"])
    report("root_angular_velocity", final_root_ang_vel, source["root_ang_vel_w"])
    report("joint_position", final_joint_pos, source["joint_pos"])
    report("joint_velocity", final_joint_vel, source["joint_vel"])
    trace_pairs = (
        ("root_position_trace", root_pos_trace, "root_pos_trace"),
        ("root_quaternion_trace", root_quat_trace, "root_quat_trace"),
        ("root_linear_velocity_trace", root_lin_vel_trace, "root_lin_vel_w_trace"),
        ("root_angular_velocity_trace", root_ang_vel_trace, "root_ang_vel_w_trace"),
        ("joint_position_trace", joint_pos_trace, "joint_pos_trace"),
        ("joint_velocity_trace", joint_vel_trace, "joint_vel_trace"),
    )
    for name, native, source_name in trace_pairs:
        if source_name in source:
            report(name, native, source[source_name])

    args.output.expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.output.expanduser().resolve(),
        torques=torque_trace,
        root_pos=final_root_pos,
        root_quat=final_root_quat,
        root_lin_vel_w=final_root_lin_vel,
        root_ang_vel_w=final_root_ang_vel,
        joint_pos=final_joint_pos,
        joint_vel=final_joint_vel,
        root_pos_trace=root_pos_trace,
        root_quat_trace=root_quat_trace,
        root_lin_vel_w_trace=root_lin_vel_trace,
        root_ang_vel_w_trace=root_ang_vel_trace,
        joint_pos_trace=joint_pos_trace,
        joint_vel_trace=joint_vel_trace,
    )
    print(f"[PASS] Wrote MuJoCo control probe: {args.output.expanduser().resolve()}")


if __name__ == "__main__":
    main()
