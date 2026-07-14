"""Record one deterministic G1 control step from Isaac for MuJoCo parity checks."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", default="LeggedLab-Isaac-AMP-G1-SplitPolicy-HeightScan-v0")
parser.add_argument("--output", type=Path, required=True)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import numpy as np
import torch

import isaaclab_tasks  # noqa: F401
import legged_lab.tasks  # noqa: F401
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab_tasks.utils.hydra import hydra_task_config


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg, _agent_cfg) -> None:
    env_cfg.scene.num_envs = 1
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    env_cfg.seed = 42
    # This is a controller/simulator parity probe, not a training rollout.  Keep
    # Isaac's startup model randomization out of the comparison with fixed MJX.
    for event_name in (
        "physics_material",
        "add_base_mass",
        "randomize_rigid_body_com",
        "scale_link_mass",
        "scale_actuator_gains",
        "scale_joint_parameters",
    ):
        if hasattr(env_cfg.events, event_name):
            setattr(env_cfg.events, event_name, None)
    env = gym.make(args_cli.task, cfg=env_cfg)
    base_env = env.unwrapped
    env.reset(seed=42)
    robot = base_env.scene["robot"]

    root_pose = torch.zeros((1, 7), device=base_env.device)
    root_pose[:, :3] = base_env.scene.env_origins
    root_pose[:, 2] += 0.8
    root_pose[:, 3] = 1.0
    joint_pos = robot.data.default_joint_pos.clone()
    joint_vel = torch.zeros_like(joint_pos)
    robot.write_root_pose_to_sim(root_pose)
    robot.write_root_velocity_to_sim(torch.zeros((1, 6), device=base_env.device))
    robot.write_joint_state_to_sim(joint_pos, joint_vel)
    base_env.scene.write_data_to_sim()
    base_env.sim.step(render=False)
    base_env.scene.update(dt=base_env.physics_dt)

    initial_root_pos = robot.data.root_pos_w[0].detach().cpu().numpy().copy()
    initial_root_quat = robot.data.root_quat_w[0].detach().cpu().numpy().copy()
    initial_root_lin_vel_w = robot.data.root_lin_vel_w[0].detach().cpu().numpy().copy()
    initial_root_ang_vel_w = robot.data.root_ang_vel_w[0].detach().cpu().numpy().copy()
    initial_joint_pos = robot.data.joint_pos[0].detach().cpu().numpy().copy()
    initial_joint_vel = robot.data.joint_vel[0].detach().cpu().numpy().copy()

    action = torch.linspace(-0.5, 0.5, robot.num_joints, device=base_env.device).unsqueeze(0)
    base_env.action_manager.process_action(action)
    torques = []
    root_pos_trace = []
    root_quat_trace = []
    root_lin_vel_trace = []
    root_ang_vel_trace = []
    joint_pos_trace = []
    joint_vel_trace = []
    for _ in range(base_env.cfg.decimation):
        base_env.action_manager.apply_action()
        base_env.scene.write_data_to_sim()
        torques.append(robot.data.applied_torque[0].detach().cpu().numpy().copy())
        base_env.sim.step(render=False)
        base_env.scene.update(dt=base_env.physics_dt)
        root_pos_trace.append(robot.data.root_pos_w[0].detach().cpu().numpy().copy())
        root_quat_trace.append(robot.data.root_quat_w[0].detach().cpu().numpy().copy())
        root_lin_vel_trace.append(robot.data.root_lin_vel_w[0].detach().cpu().numpy().copy())
        root_ang_vel_trace.append(robot.data.root_ang_vel_w[0].detach().cpu().numpy().copy())
        joint_pos_trace.append(robot.data.joint_pos[0].detach().cpu().numpy().copy())
        joint_vel_trace.append(robot.data.joint_vel[0].detach().cpu().numpy().copy())

    args_cli.output.expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args_cli.output.expanduser().resolve(),
        action=action[0].detach().cpu().numpy(),
        joint_names=np.asarray(robot.joint_names),
        torques=np.asarray(torques),
        initial_root_pos=initial_root_pos,
        initial_root_quat=initial_root_quat,
        initial_root_lin_vel_w=initial_root_lin_vel_w,
        initial_root_ang_vel_w=initial_root_ang_vel_w,
        initial_joint_pos=initial_joint_pos,
        initial_joint_vel=initial_joint_vel,
        root_pos_trace=np.asarray(root_pos_trace),
        root_quat_trace=np.asarray(root_quat_trace),
        root_lin_vel_w_trace=np.asarray(root_lin_vel_trace),
        root_ang_vel_w_trace=np.asarray(root_ang_vel_trace),
        joint_pos_trace=np.asarray(joint_pos_trace),
        joint_vel_trace=np.asarray(joint_vel_trace),
        root_pos=robot.data.root_pos_w[0].detach().cpu().numpy(),
        root_quat=robot.data.root_quat_w[0].detach().cpu().numpy(),
        root_lin_vel_w=robot.data.root_lin_vel_w[0].detach().cpu().numpy(),
        root_ang_vel_w=robot.data.root_ang_vel_w[0].detach().cpu().numpy(),
        joint_pos=robot.data.joint_pos[0].detach().cpu().numpy(),
        joint_vel=robot.data.joint_vel[0].detach().cpu().numpy(),
    )
    print(f"[PASS] Wrote Isaac control parity probe: {args_cli.output.expanduser().resolve()}")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
