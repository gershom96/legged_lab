# Copyright (c) 2026
# SPDX-License-Identifier: BSD-3-Clause

"""Play a G1 actor checkpoint with keyboard velocity commands."""

from __future__ import annotations

import argparse
import os
import sys
import time
import weakref
from pathlib import Path

from isaaclab.app import AppLauncher

import cli_args  # isort: skip


parser = argparse.ArgumentParser(description="Run a G1 actor checkpoint with keyboard vx/vy/yaw-rate control.")
parser.add_argument("--video", action="store_true", default=False, help="Record a video from the play rollout.")
parser.add_argument("--video_length", type=int, default=600, help="Length of the recorded video in env steps.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default="LeggedLab-Isaac-AMP-G1-Play-v0", help="Task name.")
parser.add_argument("--agent", type=str, default="rsl_rl_cfg_entry_point", help="RSL-RL config entry point.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment.")
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
parser.add_argument("--vx", type=float, default=0.0, help="Initial/fixed forward velocity command.")
parser.add_argument("--vy", type=float, default=0.0, help="Initial/fixed lateral velocity command.")
parser.add_argument("--omega", type=float, default=0.0, help="Initial/fixed yaw-rate command.")
parser.add_argument("--vx_step", type=float, default=0.1, help="Keyboard increment for forward velocity.")
parser.add_argument("--vy_step", type=float, default=0.1, help="Keyboard increment for lateral velocity.")
parser.add_argument("--omega_step", type=float, default=0.2, help="Keyboard increment for yaw-rate.")
parser.add_argument("--max_vx", type=float, default=1.0, help="Absolute clamp for vx.")
parser.add_argument("--max_vy", type=float, default=1.0, help="Absolute clamp for vy.")
parser.add_argument("--max_omega", type=float, default=1.0, help="Absolute clamp for yaw-rate.")
parser.add_argument("--max_steps", type=int, default=0, help="Stop after this many env steps. Use 0 to run forever.")
parser.add_argument(
    "--load_references",
    action="store_true",
    default=False,
    help="Keep AMP reference motion/animation managers enabled. Actor keyboard playback disables them by default.",
)

cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

if args_cli.video:
    args_cli.enable_cameras = True

sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import carb  # noqa: E402
import gymnasium as gym  # noqa: E402
import omni  # noqa: E402
import torch  # noqa: E402

from isaaclab.envs import DirectMARLEnv, DirectMARLEnvCfg, DirectRLEnvCfg, ManagerBasedRLEnvCfg, multi_agent_to_single_agent  # noqa: E402
from isaaclab.utils.assets import retrieve_file_path  # noqa: E402
from isaaclab.utils.dict import print_dict  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper  # noqa: E402
from isaaclab_tasks.utils.hydra import hydra_task_config  # noqa: E402

import isaaclab_tasks  # noqa: F401,E402
import legged_lab.tasks  # noqa: F401,E402
from train_bc import make_actor_critic  # noqa: E402


class KeyboardVelocity:
    """Omniverse keyboard listener for a shared [vx, vy, omega] command."""

    def __init__(
        self,
        device: torch.device,
        initial_command: tuple[float, float, float],
        steps: tuple[float, float, float],
        limits: tuple[float, float, float],
    ):
        self.device = device
        self.command = torch.tensor(initial_command, dtype=torch.float32, device=device)
        self.steps = torch.tensor(steps, dtype=torch.float32, device=device)
        self.limits = torch.tensor(limits, dtype=torch.float32, device=device)

        self._appwindow = omni.appwindow.get_default_app_window()
        self._input = carb.input.acquire_input_interface()
        self._keyboard = self._appwindow.get_keyboard()
        self._sub = self._input.subscribe_to_keyboard_events(
            self._keyboard,
            lambda event, *args, obj=weakref.proxy(self): obj._on_keyboard_event(event, *args),
        )
        self._print_help()

    def close(self):
        if getattr(self, "_sub", None) is not None:
            self._input.unsubscribe_from_keyboard_events(self._keyboard, self._sub)
            self._sub = None

    def _print_help(self):
        print("[KEYBOARD] W/S or Up/Down: vx +/-")
        print("[KEYBOARD] A/D or Left/Right: vy +/-")
        print("[KEYBOARD] Q/E: yaw-rate +/-")
        print("[KEYBOARD] X or Space: zero command")
        self._print_command()

    def _print_command(self):
        print(
            f"[KEYBOARD] command vx={self.command[0].item():+.2f}, "
            f"vy={self.command[1].item():+.2f}, omega={self.command[2].item():+.2f}"
        )

    def _on_keyboard_event(self, event, *args, **kwargs):
        if event.type != carb.input.KeyboardEventType.KEY_PRESS:
            return True

        key = event.input.name.upper()
        delta = torch.zeros(3, dtype=torch.float32, device=self.device)
        if key in {"W", "UP"}:
            delta[0] = self.steps[0]
        elif key in {"S", "DOWN"}:
            delta[0] = -self.steps[0]
        elif key in {"A", "LEFT"}:
            delta[1] = self.steps[1]
        elif key in {"D", "RIGHT"}:
            delta[1] = -self.steps[1]
        elif key == "Q":
            delta[2] = self.steps[2]
        elif key == "E":
            delta[2] = -self.steps[2]
        elif key in {"X", "SPACE"}:
            self.command.zero_()
            self._print_command()
            return True
        else:
            return True

        self.command = torch.clamp(self.command + delta, -self.limits, self.limits)
        self._print_command()
        return True


def _set_manual_command(raw_env, command: torch.Tensor):
    command_term = raw_env.command_manager.get_term("base_velocity")
    command_term.vel_command_b[:, :] = command.to(command_term.vel_command_b.device).unsqueeze(0)
    if hasattr(command_term, "is_standing_env"):
        command_term.is_standing_env[:] = False
    if hasattr(command_term, "is_heading_env"):
        command_term.is_heading_env[:] = False


def _configure_play_env(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    if hasattr(env_cfg, "observations"):
        if hasattr(env_cfg.observations, "policy"):
            env_cfg.observations.policy.enable_corruption = False
        if hasattr(env_cfg.observations, "critic"):
            env_cfg.observations.critic.enable_corruption = False

    if not args_cli.load_references:
        if hasattr(env_cfg, "events") and hasattr(env_cfg.events, "reset_from_ref"):
            env_cfg.events.reset_from_ref = None
        if hasattr(env_cfg, "observations"):
            if hasattr(env_cfg.observations, "disc"):
                env_cfg.observations.disc = None
            if hasattr(env_cfg.observations, "disc_demo"):
                env_cfg.observations.disc_demo = None
        if hasattr(env_cfg, "motion_data") and hasattr(env_cfg.motion_data, "motion_dataset"):
            env_cfg.motion_data.motion_dataset = None
        if hasattr(env_cfg, "animation") and hasattr(env_cfg.animation, "animation"):
            env_cfg.animation.animation = None

    if hasattr(env_cfg, "commands") and hasattr(env_cfg.commands, "base_velocity"):
        command_cfg = env_cfg.commands.base_velocity
        command_cfg.resampling_time_range = (1.0e9, 1.0e9)
        command_cfg.rel_standing_envs = 0.0
        command_cfg.rel_heading_envs = 0.0
        command_cfg.heading_command = False
        command_cfg.debug_vis = True
        command_cfg.ranges.lin_vel_x = (-args_cli.max_vx, args_cli.max_vx)
        command_cfg.ranges.lin_vel_y = (-args_cli.max_vy, args_cli.max_vy)
        command_cfg.ranges.ang_vel_z = (-args_cli.max_omega, args_cli.max_omega)


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    _configure_play_env(env_cfg, agent_cfg)

    if args_cli.checkpoint is None:
        raise ValueError("Pass an actor checkpoint with --checkpoint, for example logs/rsl_rl/.../model_bc.pt")
    checkpoint_path = retrieve_file_path(args_cli.checkpoint)
    log_dir = os.path.dirname(checkpoint_path)
    env_cfg.log_dir = log_dir

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "actor_keyboard"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording video.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    raw_env = env.unwrapped
    device = torch.device(raw_env.device)

    model = make_actor_critic(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    print(f"[INFO] Loaded actor checkpoint: {checkpoint_path}")

    keyboard = None
    fixed_command = torch.tensor([args_cli.vx, args_cli.vy, args_cli.omega], dtype=torch.float32, device=device)
    if args_cli.headless:
        print(
            f"[WARN] Headless mode: keyboard unavailable. Using fixed command "
            f"vx={args_cli.vx:+.2f}, vy={args_cli.vy:+.2f}, omega={args_cli.omega:+.2f}."
        )
    else:
        keyboard = KeyboardVelocity(
            device=device,
            initial_command=(args_cli.vx, args_cli.vy, args_cli.omega),
            steps=(args_cli.vx_step, args_cli.vy_step, args_cli.omega_step),
            limits=(args_cli.max_vx, args_cli.max_vy, args_cli.max_omega),
        )

    dt = raw_env.step_dt
    obs = env.get_observations()
    step = 0
    try:
        while simulation_app.is_running():
            start_time = time.time()
            command = keyboard.command if keyboard is not None else fixed_command
            _set_manual_command(raw_env, command)
            with torch.inference_mode():
                actions = model.act_inference(obs)
                obs, _, dones, _ = env.step(actions)
                model.reset(dones)
            step += 1
            if args_cli.video and step >= args_cli.video_length:
                break
            if args_cli.max_steps > 0 and step >= args_cli.max_steps:
                break
            sleep_time = dt - (time.time() - start_time)
            if args_cli.real_time and sleep_time > 0:
                time.sleep(sleep_time)
    finally:
        if keyboard is not None:
            keyboard.close()
        env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
