"""Play a G1 actor checkpoint in MuJoCo with keyboard velocity commands."""

from __future__ import annotations

import argparse
import os
import tempfile
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np
import torch
from tensordict import TensorDict

from train_bc import ACTION_SCALE, G1_DEFAULT_DOF_POS, NUM_ACTIONS, NUM_CRITIC_OBS, NUM_POLICY_OBS, make_actor_critic


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL = REPO_ROOT / "source/legged_lab/legged_lab/data/Robots/Unitree/g1_29dof/g1_29dof.xml"

JOINT_NAMES = [
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
]

KEY_BODY_NAMES = [
    "left_ankle_roll_link",
    "right_ankle_roll_link",
    "left_wrist_yaw_link",
    "right_wrist_yaw_link",
    "left_shoulder_roll_link",
    "right_shoulder_roll_link",
]

GLFW_KEY_UP = 265
GLFW_KEY_DOWN = 264
GLFW_KEY_LEFT = 263
GLFW_KEY_RIGHT = 262
GLFW_KEY_SPACE = 32


class KeyboardCommand:
    def __init__(
        self,
        initial: tuple[float, float, float],
        step: tuple[float, float, float],
        limits: tuple[float, float, float],
    ):
        self.command = np.asarray(initial, dtype=np.float32)
        self.step = np.asarray(step, dtype=np.float32)
        self.limits = np.asarray(limits, dtype=np.float32)

    def key_callback(self, keycode: int):
        delta = np.zeros(3, dtype=np.float32)
        key = chr(keycode).upper() if 0 <= keycode < 256 else ""
        if key == "W" or keycode == GLFW_KEY_UP:
            delta[0] = self.step[0]
        elif key == "S" or keycode == GLFW_KEY_DOWN:
            delta[0] = -self.step[0]
        elif key == "A" or keycode == GLFW_KEY_LEFT:
            delta[1] = self.step[1]
        elif key == "D" or keycode == GLFW_KEY_RIGHT:
            delta[1] = -self.step[1]
        elif key == "Q":
            delta[2] = self.step[2]
        elif key == "E":
            delta[2] = -self.step[2]
        elif key == "X" or keycode == GLFW_KEY_SPACE:
            self.command[:] = 0.0
            self.print_command()
            return
        else:
            return
        self.command[:] = np.clip(self.command + delta, -self.limits, self.limits)
        self.print_command()

    def print_help(self):
        print("[KEYBOARD] W/S or Up/Down: vx +/-")
        print("[KEYBOARD] A/D or Left/Right: vy +/-")
        print("[KEYBOARD] Q/E: yaw-rate +/-")
        print("[KEYBOARD] X or Space: zero command")
        self.print_command()

    def print_command(self):
        print(f"[KEYBOARD] vx={self.command[0]:+.2f}, vy={self.command[1]:+.2f}, omega={self.command[2]:+.2f}")


def quat_normalize(q: np.ndarray) -> np.ndarray:
    return q / max(float(np.linalg.norm(q)), 1.0e-8)


def quat_conjugate(q: np.ndarray) -> np.ndarray:
    q = quat_normalize(q)
    return np.asarray([q[0], -q[1], -q[2], -q[3]], dtype=np.float64)


def quat_apply(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    q = quat_normalize(q)
    v = np.asarray(v, dtype=np.float64)
    q_xyz = q[1:4]
    t = 2.0 * np.cross(q_xyz, v)
    return v + q[0] * t + np.cross(q_xyz, t)


def quat_apply_inverse(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    return quat_apply(quat_conjugate(q), v)


def add_ground_to_model_xml(model_path: Path) -> str:
    model_path = model_path.resolve()
    text = model_path.read_text(encoding="utf-8")
    meshdir = (model_path.parent / "meshes").as_posix()
    text = text.replace('meshdir="meshes"', f'meshdir="{meshdir}"')
    ground = (
        '<light name="tracking_light" pos="0 -3 4" dir="0 1 -1" diffuse="0.8 0.8 0.8"/>\n'
        '    <geom name="ground" type="plane" pos="0 0 0" size="200 200 0.1" '
        'rgba="0.35 0.35 0.35 1" friction="1.0 0.01 0.001"/>\n'
    )
    text = text.replace("<worldbody>", "<worldbody>\n    " + ground, 1)
    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".xml", delete=False, encoding="utf-8")
    tmp.write(text)
    tmp.close()
    return tmp.name


def joint_indices(model: mujoco.MjModel) -> tuple[np.ndarray, np.ndarray]:
    qpos_indices = []
    qvel_indices = []
    for name in JOINT_NAMES:
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if joint_id < 0:
            raise KeyError(f"Missing joint in MuJoCo model: {name}")
        qpos_indices.append(int(model.jnt_qposadr[joint_id]))
        qvel_indices.append(int(model.jnt_dofadr[joint_id]))
    return np.asarray(qpos_indices, dtype=np.int64), np.asarray(qvel_indices, dtype=np.int64)


def body_ids(model: mujoco.MjModel) -> np.ndarray:
    ids = []
    for name in KEY_BODY_NAMES:
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        if body_id < 0:
            raise KeyError(f"Missing body in MuJoCo model: {name}")
        ids.append(int(body_id))
    return np.asarray(ids, dtype=np.int64)


def actuator_gains() -> tuple[np.ndarray, np.ndarray]:
    kp = np.zeros(NUM_ACTIONS, dtype=np.float64)
    kd = np.zeros(NUM_ACTIONS, dtype=np.float64)
    for i, name in enumerate(JOINT_NAMES):
        if "knee" in name:
            kp[i], kd[i] = 150.0, 4.0
        elif "hip" in name:
            kp[i], kd[i] = 100.0, 2.0
        elif name == "waist_yaw_joint":
            kp[i], kd[i] = 200.0, 5.0
        elif name in {"waist_roll_joint", "waist_pitch_joint"}:
            kp[i], kd[i] = 40.0, 5.0
        elif "ankle" in name:
            kp[i], kd[i] = 40.0, 2.0
        else:
            kp[i], kd[i] = 40.0, 1.0
    return kp, kd


def reset_robot(data: mujoco.MjData, qpos_indices: np.ndarray, default_dof_pos: np.ndarray):
    data.qpos[:] = 0.0
    data.qvel[:] = 0.0
    data.qpos[0:3] = np.asarray([0.0, 0.0, 0.8], dtype=np.float64)
    data.qpos[3:7] = np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    data.qpos[qpos_indices] = default_dof_pos


def build_observation(
    data: mujoco.MjData,
    qpos_indices: np.ndarray,
    qvel_indices: np.ndarray,
    key_body_ids: np.ndarray,
    command: np.ndarray,
    previous_action: np.ndarray,
    default_dof_pos: np.ndarray,
) -> np.ndarray:
    root_pos = np.asarray(data.qpos[0:3], dtype=np.float64)
    root_quat = quat_normalize(np.asarray(data.qpos[3:7], dtype=np.float64))
    joint_pos = np.asarray(data.qpos[qpos_indices], dtype=np.float64)
    joint_vel = np.asarray(data.qvel[qvel_indices], dtype=np.float64)
    root_ang_vel_b = np.asarray(data.sensor("imu_gyro").data, dtype=np.float64)
    projected_gravity = quat_apply_inverse(root_quat, np.asarray([0.0, 0.0, -1.0], dtype=np.float64))

    key_body_pos_b = []
    for body_id in key_body_ids:
        pos_w = np.asarray(data.xpos[body_id], dtype=np.float64)
        key_body_pos_b.append(quat_apply_inverse(root_quat, pos_w - root_pos))

    obs = np.concatenate(
        [
            root_ang_vel_b,
            projected_gravity,
            command.astype(np.float64),
            joint_pos - default_dof_pos,
            joint_vel,
            previous_action.astype(np.float64),
            np.concatenate(key_body_pos_b),
        ],
        axis=0,
    ).astype(np.float32)
    if obs.shape != (NUM_POLICY_OBS,):
        raise RuntimeError(f"Expected {NUM_POLICY_OBS} obs dims, got {obs.shape}")
    return obs


def parse_args():
    parser = argparse.ArgumentParser(description="Run a G1 actor checkpoint in MuJoCo with keyboard commands.")
    parser.add_argument("--checkpoint", required=True, help="Path to model_bc.pt or an RSL-RL model_*.pt checkpoint.")
    parser.add_argument("--model", default=str(DEFAULT_MODEL), help="G1 MuJoCo XML path.")
    parser.add_argument("--device", default="cpu", help="Torch device for actor inference.")
    parser.add_argument("--headless", action="store_true", help="Run without the MuJoCo viewer.")
    parser.add_argument("--duration", type=float, default=120.0, help="Simulation duration in seconds.")
    parser.add_argument("--dt", type=float, default=0.002, help="MuJoCo physics timestep.")
    parser.add_argument("--decimation", type=int, default=10, help="Physics steps per policy step.")
    parser.add_argument("--real-time", action="store_true", default=True, help="Throttle simulation to real time.")
    parser.add_argument("--no-real-time", action="store_false", dest="real_time", help="Run as fast as possible.")
    parser.add_argument("--vx", type=float, default=0.0)
    parser.add_argument("--vy", type=float, default=0.0)
    parser.add_argument("--omega", type=float, default=0.0)
    parser.add_argument("--vx_step", type=float, default=0.1)
    parser.add_argument("--vy_step", type=float, default=0.1)
    parser.add_argument("--omega_step", type=float, default=0.2)
    parser.add_argument("--max_vx", type=float, default=1.0)
    parser.add_argument("--max_vy", type=float, default=1.0)
    parser.add_argument("--max_omega", type=float, default=1.0)
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)
    default_dof_pos = G1_DEFAULT_DOF_POS.cpu().numpy().astype(np.float64)
    model_xml = add_ground_to_model_xml(Path(args.model))
    try:
        model = mujoco.MjModel.from_xml_path(model_xml)
    finally:
        os.unlink(model_xml)
    model.opt.timestep = args.dt
    data = mujoco.MjData(model)

    qpos_indices, qvel_indices = joint_indices(model)
    key_body_ids = body_ids(model)
    kp, kd = actuator_gains()
    torque_limits = np.asarray(model.actuator_ctrlrange[:, 1], dtype=np.float64)
    command = KeyboardCommand(
        initial=(args.vx, args.vy, args.omega),
        step=(args.vx_step, args.vy_step, args.omega_step),
        limits=(args.max_vx, args.max_vy, args.max_omega),
    )

    actor = make_actor_critic(device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    actor.load_state_dict(checkpoint["model_state_dict"], strict=True)
    actor.eval()
    print(f"[INFO] Loaded checkpoint: {args.checkpoint}")

    reset_robot(data, qpos_indices, default_dof_pos)
    mujoco.mj_forward(model, data)
    previous_action = np.zeros(NUM_ACTIONS, dtype=np.float32)
    target_joint_pos = default_dof_pos.copy()
    sim_steps = int(args.duration / args.dt)

    command.print_help()
    viewer = None
    if not args.headless:
        viewer = mujoco.viewer.launch_passive(model, data, key_callback=command.key_callback)
        viewer.cam.distance = 4.0
        viewer.cam.azimuth = 135.0
        viewer.cam.elevation = -20.0

    try:
        start = time.time()
        for step in range(sim_steps):
            if step % args.decimation == 0:
                obs = build_observation(
                    data,
                    qpos_indices,
                    qvel_indices,
                    key_body_ids,
                    command.command,
                    previous_action,
                    default_dof_pos,
                )
                obs_td = TensorDict(
                    {
                        "policy": torch.from_numpy(obs).to(device).unsqueeze(0),
                        "critic": torch.zeros(1, NUM_CRITIC_OBS, device=device),
                    },
                    batch_size=[1],
                    device=device,
                )
                with torch.inference_mode():
                    action = actor.act_inference(obs_td)[0].detach().cpu().numpy()
                previous_action = action.astype(np.float32)
                target_joint_pos = default_dof_pos + ACTION_SCALE * action

            joint_pos = np.asarray(data.qpos[qpos_indices], dtype=np.float64)
            joint_vel = np.asarray(data.qvel[qvel_indices], dtype=np.float64)
            tau = kp * (target_joint_pos - joint_pos) - kd * joint_vel
            data.ctrl[:] = np.clip(tau, -torque_limits, torque_limits)
            mujoco.mj_step(model, data)

            if viewer is not None:
                viewer.cam.lookat[:] = data.qpos[0:3]
                viewer.sync()
                if not viewer.is_running():
                    break

            if args.real_time:
                target_time = (step + 1) * args.dt
                sleep_time = start + target_time - time.time()
                if sleep_time > 0.0:
                    time.sleep(sleep_time)
    finally:
        if viewer is not None:
            viewer.close()


if __name__ == "__main__":
    main()
