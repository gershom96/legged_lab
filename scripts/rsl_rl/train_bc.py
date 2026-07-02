"""Offline behavior cloning for the G1 AMP actor.

This trains the same rsl_rl ActorCritic architecture used by the AMP tasks, but
supervises the actor from retargeted motion files.  The saved checkpoint uses the
same ``model_state_dict`` key as AMP checkpoints so it can be passed to
``scripts/rsl_rl/train.py --warm_start``.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import OrderedDict
from datetime import datetime
from pathlib import Path

import joblib
import torch
import torch.nn.functional as F
from tensordict import TensorDict

from rsl_rl.modules import ActorCritic


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_AMP_DIR = REPO_ROOT / "source/legged_lab/legged_lab/data/MotionData/g1_29dof/amp/walk_and_run"
DEFAULT_MOTIONBRICKS_DIR = (
    Path.home() / "Documents/shared_datasets/motionbricks/motionbricks_sonic_grid_walk_dense/legged_lab_g1"
)

NUM_POLICY_OBS = 114
NUM_CRITIC_OBS = 351
NUM_ACTIONS = 29
ACTION_SCALE = 0.25

G1_DEFAULT_DOF_POS = torch.tensor(
    [
        -0.1,
        -0.1,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.3,
        0.3,
        0.3,
        0.3,
        -0.2,
        -0.2,
        0.25,
        -0.25,
        0.0,
        0.0,
        0.0,
        0.0,
        0.97,
        0.97,
        0.15,
        -0.15,
        0.0,
        0.0,
        0.0,
        0.0,
    ],
    dtype=torch.float32,
)


def quat_normalize(q: torch.Tensor) -> torch.Tensor:
    return q / q.norm(dim=-1, keepdim=True).clamp_min(1.0e-8)


def quat_conjugate(q: torch.Tensor) -> torch.Tensor:
    return torch.cat([q[..., :1], -q[..., 1:]], dim=-1)


def quat_mul(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    w1, x1, y1, z1 = q1.unbind(dim=-1)
    w2, x2, y2, z2 = q2.unbind(dim=-1)
    return torch.stack(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        dim=-1,
    )


def quat_apply(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    q = quat_normalize(q)
    q_xyz = q[..., 1:]
    t = 2.0 * torch.cross(q_xyz, v, dim=-1)
    return v + q[..., :1] * t + torch.cross(q_xyz, t, dim=-1)


def quat_apply_inverse(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    return quat_apply(quat_conjugate(q), v)


def next_indices(indices: torch.Tensor, num_frames: int, loop_mode: int) -> torch.Tensor:
    if loop_mode == 1:
        return (indices + 1) % num_frames
    return torch.clamp(indices + 1, max=num_frames - 1)


def prev_indices(indices: torch.Tensor, num_frames: int, loop_mode: int) -> torch.Tensor:
    if loop_mode == 1:
        return (indices - 1) % num_frames
    return torch.clamp(indices - 1, min=0)


def angular_velocity_world(root_quat: torch.Tensor, next_root_quat: torch.Tensor, dt: float) -> torch.Tensor:
    root_quat = quat_normalize(root_quat)
    next_root_quat = quat_normalize(next_root_quat)
    same_hemisphere = (root_quat * next_root_quat).sum(dim=-1, keepdim=True) >= 0.0
    next_root_quat = torch.where(same_hemisphere, next_root_quat, -next_root_quat)

    delta = quat_normalize(quat_mul(next_root_quat, quat_conjugate(root_quat)))
    xyz = delta[..., 1:]
    xyz_norm = xyz.norm(dim=-1, keepdim=True)
    angle = 2.0 * torch.atan2(xyz_norm, delta[..., :1].clamp(min=-1.0, max=1.0))
    axis = xyz / xyz_norm.clamp_min(1.0e-8)
    return axis * angle / dt


class MotionFileCache:
    def __init__(self, max_items: int):
        self.max_items = max_items
        self._items: OrderedDict[Path, dict[str, torch.Tensor | float | int]] = OrderedDict()

    def get(self, path: Path, device: torch.device) -> dict[str, torch.Tensor | float | int]:
        if path in self._items:
            item = self._items.pop(path)
            self._items[path] = item
            return item

        raw = joblib.load(path)
        item = {
            "fps": float(raw["fps"]),
            "dt": 1.0 / float(raw["fps"]),
            "loop_mode": int(raw["loop_mode"]),
            "root_pos": torch.as_tensor(raw["root_pos"], dtype=torch.float32, device=device),
            "root_rot": quat_normalize(torch.as_tensor(raw["root_rot"], dtype=torch.float32, device=device)),
            "dof_pos": torch.as_tensor(raw["dof_pos"], dtype=torch.float32, device=device),
            "key_body_pos": torch.as_tensor(raw["key_body_pos"], dtype=torch.float32, device=device),
        }
        self._items[path] = item
        while len(self._items) > self.max_items:
            self._items.popitem(last=False)
        return item


class MixedMotionSampler:
    def __init__(
        self,
        default_dir: Path,
        motionbricks_dir: Path,
        default_weight: float,
        motionbricks_weight: float,
        cache_size: int,
        device: torch.device,
    ):
        self.datasets: list[tuple[str, list[Path], float]] = []
        if default_weight > 0.0:
            self.datasets.append(("default", sorted(default_dir.glob("*.pkl")), default_weight))
        if motionbricks_weight > 0.0:
            self.datasets.append(("motionbricks", sorted(motionbricks_dir.glob("*.pkl")), motionbricks_weight))
        if not self.datasets:
            raise ValueError("At least one dataset weight must be positive.")
        for name, files, _ in self.datasets:
            if not files:
                raise FileNotFoundError(f"No .pkl motion files found for dataset '{name}'.")

        total_weight = sum(weight for _, _, weight in self.datasets)
        self.dataset_probs = [weight / total_weight for _, _, weight in self.datasets]
        self.cache = MotionFileCache(cache_size)
        self.device = device
        self.default_dof_pos = G1_DEFAULT_DOF_POS.to(device)

    def _sample_file(self) -> Path:
        dataset_idx = random.choices(range(len(self.datasets)), weights=self.dataset_probs, k=1)[0]
        _, files, _ = self.datasets[dataset_idx]
        return random.choice(files)

    def sample(self, batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
        obs_chunks = []
        action_chunks = []
        remaining = batch_size

        while remaining > 0:
            path = self._sample_file()
            motion = self.cache.get(path, self.device)
            dof_pos = motion["dof_pos"]
            num_frames = int(dof_pos.shape[0])
            if num_frames < 3:
                continue

            count = min(remaining, max(1, batch_size // 8))
            frame_idx = torch.randint(0, num_frames, (count,), device=self.device)
            loop_mode = int(motion["loop_mode"])
            next_idx = next_indices(frame_idx, num_frames, loop_mode)
            prev_idx = prev_indices(frame_idx, num_frames, loop_mode)

            root_pos = motion["root_pos"][frame_idx]
            root_pos_next = motion["root_pos"][next_idx]
            root_quat = motion["root_rot"][frame_idx]
            root_quat_next = motion["root_rot"][next_idx]
            dt = float(motion["dt"])

            root_vel_w = (root_pos_next - root_pos) / dt
            root_vel_b = quat_apply_inverse(root_quat, root_vel_w)
            root_ang_vel_w = angular_velocity_world(root_quat, root_quat_next, dt)
            root_ang_vel_b = quat_apply_inverse(root_quat, root_ang_vel_w)

            gravity = torch.tensor([0.0, 0.0, -1.0], dtype=torch.float32, device=self.device).expand(count, -1)
            projected_gravity = quat_apply_inverse(root_quat, gravity)
            velocity_commands = torch.stack([root_vel_b[:, 0], root_vel_b[:, 1], root_ang_vel_b[:, 2]], dim=-1)

            cur_dof_pos = dof_pos[frame_idx]
            next_dof_pos = dof_pos[next_idx]
            prev_dof_pos = dof_pos[prev_idx]
            dof_vel = (next_dof_pos - cur_dof_pos) / dt
            joint_pos_rel = cur_dof_pos - self.default_dof_pos
            prev_action = (prev_dof_pos - self.default_dof_pos) / ACTION_SCALE
            target_action = (next_dof_pos - self.default_dof_pos) / ACTION_SCALE

            key_body_pos_w = motion["key_body_pos"][frame_idx]
            key_body_pos_b = quat_apply_inverse(
                root_quat[:, None, :].expand(-1, key_body_pos_w.shape[1], -1),
                key_body_pos_w - root_pos[:, None, :],
            ).reshape(count, -1)

            obs = torch.cat(
                [
                    root_ang_vel_b,
                    projected_gravity,
                    velocity_commands,
                    joint_pos_rel,
                    dof_vel,
                    prev_action,
                    key_body_pos_b,
                ],
                dim=-1,
            )
            if obs.shape[-1] != NUM_POLICY_OBS:
                raise RuntimeError(f"Expected {NUM_POLICY_OBS} policy obs dims, got {obs.shape[-1]} from {path}.")

            obs_chunks.append(obs)
            action_chunks.append(target_action)
            remaining -= count

        return torch.cat(obs_chunks, dim=0)[:batch_size], torch.cat(action_chunks, dim=0)[:batch_size]


def make_actor_critic(device: torch.device) -> ActorCritic:
    obs = TensorDict(
        {
            "policy": torch.zeros(1, NUM_POLICY_OBS, device=device),
            "critic": torch.zeros(1, NUM_CRITIC_OBS, device=device),
        },
        batch_size=[1],
        device=device,
    )
    return ActorCritic(
        obs=obs,
        obs_groups={"policy": ["policy"], "critic": ["critic"]},
        num_actions=NUM_ACTIONS,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
        init_noise_std=1.0,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
    ).to(device)


def resolve_checkpoint(path: str | None) -> Path | None:
    if path is None:
        return None
    checkpoint = Path(path).expanduser()
    if not checkpoint.is_absolute():
        checkpoint = REPO_ROOT / checkpoint
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint}")
    return checkpoint


def train(args: argparse.Namespace) -> Path:
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    device = torch.device(args.device)

    sampler = MixedMotionSampler(
        default_dir=Path(args.default_dir).expanduser(),
        motionbricks_dir=Path(args.motionbricks_dir).expanduser(),
        default_weight=args.default_weight,
        motionbricks_weight=args.motionbricks_weight,
        cache_size=args.cache_size,
        device=device,
    )
    model = make_actor_critic(device)

    init_checkpoint = resolve_checkpoint(args.init_checkpoint)
    if init_checkpoint is not None:
        checkpoint = torch.load(init_checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        print(f"[BC] Initialized actor-critic from {init_checkpoint}")

    if args.freeze_critic:
        for parameter in model.critic.parameters():
            parameter.requires_grad_(False)

    optimizer = torch.optim.Adam(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    run_dir = Path(args.log_dir).expanduser() / args.experiment_name / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "params").mkdir()
    config = vars(args).copy()
    config["init_checkpoint"] = str(init_checkpoint) if init_checkpoint else None
    (run_dir / "params" / "bc_args.json").write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")

    wandb_run = None
    if args.logger == "wandb":
        try:
            import wandb
        except ImportError as exc:
            raise ImportError("WandB logging requested but wandb is not installed. Run `uv sync`.") from exc
        wandb_run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.run_name or run_dir.name,
            id=args.wandb_run_id,
            resume="allow" if args.wandb_run_id else None,
            dir=str(run_dir),
            config=config,
        )

    for iteration in range(1, args.iterations + 1):
        obs, target_action = sampler.sample(args.batch_size)
        td = TensorDict(
            {
                "policy": obs,
                "critic": torch.zeros(obs.shape[0], NUM_CRITIC_OBS, device=device),
            },
            batch_size=[obs.shape[0]],
            device=device,
        )
        pred_action = model.act_inference(td)
        loss = F.mse_loss(pred_action, target_action)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
        optimizer.step()

        if iteration == 1 or iteration % args.log_interval == 0:
            mae = (pred_action.detach() - target_action).abs().mean().item()
            target_abs = target_action.abs().mean().item()
            pred_abs = pred_action.detach().abs().mean().item()
            if wandb_run is not None:
                wandb_run.log(
                    {
                        "BC/loss": loss.item(),
                        "BC/action_mae": mae,
                        "BC/target_action_abs_mean": target_abs,
                        "BC/pred_action_abs_mean": pred_abs,
                        "BC/grad_norm": float(grad_norm),
                    },
                    step=iteration,
                )
            print(
                f"[BC] iter={iteration:06d} loss={loss.item():.6f} mae={mae:.4f} "
                f"|target|={target_abs:.3f} |pred|={pred_abs:.3f} grad={float(grad_norm):.3f}"
            )

        if iteration % args.save_interval == 0 or iteration == args.iterations:
            checkpoint_path = run_dir / f"model_{iteration}.pt"
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "iter": iteration,
                    "infos": {"bc_loss": loss.item()},
                },
                checkpoint_path,
            )

    final_path = run_dir / "model_bc.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "iter": args.iterations,
            "infos": {"bc_loss": loss.item()},
        },
        final_path,
    )
    print(f"[BC] Saved final checkpoint to {final_path}")
    if wandb_run is not None:
        wandb_run.summary["final_checkpoint"] = str(final_path)
        wandb_run.finish()
    return final_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a G1 AMP actor with offline behavior cloning.")
    parser.add_argument("--default_dir", type=str, default=str(DEFAULT_AMP_DIR))
    parser.add_argument("--motionbricks_dir", type=str, default=str(DEFAULT_MOTIONBRICKS_DIR))
    parser.add_argument("--default_weight", type=float, default=0.2)
    parser.add_argument("--motionbricks_weight", type=float, default=0.8)
    parser.add_argument("--iterations", type=int, default=20000)
    parser.add_argument("--batch_size", type=int, default=8192)
    parser.add_argument("--learning_rate", type=float, default=1.0e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--cache_size", type=int, default=512)
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--log_dir", type=str, default=str(REPO_ROOT / "logs/rsl_rl"))
    parser.add_argument("--experiment_name", type=str, default="g1_bc_mixed_motion")
    parser.add_argument("--init_checkpoint", type=str, default=None)
    parser.add_argument("--freeze_critic", action="store_true", help="Keep critic weights fixed if an init checkpoint is used.")
    parser.add_argument("--logger", choices=["wandb", "none"], default="wandb")
    parser.add_argument("--wandb_project", type=str, default="g1_bc_mixed_motion")
    parser.add_argument("--wandb_entity", type=str, default=None)
    parser.add_argument("--wandb_run_id", type=str, default=None)
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--log_interval", type=int, default=100)
    parser.add_argument("--save_interval", type=int, default=1000)
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
