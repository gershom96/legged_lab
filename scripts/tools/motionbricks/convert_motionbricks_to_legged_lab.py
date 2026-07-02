"""
Convert MotionBricks G1 .npz motions into the Legged Lab AMP .pkl format.

The MotionBricks files contain root pose and 29-DOF joint positions, but this
repo's AMP environment also needs Isaac-computed key body positions. This script
runs the motions through Isaac in batches and writes one .pkl per input .npz.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Convert MotionBricks G1 motions to Legged Lab AMP motion data.")
parser.add_argument(
    "--input_dir",
    type=str,
    default="~/Documents/shared_datasets/motionbricks/motionbricks_sonic_grid_walk_dense/motions",
    help="Directory containing MotionBricks .npz files.",
)
parser.add_argument(
    "--output_dir",
    type=str,
    default="~/Documents/shared_datasets/motionbricks/motionbricks_sonic_grid_walk_dense/legged_lab_g1",
    help="Directory to write converted Legged Lab .pkl files.",
)
parser.add_argument("--batch_size", type=int, default=64, help="Number of motions to process per Isaac scene.")
parser.add_argument("--limit", type=int, default=None, help="Optional max number of motions to convert.")
parser.add_argument("--overwrite", action="store_true", help="Rewrite existing output .pkl files.")
parser.add_argument("--loop", choices=["clamp", "wrap"], default="wrap", help="Loop mode stored in the output files.")

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import numpy as np
import joblib
import isaaclab.sim as sim_utils
from isaaclab.scene import InteractiveScene

from legged_lab.assets.unitree import UNITREE_G1_29DOF_CFG as ROBOT_CFG


RETARGET_DIR = Path(__file__).resolve().parents[1] / "retarget"
sys.path.insert(0, str(RETARGET_DIR))
from gmr_to_lab import LoopMode, ReplayMotionsSceneCfg, run_simulator  # noqa: E402


KEY_BODY_NAMES = [
    "left_ankle_roll_link",
    "right_ankle_roll_link",
    "left_wrist_yaw_link",
    "right_wrist_yaw_link",
    "left_shoulder_roll_link",
    "right_shoulder_roll_link",
]


def motionbricks_to_lab_dict(path: Path, loop_mode: LoopMode) -> dict:
    with np.load(path) as motion:
        fps = int(round(float(motion["frequency"])))
        root_pos = np.asarray(motion["root_pos"], dtype=np.float32)
        root_rot_wxyz = np.asarray(motion["root_rot"], dtype=np.float32)
        dof_pos = np.asarray(motion["dof_pos"], dtype=np.float32)

    if root_pos.ndim != 2 or root_pos.shape[1] != 3:
        raise ValueError(f"{path} root_pos must have shape (frames, 3), got {root_pos.shape}")
    if root_rot_wxyz.ndim != 2 or root_rot_wxyz.shape[1] != 4:
        raise ValueError(f"{path} root_rot must have shape (frames, 4), got {root_rot_wxyz.shape}")
    if dof_pos.ndim != 2 or dof_pos.shape[1] != 29:
        raise ValueError(f"{path} dof_pos must have shape (frames, 29), got {dof_pos.shape}")

    # gmr_to_lab.run_simulator expects xyzw input and stores wxyz output.
    root_rot_xyzw = root_rot_wxyz[:, [1, 2, 3, 0]]
    return {
        "fps": fps,
        "root_pos": root_pos,
        "root_rot": root_rot_xyzw,
        "dof_pos": dof_pos,
        "loop_mode": loop_mode.value,
    }


def iter_batches(items: list[Path], batch_size: int):
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def convert_batch(input_paths: list[Path], output_dir: Path, loop_mode: LoopMode):
    motions = [motionbricks_to_lab_dict(path, loop_mode) for path in input_paths]
    fps_values = {motion["fps"] for motion in motions}
    if len(fps_values) != 1:
        raise ValueError(f"All motions in a batch must have the same fps, got {sorted(fps_values)}")

    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=1.0 / motions[0]["fps"], device=args_cli.device))
    scene_cfg = ReplayMotionsSceneCfg(
        num_envs=len(motions),
        env_spacing=3.0,
        robot=ROBOT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot"),
    )
    scene = InteractiveScene(scene_cfg)
    sim.set_camera_view([2.0, 0.0, 2.5], [-0.5, 0.0, 0.5])
    sim.reset()

    converted = run_simulator(simulation_app, sim, scene, motions, KEY_BODY_NAMES)
    for input_path, motion in zip(input_paths, converted):
        output_path = output_dir / f"{input_path.stem}.pkl"
        joblib.dump(motion, output_path)
        print(f"Saved: {output_path}")

    sim.clear_all_callbacks()
    sim.clear_instance()


def main():
    input_dir = Path(args_cli.input_dir).expanduser()
    output_dir = Path(args_cli.output_dir).expanduser()
    if not input_dir.exists():
        raise FileNotFoundError(f"MotionBricks input directory does not exist: {input_dir}")

    input_paths = sorted(input_dir.glob("*.npz"))
    if args_cli.limit is not None:
        input_paths = input_paths[: args_cli.limit]
    if not args_cli.overwrite:
        input_paths = [path for path in input_paths if not (output_dir / f"{path.stem}.pkl").exists()]

    if not input_paths:
        print("No MotionBricks files to convert.")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    loop_mode = LoopMode.WRAP if args_cli.loop == "wrap" else LoopMode.CLAMP

    print(f"Converting {len(input_paths)} MotionBricks files to {output_dir}")
    for batch_idx, batch in enumerate(iter_batches(input_paths, args_cli.batch_size), start=1):
        print(f"Converting batch {batch_idx}: {len(batch)} motions")
        convert_batch(batch, output_dir, loop_mode)


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
