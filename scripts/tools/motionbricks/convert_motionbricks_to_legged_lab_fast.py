"""
Fast CPU-parallel MotionBricks -> Legged Lab AMP converter.

MotionBricks .npz files already contain root pose, compact MuJoCo qpos, and G1
29-DOF joint positions. The Legged Lab AMP loader additionally wants
key_body_pos, so this script uses MuJoCo forward kinematics to compute those
positions and writes the .pkl files directly.
"""

from __future__ import annotations

import argparse
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import joblib
import numpy as np


G1_BODY_JOINT_NAMES = [
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

LOOP_MODES = {"clamp": 0, "wrap": 1}

_MODEL = None
_DATA = None
_BODY_QPOS_INDICES = None
_KEY_BODY_IDS = None


def _init_worker(model_path: str):
    global _MODEL, _DATA, _BODY_QPOS_INDICES, _KEY_BODY_IDS

    import mujoco

    _MODEL = mujoco.MjModel.from_xml_path(model_path)
    _DATA = mujoco.MjData(_MODEL)

    body_qpos_indices = []
    for joint_name in G1_BODY_JOINT_NAMES:
        joint_id = mujoco.mj_name2id(_MODEL, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        if joint_id < 0:
            raise KeyError(f"Joint not found in G1 MuJoCo model: {joint_name}")
        body_qpos_indices.append(int(_MODEL.jnt_qposadr[joint_id]))
    _BODY_QPOS_INDICES = np.asarray(body_qpos_indices, dtype=np.int64)

    key_body_ids = []
    for body_name in KEY_BODY_NAMES:
        body_id = mujoco.mj_name2id(_MODEL, mujoco.mjtObj.mjOBJ_BODY, body_name)
        if body_id < 0:
            raise KeyError(f"Body not found in G1 MuJoCo model: {body_name}")
        key_body_ids.append(int(body_id))
    _KEY_BODY_IDS = np.asarray(key_body_ids, dtype=np.int64)


def _compute_key_body_pos(qpos: np.ndarray) -> np.ndarray:
    import mujoco

    assert _MODEL is not None
    assert _DATA is not None
    assert _BODY_QPOS_INDICES is not None
    assert _KEY_BODY_IDS is not None

    qpos = np.asarray(qpos, dtype=np.float64)
    key_body_pos = np.zeros((qpos.shape[0], len(_KEY_BODY_IDS), 3), dtype=np.float32)

    for frame_idx, compact_qpos in enumerate(qpos):
        full_qpos = np.asarray(_DATA.qpos, dtype=np.float64).copy()
        full_qpos[0:7] = compact_qpos[0:7]
        full_qpos[_BODY_QPOS_INDICES] = compact_qpos[7:36]
        _DATA.qpos[:] = full_qpos
        _DATA.qvel[:] = 0.0
        mujoco.mj_forward(_MODEL, _DATA)
        key_body_pos[frame_idx] = np.asarray(_DATA.xpos[_KEY_BODY_IDS], dtype=np.float32)

    return key_body_pos


def _convert_one(args: tuple[str, str, int, bool]) -> tuple[str, bool]:
    input_path_str, output_dir_str, loop_mode, overwrite = args
    input_path = Path(input_path_str)
    output_path = Path(output_dir_str) / f"{input_path.stem}.pkl"

    if output_path.exists() and not overwrite:
        return str(output_path), False

    with np.load(input_path) as motion:
        qpos = np.asarray(motion["qpos"], dtype=np.float32)
        root_pos = np.asarray(motion["root_pos"], dtype=np.float32)
        root_rot = np.asarray(motion["root_rot"], dtype=np.float32)
        dof_pos = np.asarray(motion["dof_pos"], dtype=np.float32)
        fps = int(round(float(motion["frequency"])))

    if qpos.ndim != 2 or qpos.shape[1] != 36:
        raise ValueError(f"{input_path} qpos must have shape (frames, 36), got {qpos.shape}")
    if dof_pos.ndim != 2 or dof_pos.shape[1] != 29:
        raise ValueError(f"{input_path} dof_pos must have shape (frames, 29), got {dof_pos.shape}")

    output = {
        "fps": fps,
        "root_pos": root_pos,
        "root_rot": root_rot,
        "dof_pos": dof_pos,
        "loop_mode": loop_mode,
        "key_body_pos": _compute_key_body_pos(qpos),
    }
    joblib.dump(output, output_path)
    return str(output_path), True


def parse_args():
    parser = argparse.ArgumentParser(description="Fast CPU-parallel MotionBricks G1 converter.")
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
    parser.add_argument(
        "--model",
        type=str,
        default="source/legged_lab/legged_lab/data/Robots/Unitree/g1_29dof/g1_29dof.xml",
        help="G1 MuJoCo XML model path.",
    )
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    parser.add_argument("--limit", type=int, default=None, help="Optional max number of motions to convert.")
    parser.add_argument("--overwrite", action="store_true", help="Rewrite existing output .pkl files.")
    parser.add_argument("--loop", choices=sorted(LOOP_MODES), default="wrap")
    return parser.parse_args()


def main():
    args = parse_args()
    input_dir = Path(args.input_dir).expanduser()
    output_dir = Path(args.output_dir).expanduser()
    model_path = Path(args.model).expanduser()
    if not model_path.is_absolute():
        model_path = Path.cwd() / model_path

    if not input_dir.exists():
        raise FileNotFoundError(f"MotionBricks input directory does not exist: {input_dir}")
    if not model_path.exists():
        raise FileNotFoundError(f"G1 MuJoCo model does not exist: {model_path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    input_paths = sorted(input_dir.glob("*.npz"))
    if not args.overwrite:
        input_paths = [path for path in input_paths if not (output_dir / f"{path.stem}.pkl").exists()]
    if args.limit is not None:
        input_paths = input_paths[: args.limit]

    if not input_paths:
        print("No MotionBricks files to convert.")
        return

    workers = max(1, int(args.workers))
    loop_mode = LOOP_MODES[args.loop]
    print(f"Converting {len(input_paths)} files with {workers} CPU workers")
    print(f"Input:  {input_dir}")
    print(f"Output: {output_dir}")
    print(f"Model:  {model_path}")

    tasks = [(str(path), str(output_dir), loop_mode, args.overwrite) for path in input_paths]
    converted = 0
    with ProcessPoolExecutor(max_workers=workers, initializer=_init_worker, initargs=(str(model_path),)) as executor:
        futures = [executor.submit(_convert_one, task) for task in tasks]
        for index, future in enumerate(as_completed(futures), start=1):
            output_path, wrote_file = future.result()
            converted += int(wrote_file)
            if index == 1 or index % 100 == 0 or index == len(futures):
                print(f"[{index}/{len(futures)}] converted={converted} latest={output_path}")

    print(f"Done. Converted {converted} files.")


if __name__ == "__main__":
    main()
