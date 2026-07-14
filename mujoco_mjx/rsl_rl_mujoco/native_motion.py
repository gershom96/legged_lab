from __future__ import annotations

from pathlib import Path
from typing import Any, NamedTuple

import joblib
import numpy as np

from .constants import UPPER_ACTION_INDICES, UPPER_KEY_BODY_INDICES
from .native_math import quat_apply_inverse, quat_slerp, root_local_rot_tan_norm
from .spec import MotionSpec


class NativeMotionData(NamedTuple):
    root_pos: Any
    root_quat: Any
    root_vel: Any
    root_ang_vel: Any
    dof_pos: Any
    dof_vel: Any
    key_body_pos: Any
    starts: Any
    frames: Any
    durations: Any
    weights: Any


def load_motion_data(spec: MotionSpec) -> tuple[NativeMotionData, int]:
    paths = _select_paths(spec)
    if not paths:
        raise FileNotFoundError(f"No AMP pickle files found in {spec.directory}")
    arrays: dict[str, list[np.ndarray]] = {
        "root_pos": [],
        "root_quat": [],
        "root_vel": [],
        "root_ang_vel": [],
        "dof_pos": [],
        "dof_vel": [],
        "key_body_pos": [],
    }
    starts: list[int] = []
    frames: list[int] = []
    durations: list[float] = []
    weights: list[float] = []
    source_counts = _source_counts(paths)
    cursor = 0
    for path in paths:
        raw = joblib.load(path)
        _validate_motion(path, raw)
        dt = 1.0 / float(raw["fps"])
        root_pos = np.asarray(raw["root_pos"], dtype=np.float32)
        root_quat = np.asarray(raw["root_rot"], dtype=np.float32)
        dof_pos = np.asarray(raw["dof_pos"], dtype=np.float32)
        key_body_pos = np.asarray(raw["key_body_pos"], dtype=np.float32)
        count = root_pos.shape[0]
        arrays["root_pos"].append(root_pos)
        arrays["root_quat"].append(root_quat)
        arrays["root_vel"].append(_forward_difference(root_pos, dt))
        arrays["root_ang_vel"].append(_angular_velocity(root_quat, dt))
        arrays["dof_pos"].append(dof_pos)
        arrays["dof_vel"].append(_forward_difference(dof_pos, dt))
        arrays["key_body_pos"].append(key_body_pos)
        starts.append(cursor)
        frames.append(count)
        durations.append(dt * (count - 1))
        source = _source(path)
        source_weight = spec.default_weight if source == "default" else spec.motionbricks_weight
        if source not in {"default", "motionbricks"}:
            source_weight = 1.0
        weights.append(source_weight / source_counts[source])
        cursor += count
    probabilities = np.asarray(weights, dtype=np.float32)
    probabilities /= probabilities.sum()
    return (
        NativeMotionData(
            root_pos=np.concatenate(arrays["root_pos"]),
            root_quat=np.concatenate(arrays["root_quat"]),
            root_vel=np.concatenate(arrays["root_vel"]),
            root_ang_vel=np.concatenate(arrays["root_ang_vel"]),
            dof_pos=np.concatenate(arrays["dof_pos"]),
            dof_vel=np.concatenate(arrays["dof_vel"]),
            key_body_pos=np.concatenate(arrays["key_body_pos"]),
            starts=np.asarray(starts, dtype=np.int32),
            frames=np.asarray(frames, dtype=np.int32),
            durations=np.asarray(durations, dtype=np.float32),
            weights=probabilities,
        ),
        len(paths),
    )


def sample_discriminator_observations_numpy(
    data: NativeMotionData,
    rng: np.random.Generator,
    batch_size: int,
    history_length: int,
    step_dt: float,
) -> np.ndarray:
    """Sample exact AMP histories on the host to keep the corpus out of GPU physics memory."""
    motion_ids = rng.choice(data.weights.shape[0], size=batch_size, replace=True, p=data.weights)
    usable = np.maximum(data.durations[motion_ids] - history_length * float(step_dt), 1.0e-6)
    times = rng.random(batch_size, dtype=np.float32) * usable
    offsets = np.arange(history_length, dtype=np.float32) * float(step_dt)
    sample_times = times[:, None] + offsets[None, :]
    ids = np.broadcast_to(motion_ids[:, None], sample_times.shape)
    state = _sample_state_numpy(data, ids, sample_times)
    joint_ids = np.asarray(UPPER_ACTION_INDICES)
    key_ids = np.asarray(UPPER_KEY_BODY_INDICES)
    return np.concatenate(
        (
            _root_local_rot_tan_norm_numpy(state["root_quat"]),
            state["root_ang_vel_b"],
            state["dof_pos"][..., joint_ids],
            state["dof_vel"][..., joint_ids],
            state["key_body_pos_b"][..., key_ids, :].reshape(batch_size, history_length, -1),
        ),
        axis=-1,
    ).astype(np.float32, copy=False)


def sample_discriminator_observations(
    data: NativeMotionData,
    key: Any,
    batch_size: int,
    history_length: int,
    step_dt: float,
) -> Any:
    import jax
    import jax.numpy as jnp

    motion_key, time_key = jax.random.split(key)
    motion_ids = jax.random.choice(
        motion_key,
        data.weights.shape[0],
        shape=(batch_size,),
        replace=True,
        p=data.weights,
    )
    usable = jnp.maximum(data.durations[motion_ids] - history_length * float(step_dt), 1.0e-6)
    times = jax.random.uniform(time_key, (batch_size,), dtype=jnp.float32) * usable
    offsets = jnp.arange(history_length, dtype=jnp.float32) * float(step_dt)
    sample_times = times[:, None] + offsets[None, :]
    state = _sample_state(data, jnp.broadcast_to(motion_ids[:, None], sample_times.shape), sample_times)
    joint_ids = jnp.asarray(UPPER_ACTION_INDICES)
    key_ids = jnp.asarray(UPPER_KEY_BODY_INDICES)
    return jnp.concatenate(
        (
            root_local_rot_tan_norm(state["root_quat"]),
            state["root_ang_vel_b"],
            state["dof_pos"][..., joint_ids],
            state["dof_vel"][..., joint_ids],
            state["key_body_pos_b"][..., key_ids, :].reshape(batch_size, history_length, -1),
        ),
        axis=-1,
    )


def _sample_state(data: NativeMotionData, motion_ids: Any, times: Any) -> dict[str, Any]:
    import jax.numpy as jnp

    frames = data.frames[motion_ids]
    phase = jnp.clip(times / jnp.maximum(data.durations[motion_ids], 1.0e-6), 0.0, 1.0)
    frame_float = phase * (frames - 1).astype(jnp.float32)
    frame0 = jnp.floor(frame_float).astype(jnp.int32)
    frame1 = jnp.minimum(frame0 + 1, frames - 1)
    blend = frame_float - frame0.astype(jnp.float32)
    index0 = frame0 + data.starts[motion_ids]
    index1 = frame1 + data.starts[motion_ids]

    def lerp(values: Any, extra_dims: int = 1) -> Any:
        amount = blend.reshape(*blend.shape, *((1,) * extra_dims))
        return values[index0] + amount * (values[index1] - values[index0])

    root_pos = lerp(data.root_pos)
    root_quat = quat_slerp(data.root_quat[index0], data.root_quat[index1], blend)
    root_vel = lerp(data.root_vel)
    root_ang_vel = lerp(data.root_ang_vel)
    dof_pos = lerp(data.dof_pos)
    dof_vel = lerp(data.dof_vel)
    key_body_pos = lerp(data.key_body_pos, extra_dims=2)
    expanded_quat = jnp.broadcast_to(root_quat[..., None, :], (*root_quat.shape[:-1], key_body_pos.shape[-2], 4))
    return {
        "root_quat": root_quat,
        "root_ang_vel_b": quat_apply_inverse(root_quat, root_ang_vel),
        "dof_pos": dof_pos,
        "dof_vel": dof_vel,
        "key_body_pos_b": quat_apply_inverse(expanded_quat, key_body_pos - root_pos[..., None, :]),
        "root_vel_b": quat_apply_inverse(root_quat, root_vel),
    }


def _sample_state_numpy(data: NativeMotionData, motion_ids: np.ndarray, times: np.ndarray) -> dict[str, np.ndarray]:
    frames = data.frames[motion_ids]
    phase = np.clip(times / np.maximum(data.durations[motion_ids], 1.0e-6), 0.0, 1.0)
    frame_float = phase * (frames - 1).astype(np.float32)
    frame0 = np.floor(frame_float).astype(np.int32)
    frame1 = np.minimum(frame0 + 1, frames - 1)
    blend = frame_float - frame0.astype(np.float32)
    index0 = frame0 + data.starts[motion_ids]
    index1 = frame1 + data.starts[motion_ids]

    def lerp(values: np.ndarray, extra_dims: int = 1) -> np.ndarray:
        amount = blend.reshape(*blend.shape, *((1,) * extra_dims))
        return values[index0] + amount * (values[index1] - values[index0])

    root_pos = lerp(data.root_pos)
    root_quat = _quat_slerp_numpy(data.root_quat[index0], data.root_quat[index1], blend)
    root_ang_vel = lerp(data.root_ang_vel)
    dof_pos = lerp(data.dof_pos)
    dof_vel = lerp(data.dof_vel)
    key_body_pos = lerp(data.key_body_pos, extra_dims=2)
    expanded_quat = np.broadcast_to(root_quat[..., None, :], (*root_quat.shape[:-1], key_body_pos.shape[-2], 4))
    return {
        "root_quat": root_quat,
        "root_ang_vel_b": _quat_apply_inverse_numpy(root_quat, root_ang_vel),
        "dof_pos": dof_pos,
        "dof_vel": dof_vel,
        "key_body_pos_b": _quat_apply_inverse_numpy(expanded_quat, key_body_pos - root_pos[..., None, :]),
    }


def _quat_normalize_numpy(quat: np.ndarray) -> np.ndarray:
    return quat / np.maximum(np.linalg.norm(quat, axis=-1, keepdims=True), 1.0e-8)


def _quat_conjugate_numpy(quat: np.ndarray) -> np.ndarray:
    result = quat.copy()
    result[..., 1:] *= -1.0
    return result


def _quat_mul_numpy(lhs: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    lw, lx, ly, lz = np.moveaxis(lhs, -1, 0)
    rw, rx, ry, rz = np.moveaxis(rhs, -1, 0)
    return np.stack(
        (
            lw * rw - lx * rx - ly * ry - lz * rz,
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
        ),
        axis=-1,
    )


def _quat_apply_inverse_numpy(quat: np.ndarray, vector: np.ndarray) -> np.ndarray:
    quat = _quat_normalize_numpy(_quat_conjugate_numpy(quat))
    xyz = quat[..., 1:]
    uv = np.cross(xyz, vector, axis=-1)
    uuv = np.cross(xyz, uv, axis=-1)
    return vector + 2.0 * (quat[..., :1] * uv + uuv)


def _quat_slerp_numpy(q0: np.ndarray, q1: np.ndarray, blend: np.ndarray) -> np.ndarray:
    q0 = _quat_normalize_numpy(q0)
    q1 = _quat_normalize_numpy(q1)
    dot = np.sum(q0 * q1, axis=-1, keepdims=True)
    q1 = np.where(dot < 0.0, -q1, q1)
    dot = np.minimum(np.abs(dot), 1.0)
    theta = np.arccos(dot)
    sin_theta = np.sin(theta)
    amount = blend[..., None]
    lerped = _quat_normalize_numpy((1.0 - amount) * q0 + amount * q1)
    spherical = (
        np.sin((1.0 - amount) * theta) * q0 + np.sin(amount * theta) * q1
    ) / np.maximum(sin_theta, 1.0e-8)
    return np.where(np.abs(sin_theta) < 1.0e-5, lerped, spherical)


def _root_local_rot_tan_norm_numpy(quat: np.ndarray) -> np.ndarray:
    quat = _quat_normalize_numpy(quat)
    w, x, y, z = np.moveaxis(quat, -1, 0)
    yaw = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    yaw_quat = np.zeros_like(quat)
    yaw_quat[..., 0] = np.cos(0.5 * yaw)
    yaw_quat[..., 3] = np.sin(0.5 * yaw)
    local = _quat_normalize_numpy(_quat_mul_numpy(_quat_conjugate_numpy(yaw_quat), quat))
    w, x, y, z = np.moveaxis(local, -1, 0)
    first_column = np.stack(
        (1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y + z * w), 2.0 * (x * z - y * w)),
        axis=-1,
    )
    third_column = np.stack(
        (2.0 * (x * z + y * w), 2.0 * (y * z - x * w), 1.0 - 2.0 * (x * x + y * y)),
        axis=-1,
    )
    return np.concatenate((first_column, third_column), axis=-1)


def _source(path: Path) -> str:
    return path.stem.split("__", 1)[0] if "__" in path.stem else "other"


def _source_counts(paths: list[Path]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for path in paths:
        counts[_source(path)] = counts.get(_source(path), 0) + 1
    return counts


def _select_paths(spec: MotionSpec) -> list[Path]:
    paths = sorted(spec.directory.expanduser().glob("*.pkl"))
    if spec.max_files <= 0 or len(paths) <= spec.max_files:
        return paths
    rng = np.random.default_rng(spec.seed)
    by_source: dict[str, list[Path]] = {}
    for path in paths:
        by_source.setdefault(_source(path), []).append(path)
    selected: list[Path] = []
    remaining = spec.max_files
    for source, candidates in sorted(by_source.items()):
        take = max(1, round(spec.max_files * len(candidates) / len(paths)))
        take = min(take, len(candidates), remaining)
        selected.extend(Path(path) for path in rng.choice(np.asarray(candidates, dtype=object), size=take, replace=False))
        remaining -= take
    if remaining > 0:
        selected.extend(sorted(set(paths) - set(selected))[:remaining])
    return sorted(selected[: spec.max_files])


def _validate_motion(path: Path, raw: object) -> None:
    required = {"fps", "root_pos", "root_rot", "dof_pos", "key_body_pos"}
    if not isinstance(raw, dict) or not required.issubset(raw):
        raise ValueError(f"Invalid AMP motion file {path}; expected keys {sorted(required)}")
    is_motionbricks = path.stem.startswith("motionbricks__") or raw.get("source_joint_order") == "mujoco"
    if is_motionbricks and raw.get("joint_order") != "isaaclab":
        raise ValueError(f"MotionBricks file {path} is not converted to IsaacLab joint order")
    if np.asarray(raw["dof_pos"]).shape[1:] != (29,) or np.asarray(raw["key_body_pos"]).shape[1:] != (6, 3):
        raise ValueError(f"AMP motion {path} has an incompatible body layout")


def _forward_difference(values: np.ndarray, dt: float) -> np.ndarray:
    velocity = np.empty_like(values, dtype=np.float32)
    velocity[:-1] = (values[1:] - values[:-1]) / dt
    velocity[-1] = velocity[-2]
    return velocity


def _quat_mul_np(lhs: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    lw, lx, ly, lz = np.moveaxis(lhs, -1, 0)
    rw, rx, ry, rz = np.moveaxis(rhs, -1, 0)
    return np.stack(
        (
            lw * rw - lx * rx - ly * ry - lz * rz,
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
        ),
        axis=-1,
    )


def _angular_velocity(quat: np.ndarray, dt: float) -> np.ndarray:
    normalized = quat / np.maximum(np.linalg.norm(quat, axis=-1, keepdims=True), 1.0e-8)
    conjugate = normalized[:-1].copy()
    conjugate[:, 1:] *= -1.0
    delta = _quat_mul_np(conjugate, normalized[1:])
    delta /= np.maximum(np.linalg.norm(delta, axis=-1, keepdims=True), 1.0e-8)
    delta = np.where(delta[:, :1] < 0.0, -delta, delta)
    sin_half = np.linalg.norm(delta[:, 1:], axis=-1, keepdims=True)
    angle = 2.0 * np.arctan2(sin_half, np.maximum(delta[:, :1], 1.0e-8))
    scale = np.where(sin_half > 1.0e-7, angle / np.maximum(sin_half, 1.0e-8), 2.0)
    axis_angle = delta[:, 1:] * scale
    xyz = normalized[:-1, 1:]
    uv = np.cross(xyz, axis_angle)
    uuv = np.cross(xyz, uv)
    world = axis_angle + 2.0 * (normalized[:-1, :1] * uv + uuv)
    velocity = np.empty((quat.shape[0], 3), dtype=np.float32)
    velocity[:-1] = world / dt
    velocity[-1] = velocity[-2]
    return velocity
