from __future__ import annotations

from typing import Any

import numpy as np


_LEFT = np.asarray((0, 3, 6, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27), dtype=np.int32)
_RIGHT = np.asarray((1, 4, 7, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28), dtype=np.int32)
_JOINT_PERMUTATION = np.arange(29, dtype=np.int32)
_JOINT_PERMUTATION[_LEFT] = _RIGHT
_JOINT_PERMUTATION[_RIGHT] = _LEFT
_JOINT_SIGNS = np.ones(29, dtype=np.float32)
_JOINT_SIGNS[[3, 4, 15, 16, 17, 18, 23, 24, 6, 7, 19, 20, 27, 28, 2, 5]] = -1.0


def switch_joints(values: Any) -> Any:
    import jax.numpy as jnp

    return values[..., jnp.asarray(_JOINT_PERMUTATION)] * jnp.asarray(_JOINT_SIGNS)


def switch_key_bodies(values: Any) -> Any:
    import jax.numpy as jnp

    shape = values.shape
    bodies = values.reshape(*shape[:-1], 6, 3)
    bodies = bodies[..., jnp.asarray((1, 0, 3, 2, 5, 4)), :]
    signs = jnp.asarray((1.0, -1.0, 1.0))
    return (bodies * signs).reshape(shape)


def mirror_policy_observation(
    observation: Any, observation_layout: str = "isaac_scan_first_v2"
) -> Any:
    import jax.numpy as jnp

    if observation_layout == "isaac_scan_first_v2":
        raw_scan = observation[:, :187].reshape(-1, 1, 11, 17)
        packed = observation[:, 187:].reshape(-1, 5, 114)
    elif observation_layout == "legacy_native_history_first_v1":
        packed = observation[:, :570].reshape(-1, 5, 114)
        raw_scan = observation[:, 570:].reshape(-1, 1, 11, 17)
    else:
        raise ValueError(f"Unsupported observation layout: {observation_layout}")
    pieces = (
        packed[..., 0:3] * jnp.asarray((-1.0, 1.0, -1.0)),
        packed[..., 3:6] * jnp.asarray((1.0, -1.0, 1.0)),
        packed[..., 6:9] * jnp.asarray((1.0, -1.0, -1.0)),
        switch_joints(packed[..., 9:38]),
        switch_joints(packed[..., 38:67]),
        switch_joints(packed[..., 67:96]),
        switch_key_bodies(packed[..., 96:114]),
    )
    mirrored_packed = jnp.concatenate(pieces, axis=-1).reshape(-1, 570)
    mirrored_scan = jnp.flip(raw_scan, axis=2).reshape(-1, 187)
    if observation_layout == "legacy_native_history_first_v1":
        return jnp.concatenate((mirrored_packed, mirrored_scan), axis=-1)
    return jnp.concatenate((mirrored_scan, mirrored_packed), axis=-1)


def augment_policy_batch(
    policy_obs: Any, actions: Any, observation_layout: str = "isaac_scan_first_v2"
) -> tuple[Any, Any]:
    import jax.numpy as jnp

    return (
        jnp.concatenate((policy_obs, mirror_policy_observation(policy_obs, observation_layout)), axis=0),
        jnp.concatenate((actions, switch_joints(actions)), axis=0),
    )
