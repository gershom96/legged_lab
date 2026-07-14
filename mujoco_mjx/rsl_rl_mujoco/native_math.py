from __future__ import annotations

from typing import Any


def quat_normalize(quat: Any) -> Any:
    import jax.numpy as jnp

    return quat / jnp.maximum(jnp.linalg.norm(quat, axis=-1, keepdims=True), 1.0e-8)


def quat_conjugate(quat: Any) -> Any:
    import jax.numpy as jnp

    return jnp.concatenate((quat[..., :1], -quat[..., 1:]), axis=-1)


def quat_mul(lhs: Any, rhs: Any) -> Any:
    import jax.numpy as jnp

    lw, lx, ly, lz = [lhs[..., index] for index in range(4)]
    rw, rx, ry, rz = [rhs[..., index] for index in range(4)]
    return jnp.stack(
        (
            lw * rw - lx * rx - ly * ry - lz * rz,
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
        ),
        axis=-1,
    )


def quat_apply(quat: Any, vector: Any) -> Any:
    import jax.numpy as jnp

    quat = quat_normalize(quat)
    xyz = quat[..., 1:]
    uv = jnp.cross(xyz, vector, axis=-1)
    uuv = jnp.cross(xyz, uv, axis=-1)
    return vector + 2.0 * (quat[..., :1] * uv + uuv)


def quat_apply_inverse(quat: Any, vector: Any) -> Any:
    return quat_apply(quat_conjugate(quat), vector)


def yaw_from_quat(quat: Any) -> Any:
    import jax.numpy as jnp

    quat = quat_normalize(quat)
    w, x, y, z = [quat[..., index] for index in range(4)]
    return jnp.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def quat_from_yaw(yaw: Any) -> Any:
    import jax.numpy as jnp

    half = 0.5 * yaw
    zeros = jnp.zeros_like(half)
    return jnp.stack((jnp.cos(half), zeros, zeros, jnp.sin(half)), axis=-1)


def quat_to_matrix(quat: Any) -> Any:
    import jax.numpy as jnp

    quat = quat_normalize(quat)
    w, x, y, z = [quat[..., index] for index in range(4)]
    values = jnp.stack(
        (
            1 - 2 * (y * y + z * z),
            2 * (x * y - z * w),
            2 * (x * z + y * w),
            2 * (x * y + z * w),
            1 - 2 * (x * x + z * z),
            2 * (y * z - x * w),
            2 * (x * z - y * w),
            2 * (y * z + x * w),
            1 - 2 * (x * x + y * y),
        ),
        axis=-1,
    )
    return values.reshape(*quat.shape[:-1], 3, 3)


def root_local_rot_tan_norm(quat: Any) -> Any:
    import jax.numpy as jnp

    local = quat_mul(quat_conjugate(quat_from_yaw(yaw_from_quat(quat))), quat)
    matrix = quat_to_matrix(local)
    return jnp.concatenate((matrix[..., :, 0], matrix[..., :, 2]), axis=-1)


def projected_gravity(quat: Any) -> Any:
    import jax.numpy as jnp

    gravity = jnp.zeros((*quat.shape[:-1], 3), dtype=quat.dtype).at[..., 2].set(-1.0)
    return quat_apply_inverse(quat, gravity)


def wrap_to_pi(angle: Any) -> Any:
    import jax.numpy as jnp

    return jnp.remainder(angle + jnp.pi, 2.0 * jnp.pi) - jnp.pi


def quat_slerp(q0: Any, q1: Any, blend: Any) -> Any:
    import jax.numpy as jnp

    q0 = quat_normalize(q0)
    q1 = quat_normalize(q1)
    dot = jnp.sum(q0 * q1, axis=-1, keepdims=True)
    q1 = jnp.where(dot < 0.0, -q1, q1)
    dot = jnp.minimum(jnp.abs(dot), 1.0)
    theta = jnp.arccos(dot)
    sin_theta = jnp.sin(theta)
    if blend.ndim == q0.ndim - 1:
        blend = blend[..., None]
    lerped = quat_normalize((1.0 - blend) * q0 + blend * q1)
    spherical = (
        jnp.sin((1.0 - blend) * theta) * q0 + jnp.sin(blend * theta) * q1
    ) / jnp.maximum(sin_theta, 1.0e-8)
    return jnp.where(jnp.abs(sin_theta) < 1.0e-5, lerped, spherical)
