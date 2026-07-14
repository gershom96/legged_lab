from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from .constants import LOWER_ACTION_INDICES, UPPER_ACTION_INDICES


CHECKPOINT_FORMAT = "legged_lab_mjx_native_jax_v1"
POLICY_OBS_DIM = 757
CRITIC_OBS_DIM = 1052
DISC_HISTORY_LENGTH = 4
DISC_FRAME_DIM = 49


def elu(x: Any) -> Any:
    import jax

    return jax.nn.elu(x)


def linear(x: Any, weight: Any, bias: Any) -> Any:
    import jax.numpy as jnp

    return jnp.matmul(x, jnp.swapaxes(weight, -1, -2)) + bias


def _conv2d(x: Any, weight: Any, bias: Any, stride: int, padding: tuple[tuple[int, int], tuple[int, int]]) -> Any:
    from jax import lax
    import jax.numpy as jnp

    # Checkpoints retain PyTorch OIHW layout. JAX consumes HWIO here.
    kernel = jnp.transpose(weight, (2, 3, 1, 0))
    result = lax.conv_general_dilated(
        x,
        kernel,
        window_strides=(stride, stride),
        padding=padding,
        dimension_numbers=("NHWC", "HWIO", "NHWC"),
    )
    return result + bias.reshape(1, 1, 1, -1)


def _cnn_apply(params: dict[str, Any], prefix: str, scan: Any) -> Any:
    import jax.numpy as jnp

    x = elu(
        _conv2d(
            scan,
            params[f"{prefix}.0.weight"],
            params[f"{prefix}.0.bias"],
            stride=1,
            padding=((1, 1), (1, 1)),
        )
    )
    x = elu(
        _conv2d(
            x,
            params[f"{prefix}.2.weight"],
            params[f"{prefix}.2.bias"],
            stride=2,
            padding=((0, 0), (0, 0)),
        )
    )
    x = elu(
        _conv2d(
            x,
            params[f"{prefix}.4.weight"],
            params[f"{prefix}.4.bias"],
            stride=2,
            padding=((1, 1), (0, 0)),
        )
    )
    return jnp.mean(x, axis=(1, 2))


def _mlp_apply(params: dict[str, Any], prefix: str, x: Any) -> Any:
    for layer in (0, 2, 4):
        x = elu(linear(x, params[f"{prefix}.{layer}.weight"], params[f"{prefix}.{layer}.bias"]))
    return linear(x, params[f"{prefix}.6.weight"], params[f"{prefix}.6.bias"])


def actor_apply(
    params: dict[str, Any], policy_obs: Any, observation_layout: str = "isaac_scan_first_v2"
) -> Any:
    import jax.numpy as jnp

    if policy_obs.shape[-1] != POLICY_OBS_DIM:
        raise ValueError(f"Expected policy observations of width {POLICY_OBS_DIM}, got {policy_obs.shape}")
    if observation_layout == "isaac_scan_first_v2":
        # OnPolicyRunner discovers height_scan in the Isaac observation manager
        # and injects actor_height_scan_slice=(0, 187).
        scan = policy_obs[:, :187].reshape(-1, 1, 11, 17)
        one_dimensional = policy_obs[:, 187:]
    elif observation_layout == "legacy_native_history_first_v1":
        scan = policy_obs[:, 570:].reshape(-1, 1, 11, 17)
        one_dimensional = policy_obs[:, :570]
    else:
        raise ValueError(f"Unsupported observation layout: {observation_layout}")
    # The scan is flattened [y, x], then transposed to CNN [x, y].
    scan = jnp.transpose(scan, (0, 3, 2, 1))
    features = jnp.concatenate((one_dimensional, _cnn_apply(params, "actor_cnn", scan)), axis=-1)
    lower = _mlp_apply(params, "actor_lower", features)
    upper = _mlp_apply(params, "actor_upper", features)
    output = jnp.zeros((policy_obs.shape[0], 29), dtype=features.dtype)
    output = output.at[:, jnp.asarray(LOWER_ACTION_INDICES)].set(lower)
    return output.at[:, jnp.asarray(UPPER_ACTION_INDICES)].set(upper)


def critic_apply(
    params: dict[str, Any], critic_obs: Any, observation_layout: str = "isaac_scan_first_v2"
) -> Any:
    import jax.numpy as jnp

    if critic_obs.shape[-1] != CRITIC_OBS_DIM:
        raise ValueError(f"Expected critic observations of width {CRITIC_OBS_DIM}, got {critic_obs.shape}")
    if observation_layout == "isaac_scan_first_v2":
        # As above, the source runner injects critic_height_scan_slice=(0, 935).
        scan = critic_obs[:, :935].reshape(-1, 5, 11, 17)
        one_dimensional = critic_obs[:, 935:]
    elif observation_layout == "legacy_native_history_first_v1":
        scan = critic_obs[:, 117:].reshape(-1, 5, 11, 17)
        one_dimensional = critic_obs[:, :117]
    else:
        raise ValueError(f"Unsupported observation layout: {observation_layout}")
    scan = jnp.transpose(scan, (0, 3, 2, 1))
    features = jnp.concatenate((one_dimensional, _cnn_apply(params, "critic_cnn", scan)), axis=-1)
    return _mlp_apply(params, "critic", features)


def gaussian_log_prob_parts(mean: Any, std: Any, actions: Any) -> tuple[Any, Any]:
    import jax.numpy as jnp

    log_prob = -0.5 * (((actions - mean) / std) ** 2 + 2.0 * jnp.log(std) + np.log(2.0 * np.pi))
    lower = jnp.sum(log_prob[:, jnp.asarray(LOWER_ACTION_INDICES)], axis=-1)
    upper = jnp.sum(log_prob[:, jnp.asarray(UPPER_ACTION_INDICES)], axis=-1)
    return lower, upper


def gaussian_entropy(std: Any) -> Any:
    import jax.numpy as jnp

    return jnp.sum(jnp.log(std) + 0.5 * np.log(2.0 * np.pi * np.e), axis=-1)


def discriminator_apply(params: dict[str, Any], normalized_flat_obs: Any) -> Any:
    x = elu(linear(normalized_flat_obs, params["disc_trunk.0.weight"], params["disc_trunk.0.bias"]))
    x = elu(linear(x, params["disc_trunk.2.weight"], params["disc_trunk.2.bias"]))
    return linear(x, params["disc_linear.weight"], params["disc_linear.bias"]).squeeze(-1)


def normalize_discriminator_obs(obs: Any, normalizer: dict[str, Any]) -> Any:
    return (obs - normalizer["mean"]) / (normalizer["std"] + 1.0e-2)


def discriminator_score(
    params: dict[str, Any],
    normalizer: dict[str, Any],
    observations: Any,
) -> Any:
    normalized = normalize_discriminator_obs(observations, normalizer)
    return discriminator_apply(params, normalized.reshape(normalized.shape[0], -1))


def style_reward(score: Any, step_dt: float, scale: float = 5.0) -> Any:
    import jax.numpy as jnp

    return float(step_dt) * float(scale) * jnp.maximum(1.0 - 0.25 * (score - 1.0) ** 2, 0.0)


def update_normalizer(normalizer: dict[str, Any], observations: Any) -> dict[str, Any]:
    import jax.numpy as jnp

    values = observations.reshape(-1, observations.shape[-1])
    count_x = jnp.asarray(values.shape[0], dtype=jnp.float32)
    old_count = normalizer["count"].astype(jnp.float32)
    new_count = old_count + count_x
    rate = count_x / jnp.maximum(new_count, 1.0)
    batch_var = jnp.var(values, axis=0, keepdims=True)
    batch_mean = jnp.mean(values, axis=0, keepdims=True)
    delta = batch_mean - normalizer["mean"]
    mean = normalizer["mean"] + rate * delta
    var = normalizer["var"] + rate * (batch_var - normalizer["var"] + delta * (batch_mean - mean))
    should_update = old_count < 1.0e8
    return {
        "mean": jnp.where(should_update, mean, normalizer["mean"]),
        "var": jnp.where(should_update, var, normalizer["var"]),
        "std": jnp.where(should_update, jnp.sqrt(var), normalizer["std"]),
        "count": jnp.where(should_update, new_count, old_count),
    }


def load_checkpoint(path: str | Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    import jax.numpy as jnp

    checkpoint = Path(path).expanduser().resolve()
    with np.load(checkpoint, allow_pickle=False) as payload:
        metadata = json.loads(str(payload["__metadata_json__"].item()))
        if metadata.get("format") != CHECKPOINT_FORMAT:
            raise ValueError(f"Unsupported native JAX checkpoint format: {metadata.get('format')!r}")
        policy = {
            key.removeprefix("policy::"): jnp.asarray(np.asarray(payload[key], dtype=np.float32))
            for key in payload.files
            if key.startswith("policy::")
        }
        discriminator = {
            key.removeprefix("discriminator::"): jnp.asarray(np.asarray(payload[key], dtype=np.float32))
            for key in payload.files
            if key.startswith("discriminator::")
        }
        normalizer = {
            "mean": jnp.asarray(np.asarray(payload["normalizer::mean"], dtype=np.float32)),
            "var": jnp.asarray(np.asarray(payload["normalizer::var"], dtype=np.float32)),
            "std": jnp.asarray(np.asarray(payload["normalizer::std"], dtype=np.float32)),
            # JAX runs with x64 disabled. Float32 avoids int32 overflow while
            # preserving the running-moment update, which only uses this as a ratio.
            "count": jnp.asarray(np.asarray(payload["normalizer::count"], dtype=np.float32)),
        }
    _validate_params(policy, discriminator, normalizer)
    return policy, discriminator, normalizer, metadata


def save_converted_checkpoint(
    path: str | Path,
    policy: dict[str, np.ndarray],
    discriminator: dict[str, np.ndarray],
    normalizer: dict[str, np.ndarray],
    metadata: dict[str, Any],
) -> Path:
    output = Path(path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    flat: dict[str, np.ndarray] = {
        "__metadata_json__": np.asarray(json.dumps({"format": CHECKPOINT_FORMAT, **metadata}), dtype=np.str_),
        **{f"policy::{key}": np.asarray(value) for key, value in policy.items()},
        **{f"discriminator::{key}": np.asarray(value) for key, value in discriminator.items()},
        **{f"normalizer::{key}": np.asarray(value) for key, value in normalizer.items()},
    }
    np.savez_compressed(output, **flat)
    return output


def _validate_params(
    policy: dict[str, Any], discriminator: dict[str, Any], normalizer: dict[str, Any]
) -> None:
    required_policy = {
        "std",
        *(f"actor_cnn.{layer}.{suffix}" for layer in (0, 2, 4) for suffix in ("weight", "bias")),
        *(f"critic_cnn.{layer}.{suffix}" for layer in (0, 2, 4) for suffix in ("weight", "bias")),
        *(f"actor_lower.{layer}.{suffix}" for layer in (0, 2, 4, 6) for suffix in ("weight", "bias")),
        *(f"actor_upper.{layer}.{suffix}" for layer in (0, 2, 4, 6) for suffix in ("weight", "bias")),
        *(f"critic.{layer}.{suffix}" for layer in (0, 2, 4, 6) for suffix in ("weight", "bias")),
    }
    required_discriminator = {
        "disc_trunk.0.weight",
        "disc_trunk.0.bias",
        "disc_trunk.2.weight",
        "disc_trunk.2.bias",
        "disc_linear.weight",
        "disc_linear.bias",
    }
    missing_policy = sorted(required_policy - set(policy))
    missing_discriminator = sorted(required_discriminator - set(discriminator))
    if missing_policy or missing_discriminator:
        raise ValueError(
            f"Incomplete native checkpoint: missing policy={missing_policy}, discriminator={missing_discriminator}"
        )
    if tuple(policy["std"].shape) != (29,):
        raise ValueError(f"Expected 29 action standard deviations, got {policy['std'].shape}")
    if tuple(normalizer["mean"].shape) != (1, DISC_FRAME_DIM):
        raise ValueError(f"Expected discriminator normalizer shape (1, {DISC_FRAME_DIM}), got {normalizer['mean'].shape}")
