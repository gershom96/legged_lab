#!/usr/bin/env python3
"""Convert RSL-RL height-scan checkpoints into the native JAX split-policy format."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _array(value: Any) -> np.ndarray:
    return np.asarray(value.detach().cpu(), dtype=np.float32)


def _arrays(values: dict[str, Any], keys: list[str]) -> dict[str, np.ndarray]:
    return {key: _array(values[key]) for key in keys if hasattr(values[key], "detach")}


def _load_amp_checkpoint(torch: Any, path: Path) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    required = (
        "model_state_dict",
        "amp_discriminator_state_dict",
        "amp_discriminator_normalizer_state_dict",
    )
    missing = [key for key in required if not isinstance(payload.get(key), dict)]
    if missing:
        raise ValueError(f"{path} is missing AMP checkpoint entries: {missing}")
    return payload


def _is_split_policy(model_state: dict[str, Any]) -> bool:
    return any(key.startswith("actor_lower.") for key in model_state)


def _action_std(model_state: dict[str, Any]) -> np.ndarray:
    if "std" in model_state:
        return _array(model_state["std"])
    if "log_std" in model_state:
        return np.exp(_array(model_state["log_std"]))
    raise ValueError("Source policy contains neither std nor log_std")


def _single_head_to_split_policy(
    model_state: dict[str, Any],
    *,
    std_source: str,
    lower_init_std: float,
    upper_init_std: float,
) -> dict[str, np.ndarray]:
    from mujoco_mjx.rsl_rl_mujoco.constants import LOWER_ACTION_INDICES, UPPER_ACTION_INDICES

    policy: dict[str, np.ndarray] = {}
    for prefix in ("actor_cnn.", "critic_cnn."):
        policy.update(_arrays(model_state, [key for key in model_state if key.startswith(prefix)]))

    for target in ("actor_lower", "actor_upper"):
        for layer in (0, 2, 4):
            for suffix in ("weight", "bias"):
                policy[f"{target}.{layer}.{suffix}"] = _array(model_state[f"actor.{layer}.{suffix}"])

    for target, indices in (("actor_lower", LOWER_ACTION_INDICES), ("actor_upper", UPPER_ACTION_INDICES)):
        policy[f"{target}.6.weight"] = _array(model_state["actor.6.weight"])[list(indices)].copy()
        policy[f"{target}.6.bias"] = _array(model_state["actor.6.bias"])[list(indices)].copy()

    for layer in (0, 2, 4):
        for suffix in ("weight", "bias"):
            policy[f"critic.{layer}.{suffix}"] = _array(model_state[f"critic.{layer}.{suffix}"])
    policy["critic.6.weight"] = np.repeat(_array(model_state["critic.6.weight"]), 2, axis=0)
    policy["critic.6.bias"] = np.repeat(_array(model_state["critic.6.bias"]), 2, axis=0)

    if std_source == "checkpoint":
        policy["std"] = _action_std(model_state)
    else:
        std = np.empty(29, dtype=np.float32)
        std[list(LOWER_ACTION_INDICES)] = float(lower_init_std)
        std[list(UPPER_ACTION_INDICES)] = float(upper_init_std)
        policy["std"] = std
    return policy


def _discriminator_state(payload: dict[str, Any]) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    disc_state = payload["amp_discriminator_state_dict"]
    norm_state = payload["amp_discriminator_normalizer_state_dict"]
    discriminator = _arrays(
        disc_state,
        [key for key in disc_state if key.startswith("disc_trunk.") or key.startswith("disc_linear.")],
    )
    normalizer = {
        "mean": _array(norm_state["_mean"]),
        "var": _array(norm_state["_var"]),
        "std": _array(norm_state["_std"]),
        "count": np.asarray(norm_state["count"].detach().cpu(), dtype=np.int64),
    }
    return discriminator, normalizer


def _torch_cnn(functional: Any, params: dict[str, Any], prefix: str, scan: Any) -> Any:
    scan = functional.elu(
        functional.conv2d(scan, params[f"{prefix}.0.weight"], params[f"{prefix}.0.bias"], padding=1)
    )
    scan = functional.elu(
        functional.conv2d(scan, params[f"{prefix}.2.weight"], params[f"{prefix}.2.bias"], stride=2)
    )
    scan = functional.elu(
        functional.conv2d(
            scan,
            params[f"{prefix}.4.weight"],
            params[f"{prefix}.4.bias"],
            stride=2,
            padding=(1, 0),
        )
    )
    return functional.adaptive_avg_pool2d(scan, (1, 1)).flatten(1)


def _torch_mlp(functional: Any, params: dict[str, Any], prefix: str, values: Any) -> Any:
    for layer in (0, 2, 4):
        values = functional.elu(
            functional.linear(values, params[f"{prefix}.{layer}.weight"], params[f"{prefix}.{layer}.bias"])
        )
    return functional.linear(values, params[f"{prefix}.6.weight"], params[f"{prefix}.6.bias"])


def _save_verification_fixture(
    torch: Any,
    output: Path,
    source_payload: dict[str, Any],
    discriminator_payload: dict[str, Any],
    policy: dict[str, np.ndarray],
    *,
    batch_size: int,
    seed: int,
) -> Path:
    import torch.nn.functional as functional

    from mujoco_mjx.rsl_rl_mujoco.constants import LOWER_ACTION_INDICES, UPPER_ACTION_INDICES

    source_policy = source_payload["model_state_dict"]
    disc_state = discriminator_payload["amp_discriminator_state_dict"]
    norm_state = discriminator_payload["amp_discriminator_normalizer_state_dict"]
    rng = np.random.default_rng(seed)
    policy_obs = rng.normal(size=(batch_size, 757)).astype(np.float32)
    critic_obs = rng.normal(size=(batch_size, 1052)).astype(np.float32)
    disc_obs = rng.normal(size=(batch_size, 4, 49)).astype(np.float32)

    with torch.inference_mode():
        policy_tensor = torch.from_numpy(policy_obs)
        scan = policy_tensor[:, :187].reshape(-1, 1, 11, 17).transpose(-1, -2).contiguous()
        actor_features = torch.cat((policy_tensor[:, 187:], _torch_cnn(functional, source_policy, "actor_cnn", scan)), dim=-1)
        if _is_split_policy(source_policy):
            expected_actor = torch.zeros((batch_size, 29), dtype=policy_tensor.dtype)
            expected_actor[:, list(LOWER_ACTION_INDICES)] = _torch_mlp(
                functional, source_policy, "actor_lower", actor_features
            )
            expected_actor[:, list(UPPER_ACTION_INDICES)] = _torch_mlp(
                functional, source_policy, "actor_upper", actor_features
            )
        else:
            expected_actor = _torch_mlp(functional, source_policy, "actor", actor_features)

        critic_tensor = torch.from_numpy(critic_obs)
        critic_scan = critic_tensor[:, :935].reshape(-1, 5, 11, 17).transpose(-1, -2).contiguous()
        critic_features = torch.cat(
            (critic_tensor[:, 935:], _torch_cnn(functional, source_policy, "critic_cnn", critic_scan)), dim=-1
        )
        expected_critic = _torch_mlp(functional, source_policy, "critic", critic_features)
        if expected_critic.shape[1] == 1:
            expected_critic = expected_critic.repeat(1, 2)

        disc_tensor = torch.from_numpy(disc_obs)
        normalized = (disc_tensor - norm_state["_mean"]) / (norm_state["_std"] + 1.0e-2)
        hidden = functional.elu(
            functional.linear(normalized.flatten(1), disc_state["disc_trunk.0.weight"], disc_state["disc_trunk.0.bias"])
        )
        hidden = functional.elu(
            functional.linear(hidden, disc_state["disc_trunk.2.weight"], disc_state["disc_trunk.2.bias"])
        )
        expected_discriminator = functional.linear(
            hidden, disc_state["disc_linear.weight"], disc_state["disc_linear.bias"]
        ).squeeze(-1)

    fixture = output.expanduser().resolve()
    fixture.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        fixture,
        policy_obs=policy_obs,
        critic_obs=critic_obs,
        disc_obs=disc_obs,
        expected_actor=np.asarray(expected_actor),
        expected_critic=np.asarray(expected_critic),
        expected_discriminator=np.asarray(expected_discriminator),
        expected_std=np.asarray(policy["std"]),
        metadata_json=np.asarray(json.dumps({"seed": seed, "batch_size": batch_size}), dtype=np.str_),
    )
    return fixture


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path, help="Split-policy checkpoint or single-head height-scan checkpoint")
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--split-template",
        type=Path,
        help="Split checkpoint supplying the upper-body discriminator when converting a single-head checkpoint",
    )
    parser.add_argument(
        "--std-source",
        choices=("checkpoint", "split-config"),
        default="checkpoint",
        help="Preserve learned source std (default) or reset lower/upper std to split-policy config values",
    )
    parser.add_argument("--lower-init-std", type=float, default=1.0)
    parser.add_argument("--upper-init-std", type=float, default=0.6)
    parser.add_argument("--verification-fixture", type=Path)
    parser.add_argument("--verification-batch-size", type=int, default=32)
    parser.add_argument("--verification-seed", type=int, default=7)
    args = parser.parse_args()

    import torch

    from mujoco_mjx.rsl_rl_mujoco.native_model import save_converted_checkpoint

    source = args.checkpoint.expanduser().resolve()
    output = (args.output or source.with_name(f"{source.stem}_jax.npz")).expanduser().resolve()
    source_payload = _load_amp_checkpoint(torch, source)
    source_model = source_payload["model_state_dict"]
    source_is_split = _is_split_policy(source_model)

    if source_is_split:
        policy = _arrays(source_model, [key for key in source_model if not key.startswith("_")])
        discriminator_payload = source_payload
        transfer = "direct_split_checkpoint"
        template = None
    else:
        if args.split_template is None:
            raise ValueError("A single-head checkpoint requires --split-template for the upper-body discriminator")
        template = args.split_template.expanduser().resolve()
        discriminator_payload = _load_amp_checkpoint(torch, template)
        if not _is_split_policy(discriminator_payload["model_state_dict"]):
            raise ValueError(f"Split template {template} does not contain a split policy")
        policy = _single_head_to_split_policy(
            source_model,
            std_source=args.std_source,
            lower_init_std=args.lower_init_std,
            upper_init_std=args.upper_init_std,
        )
        transfer = "single_head_actor_critic_to_split"

    discriminator, normalizer = _discriminator_state(discriminator_payload)
    if normalizer["mean"].shape != (1, 49):
        raise ValueError(
            "Native split training requires an upper-body discriminator normalizer of shape (1, 49); "
            f"got {normalizer['mean'].shape}"
        )
    result = save_converted_checkpoint(
        output,
        policy,
        discriminator,
        normalizer,
        {
            "source_checkpoint": str(source),
            "source_iteration": int(source_payload.get("iter", 0)),
            "transfer": transfer,
            "std_source": "source_split_checkpoint" if source_is_split else args.std_source,
            "split_discriminator_template": str(template) if template is not None else None,
            "split_discriminator_template_iteration": (
                int(discriminator_payload.get("iter", 0)) if template is not None else None
            ),
            "source_full_body_discriminator_ignored": not source_is_split,
            "optimizer_state": "fresh_native_jax_adam",
            "policy_class": "ActorCriticSplitHeightScan",
            "observation_layout": "isaac_scan_first_v2",
            "height_scan_reference": "torso_link_frame_z_minus_terrain_hit_z_minus_offset",
            "height_scan_offset": 0.5,
            "discriminator_loss": "LSGAN",
        },
    )
    print(f"[PASS] Converted {source} -> {result}")
    print(f"[INFO] Transfer: {transfer}; action std: {args.std_source if not source_is_split else 'checkpoint'}")
    if template is not None:
        print(f"[INFO] Upper-body discriminator template: {template}")
    if args.verification_fixture is not None:
        fixture = _save_verification_fixture(
            torch,
            args.verification_fixture,
            source_payload,
            discriminator_payload,
            policy,
            batch_size=args.verification_batch_size,
            seed=args.verification_seed,
        )
        print(f"[PASS] Wrote framework-handoff verification fixture: {fixture}")


if __name__ == "__main__":
    main()
