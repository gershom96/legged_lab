#!/usr/bin/env python3
"""Verify native JAX outputs against a PyTorch fixture emitted during conversion."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

import numpy as np


os.environ.setdefault("JAX_PLATFORMS", "cpu")
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("converted", type=Path, help="Converted native-JAX .npz checkpoint")
    parser.add_argument("fixture", type=Path, help="PyTorch reference fixture written by convert_pt_to_jax.py")
    parser.add_argument("--atol", type=float, default=1.0e-4)
    args = parser.parse_args()

    import jax

    from mujoco_mjx.rsl_rl_mujoco.native_model import (
        actor_apply,
        critic_apply,
        discriminator_score,
        load_checkpoint,
    )

    policy, discriminator, normalizer, _ = load_checkpoint(args.converted)
    with np.load(args.fixture.expanduser().resolve(), allow_pickle=False) as fixture:
        comparisons = {
            "actor": (
                np.asarray(fixture["expected_actor"]),
                np.asarray(actor_apply(policy, np.asarray(fixture["policy_obs"]))),
            ),
            "critic": (
                np.asarray(fixture["expected_critic"]),
                np.asarray(critic_apply(policy, np.asarray(fixture["critic_obs"]))),
            ),
            "discriminator": (
                np.asarray(fixture["expected_discriminator"]),
                np.asarray(discriminator_score(discriminator, normalizer, np.asarray(fixture["disc_obs"]))),
            ),
            "std": (np.asarray(fixture["expected_std"]), np.asarray(policy["std"])),
        }

    failed = []
    for name, (expected, actual) in comparisons.items():
        maximum = float(np.max(np.abs(expected - actual)))
        print(f"{name:14s} max_abs_error={maximum:.9g}")
        if not np.allclose(expected, actual, rtol=1.0e-5, atol=args.atol):
            failed.append(name)
    if failed:
        raise RuntimeError(f"Torch/JAX parity failed for: {', '.join(failed)}")
    jax.clear_caches()
    print("[PASS] Native actor, critic, action std, and AMP discriminator match the PyTorch handoff fixture.")


if __name__ == "__main__":
    main()
