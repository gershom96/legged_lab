from __future__ import annotations

from typing import Any


def adam_init(params: Any) -> dict[str, Any]:
    import jax
    import jax.numpy as jnp

    return {
        "m": jax.tree_util.tree_map(jnp.zeros_like, params),
        "v": jax.tree_util.tree_map(jnp.zeros_like, params),
        "t": jnp.asarray(0, dtype=jnp.int32),
    }


def tree_l2_norm(tree: Any) -> Any:
    import jax
    import jax.numpy as jnp

    leaves = jax.tree_util.tree_leaves(tree)
    return jnp.sqrt(sum(jnp.sum(value * value) for value in leaves))


def adam_update(
    params: Any,
    grads: Any,
    state: dict[str, Any],
    learning_rate: Any,
    max_grad_norm: float,
    weight_decay: Any | None = None,
    beta1: float = 0.9,
    beta2: float = 0.999,
    epsilon: float = 1.0e-8,
) -> tuple[Any, dict[str, Any]]:
    import jax
    import jax.numpy as jnp

    norm = tree_l2_norm(grads)
    scale = jnp.minimum(1.0, float(max_grad_norm) / (norm + 1.0e-8))
    grads = jax.tree_util.tree_map(lambda gradient: gradient * scale, grads)
    if weight_decay is not None:
        grads = jax.tree_util.tree_map(
            lambda gradient, parameter, decay: gradient + decay * parameter,
            grads,
            params,
            weight_decay,
        )
    time = state["t"] + 1
    first = jax.tree_util.tree_map(
        lambda old, gradient: beta1 * old + (1.0 - beta1) * gradient,
        state["m"],
        grads,
    )
    second = jax.tree_util.tree_map(
        lambda old, gradient: beta2 * old + (1.0 - beta2) * gradient * gradient,
        state["v"],
        grads,
    )
    first_correction = 1.0 - beta1**time
    second_correction = 1.0 - beta2**time
    updated = jax.tree_util.tree_map(
        lambda parameter, first_value, second_value: parameter
        - learning_rate
        * (first_value / first_correction)
        / (jnp.sqrt(second_value / second_correction) + epsilon),
        params,
        first,
        second,
    )
    return updated, {"m": first, "v": second, "t": time}
