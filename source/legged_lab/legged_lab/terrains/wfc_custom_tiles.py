"""Optional custom WFC tiles for cluttered locomotion scenes."""

from __future__ import annotations

from math import cos, sin
from typing import Any

import numpy as np


DEFAULT_CUSTOM_PRIMITIVES = (
    "star_clutter",
    "payload_gate",
    "overhead_barrier",
    "low_tunnel",
    "sidestep_corridor",
    "stepping_clutter",
    "random_boxes",
    "combined_challenge",
)


def build_custom_mesh_parts(config: dict[str, Any] | None, *, dim: tuple[float, float, float], seed: int) -> tuple[Any, ...]:
    """Create terrain-generator mesh-part configs for custom WFC tiles."""
    if config is None or not bool(config.get("enabled", False)):
        return ()

    from terrain_generator.trimesh_tiles.mesh_parts.mesh_parts_cfg import BoxMeshPartsCfg

    names = config.get("types")
    if names is None or names == "all":
        selected = DEFAULT_CUSTOM_PRIMITIVES
    else:
        selected = tuple(str(name) for name in names)

    weight = float(config.get("weight", 0.35))
    floor_edge = np.ones((5, 5), dtype=np.float64) * float(config.get("edge_height", 0.1))
    rng = np.random.default_rng(seed)
    builders = {
        "star_clutter": lambda: _star_clutter(BoxMeshPartsCfg, dim, floor_edge, weight),
        "payload_gate": lambda: _payload_gate(BoxMeshPartsCfg, dim, floor_edge, weight),
        "overhead_barrier": lambda: _overhead_barrier(BoxMeshPartsCfg, dim, floor_edge, weight),
        "low_tunnel": lambda: _low_tunnel(BoxMeshPartsCfg, dim, floor_edge, weight),
        "sidestep_corridor": lambda: _sidestep_corridor(BoxMeshPartsCfg, dim, floor_edge, weight),
        "stepping_clutter": lambda: _stepping_clutter(BoxMeshPartsCfg, dim, floor_edge, weight),
        "random_boxes": lambda: _random_boxes(BoxMeshPartsCfg, dim, floor_edge, weight, rng),
        "combined_challenge": lambda: _combined_challenge(BoxMeshPartsCfg, dim, floor_edge, weight),
    }

    unknown = sorted(set(selected) - set(builders))
    if unknown:
        raise ValueError(f"Unknown custom WFC primitive(s): {', '.join(unknown)}")

    return tuple(builders[name]() for name in selected)


def _box_cfg(
    box_cls: Any,
    *,
    name: str,
    dim: tuple[float, float, float],
    edge_array: np.ndarray,
    boxes: list[tuple[tuple[float, float, float], tuple[float, float, float], float]],
    weight: float,
) -> Any:
    box_dims = []
    transforms = []
    for size, center_xyz, yaw in boxes:
        transform = _transform(
            x=center_xyz[0],
            y=center_xyz[1],
            z_center_above_floor=float(size[2]) / 2.0 + float(center_xyz[2]),
            yaw=yaw,
            floor_thickness=0.1,
        )
        box_dims.append(tuple(float(value) for value in size))
        transforms.append(transform)

    return box_cls(
        name=name,
        dim=dim,
        box_dims=tuple(box_dims),
        transformations=tuple(transforms),
        add_floor=True,
        edge_array=edge_array.copy(),
        rotations=(90, 180, 270),
        flips=(),
        weight=weight,
        minimal_triangles=False,
        load_from_cache=False,
    )


def _transform(
    *,
    x: float,
    y: float,
    z_center_above_floor: float,
    yaw: float = 0.0,
    floor_thickness: float = 0.1,
) -> np.ndarray:
    transform = np.eye(4)
    c = cos(yaw)
    s = sin(yaw)
    transform[:3, :3] = np.array(
        [
            [c, -s, 0.0],
            [s, c, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    transform[:3, 3] = [x, y, floor_thickness + z_center_above_floor]
    return transform


def _star_clutter(box_cls: Any, dim: tuple[float, float, float], edge: np.ndarray, weight: float) -> Any:
    boxes = []
    for i in range(8):
        yaw = i * np.pi / 4.0
        boxes.append(((0.75, 0.08, 0.45), (0.0, 0.0, 0.0), yaw))
    boxes.append(((0.22, 0.22, 0.55), (0.0, 0.0, 0.0), 0.0))
    return _box_cfg(box_cls, name="ean_star_clutter", dim=dim, edge_array=edge, boxes=boxes, weight=weight)


def _payload_gate(box_cls: Any, dim: tuple[float, float, float], edge: np.ndarray, weight: float) -> Any:
    boxes = [
        ((0.24, 1.55, 1.15), (-0.58, 0.0, 0.0), 0.0),
        ((0.24, 1.55, 1.15), (0.58, 0.0, 0.0), 0.0),
    ]
    return _box_cfg(box_cls, name="ean_payload_gate", dim=dim, edge_array=edge, boxes=boxes, weight=weight)


def _overhead_barrier(box_cls: Any, dim: tuple[float, float, float], edge: np.ndarray, weight: float) -> Any:
    boxes = [((1.35, 0.32, 0.22), (0.0, 0.0, 1.05), 0.0)]
    return _box_cfg(box_cls, name="ean_overhead_barrier", dim=dim, edge_array=edge, boxes=boxes, weight=weight)


def _low_tunnel(box_cls: Any, dim: tuple[float, float, float], edge: np.ndarray, weight: float) -> Any:
    boxes = [
        ((0.18, 1.6, 1.0), (-0.72, 0.0, 0.0), 0.0),
        ((0.18, 1.6, 1.0), (0.72, 0.0, 0.0), 0.0),
        ((1.6, 1.6, 0.18), (0.0, 0.0, 1.05), 0.0),
    ]
    return _box_cfg(box_cls, name="ean_low_tunnel", dim=dim, edge_array=edge, boxes=boxes, weight=weight)


def _sidestep_corridor(box_cls: Any, dim: tuple[float, float, float], edge: np.ndarray, weight: float) -> Any:
    boxes = [
        ((0.28, 0.95, 1.1), (-0.52, -0.42, 0.0), 0.0),
        ((0.28, 0.95, 1.1), (0.52, 0.42, 0.0), 0.0),
        ((0.75, 0.18, 1.1), (0.0, -0.76, 0.0), 0.0),
        ((0.75, 0.18, 1.1), (0.0, 0.76, 0.0), 0.0),
    ]
    return _box_cfg(box_cls, name="ean_sidestep_corridor", dim=dim, edge_array=edge, boxes=boxes, weight=weight)


def _stepping_clutter(box_cls: Any, dim: tuple[float, float, float], edge: np.ndarray, weight: float) -> Any:
    boxes = [
        ((0.24, 0.24, 0.18), (-0.5, -0.45, 0.0), 0.0),
        ((0.30, 0.22, 0.22), (0.1, -0.18, 0.0), 0.25),
        ((0.26, 0.30, 0.16), (0.55, 0.12, 0.0), -0.3),
        ((0.20, 0.38, 0.20), (-0.18, 0.52, 0.0), 0.1),
    ]
    return _box_cfg(box_cls, name="ean_stepping_clutter", dim=dim, edge_array=edge, boxes=boxes, weight=weight)


def _random_boxes(
    box_cls: Any,
    dim: tuple[float, float, float],
    edge: np.ndarray,
    weight: float,
    rng: np.random.Generator,
) -> Any:
    boxes = []
    for _ in range(6):
        sx = float(rng.uniform(0.12, 0.35))
        sy = float(rng.uniform(0.12, 0.45))
        sz = float(rng.uniform(0.15, 0.65))
        x = float(rng.uniform(-0.55, 0.55))
        y = float(rng.uniform(-0.55, 0.55))
        yaw = float(rng.uniform(-np.pi, np.pi))
        boxes.append(((sx, sy, sz), (x, y, 0.0), yaw))
    return _box_cfg(box_cls, name="ean_random_boxes", dim=dim, edge_array=edge, boxes=boxes, weight=weight)


def _combined_challenge(box_cls: Any, dim: tuple[float, float, float], edge: np.ndarray, weight: float) -> Any:
    boxes = [
        ((0.2, 1.5, 1.0), (-0.66, 0.0, 0.0), 0.0),
        ((0.2, 1.5, 1.0), (0.66, 0.0, 0.0), 0.0),
        ((1.2, 0.18, 0.2), (0.0, 0.0, 1.0), 0.0),
        ((0.25, 0.25, 0.2), (-0.2, -0.45, 0.0), 0.0),
        ((0.28, 0.2, 0.25), (0.25, 0.35, 0.0), 0.4),
    ]
    return _box_cfg(box_cls, name="ean_combined_challenge", dim=dim, edge_array=edge, boxes=boxes, weight=weight)
