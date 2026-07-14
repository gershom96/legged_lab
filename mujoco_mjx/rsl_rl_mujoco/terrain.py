from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
import sys
from typing import Any

import numpy as np
import yaml

from .spec import TerrainSpec


@dataclass(frozen=True)
class TerrainBox:
    center: tuple[float, float, float]
    half_size: tuple[float, float, float]


@dataclass(frozen=True)
class TerrainPatch:
    name: str
    stage: int
    origin: tuple[float, float]
    safe_half_extent: tuple[float, float]
    difficulty: float


@dataclass(frozen=True)
class TerrainData:
    heights: np.ndarray
    size: tuple[float, float]
    horizontal_scale: float
    terrain_type: str
    mesh_path: Path | None = None
    collision_mesh_paths: tuple[Path, ...] = ()
    mesh_z_offset: float = 0.0
    boxes: tuple[TerrainBox, ...] = ()
    spawn_xy: tuple[float, float] = (0.0, 0.0)
    collision_heights: np.ndarray | None = None
    perception_heights: np.ndarray | None = None
    patches: tuple[TerrainPatch, ...] = ()

    @property
    def x_coordinates(self) -> np.ndarray:
        return np.linspace(-self.size[0] / 2.0, self.size[0] / 2.0, self.heights.shape[0])

    @property
    def y_coordinates(self) -> np.ndarray:
        return np.linspace(-self.size[1] / 2.0, self.size[1] / 2.0, self.heights.shape[1])

    def height_at(self, x: float, y: float) -> float:
        x_index = int(np.clip(round((x + self.size[0] / 2.0) / self.horizontal_scale), 0, self.heights.shape[0] - 1))
        y_index = int(np.clip(round((y + self.size[1] / 2.0) / self.horizontal_scale), 0, self.heights.shape[1] - 1))
        return float(self.heights[x_index, y_index])

    def scan_surface_heights(self) -> np.ndarray:
        """Return the top-most static collision surface represented by the height scan.

        The MuJoCo terrain may contain a height field/mesh plus standalone box geoms.
        Isaac's downward rays see the highest of those surfaces, so preserve that
        convention in the native analytic sampler.
        """
        heights = np.asarray(
            self.heights if self.perception_heights is None else self.perception_heights,
            dtype=np.float32,
        ).copy()
        if self.boxes:
            _rasterize_box_tops(heights, self.boxes, self.size)
        return heights


_ALIASES = {
    "flat_empty": "flat",
    "random_rough_heightfield": "random_rough",
    "inverted_pyramid_slope": "pyramid_slope_inv",
    "inverted_pyramid_stairs": "pyramid_stairs_inv",
    "random_grid_boxes": "boxes_low",
    "discrete_floor_obstacles": "discrete_obstacles",
    "wave_ground": "wave",
    "rails_terrain": "rails",
    "pit_terrain": "pit",
    "box_terrain": "box",
    "gap_terrain": "gap",
}


def generate_terrain(spec: TerrainSpec) -> TerrainData:
    terrain_type = _ALIASES.get(spec.type, spec.type)
    if terrain_type == "curriculum":
        return _generate_curriculum_board(spec)
    if terrain_type in {"ean", "ean_scene", "wfc"}:
        return _generate_from_ean(spec)

    nx = int(round(spec.size[0] / spec.horizontal_scale)) + 1
    ny = int(round(spec.size[1] / spec.horizontal_scale)) + 1
    x = np.linspace(-spec.size[0] / 2.0, spec.size[0] / 2.0, nx, dtype=np.float32)
    y = np.linspace(-spec.size[1] / 2.0, spec.size[1] / 2.0, ny, dtype=np.float32)
    xx, yy = np.meshgrid(x, y, indexing="ij")
    heights = np.zeros((nx, ny), dtype=np.float32)
    rng = np.random.default_rng(spec.seed)
    difficulty = float(np.clip(spec.difficulty, 0.0, 1.0))

    if terrain_type == "flat":
        pass
    elif terrain_type in {"random_rough_mild", "random_rough"}:
        limits = (0.005, 0.035) if terrain_type.endswith("mild") else (0.02, 0.10)
        maximum = float(np.interp(difficulty, (0.0, 1.0), limits))
        cell = max(1, int(round(0.35 / spec.horizontal_scale)))
        coarse = rng.uniform(0.0, maximum, size=((nx + cell - 1) // cell, (ny + cell - 1) // cell))
        heights = np.repeat(np.repeat(coarse, cell, axis=0), cell, axis=1)[:nx, :ny].astype(np.float32)
    elif terrain_type in {"wave_mild", "wave"}:
        limits = (0.005, 0.03) if terrain_type.endswith("mild") else (0.03, 0.12)
        amplitude = float(np.interp(difficulty, (0.0, 1.0), limits))
        waves = 2.0 if terrain_type.endswith("mild") else 3.0
        phase = rng.uniform(0.0, 2.0 * np.pi)
        heights = amplitude * np.maximum(
            0.0,
            0.5
            * (
                np.sin(2.0 * np.pi * waves * (xx / spec.size[0] + 0.5) + phase)
                + np.cos(2.0 * np.pi * waves * (yy / spec.size[1] + 0.5) - phase)
            ),
        )
    elif terrain_type in {"pyramid_slope", "pyramid_slope_inv"}:
        distance = np.maximum(np.abs(2.0 * xx / spec.size[0]), np.abs(2.0 * yy / spec.size[1]))
        profile = np.clip(distance if terrain_type.endswith("inv") else 1.0 - distance, 0.0, 1.0)
        heights = (0.04 + 0.30 * difficulty) * profile
    elif terrain_type in {"pyramid_stairs", "pyramid_stairs_inv"}:
        distance = np.maximum(np.abs(2.0 * xx / spec.size[0]), np.abs(2.0 * yy / spec.size[1]))
        profile = np.clip(distance if terrain_type.endswith("inv") else 1.0 - distance, 0.0, 1.0)
        step_height = 0.04 + 0.10 * difficulty
        num_steps = max(4, int(round(min(spec.size) / 1.5)))
        heights = np.floor(profile * num_steps) * step_height
    elif terrain_type in {"boxes_low", "discrete_obstacles"}:
        count = int(round(20 + 100 * difficulty)) if terrain_type == "discrete_obstacles" else int(round(80 + 160 * difficulty))
        maximum = 0.05 + 0.18 * difficulty
        for _ in range(count):
            center_x = rng.uniform(x[0] + 0.5, x[-1] - 0.5)
            center_y = rng.uniform(y[0] + 0.5, y[-1] - 0.5)
            width_x = rng.uniform(0.18, 0.55)
            width_y = rng.uniform(0.18, 0.55)
            box_height = rng.uniform(0.02, maximum)
            mask = (np.abs(xx - center_x) <= width_x / 2.0) & (np.abs(yy - center_y) <= width_y / 2.0)
            heights[mask] = np.maximum(heights[mask], box_height)
    elif terrain_type == "stepping_stones":
        heights.fill(-0.35 - 0.25 * difficulty)
        count = max(10, int(round(spec.size[0] / 0.7)))
        stone_length = 0.45
        stone_width = 0.65 - 0.20 * difficulty
        for stone_x in np.linspace(x[0] + 0.8, x[-1] - 0.8, count):
            stone_y = rng.uniform(-0.05 - 0.4 * difficulty, 0.05 + 0.4 * difficulty)
            mask = (np.abs(xx - stone_x) <= stone_length / 2.0) & (np.abs(yy - stone_y) <= stone_width / 2.0)
            heights[mask] = rng.uniform(0.03, 0.04 + 0.16 * difficulty)
    elif terrain_type == "gap":
        gap_width = 0.25 + 0.75 * difficulty
        heights[np.abs(xx) < gap_width / 2.0] = -0.5
    elif terrain_type == "pit":
        pit_width = 1.0 + 1.5 * difficulty
        heights[(np.abs(xx) < pit_width / 2.0) & (np.abs(yy) < pit_width / 2.0)] = -0.5
    elif terrain_type == "rails":
        rail_height = 0.06 + 0.20 * difficulty
        thickness = 0.10 + 0.12 * difficulty
        for rail_y in (-0.6, 0.6):
            heights[np.abs(yy - rail_y) < thickness / 2.0] = rail_height
    elif terrain_type == "box":
        width = 1.0 + difficulty
        heights[(np.abs(xx) < width / 2.0) & (np.abs(yy) < width / 2.0)] = 0.06 + 0.20 * difficulty
    else:
        raise ValueError(
            f"Unsupported terrain type {spec.type!r}. Supported types: "
            "flat, random_rough[_mild], wave[_mild], pyramid_slope[_inv], "
            "pyramid_stairs[_inv], boxes_low, discrete_obstacles, stepping_stones, gap, pit, rails, box, ean, wfc"
        )

    center_mask = xx * xx + yy * yy <= spec.center_platform_radius**2
    hazardous_center = terrain_type in {"stepping_stones", "gap", "pit"}
    center_height = 0.0 if hazardous_center else float(np.median(heights[center_mask]))
    heights[center_mask] = center_height
    return TerrainData(
        heights=np.asarray(heights, dtype=np.float32),
        size=spec.size,
        horizontal_scale=spec.horizontal_scale,
        terrain_type=terrain_type,
    )


_CURRICULUM_PATCHES = (
    ("flat", 0, 0.0),
    ("random_rough_mild", 1, 0.35),
    ("wave_mild", 1, 0.35),
    ("random_rough", 2, 0.45),
    ("discrete_obstacles", 2, 0.40),
    ("boxes_low", 2, 0.40),
    ("pyramid_slope", 3, 0.50),
    ("pyramid_slope_inv", 3, 0.50),
    ("wave", 3, 0.50),
    ("pyramid_stairs", 4, 0.55),
    ("pyramid_stairs_inv", 4, 0.55),
    ("stepping_stones", 5, 0.45),
    ("gap", 5, 0.45),
    ("pit", 5, 0.45),
    ("rails", 5, 0.45),
)


def _generate_curriculum_board(spec: TerrainSpec) -> TerrainData:
    if not spec.curriculum.enabled:
        raise ValueError("terrain.type='curriculum' requires terrain.curriculum.enabled=true")

    rows = columns = 4
    patch_nx = int(round(spec.size[0] / spec.horizontal_scale)) + 1
    patch_ny = int(round(spec.size[1] / spec.horizontal_scale)) + 1
    board_nx = rows * (patch_nx - 1) + 1
    board_ny = columns * (patch_ny - 1) + 1
    board = np.zeros((board_nx, board_ny), dtype=np.float32)
    collision_board = np.zeros_like(board)
    board_size = (rows * spec.size[0], columns * spec.size[1])
    patches: list[TerrainPatch] = []

    x = np.linspace(-spec.size[0] / 2.0, spec.size[0] / 2.0, patch_nx, dtype=np.float32)
    y = np.linspace(-spec.size[1] / 2.0, spec.size[1] / 2.0, patch_ny, dtype=np.float32)
    xx, yy = np.meshgrid(x, y, indexing="ij")
    edge_distance = np.minimum(spec.size[0] / 2.0 - np.abs(xx), spec.size[1] / 2.0 - np.abs(yy))
    border_width = min(spec.size) / 2.0 - spec.curriculum.safe_half_extent
    blend = np.clip(edge_distance / border_width, 0.0, 1.0)
    blend = blend * blend * (3.0 - 2.0 * blend)

    for patch_index, (name, stage, difficulty) in enumerate(_CURRICULUM_PATCHES):
        row, column = divmod(patch_index, columns)
        local_spec = replace(
            spec,
            type=name,
            difficulty=difficulty,
            seed=spec.seed + 7919 * (patch_index + 1),
            curriculum=replace(spec.curriculum, enabled=False),
        )
        local = generate_terrain(local_spec)
        local_heights = np.asarray(local.heights, dtype=np.float32) * blend
        x_start = row * (patch_nx - 1)
        y_start = column * (patch_ny - 1)
        board[x_start : x_start + patch_nx, y_start : y_start + patch_ny] = local_heights
        collision_board[x_start : x_start + patch_nx, y_start : y_start + patch_ny] = local_heights
        origin = (
            -board_size[0] / 2.0 + (row + 0.5) * spec.size[0],
            -board_size[1] / 2.0 + (column + 0.5) * spec.size[1],
        )
        patches.append(
            TerrainPatch(
                name=name,
                stage=stage,
                origin=origin,
                safe_half_extent=(spec.curriculum.safe_half_extent, spec.curriculum.safe_half_extent),
                difficulty=difficulty,
            )
        )

    flat_patch = patches[0]
    flat_depth = 0.05
    collision_board[:patch_nx, :patch_ny] = -flat_depth
    flat_box = TerrainBox(
        center=(flat_patch.origin[0], flat_patch.origin[1], -flat_depth / 2.0),
        half_size=(spec.size[0] / 2.0, spec.size[1] / 2.0, flat_depth / 2.0),
    )
    return TerrainData(
        heights=board,
        size=board_size,
        horizontal_scale=spec.horizontal_scale,
        terrain_type="curriculum",
        boxes=(flat_box,),
        collision_heights=collision_board,
        patches=tuple(patches),
    )


def _generate_from_ean(spec: TerrainSpec) -> TerrainData:
    if spec.ean_root is None or spec.ean_scene_config is None:
        raise ValueError("EAN terrain integration requires terrain.ean_root and terrain.ean_scene_config")
    ean_root = spec.ean_root.expanduser().resolve()
    source_root = ean_root / "source"
    if not source_root.is_dir():
        raise FileNotFoundError(f"Embodiment-Aware-Nav source directory not found: {source_root}")
    sys.path.insert(0, str(source_root))
    try:
        with spec.ean_scene_config.open("r", encoding="utf-8") as stream:
            config: dict[str, Any] = yaml.safe_load(stream)
        if str(config.get("kind")) == "wfc_terrain" or spec.type == "wfc":
            from embodiment_aware_nav.scene_generation.generators.wfc_terrain import generate_wfc_terrain_mesh

            output = spec.wfc_output_dir or Path("mujoco_mjx/outputs/wfc")
            wfc_config = dict(config.get("wfc", {}))
            wfc_config.setdefault("seed", int(config.get("seed", spec.seed)))
            wfc_config.setdefault("save_collision_parts", True)
            artifacts = generate_wfc_terrain_mesh(wfc_config, output)
            mesh_path = Path(artifacts["mesh_path"]).resolve()
            collision_paths = tuple(Path(path).resolve() for path in artifacts.get("collision_mesh_paths", []))
            return _terrain_from_mesh(
                mesh_path,
                collision_paths,
                spec,
                spawn_height=float(config.get("terrain_mesh_spawn_height", 0.0)),
            )

        from embodiment_aware_nav.scene_generation import PayloadSpec, generate_scene

        scene = generate_scene(config, PayloadSpec.from_mapping({"name": "none", "enabled": False}))
        if scene.terrain_mesh_path:
            return _terrain_from_mesh(
                Path(scene.terrain_mesh_path),
                tuple(Path(path) for path in scene.terrain_collision_mesh_paths),
                spec,
                spawn_height=float(scene.terrain_mesh_spawn_height),
            )

        shape_xy = scene.occupancy.shape[:2]
        size = (shape_xy[0] * scene.voxel_size, shape_xy[1] * scene.voxel_size)
        center_xy = (
            scene.origin[0] + size[0] / 2.0,
            scene.origin[1] + size[1] / 2.0,
        )
        if scene.terrain_heightfield is None:
            heights = np.zeros(shape_xy, dtype=np.float32)
            obstacle_occupancy = scene.occupancy
        else:
            heights = np.asarray(scene.terrain_heightfield, dtype=np.float32) + float(scene.origin[2])
            terrain_occupancy = scene.terrain_occupancy
            obstacle_occupancy = scene.occupancy if terrain_occupancy is None else scene.occupancy & ~terrain_occupancy

        boxes = _boxes_from_occupancy(obstacle_occupancy, scene.origin, scene.voxel_size, center_xy)
        collision_heights = np.asarray(heights, dtype=np.float32).copy()
        heights = collision_heights.copy()
        _rasterize_box_tops(heights, boxes, size)
        return TerrainData(
            heights=heights,
            size=size,
            horizontal_scale=float(scene.voxel_size),
            terrain_type=f"ean:{config.get('name', config.get('kind', 'scene'))}",
            boxes=boxes,
            spawn_xy=(float(scene.start.x - center_xy[0]), float(scene.start.y - center_xy[1])),
            collision_heights=collision_heights,
        )
    finally:
        if sys.path and sys.path[0] == str(source_root):
            sys.path.pop(0)


def _terrain_from_mesh(
    mesh_path: Path,
    collision_paths: tuple[Path, ...],
    spec: TerrainSpec,
    spawn_height: float = 0.0,
) -> TerrainData:
    import trimesh

    mesh = trimesh.load(mesh_path.expanduser().resolve(), force="mesh")
    bounds = np.asarray(mesh.bounds, dtype=np.float64)
    size = (float(bounds[1, 0] - bounds[0, 0]), float(bounds[1, 1] - bounds[0, 1]))
    nx = int(round(size[0] / spec.horizontal_scale)) + 1
    ny = int(round(size[1] / spec.horizontal_scale)) + 1
    x = np.linspace(bounds[0, 0], bounds[1, 0], nx)
    y = np.linspace(bounds[0, 1], bounds[1, 1], ny)
    xx, yy = np.meshgrid(x, y, indexing="ij")
    origins = np.column_stack((xx.ravel(), yy.ravel(), np.full(xx.size, bounds[1, 2] + 1.0)))
    directions = np.zeros_like(origins)
    directions[:, 2] = -1.0
    heights = _sample_mesh_surface((mesh,), origins, directions, float(bounds[0, 2]), mesh_path)
    scan_meshes = tuple(
        trimesh.load(collision_path.expanduser().resolve(), force="mesh") for collision_path in collision_paths
    )
    scan_heights = _sample_mesh_surface(
        scan_meshes or (mesh,), origins, directions, float(bounds[0, 2]), mesh_path
    )
    center = 0.5 * (bounds[0, :2] + bounds[1, :2])
    z_offset = float(spawn_height - bounds[0, 2])
    centered_mesh = mesh.copy()
    centered_mesh.apply_translation((-center[0], -center[1], z_offset))
    centered_path = mesh_path.parent / f"{mesh_path.stem}_centered.obj"
    centered_mesh.export(centered_path)
    centered_collision_paths: list[Path] = []
    for index, collision_mesh in enumerate(scan_meshes):
        collision_mesh.apply_translation((-center[0], -center[1], z_offset))
        centered_collision_path = mesh_path.parent / f"{mesh_path.stem}_collision_{index:05d}_centered.obj"
        collision_mesh.export(centered_collision_path)
        centered_collision_paths.append(centered_collision_path)
    return TerrainData(
        heights=heights.reshape(nx, ny) + z_offset,
        size=size,
        horizontal_scale=spec.horizontal_scale,
        terrain_type="wfc",
        mesh_path=centered_path,
        collision_mesh_paths=tuple(centered_collision_paths),
        mesh_z_offset=z_offset,
        perception_heights=scan_heights.reshape(nx, ny) + z_offset,
    )


def _sample_mesh_surface(
    meshes: tuple[Any, ...],
    origins: np.ndarray,
    directions: np.ndarray,
    fallback_height: float,
    source_path: Path,
) -> np.ndarray:
    """Rasterize the first downward hit across static collision meshes."""
    heights = np.full(origins.shape[0], -np.inf, dtype=np.float32)
    try:
        for mesh in meshes:
            locations, ray_ids, _ = mesh.ray.intersects_location(origins, directions, multiple_hits=True)
            if len(locations):
                np.maximum.at(heights, ray_ids, locations[:, 2].astype(np.float32))
    except Exception as exc:
        raise RuntimeError(f"Could not sample WFC collision mesh for height scans: {source_path}") from exc
    heights[~np.isfinite(heights)] = fallback_height
    return heights


def _boxes_from_occupancy(
    occupancy: np.ndarray,
    origin: tuple[float, float, float],
    voxel_size: float,
    center_xy: tuple[float, float],
) -> tuple[TerrainBox, ...]:
    remaining = np.asarray(occupancy, dtype=bool).copy()
    boxes: list[TerrainBox] = []
    origin_array = np.asarray(origin, dtype=np.float64)
    for start in np.argwhere(remaining):
        x0, y0, z0 = (int(value) for value in start)
        if not remaining[x0, y0, z0]:
            continue
        x1 = x0 + 1
        while x1 < remaining.shape[0] and remaining[x1, y0, z0]:
            x1 += 1
        y1 = y0 + 1
        while y1 < remaining.shape[1] and remaining[x0:x1, y1, z0].all():
            y1 += 1
        z1 = z0 + 1
        while z1 < remaining.shape[2] and remaining[x0:x1, y0:y1, z1].all():
            z1 += 1
        remaining[x0:x1, y0:y1, z0:z1] = False
        lower = np.asarray((x0, y0, z0), dtype=np.float64)
        upper = np.asarray((x1, y1, z1), dtype=np.float64)
        center = origin_array + voxel_size * (lower + upper) / 2.0
        center[:2] -= np.asarray(center_xy)
        half_size = voxel_size * (upper - lower) / 2.0
        boxes.append(TerrainBox(tuple(center.tolist()), tuple(half_size.tolist())))
    return tuple(boxes)


def _rasterize_box_tops(
    heights: np.ndarray,
    boxes: tuple[TerrainBox, ...],
    size: tuple[float, float],
) -> None:
    nx, ny = heights.shape
    for box in boxes:
        center = np.asarray(box.center)
        half_size = np.asarray(box.half_size)
        x0 = int(np.floor((center[0] - half_size[0] + size[0] / 2.0) * nx / size[0]))
        x1 = int(np.ceil((center[0] + half_size[0] + size[0] / 2.0) * nx / size[0]))
        y0 = int(np.floor((center[1] - half_size[1] + size[1] / 2.0) * ny / size[1]))
        y1 = int(np.ceil((center[1] + half_size[1] + size[1] / 2.0) * ny / size[1]))
        x0, x1 = max(0, x0), min(nx, x1)
        y0, y1 = max(0, y0), min(ny, y1)
        if x0 < x1 and y0 < y1:
            heights[x0:x1, y0:y1] = np.maximum(heights[x0:x1, y0:y1], center[2] + half_size[2])
