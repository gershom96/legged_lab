"""IsaacLab terrain-generator wrapper for cached WFC worlds."""

from __future__ import annotations

from contextlib import contextmanager
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import random
import sys
import types
from typing import Any, Iterator, Literal

import numpy as np
import torch
from isaaclab.utils import configclass

from legged_lab.terrains.wfc_custom_tiles import build_custom_mesh_parts


@configclass
class WfcTerrainGeneratorCfg:
    """Configuration for :class:`WfcTerrainGenerator`.

    This config intentionally mirrors the fields IsaacLab terrain importers need:
    the generator produces ``terrain_mesh``, ``terrain_origins`` and ``flat_patches``.
    """

    class_type: type = None

    terrain_generator_root: str | None = None
    """Path to the terrain-generator checkout. Defaults to ``third_party/terrain-generator`` in this repo."""

    cache_dir: str = "logs/wfc_terrain_cache"
    """Directory where generated OBJ meshes and metadata are cached."""

    seed: int = 0
    cfg: Literal["indoor_navigation", "indoor", "overhanging", "overhanging_floor"] = "indoor_navigation"
    shape: tuple[int, int] = (5, 5)
    tile_dim: tuple[float, float, float] = (2.0, 2.0, 2.0)
    wall_height: float = 2.0
    initial_tile_name: str = "floor"
    over_cfg: bool = False
    overhanging_initial_tile_name: str = "walls_empty"
    enable_sdf: bool = False
    enable_history: bool = False
    save_collision_parts: bool = False
    sdf_resolution: float = 0.1
    use_boolean_merges: bool = False
    visualize: bool = False
    custom_primitives: dict[str, Any] | None = None

    num_rows: int = 1
    num_cols: int = 4
    """Number of cached WFC worlds tiled into the IsaacLab terrain grid."""

    origin_z: float = 0.0
    """Spawn origin height for each WFC sub-world."""

    spacing_margin: float = 0.0
    """Extra xy spacing between WFC sub-worlds."""

    load_cache: bool = True
    """Reuse already-generated OBJ files when the generation config hash matches."""

    def __post_init__(self):
        self.class_type = WfcTerrainGenerator


class WfcTerrainGenerator:
    """Generate a grid of WFC worlds and expose them as an IsaacLab terrain mesh."""

    terrain_mesh: Any
    terrain_meshes: list[Any]
    terrain_origins: np.ndarray
    flat_patches: dict[str, torch.Tensor]

    def __init__(self, cfg: WfcTerrainGeneratorCfg, device: str = "cpu"):
        self.cfg = cfg
        self.device = device
        self.flat_patches = {}
        self.terrain_meshes = []
        self.terrain_origins = np.zeros((cfg.num_rows, cfg.num_cols, 3), dtype=np.float32)

        trimesh = _import_trimesh()
        terrain_root = _resolve_terrain_generator_root(cfg.terrain_generator_root)
        cache_root = _resolve_repo_path(cfg.cache_dir)
        cache_root.mkdir(parents=True, exist_ok=True)

        generation_key = _generation_key(cfg, terrain_root)
        sub_size_x = float(cfg.shape[1]) * float(cfg.tile_dim[0]) + float(cfg.spacing_margin)
        sub_size_y = float(cfg.shape[0]) * float(cfg.tile_dim[1]) + float(cfg.spacing_margin)

        for row in range(cfg.num_rows):
            for col in range(cfg.num_cols):
                sub_seed = int(cfg.seed) + row * cfg.num_cols + col
                sub_dir = cache_root / generation_key / f"r{row:02d}_c{col:02d}_seed{sub_seed}"
                mesh_path = _ensure_wfc_mesh(cfg, terrain_root, sub_dir, sub_seed)

                mesh = trimesh.load(mesh_path, force="mesh")
                if not hasattr(mesh, "vertices") or len(mesh.vertices) == 0:
                    raise RuntimeError(f"WFC mesh is empty or invalid: {mesh_path}")

                x = (row - (cfg.num_rows - 1) * 0.5) * sub_size_x
                y = (col - (cfg.num_cols - 1) * 0.5) * sub_size_y
                mesh.apply_translation((x, y, 0.0))
                self.terrain_meshes.append(mesh)
                self.terrain_origins[row, col] = (x, y, float(cfg.origin_z))

        self.terrain_mesh = trimesh.util.concatenate(self.terrain_meshes)

    def __str__(self) -> str:
        return (
            "WFC Terrain Generator:"
            f"\n\tSeed: {self.cfg.seed}"
            f"\n\tGrid: {self.cfg.num_rows} x {self.cfg.num_cols}"
            f"\n\tWFC cfg: {self.cfg.cfg}"
            f"\n\tWFC shape: {self.cfg.shape}"
            f"\n\tTile dim: {self.cfg.tile_dim}"
            f"\n\tCache dir: {self.cfg.cache_dir}"
        )


WfcTerrainGeneratorCfg.class_type = WfcTerrainGenerator


def _ensure_wfc_mesh(
    cfg: WfcTerrainGeneratorCfg,
    terrain_root: Path,
    output_dir: Path,
    seed: int,
) -> Path:
    mesh_dir = output_dir / "mesh"
    mesh_path = mesh_dir / "mesh.obj"
    manifest_path = output_dir / "manifest.json"
    manifest = _manifest(cfg, terrain_root, seed)

    if cfg.load_cache and mesh_path.is_file() and manifest_path.is_file():
        try:
            if json.loads(manifest_path.read_text()) == manifest:
                return mesh_path
        except json.JSONDecodeError:
            pass

    output_dir.mkdir(parents=True, exist_ok=True)
    _generate_wfc_mesh(cfg, terrain_root, output_dir, seed)
    if not mesh_path.is_file():
        raise FileNotFoundError(f"terrain-generator did not write expected mesh: {mesh_path}")
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    return mesh_path


def _generate_wfc_mesh(cfg: WfcTerrainGeneratorCfg, terrain_root: Path, output_dir: Path, seed: int) -> None:
    examples_dir = terrain_root / "examples"
    generator_path = examples_dir / "generate_with_wfc.py"
    navigation_cfg_path = examples_dir / "configs" / "navigation_cfg.py"
    overhanging_cfg_path = examples_dir / "configs" / "overhanging_cfg.py"
    if not generator_path.is_file():
        raise FileNotFoundError(f"Missing terrain-generator example: {generator_path}")

    matplotlib_config_dir = output_dir / ".matplotlib"
    matplotlib_config_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(matplotlib_config_dir))

    np.random.seed(seed)
    random.seed(seed)

    with _temporary_sys_path(terrain_root, examples_dir), _temporary_working_directory(terrain_root):
        _install_optional_dependency_stubs()
        try:
            generator = _load_module(generator_path, f"legged_lab_wfc_example_{seed}")
            navigation_cfg = _load_module(navigation_cfg_path, f"legged_lab_wfc_navigation_cfg_{seed}")
            overhanging_cfg = _load_module(overhanging_cfg_path, f"legged_lab_wfc_overhanging_cfg_{seed}")
        except ImportError as exc:
            raise RuntimeError(
                "terrain-generator dependencies are not installed in this Python environment. "
                "Install the submodule package with `pip install -e third_party/terrain-generator` "
                "or run from an environment that already has its requirements."
            ) from exc

        if cfg.cfg in {"indoor", "indoor_navigation"}:
            pattern_cfg = navigation_cfg.IndoorNavigationPatternLevels(
                dim=cfg.tile_dim,
                seed=seed,
                wall_height=float(cfg.wall_height),
            )
        elif cfg.cfg == "overhanging":
            pattern_cfg = overhanging_cfg.OverhangingTerrainPattern(dim=cfg.tile_dim)
        elif cfg.cfg == "overhanging_floor":
            pattern_cfg = overhanging_cfg.OverhangingFloorPattern(dim=cfg.tile_dim)
        else:
            raise ValueError(f"Unsupported WFC cfg: {cfg.cfg}")

        custom_parts = build_custom_mesh_parts(cfg.custom_primitives, dim=cfg.tile_dim, seed=seed)
        if custom_parts:
            pattern_cfg.mesh_parts = tuple(pattern_cfg.mesh_parts) + custom_parts

        over_cfg = overhanging_cfg.OverhangingPattern() if cfg.over_cfg else None
        if not cfg.use_boolean_merges:
            _disable_boolean_merges(pattern_cfg)
            if over_cfg is not None:
                _disable_boolean_merges(over_cfg)

        generator.create_mesh_from_cfg(
            pattern_cfg,
            over_cfg,
            prefix="mesh",
            mesh_dir=str(output_dir),
            shape=tuple(int(value) for value in cfg.shape),
            initial_tile_name=str(cfg.initial_tile_name),
            overhanging_initial_tile_name=str(cfg.overhanging_initial_tile_name),
            visualize=bool(cfg.visualize),
            enable_history=bool(cfg.enable_history or cfg.save_collision_parts),
            enable_sdf=bool(cfg.enable_sdf),
            sdf_resolution=float(cfg.sdf_resolution),
        )


def _generation_key(cfg: WfcTerrainGeneratorCfg, terrain_root: Path) -> str:
    payload = _manifest(cfg, terrain_root, int(cfg.seed))
    payload.pop("seed")
    text = json.dumps(payload, sort_keys=True, default=str)
    return "wfc_" + hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]


def _manifest(cfg: WfcTerrainGeneratorCfg, terrain_root: Path, seed: int) -> dict[str, Any]:
    return {
        "terrain_generator_root": str(terrain_root),
        "terrain_generator_commit": _git_commit(terrain_root),
        "seed": int(seed),
        "cfg": str(cfg.cfg),
        "shape": [int(value) for value in cfg.shape],
        "tile_dim": [float(value) for value in cfg.tile_dim],
        "wall_height": float(cfg.wall_height),
        "initial_tile_name": str(cfg.initial_tile_name),
        "over_cfg": bool(cfg.over_cfg),
        "overhanging_initial_tile_name": str(cfg.overhanging_initial_tile_name),
        "enable_sdf": bool(cfg.enable_sdf),
        "enable_history": bool(cfg.enable_history),
        "save_collision_parts": bool(cfg.save_collision_parts),
        "sdf_resolution": float(cfg.sdf_resolution),
        "use_boolean_merges": bool(cfg.use_boolean_merges),
        "custom_primitives": cfg.custom_primitives,
    }


def _git_commit(path: Path) -> str | None:
    head_path = path / ".git"
    if head_path.is_file():
        gitdir_text = head_path.read_text().strip()
        if gitdir_text.startswith("gitdir:"):
            git_dir = (path / gitdir_text.split(":", 1)[1].strip()).resolve()
        else:
            git_dir = head_path
    elif head_path.is_dir():
        git_dir = head_path
    else:
        return None

    head_file = git_dir / "HEAD"
    if not head_file.is_file():
        return None
    head = head_file.read_text().strip()
    if head.startswith("ref:"):
        ref_file = git_dir / head.split(":", 1)[1].strip()
        return ref_file.read_text().strip() if ref_file.is_file() else None
    return head


def _resolve_terrain_generator_root(path: str | None) -> Path:
    if path is None:
        return (_repo_root() / "third_party" / "terrain-generator").resolve()
    return _resolve_repo_path(path)


def _resolve_repo_path(path: str) -> Path:
    expanded = Path(path).expanduser()
    if expanded.is_absolute():
        return expanded.resolve()
    return (_repo_root() / expanded).resolve()


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


def _import_trimesh() -> Any:
    try:
        import trimesh
    except ImportError as exc:
        raise RuntimeError(
            "trimesh is required for WFC terrain import. Install the terrain-generator requirements "
            "or run through the IsaacLab environment."
        ) from exc
    return trimesh


@contextmanager
def _temporary_sys_path(*paths: Path) -> Iterator[None]:
    original = list(sys.path)
    sys.path[:0] = [str(path) for path in paths]
    try:
        yield
    finally:
        sys.path[:] = original


@contextmanager
def _temporary_working_directory(path: Path) -> Iterator[None]:
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


def _load_module(path: Path, module_name: str) -> Any:
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _install_optional_dependency_stubs() -> None:
    """Provide tiny stubs for terrain-generator imports that are unused in our basic path."""
    if "alive_progress" not in sys.modules:
        try:
            import alive_progress  # noqa: F401
        except ImportError:
            alive_progress_stub = types.ModuleType("alive_progress")

            class _AliveBar:
                def __init__(self, *args, **kwargs):
                    pass

                def __enter__(self):
                    return lambda *args, **kwargs: None

                def __exit__(self, exc_type, exc, tb):
                    return False

            def _alive_it(iterable, *args, **kwargs):
                return iterable

            alive_progress_stub.alive_bar = _AliveBar
            alive_progress_stub.alive_it = _alive_it
            sys.modules["alive_progress"] = alive_progress_stub

    if "open3d" not in sys.modules:
        try:
            import open3d  # noqa: F401
        except ImportError:
            open3d_stub = types.ModuleType("open3d")
            open3d_stub.geometry = types.SimpleNamespace(TriangleMesh=type("TriangleMesh", (), {}))
            open3d_stub.t = types.SimpleNamespace(
                geometry=types.SimpleNamespace(RaycastingScene=type("RaycastingScene", (), {}))
            )
            sys.modules["open3d"] = open3d_stub

    if "perlin_numpy" not in sys.modules:
        try:
            import perlin_numpy  # noqa: F401
        except ImportError:
            perlin_stub = types.ModuleType("perlin_numpy")

            def _zeros(shape, *args, **kwargs):
                return np.zeros(shape, dtype=np.float64)

            perlin_stub.generate_perlin_noise_2d = _zeros
            perlin_stub.generate_fractal_noise_2d = _zeros
            sys.modules["perlin_numpy"] = perlin_stub


def _disable_boolean_merges(config_obj: Any) -> None:
    """Disable Blender-backed boolean mesh unions on terrain-generator configs."""
    seen: set[int] = set()

    def visit(value: Any) -> None:
        if value is None:
            return
        value_id = id(value)
        if value_id in seen:
            return
        seen.add(value_id)

        if hasattr(value, "minimal_triangles"):
            value.minimal_triangles = False
        if hasattr(value, "create_door"):
            value.create_door = False

        if isinstance(value, dict):
            children = value.values()
        elif isinstance(value, (list, tuple)):
            children = value
        else:
            children = []
            for attr in (
                "mesh_parts",
                "cfgs",
                "stairs",
                "wall",
                "overhanging_cfg_list",
                "floor_cfg",
                "mesh_cfg",
            ):
                if hasattr(value, attr):
                    children.append(getattr(value, attr))

        for child in children:
            visit(child)

    visit(config_obj)
