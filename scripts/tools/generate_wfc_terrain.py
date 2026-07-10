"""Generate cached WFC terrain meshes for the LeggedLab IsaacLab terrain adapter."""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher


def _tuple_arg(value: str, cast):
    return tuple(cast(part.strip()) for part in value.split(",") if part.strip())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--terrain-generator-root", default=None)
    parser.add_argument("--cache-dir", default="logs/wfc_terrain_cache")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cfg", default="indoor_navigation", choices=["indoor_navigation", "indoor", "overhanging", "overhanging_floor"])
    parser.add_argument("--shape", default="5,5", help="WFC wave shape as rows,cols.")
    parser.add_argument("--tile-dim", default="2.0,2.0,2.0", help="Tile dimensions as x,y,z meters.")
    parser.add_argument("--rows", type=int, default=1, help="Number of WFC sub-world rows in the IsaacLab terrain grid.")
    parser.add_argument("--cols", type=int, default=1, help="Number of WFC sub-world cols in the IsaacLab terrain grid.")
    parser.add_argument("--wall-height", type=float, default=2.0)
    parser.add_argument("--initial-tile", default="floor")
    parser.add_argument("--custom-primitives", action="store_true")
    parser.add_argument("--custom-weight", type=float, default=0.35)
    parser.add_argument("--over-cfg", action="store_true")
    parser.add_argument("--enable-sdf", action="store_true")
    parser.add_argument("--enable-history", action="store_true")
    parser.add_argument("--no-cache", action="store_true")
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()

    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app

    from legged_lab.terrains import WfcTerrainGenerator, WfcTerrainGeneratorCfg

    custom_primitives = None
    if args.custom_primitives:
        custom_primitives = {
            "enabled": True,
            "types": "all",
            "weight": args.custom_weight,
            "edge_height": 0.1,
        }

    cfg = WfcTerrainGeneratorCfg(
        terrain_generator_root=args.terrain_generator_root,
        cache_dir=args.cache_dir,
        seed=args.seed,
        cfg=args.cfg,
        shape=_tuple_arg(args.shape, int),
        tile_dim=_tuple_arg(args.tile_dim, float),
        wall_height=args.wall_height,
        initial_tile_name=args.initial_tile,
        over_cfg=args.over_cfg,
        enable_sdf=args.enable_sdf,
        enable_history=args.enable_history,
        custom_primitives=custom_primitives,
        num_rows=args.rows,
        num_cols=args.cols,
        load_cache=not args.no_cache,
    )
    try:
        terrain = WfcTerrainGenerator(cfg)
        print(terrain, flush=True)
        print(f"vertices: {len(terrain.terrain_mesh.vertices)}", flush=True)
        print(f"faces: {len(terrain.terrain_mesh.faces)}", flush=True)
        print(f"origins shape: {terrain.terrain_origins.shape}", flush=True)
        print(f"first origin: {terrain.terrain_origins.reshape(-1, 3)[0].tolist()}", flush=True)
    finally:
        simulation_app.close()


if __name__ == "__main__":
    main()
