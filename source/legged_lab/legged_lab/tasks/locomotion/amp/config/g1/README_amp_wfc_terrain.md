# AMP G1 Height-Scan WFC Terrain Integration

This integration adds the terrain-generator WFC repo as:

```text
third_party/terrain-generator
```

and exposes it through an IsaacLab terrain generator wrapper:

```text
legged_lab.terrains.WfcTerrainGeneratorCfg
legged_lab.terrains.WfcTerrainGenerator
```

The new Gym task is:

```text
LeggedLab-Isaac-AMP-G1-Mixed-HeightScan-WFC-v0
```

This does not change the existing flat height-scan task.

## What It Does

The wrapper calls terrain-generator's WFC mesh generation, caches the exported OBJ meshes, loads them with
`trimesh`, tiles them into an IsaacLab terrain grid, and provides `terrain_origins` to IsaacLab.

The current WFC task keeps the same actor/critic height-scan policy as the flat height-scan task, but replaces
the plane with cached WFC meshes.

RSI is disabled for the WFC task because reference-state initialization is not terrain-height aware yet.

## Smoke Generate

Run this before training to verify dependencies and mesh generation:

```bash
env -u CONDA_PREFIX -u CONDA_DEFAULT_ENV TERM=xterm \
  VIRTUAL_ENV=/home/gershom/Documents/IsaacLab/env_isaaclab \
  PYTHONPATH=/home/gershom/Documents/gershom96/legged_lab/source/legged_lab \
  /home/gershom/Documents/IsaacLab/isaaclab.sh -p scripts/tools/generate_wfc_terrain.py \
  --rows 1 \
  --cols 1 \
  --shape 5,5
```

Generated meshes are cached by default under:

```text
logs/wfc_terrain_cache
```

## Training Command

Start small first:

```bash
env -u CONDA_PREFIX -u CONDA_DEFAULT_ENV TERM=xterm \
  VIRTUAL_ENV=/home/gershom/Documents/IsaacLab/env_isaaclab \
  PYTHONPATH=/home/gershom/Documents/gershom96/legged_lab/rsl_rl:/home/gershom/Documents/gershom96/legged_lab/source/legged_lab \
  LEGGED_LAB_WFC_NUM_ROWS=2 \
  LEGGED_LAB_WFC_NUM_COLS=4 \
  LEGGED_LAB_WFC_SHAPE=5,5 \
  /home/gershom/Documents/IsaacLab/isaaclab.sh -p scripts/rsl_rl/train.py \
  --task LeggedLab-Isaac-AMP-G1-Mixed-HeightScan-WFC-v0 \
  --headless \
  --num_envs 1024 \
  --max_iterations 50000 \
  --video \
  --video_interval 4800 \
  --video_length 200
```

## Useful Overrides

```text
LEGGED_LAB_WFC_TERRAIN_GENERATOR_ROOT=third_party/terrain-generator
LEGGED_LAB_WFC_CACHE_DIR=logs/wfc_terrain_cache
LEGGED_LAB_WFC_SEED=0
LEGGED_LAB_WFC_CFG=indoor_navigation
LEGGED_LAB_WFC_SHAPE=5,5
LEGGED_LAB_WFC_TILE_DIM=2.0,2.0,2.0
LEGGED_LAB_WFC_NUM_ROWS=2
LEGGED_LAB_WFC_NUM_COLS=4
LEGGED_LAB_WFC_WALL_HEIGHT=2.0
LEGGED_LAB_WFC_CUSTOM_PRIMITIVES=0
LEGGED_LAB_WFC_OVER_CFG=0
LEGGED_LAB_WFC_ENABLE_SDF=0
LEGGED_LAB_WFC_LOAD_CACHE=1
```

Enable custom clutter tiles:

```text
LEGGED_LAB_WFC_CUSTOM_PRIMITIVES=1
LEGGED_LAB_WFC_CUSTOM_PRIMITIVE_TYPES=all
LEGGED_LAB_WFC_CUSTOM_PRIMITIVE_WEIGHT=0.35
```

## Caution

WFC worlds can include vertical walls, tunnels, and overhangs. The current policy still only receives a downward
height scan, so this task is an integration path and stress test. Real WFC navigation will probably need extra
perception such as depth, LiDAR, occupancy, or a traversability/goal-conditioning layer.
