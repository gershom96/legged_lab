# Isaac Terrain Experts

This is the reproducible Isaac Sim workflow for training perception-enabled G1 terrain experts. An expert can cover one terrain family or a deliberately compatible group of families.

The expert task is:

```text
LeggedLab-Isaac-AMP-G1-SplitPolicy-HeightScan-TerrainCurriculum-v0
```

It has the split-height-scan policy, current AMP/reward configuration, and the same actor/critic shapes as the flat split-policy task. The terrain-only environment variables below reduce the board to one row containing only the requested terrain family.

## What The Tile Edge Blend Does

Each generated tile is `45 x 45 m` by default. Before placing a tile into the continuous IsaacLab board, all terrain geometry in its outer `2.5 m` rim is flattened to `z=0`. Neighboring tiles therefore meet through a flat contact band; there are no overlapping collision meshes or discontinuous tile-edge heights.

The robot is reset if it leaves the central `+/-20 m` region of its tile. That leaves a `2.5 m` margin before the blended rim and is configured as a timeout, not a termination penalty.

Relevant environment variables:

```text
LEGGED_LAB_TERRAIN_SIZE=45,45
LEGGED_LAB_TERRAIN_EDGE_BLEND_WIDTH=2.5
LEGGED_LAB_TERRAIN_ALLOWED_HALF_EXTENT=20
```

Do not set the blend width to zero for normal training. `0` is useful only to deliberately reproduce seam behavior in a diagnostic run.

## One-Time Installation

This repository expects Isaac Lab `2.3.1` with Isaac Sim `5.1`, a supported NVIDIA driver, Git, Git LFS, and a CUDA-capable GPU.

### 1. Clone The Repository And Both Submodules

```bash
git clone --recurse-submodules <your-legged-lab-repository-url> legged_lab
cd legged_lab
git submodule update --init --recursive
git lfs install
git lfs pull
```

The required submodules are `rsl_rl` and `third_party/terrain-generator`. The terrain experts below do not require WFC generation, but retaining the terrain-generator submodule keeps the clone complete.

### 2. Install Isaac Lab

Install the matching Isaac Lab/Isaac Sim release following the Isaac Lab installer. For a source checkout, create the default environment and install its extensions:

```bash
cd <IsaacLab>
./isaaclab.sh --conda env_isaaclab
./isaaclab.sh --install
```

Set these paths after both repositories exist:

```bash
export ISAACLAB_ROOT=/absolute/path/to/IsaacLab
export LEGGED_LAB_ROOT=/absolute/path/to/legged_lab
```

### 3. Install Legged Lab And Its RSL-RL Fork Into Isaac Lab's Python

The `env -u` prefix is important when a Conda environment is active: without it, `isaaclab.sh` can select the wrong Python and fail with `ModuleNotFoundError: isaaclab`.

```bash
env -u CONDA_PREFIX -u CONDA_DEFAULT_ENV \
  VIRTUAL_ENV="$ISAACLAB_ROOT/env_isaaclab" \
  "$ISAACLAB_ROOT/isaaclab.sh" -p -m pip install -e "$LEGGED_LAB_ROOT/source/legged_lab"

env -u CONDA_PREFIX -u CONDA_DEFAULT_ENV \
  VIRTUAL_ENV="$ISAACLAB_ROOT/env_isaaclab" \
  "$ISAACLAB_ROOT/isaaclab.sh" -p -m pip install -e "$LEGGED_LAB_ROOT/rsl_rl"
```

Use this helper for every later command. It pins imports to the clone and the local RSL-RL fork even if another package with the same name is installed.

```bash
run_isaac() {
  env -u CONDA_PREFIX -u CONDA_DEFAULT_ENV \
    VIRTUAL_ENV="$ISAACLAB_ROOT/env_isaaclab" \
    PYTHONPATH="$LEGGED_LAB_ROOT/rsl_rl:$LEGGED_LAB_ROOT/source/legged_lab${PYTHONPATH:+:$PYTHONPATH}" \
    "$ISAACLAB_ROOT/isaaclab.sh" -p "$@"
}
```

Verify that the Isaac Lab launcher resolves from the intended environment:

```bash
cd "$LEGGED_LAB_ROOT"
run_isaac scripts/rsl_rl/train.py --help
```

The training command prints the resolved Isaac Lab, Legged Lab, and RSL-RL import paths when a real job starts. They must point into the intended checkouts.

## Train A Terrain Expert

Choose a trained flat split-policy checkpoint as the warm start. `EXPERT_TERRAINS` is a comma-separated group; the generator places the listed families in equal proportions across the tile columns.

```bash
export WARM_START=/absolute/path/to/model_XXXX.pt
export EXPERT_NAME=stairs
export EXPERT_TERRAINS=pyramid_stairs,pyramid_stairs_inv
export TERRAIN_DIFFICULTY=0.35,0.65
export TERRAIN_NUM_COLS=4
export MOTION_CACHE="$HOME/Documents/shared_datasets/legged_lab_g1_mixed_default_motionbricks_cache.pt"
export WANDB_PROJECT=g1_split_policy_heightscan_experts
```

Define this helper once in the same shell:

```bash
cd "$LEGGED_LAB_ROOT"

train_expert() {
  LEGGED_LAB_TERRAIN_STAGE_NAMES="$EXPERT_TERRAINS" \
  LEGGED_LAB_TERRAIN_NUM_COLS="$TERRAIN_NUM_COLS" \
  LEGGED_LAB_TERRAIN_STAGE_DIFFICULTY_RANGES="$TERRAIN_DIFFICULTY" \
  LEGGED_LAB_TERRAIN_SIZE=45,45 \
  LEGGED_LAB_TERRAIN_EDGE_BLEND_WIDTH=2.5 \
  LEGGED_LAB_TERRAIN_ALLOWED_HALF_EXTENT=20 \
  LEGGED_LAB_MIXED_G1_AMP_CACHE="$MOTION_CACHE" \
  run_isaac scripts/rsl_rl/train.py \
    --task LeggedLab-Isaac-AMP-G1-SplitPolicy-HeightScan-TerrainCurriculum-v0 \
    --headless \
    --num_envs 4096 \
    --max_iterations 50000 \
    --warm_start \
    --warm_start_checkpoint "$WARM_START" \
    --experiment_name g1_split_policy_heightscan_experts \
    --run_name "${EXPERT_NAME}_seed0" \
    --logger wandb \
    --log_project_name "$WANDB_PROJECT"
}
```

Run the selected expert with:

```bash
train_expert
```

On base-station, use Fabric (the default) with `--num_envs 4096`. The original open-mesh gap terrain stalled this configuration; the height-field trench replacement removes that stall. `--disable_fabric` is a diagnostic fallback only. The renderer/video path still needs a separate smoke test, so do not add `--video`, `--eval_video`, or curriculum-video environment variables to a long base-station training job yet.

For hosted Weights & Biases logging, authenticate once on the training machine before launching an expert:

```bash
uv run wandb login
```

For the footholds expert, use the three discontinuous/depressed terrain families with six columns so each family occupies two columns:

```bash
EXPERT_NAME=footholds \
EXPERT_TERRAINS=stepping_stones,gap,pit \
TERRAIN_DIFFICULTY=0.55,1.00 \
TERRAIN_NUM_COLS=6 \
train_expert
```

`--warm_start` loads the matching actor and critic only. It deliberately starts PPO optimizer state, the AMP discriminator, discriminator normalizer, and discriminator optimizer fresh for the new terrain distribution. Use `--resume --load_run ... --checkpoint ...` only to continue the *same* terrain expert after interruption.

The warm-start checkpoint must be from the same split-policy height-scan model layout. A non-split or no-height-scan checkpoint has incompatible parameter shapes and must not be used here.

`MOTION_CACHE` is optional but recommended for the mixed MotionBricks dataset. On the first run, the normal loader creates a validated packed tensor cache after loading the source clips. Later runs verify the ordered source-file manifest, then load the single cache file instead of deserializing thousands of `.pkl` files. Delete the cache or let it invalidate automatically when source clips change.

## Recommended Expert Set

Run these six commands separately. They all warm-start from the same strong flat checkpoint, but their PPO and AMP state remain independent. `TERRAIN_NUM_COLS` is a multiple of the number of families, so each family receives equal tile coverage.

| Expert | Terrain families | Skill being isolated |
| --- | --- | --- |
| `continuous` | `random_rough_mild`, `wave_mild`, `random_rough`, `wave` | Adapt foot placement and torso control to smooth and irregular height variation without discontinuities. |
| `low_obstacles` | `discrete_obstacles`, `boxes_low` | Clear low raised geometry while retaining forward velocity and a normal gait. |
| `slopes` | `pyramid_slope`, `pyramid_slope_inv` | Walk both uphill and downhill while controlling trunk pitch and foot loading. |
| `stairs` | `pyramid_stairs`, `pyramid_stairs_inv` | Learn stepped ascent and descent rather than only continuous slope traversal. |
| `footholds` | `stepping_stones`, `gap`, `pit` | Select reliable landing locations and cross discontinuous or depressed ground. |
| `rails` | `rails` | Maintain lateral balance and accurate foot placement on narrow raised supports. |

Within those groups, the terrain names mean:

- `random_rough_mild` / `random_rough`: increasingly large, irregular height perturbations.
- `wave_mild` / `wave`: increasingly large smooth undulations.
- `discrete_obstacles`: sparse low blocks; `boxes_low`: denser low raised grid geometry.
- `pyramid_slope` / `pyramid_slope_inv`: sloped terrain in opposite elevation layouts, giving both uphill and downhill exposure.
- `pyramid_stairs` / `pyramid_stairs_inv`: stepped terrain in opposite elevation layouts, giving both ascent and descent exposure.
- `stepping_stones`: separated landing regions over holes; `gap`: traversable breaks in the surface; `pit`: depressed terrain.
- `rails`: thin raised supports that penalize lateral foot-placement error.

```bash
# 1. Continuous uneven ground
EXPERT_NAME=continuous \
EXPERT_TERRAINS=random_rough_mild,wave_mild,random_rough,wave \
TERRAIN_DIFFICULTY=0.15,0.60 TERRAIN_NUM_COLS=8 \
train_expert

# 2. Low obstacles
EXPERT_NAME=low_obstacles \
EXPERT_TERRAINS=discrete_obstacles,boxes_low \
TERRAIN_DIFFICULTY=0.15,0.35 TERRAIN_NUM_COLS=4 \
train_expert

# 3. Slopes: ascent and descent
EXPERT_NAME=slopes \
EXPERT_TERRAINS=pyramid_slope,pyramid_slope_inv \
TERRAIN_DIFFICULTY=0.35,0.60 TERRAIN_NUM_COLS=4 \
train_expert

# 4. Stairs: ascent and descent
EXPERT_NAME=stairs \
EXPERT_TERRAINS=pyramid_stairs,pyramid_stairs_inv \
TERRAIN_DIFFICULTY=0.45,0.80 TERRAIN_NUM_COLS=4 \
train_expert

# 5. Footholds and discontinuities
EXPERT_NAME=footholds \
EXPERT_TERRAINS=stepping_stones,gap,pit \
TERRAIN_DIFFICULTY=0.55,1.00 TERRAIN_NUM_COLS=6 \
train_expert

# 6. Rails
EXPERT_NAME=rails \
EXPERT_TERRAINS=rails \
TERRAIN_DIFFICULTY=0.55,1.00 TERRAIN_NUM_COLS=4 \
train_expert
```

Start with `continuous`, then `slopes` and `stairs`. Train `footholds` and `rails` only after those are stable.

The helper enables one focused video per terrain family in an expert group. It temporarily hides the other robot clones on each recorded tile, then restores them when recording ends. For example, the continuous expert records one clean following-camera video for each of `random_rough_mild`, `wave_mild`, `random_rough`, and `wave`.

## Continue Or Inspect A Run

Continue the exact same expert, preserving PPO and AMP state:

```bash
LEGGED_LAB_TERRAIN_ONLY=pyramid_stairs \
LEGGED_LAB_TERRAIN_NUM_ROWS=1 \
LEGGED_LAB_TERRAIN_NUM_COLS=4 \
LEGGED_LAB_TERRAIN_STAGE_DIFFICULTY_RANGES=0.35,0.65 \
LEGGED_LAB_TERRAIN_SIZE=45,45 \
LEGGED_LAB_TERRAIN_EDGE_BLEND_WIDTH=2.5 \
LEGGED_LAB_TERRAIN_ALLOWED_HALF_EXTENT=20 \
run_isaac scripts/rsl_rl/train.py \
  --task LeggedLab-Isaac-AMP-G1-SplitPolicy-HeightScan-TerrainCurriculum-v0 \
  --headless \
  --num_envs 4096 \
  --max_iterations 50000 \
  --resume \
  --experiment_name g1_split_policy_heightscan_experts \
  --load_run <run-directory> \
  --checkpoint model_XXXX.pt
```

Play a trained expert and record a short video:

```bash
LEGGED_LAB_TERRAIN_ONLY=pyramid_stairs \
LEGGED_LAB_TERRAIN_NUM_ROWS=1 \
LEGGED_LAB_TERRAIN_NUM_COLS=4 \
LEGGED_LAB_TERRAIN_STAGE_DIFFICULTY_RANGES=0.35,0.65 \
LEGGED_LAB_TERRAIN_SIZE=45,45 \
LEGGED_LAB_TERRAIN_EDGE_BLEND_WIDTH=2.5 \
LEGGED_LAB_TERRAIN_ALLOWED_HALF_EXTENT=20 \
run_isaac scripts/rsl_rl/play.py \
  --task LeggedLab-Isaac-AMP-G1-SplitPolicy-HeightScan-TerrainCurriculum-v0 \
  --headless \
  --num_envs 16 \
  --checkpoint /absolute/path/to/model_XXXX.pt \
  --video \
  --video_length 600
```

Training logs and checkpoints are written to:

```text
logs/rsl_rl/g1_split_policy_heightscan_experts/<timestamp>_<terrain>_seed0/
```

## Guardrails

- Keep `LEGGED_LAB_TERRAIN_NUM_ROWS=1` for an individual expert. Setting it above one repeats that terrain into additional board rows; it does not create a curriculum.
- With one row, set `LEGGED_LAB_TERRAIN_STAGE_DIFFICULTY_RANGES`; otherwise the implicit first-stage range is `(0, 0)` and the expert uses only the easiest version of its terrain.
- Keep `LEGGED_LAB_TERRAIN_ALLOWED_HALF_EXTENT <= terrain_size / 2 - edge_blend_width`. The default `20 <= 22.5 - 2.5` places the reset boundary exactly at the start of the flat rim.
- Do not change observation order, history length, actor/critic architecture, or reward weights between the flat warm start and a terrain expert.
- The out-of-tile condition is a timeout reset, so it does not receive the termination penalty.
