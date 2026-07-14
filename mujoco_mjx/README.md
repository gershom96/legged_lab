# Native JAX PPO + AMP on MuJoCo/MJX

This directory contains a native JAX implementation of the current G1
perception-enabled split-policy task. MuJoCo/MJX runs physics and JAX runs the
actor, critic, PPO update, AMP discriminator, optimizers, replay sampling,
observation histories, rewards, resets, logging, and video policy evaluation.

PyTorch is not imported by the training runtime. It is used only by the
one-time `.pt` to `.npz` converter. The parity checker runs in the native JAX
environment against a reference fixture emitted by that converter.

## Model Contract

```text
policy observation:
  current 17 x 11 height scan             187
  5 x 114 packed full-observation frames  570
  total                                   757

actor:
  scan -> CNN [16, 32, 64] -> global average pool -> 64
  concat [570 history, 64 scan latent] -> MLP [512, 256, 128]
  lower head -> 15 actions
  upper head -> 14 actions

critic observation:
  5 x 17 x 11 clean height scans          935
  current privileged state                117
  total                                  1052

critic:
  scan history -> CNN [16, 32, 64] -> global average pool -> 64
  concat [117 state, 64 scan latent] -> MLP [512, 256, 128]
  output -> [lower value, upper value]

upper-body AMP discriminator:
  4 x 49 upper-body motion frames -> MLP [1024, 512] -> 1
  LSGAN + demonstration gradient penalty
```

The height scan is attached to MuJoCo's `torso_link`, yaw-aligned with that
body, and reports:

```text
torso_z - terrain_top_z - 0.5
```

The actor gets the current noisy scan. The critic gets five clean scan frames.
Noise matches the Isaac task:

```text
base angular velocity  +/- 0.35
projected gravity      +/- 0.05
relative joint pose    +/- 0.03
joint velocity         +/- 1.75
height scan            +/- 0.01
```

The network always consumes and emits the Isaac joint order used during
training. At the physics boundary, the 29 actor outputs are gathered into
MuJoCo actuator order by matching joint names. MuJoCo position, velocity,
acceleration, and torque vectors are gathered back into Isaac order before
they enter observations or rewards. AMP motion files must declare the
converted Isaac ordering.

## Environment

The MJX dependencies live in a repository-owned environment. They are kept
separate because Isaac's root `.venv` pins NumPy 1.26 while current JAX pins
NumPy 2.x:

```bash
uv sync --project mujoco_mjx
MJX_PY=$PWD/mujoco_mjx/.venv/bin/python
```

The root `.venv` remains the Isaac/PyTorch environment and is used only for
conversion. Native training uses `mujoco_mjx/.venv`, which deliberately does
not install or import PyTorch.

## Convert A Checkpoint

The recommended continuation checkpoint is the already-split `model_4400.pt`:

```bash
.venv/bin/python mujoco_mjx/scripts/convert_pt_to_jax.py \
  logs/rsl_rl/g1_split_policy_heightscan/2026-07-09_14-54-49/model_4400.pt \
  --output mujoco_mjx/outputs/checkpoints/model_4400_jax.npz \
  --verification-fixture mujoco_mjx/outputs/checkpoints/model_4400_fixture.npz
```

The `.npz` contains actor, two-output critic, action standard deviations, AMP
discriminator, and discriminator normalizer. A converted checkpoint is a warm
start, so native Adam states and the native iteration counter start fresh.

Verify the framework handoff without installing PyTorch in the MJX environment:

```bash
$MJX_PY mujoco_mjx/scripts/verify_pt_jax_parity.py \
  mujoco_mjx/outputs/checkpoints/model_4400_jax.npz \
  mujoco_mjx/outputs/checkpoints/model_4400_fixture.npz
```

The checker compares actor, critic, action standard deviation, and AMP
discriminator outputs. Its default absolute tolerance is `1e-4`.

Checkpoint lineage is:

```text
g1_mixed_amp_height_scan/2026-07-08_12-47-56/model_1000.pt
  -> g1_mixed_amp_height_scan/2026-07-08_13-55-57/model_33400.pt
  -> g1_split_policy_heightscan/2026-07-09_14-54-49/model_4400.pt
```

To start from the stable pre-split `model_33400.pt` instead, split its final
actor layer by joint rows, duplicate its scalar value output, and obtain the
compatible upper-body discriminator from the split run's `model_0.pt`:

```bash
.venv/bin/python mujoco_mjx/scripts/convert_pt_to_jax.py \
  logs/rsl_rl/g1_mixed_amp_height_scan/2026-07-08_13-55-57/model_33400.pt \
  --split-template logs/rsl_rl/g1_split_policy_heightscan/2026-07-09_14-54-49/model_0.pt \
  --std-source checkpoint \
  --output mujoco_mjx/outputs/checkpoints/model_33400_split_source_std_jax.npz \
  --verification-fixture mujoco_mjx/outputs/checkpoints/model_33400_split_source_std_fixture.npz
```

`--std-source checkpoint` preserves the high checkpoint's learned 29-action
noise. `--std-source split-config` instead resets lower-body noise to `1.0`
and upper-body noise to `0.6`, matching the original Isaac split warm start.
The old 88-feature full-body discriminator cannot be loaded into the current
49-feature upper-body discriminator and is explicitly ignored in this path.

## Smoke Test

Run a complete rollout and PPO+AMP update on CPU:

```bash
$MJX_PY mujoco_mjx/scripts/train_native_jax.py \
  --checkpoint mujoco_mjx/outputs/checkpoints/model_4400_jax.npz \
  --device cpu --mjx-impl jax \
  --num-envs 2 --motion-max-files 1 \
  --smoke-test --smoke-steps 2 --logger none
```

CPU `mjx-impl=jax` is a validation mode. It retains foot contact primitives but
disables unsupported G1 mesh collision pairs. Use CUDA/Warp for training.

## Train

Start native training from the converted checkpoint:

```bash
$MJX_PY mujoco_mjx/scripts/train_native_jax.py \
  --checkpoint mujoco_mjx/outputs/checkpoints/model_4400_jax.npz \
  --device cuda:0 --mjx-impl warp \
  --num-envs 4096 --max-iterations 50000 \
  --logger wandb \
  --video --video-interval 200 --video-length 300
```

MJX-Warp uses a global contact pool. The terrain configuration allocates 68 contact
slots per environment, so 4096 environments use `naconmax=278528`. This includes
headroom above the observed 66-contact-per-environment broadphase peak.

Rollout collection follows the Embodiment-Aware-Nav execution pattern: expert
samples are scheduled on the host, then all 24 policy steps run inside one
compiled `jax.lax.scan`. Video copies one selected world's state to the host
once per rollout and renders only after physics has finished.

PPO still uses the configured four logical mini-batches and five epochs. Each
24,576-sample logical mini-batch is evaluated in six 4,096-sample gradient
micro-batches; their gradients are averaged before one Adam update. This bounds
JAX activation memory without changing the number of optimizer updates or the
relative PPO, value, symmetry, and AMP loss weights. `micro_batch_size: 4096`
is the native-backend memory setting. `--micro-batch-size` can override it for
diagnostics; `--num-mini-batches` changes the PPO hyperparameter itself.

Videos are written to `RUN/videos/train/`. With `--logger wandb`, completed
MP4s are also uploaded to the run.

## Resume

Resume a native run from its `.npz` checkpoint:

```bash
$MJX_PY mujoco_mjx/scripts/train_native_jax.py \
  --resume logs/rsl_rl/g1_mujoco_mjx_split_policy_heightscan_native_jax/RUN/model_200.npz \
  --device cuda:0 --mjx-impl warp \
  --num-envs 4096 --max-iterations 50000 \
  --logger wandb \
  --video --video-interval 200 --video-length 300
```

Native resume restores policy, critic, action standard deviations,
discriminator, normalizer, both Adam states, learning rate, iteration, terrain
curriculum stage, stage-start iteration, and the rolling promotion window. The
environment state and AMP replay window restart, so continuation is not a
bit-for-bit replay of the interrupted process.

## Terrain

Built-in seamless heightfields include:

```text
flat
random_rough_mild, random_rough
wave_mild, wave
pyramid_slope, pyramid_slope_inv
pyramid_stairs, pyramid_stairs_inv
boxes_low, discrete_obstacles
stepping_stones, gap, pit, rails, box
```

The default config enables a global six-stage curriculum:

```text
stage 0  flat
stage 1  random_rough_mild, wave_mild
stage 2  random_rough, discrete_obstacles, boxes_low
stage 3  pyramid_slope, pyramid_slope_inv, wave
stage 4  pyramid_stairs, pyramid_stairs_inv
stage 5  stepping_stones, gap, pit, rails
```

Each independent world contains the same static 4x4 board of 45 m patches and
one robot. Patch edges blend to a flat border, and crossing the configured 20 m
safe half-extent causes a timeout reset without a termination penalty. On each
reset, stages 1-5 sample 70% from the current stage, 20% from the previous
stage, and 10% from easier replay terrain. Stage 0 uses a finite flat collision
slab so its contact behavior matches flat-ground warm-start training.

The curriculum stage is global. It advances after a 100-iteration moving
window satisfies all configured conditions and the stage has lasted at least
400 iterations: mean episode length at least 900 steps, linear tracking at
least 0.60, angular tracking at least 0.20, and termination rate at most 5%.
Promotion resets every world onto the new stage distribution; there is no
automatic demotion.

Select a terrain and continuous difficulty from `0.0` to `1.0`:

```bash
$MJX_PY mujoco_mjx/scripts/train_native_jax.py \
  --checkpoint mujoco_mjx/outputs/checkpoints/model_4400_jax.npz \
  --terrain pyramid_stairs --terrain-difficulty 0.5 \
  --device cuda:0 --mjx-impl warp --num-envs 4096 \
  --max-iterations 50000 --logger wandb --video
```

Embodiment-Aware-Nav occupancy scenes are converted into physical ground,
collision boxes, and a top-down height surface:

```bash
$MJX_PY mujoco_mjx/scripts/train_native_jax.py \
  --checkpoint mujoco_mjx/outputs/checkpoints/model_4400_jax.npz \
  --terrain ean \
  --ean-scene-config /home/gershom/Documents/gershom96/Embodiment-Aware-Nav/configs/scene/random_clutter.yaml \
  --device cuda:0 --mjx-impl warp --num-envs 4096 \
  --max-iterations 50000 --logger wandb --video
```

`wfc` delegates generation to the Embodiment-Aware-Nav adapter and its local
`third_party/terrain-generator` checkout:

```bash
$MJX_PY mujoco_mjx/scripts/train_native_jax.py \
  --checkpoint mujoco_mjx/outputs/checkpoints/model_4400_jax.npz \
  --terrain wfc \
  --ean-scene-config /home/gershom/Documents/gershom96/Embodiment-Aware-Nav/configs/scene/wfc_navigation.yaml \
  --device cuda:0 --mjx-impl warp --num-envs 4096 \
  --max-iterations 50000 --logger wandb --video
```

Each vectorized MJX world has one robot and independent physics state. Robots
in different worlds cannot see or collide with one another. Passing
`--terrain NAME` explicitly disables the curriculum and runs one homogeneous
terrain family instead.

## Regression Tests

```bash
$MJX_PY -m unittest discover -s mujoco_mjx/tests -v
```

The suite covers dimensions, joint permutation, terrain generation, height-scan
symmetry, torso scanner attachment, effort limits, EAN occupancy conversion,
AMP motion ordering, and long-run discriminator-normalizer count stability.
