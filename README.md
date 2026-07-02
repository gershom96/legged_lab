# 🤖 Legged Lab

[![IsaacSim](https://img.shields.io/badge/IsaacSim-5.1.0-silver.svg)](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/index.html)
[![Isaac Lab](https://img.shields.io/badge/IsaacLab-2.3.1-silver)](https://isaac-sim.github.io/IsaacLab/v2.3.1/index.html)
[![Python](https://img.shields.io/badge/python-3.11-blue.svg)](https://docs.python.org/3/whatsnew/3.11.html)
[![Linux platform](https://img.shields.io/badge/platform-linux--64-orange.svg)](https://releases.ubuntu.com/20.04/)
[![Windows platform](https://img.shields.io/badge/platform-windows--64-orange.svg)](https://www.microsoft.com/en-us/)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://pre-commit.com/)
[![License](https://img.shields.io/badge/license-MIT-yellow.svg)](https://opensource.org/license/mit)

## 📖 Overview

This repository is an extension for legged robot reinforcement learning based on Isaac Lab, which allows to develop in an isolated environment, outside of the core Isaac Lab repository. The RL algorithm is based on a [forked RSL-RL library](https://github.com/zitongbai/rsl_rl/tree/feature/amp). 

This project is originally developed by [zitongbai](https://github.com/zitongbai).

**Key Features:**

- `DeepMimic` for humanoid robots, including Unitree G1.
- `AMP` Adversarial Motion Priors (AMP) for humanoid robots, including Atom01, Unitree G1. We suggest retargeting the human motion data by [GMR](https://github.com/YanjieZe/GMR).

## Demo

* Adversarial Motion Priors for Unitree G1:

https://github.com/user-attachments/assets/ed84a8a3-f349-44ac-9cfd-2baab2265a25

## 🔥 News & Updates

- 2026/01/06: Add Atom01 open-source robot from Roboparty(2 version, short and long base link).
- 2025/12/16: Test in Isaac Lab 2.3.1 and RSL-RL 3.2.0. 
- 2025/12/05: Use git lfs to store large files, including motion data and robot models.
- 2025/11/23: Add Symmetry data augmentation in AMP training.
- 2025/11/22: New implementation of AMP. 
- 2025/11/19: Add DeepMimic for G1. 
- 2025/10/14: Update to support rsl_rl v3.1.1. Only walking in flat terrain is supported now.
- 2025/08/24: Support using more steps observations and motion data in AMP training.
- 2025/08/22: Compatible with Isaac Lab 2.2.0.
- 2025/08/21: Add support for retargeting human motion data by [GMR](https://github.com/YanjieZe/GMR).

## ⚙️ Installation

### Prerequisites

- **uv**: Required for creating the project Python environment from `pyproject.toml` and `uv.lock`.
- **Git LFS**: Required for downloading large model files.
- **NVIDIA GPU driver**: Required for Isaac Sim / Isaac Lab training with CUDA.

### Setup Steps

1.  **Clone the Repository**
    Clone this repository with its `rsl_rl` submodule.

    ```bash
    # Option 1: HTTPS
    git clone --recurse-submodules https://github.com/zerojuhao/legged_lab
    
    # Option 2: SSH
    git clone --recurse-submodules git@github.com:zerojuhao/legged_lab.git
    
    cd legged_lab
    ```

    If you already cloned without submodules, initialize them with:

    ```bash
    git submodule update --init --recursive
    ```

2.  **Create the Python Environment**
    Use `uv` to create the local `.venv` and install the locked dependencies, including Isaac Lab, Isaac Sim, the local `legged_lab` package, and the forked `rsl_rl` submodule.

    ```bash
    uv sync
    ```

3.  **Run Commands Through the Environment**

    ```bash
    uv run python scripts/rsl_rl/train.py -h
    ```

## 🚀 Usage

### 1. Prepare Motion Data

We have already provided some off-the-shelf motion data in the `source/legged_lab/legged_lab/data/MotionData` folder for testing. 

If you want to add more motion data, you can do so by following the steps below.

1. Retarget human motion data to the robot model. We recommend using [GMR](https://github.com/YanjieZe/GMR) for retargeting human motion data. 
2. Put the retargeted motion data in the `temp/gmr_data` folder. 
3. Use a helper script to convert the motion data to the required format:

    ```bash
    python scripts/tools/retarget/dataset_retarget.py \
        --robot g1 \
        --input_dir temp/gmr_data/ \
        --output_dir temp/lab_data/ \
        --config_file scripts/tools/retarget/config/g1_29dof.yaml \
        --loop clamp
    ```
4. Move the converted data from `temp/lab_data` to `source/legged_lab/legged_lab/data/MotionData`, and set the `MotionDataCfg` in the config file, e.g., `source/legged_lab/legged_lab/tasks/locomotion/amp/config/g1/g1_amp_env_cfg.py`. 

Please refer to the comments in the script for more details about the arguments, and refer to `scripts/tools/retarget/gmr_to_lab.py` for the data format used in this repository.

### 2. Training & Play

#### 🎭 DeepMimic

<details>
<summary>Train</summary>

To train the DeepMimic algorithm, you can run the following command:

```bash
python scripts/rsl_rl/train.py --task LeggedLab-Isaac--Deepmimic-G1-v0 --headless --max_iterations 50000
```

The `max_iterations` can be adjusted based on your needs. For more details about the arguments, run `python scripts/rsl_rl/train.py -h`.

</details>

<details>
<summary>Play</summary>

You can play the trained model in a headless mode and record the video: 

```bash
# replace the checkpoint path with the path to your trained model
python scripts/rsl_rl/play.py --task LeggedLab-Isaac-Deepmimic-G1-v0 --headless --num_envs 64 --video --checkpoint logs/rsl_rl/experiment_name/run_name/model_xxx.pt
```

</details>


#### 🏃 Adversarial Motion Priors (AMP)

##### MotionBricks G1

Convert the MotionBricks `.npz` files into the Legged Lab AMP motion format:

```bash
uv run python scripts/tools/motionbricks/convert_motionbricks_to_legged_lab_fast.py \
  --workers 22
```

Then train with the MotionBricks task:

```bash
uv run python scripts/rsl_rl/train.py \
  --task LeggedLab-Isaac-AMP-G1-MotionBricks-v0 \
  --headless \
  --max_iterations 50000 \
  --num_envs 1024
```

This task uses Weights & Biases logging by default with project `g1_motionbricks_amp`.
Use `--logger tensorboard` to disable WandB for a run, or `--log_project_name <project>` to choose another WandB project.

By default, the converter reads from `~/Documents/shared_datasets/motionbricks/motionbricks_sonic_grid_walk_dense/motions`
and writes converted `.pkl` files to `~/Documents/shared_datasets/motionbricks/motionbricks_sonic_grid_walk_dense/legged_lab_g1`.
Set `LEGGED_LAB_MOTIONBRICKS_G1_DIR` if you want the task to load converted motions from another directory.

To restart AMP training around an already-good actor/critic, use actor-only warm start instead of `--resume`.
This loads only the policy/value network and resets the PPO optimizer, AMP discriminator, discriminator normalizer, and AMP optimizer.

```bash
uv run python scripts/rsl_rl/train.py \
  --task LeggedLab-Isaac-AMP-G1-MotionBricks-SoftDisc-v0 \
  --headless \
  --max_iterations 50000 \
  --num_envs 4096 \
  --warm_start \
  --warm_start_experiment_name g1_motionbricks_amp \
  --warm_start_run 2026-07-01_14-16-48 \
  --warm_start_checkpoint model_2000.pt
```

The `SoftDisc` task uses the same MotionBricks environment with a softer AMP discriminator:
moderately larger replay buffer, lower discriminator learning rate, smaller discriminator network, and lower PPO learning rate for fine-tuning.

##### G1 Behavior Cloning Warm Start

You can pretrain the same AMP actor architecture offline from both the default AMP motions and the converted MotionBricks motions:

```bash
uv run python scripts/rsl_rl/train_bc.py \
  --default_weight 0.2 \
  --motionbricks_weight 0.8 \
  --iterations 20000 \
  --batch_size 8192 \
  --logger wandb \
  --wandb_project g1_bc_mixed_motion \
  --device cuda:0
```

The BC target is the next-frame normalized joint-position action:
`(reference_dof_pos[t+1] - default_dof_pos) / 0.25`.
The policy input is the full 114-D AMP actor observation derived from the motion file: body angular velocity, projected gravity, derived velocity command, relative joint position, joint velocity, previous action, and key-body positions in the pelvis frame.

If you want to keep the critic from a good AMP checkpoint and only nudge the actor with BC, initialize from that checkpoint and freeze the critic:

```bash
uv run python scripts/rsl_rl/train_bc.py \
  --default_weight 0.2 \
  --motionbricks_weight 0.8 \
  --iterations 10000 \
  --batch_size 8192 \
  --init_checkpoint logs/rsl_rl/g1_motionbricks_amp/2026-07-01_14-16-48/model_3600.pt \
  --freeze_critic \
  --logger wandb \
  --wandb_project g1_bc_mixed_motion \
  --device cuda:0
```

Then warm-start AMP from the BC checkpoint:

```bash
uv run python scripts/rsl_rl/train.py \
  --task LeggedLab-Isaac-AMP-G1-MotionBricks-SoftDisc-v0 \
  --headless \
  --max_iterations 50000 \
  --num_envs 4096 \
  --warm_start \
  --warm_start_experiment_name g1_bc_mixed_motion \
  --warm_start_run <BC_RUN_DIR_NAME> \
  --warm_start_checkpoint model_bc.pt
```

##### G1 Actor Keyboard Play

Use the generic actor keyboard player to inspect any G1 actor checkpoint with manual velocity commands. It loads the actor `model_state_dict` directly and disables AMP reference-motion loading by default.

```bash
uv run python scripts/rsl_rl/play_actor_keyboard.py \
  --checkpoint logs/rsl_rl/ananth/model_49999.pt \
  --num_envs 1 \
  --real-time
```

You can start with a nonzero command:

```bash
uv run python scripts/rsl_rl/play_actor_keyboard.py \
  --checkpoint logs/rsl_rl/ananth/model_49999.pt \
  --num_envs 1 \
  --vx 0.5 \
  --real-time
```

Keyboard controls:

```text
W / Up       increase forward velocity
S / Down     decrease forward velocity
A / Left     increase lateral velocity
D / Right    decrease lateral velocity
Q            increase yaw rate
E            decrease yaw rate
X / Space    zero the command
```

For a BC checkpoint, swap the checkpoint path:

```bash
uv run python scripts/rsl_rl/play_actor_keyboard.py \
  --checkpoint logs/rsl_rl/g1_bc_mixed_motion_scratch/2026-07-02_11-25-38/model_bc.pt \
  --num_envs 1 \
  --real-time
```

##### G1 Height-Scan Perception

Height-scan variants are available when you want to train or fine-tune with terrain perception. They attach a yaw-aligned ray-cast grid to `torso_link` and append the flattened height map to both actor and critic observations.

The existing non-perception tasks are unchanged. Use these task IDs only when you want the larger perception observation space:

```bash
# Default G1 AMP with height scan
uv run python scripts/rsl_rl/train.py \
  --task LeggedLab-Isaac-AMP-G1-HeightScan-v0 \
  --headless \
  --max_iterations 50000 \
  --num_envs 4096

# MotionBricks G1 AMP with height scan
uv run python scripts/rsl_rl/train.py \
  --task LeggedLab-Isaac-AMP-G1-MotionBricks-HeightScan-v0 \
  --headless \
  --max_iterations 50000 \
  --num_envs 4096
```

The default grid matches the TienKung-Lab setup: `1.6m x 1.0m` at `0.1m` resolution, with height values computed as `scanner_z - terrain_hit_z - 0.5`.

<details>
<summary>Train</summary>

To train the AMP algorithm, you can run the following command:

```bash
python scripts/rsl_rl/train.py \
  --task LeggedLab-Isaac-AMP-G1-v0 \
  --headless \
  --max_iterations 50000 \
  --num_envs 4096 \
  --video \
  --video_interval 2000 \
  --video_length 200
```

```bash
python scripts/rsl_rl/train.py --task LeggedLab-Isaac-AMP-G1-v0 --headless --max_iterations 50000
```

If you want to train it in a non-default gpu, you can pass more arguments to the command:

```bash
# replace `x` with the gpu id you want to use
python scripts/rsl_rl/train.py --task LeggedLab-Isaac-AMP-G1-v0 --headless --max_iterations 50000 --device cuda:x agent.device=cuda:x
```

For more details about the arguments, run `python scripts/rsl_rl/train.py -h`.

</details>

<details>
<summary>Play</summary>

You can play the trained model in a headless mode and record the video: 

```bash
# replace the checkpoint path with the path to your trained model
python scripts/rsl_rl/play.py --task LeggedLab-Isaac-AMP-Flat-Atom01-v0 --headless --num_envs 64 --video --checkpoint logs/rsl_rl/experiment_name/run_name/model_xxx.pt
```

```bash
# replace the checkpoint path with the path to your trained model
python scripts/rsl_rl/play.py --task LeggedLab-Isaac-AMP-G1-v0 --headless --num_envs 64 --video --checkpoint logs/rsl_rl/experiment_name/run_name/model_xxx.pt
```

The video will be saved in the `logs/rsl_rl/experiment_name/run_name/videos/play` directory.

</details>

To check sim to sim using Mujoco， you can run:

```bash
python scripts/atom01_long_base_link_lab_to_mujoco.py
```

## 🗺️ Roadmap

- [ ] Add more legged robots, such as Unitree H1
- [x] Self-contact penalty in AMP
- [x] Asymmetric Actor-Critic in AMP
- [x] Symmetric Reward
- [x] Sim2sim in mujoco (support atom01)
- [ ] Add support for image observations
- [ ] Walk in rough terrain with AMP

## 🙏 Acknowledgement

We would like to express our gratitude to the following open-source projects:
- [**legged_lab**](https://github.com/zitongbai/legged_lab) - The foundation of this project.
- [**Isaac Lab**](https://github.com/isaac-sim/IsaacLab) - The foundation of this project.
- [**RSL-RL**](https://github.com/leggedrobotics/rsl_rl) - Reinforcement learning algorithms for legged robots.
- [**AMP_for_hardware**](https://github.com/Alescontrela/AMP_for_hardware) - Inspiration for AMP implementation.
- [**GMR**](https://github.com/YanjieZe/GMR) - Excellent motion retargeting library.
- [**MimicKit**](https://github.com/xbpeng/MimicKit) - Reference for imitation learning.
