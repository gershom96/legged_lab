# G1 Mixed AMP Height-Scan Model

This note documents the model/input contract for:

```text
LeggedLab-Isaac-AMP-G1-Mixed-HeightScan-v0
```

The task uses the mixed G1 AMP motion dataset, stand-scaled rewards, and an
RPL-style terrain height scan. The policy class is:

```text
ActorCriticHeightScan
```

The key rule is: the height scan is routed through a CNN, while the remaining
proprioceptive/state observations are routed through an MLP.

## Height Scan

The height scanner is attached to:

```text
torso_link
```

with yaw-only alignment. The sample grid follows the torso position and yaw,
but does not follow torso roll/pitch.

The grid configuration is:

```text
size:       (1.6, 1.0)
resolution: 0.1
shape:      (17, 11)
flat dim:   187
ordering:   xy
```

The scalar value returned by `mdp.height_scan` is:

```text
scanner_z_world - terrain_hit_z_world - offset
```

with:

```text
offset = 0.5
```

So the sample locations are torso-yaw aligned, while the height values are
vertical world-z height differences.

IsaacLab's default `GridPatternCfg(ordering="xy")` flattens the grid with `x`
as the inner loop and `y` as the outer loop. The model and symmetry code
therefore reconstruct images as:

```text
flat -> [B, channels, 11, 17] -> transpose -> [B, channels, 17, 11]
```

This is important on non-flat terrain. A plain `flat.reshape(17, 11)` does not
preserve the physical x/y layout for `xy` ordering.

## Actor

The actor uses the `policy` observation group.

Current flattened policy observation order from the environment is:

```text
height_scan
base_ang_vel history
projected_gravity history
velocity_commands
joint_pos history
joint_vel history
actions history
key_body_pos_b
```

Dimensions:

```text
height_scan:                  187 = 17 * 11
base_ang_vel history:          15 = 5 * 3
projected_gravity history:     15 = 5 * 3
velocity_commands:              3
joint_pos history:            145 = 5 * 29
joint_vel history:            145 = 5 * 29
actions history:              145 = 5 * 29
key_body_pos_b:                18 = 6 key bodies * xyz

total policy obs:             673
non-height policy obs:        486
```

The actor model splits the observation as:

```text
height_scan -> ordering-aware [B, 1, 17, 11] -> actor CNN -> 64-D encoding
non-height obs -> [B, 486]
concat -> [B, 550] -> actor MLP -> [B, 29] action mean
```

The actor samples actions during training:

```text
action ~ Normal(action_mean, action_std)
```

For this height-scan task, `init_noise_std` is set to:

```text
0.25
```

This is intentionally lower than `1.0`, because high initial action noise caused
early bad-orientation terminations in this humanoid task.

## Critic

The critic uses the `critic` observation group.

Current flattened critic observation order from the environment is:

```text
height_scan history
base_lin_vel
base_ang_vel
projected_gravity
velocity_commands
joint_pos
joint_vel
actions
key_body_pos_b
```

Dimensions:

```text
height_scan history:          935 = 5 * 17 * 11
base_lin_vel:                   3
base_ang_vel:                   3
projected_gravity:              3
velocity_commands:              3
joint_pos:                     29
joint_vel:                     29
actions:                       29
key_body_pos_b:                18

total critic obs:            1052
non-height critic obs:        117
```

The critic model splits the observation as:

```text
height_scan history -> ordering-aware [B, 5, 17, 11] -> critic CNN -> 64-D encoding
non-height obs -> [B, 117]
concat -> [B, 181] -> critic MLP -> [B, 1] value
```

The critic is used only during training. It is not deployed as part of the
runtime policy.

## Discriminator

The AMP discriminator does not receive the height map or velocity command.
It receives the AMP style observation from the policy rollout and compares it
against reference-motion observations from the mixed motion dataset.

Per-step discriminator observation:

```text
root_local_rot_tan_norm:       6
base_lin_vel:                  3
base_ang_vel:                  3
joint_pos:                    29
joint_vel:                    29
key_body_pos_b:               18

per-step dim:                 88
history steps:                 4
discriminator input dim:     352
```

Network:

```text
[B, 352] -> MLP hidden [1024, 512] -> [B, 1] score
```

The discriminator is trained during each PPO update from:

```text
policy discriminator observations vs. demonstration discriminator observations
```

For the current config, AMP uses LSGAN loss. Its score is converted into a style
reward, then blended with the task reward:

```text
total_reward = task_style_lerp * task_reward
             + (1 - task_style_lerp) * style_reward
```

Current height-scan mixed config inherits:

```text
task_style_lerp = 0.4
style_reward_scale = 5.0
min_mean_episode_length = 0.0
```

AMP is intentionally active from the first rollout for this task. A previous
default of `min_mean_episode_length = 900.0` kept AMP rewards and discriminator
updates disabled while the humanoid was still failing at short episodes, so the
run trained only on task rewards and collapsed into bad-orientation resets.

## Observation Slicing Contract

Do not assume the height scan is the final slice of the flat observation.

IsaacLab may serialize or construct the observation terms with `height_scan`
first. The model therefore uses explicit slices resolved from
`env.observation_manager`:

```text
Actor height-scan slice:  (0, 187)
Critic height-scan slice: (0, 935)
```

The runner passes these slices into `ActorCriticHeightScan`. The model removes
that slice before feeding the remaining 1D observations to the MLP.

If these startup prints are missing, or if the slices do not match the expected
dimensions, stop the run and inspect the observation manager before training.

## Symmetry

G1 AMP symmetry is order-aware. It reads:

```text
env.observation_manager.active_terms["policy"]
env.observation_manager.group_obs_term_dim["policy"]
```

and transforms terms by name. For `height_scan`, it reconstructs the flattened
scan using the sensor's `pattern_cfg.ordering`, flips the left-right dimension,
then flattens back to the original sensor order. Internally the image layout is:

```text
[B, history_or_channels, 17, 11]
```

then flips the left-right dimension.

## Quick Sanity Checks

At startup, the height-scan model should print:

```text
Actor height-scan slice: (0, 187)
Critic height-scan slice: (0, 935)
```

The expected model sizes are:

```text
actor CNN input:   [B, 1, 17, 11]
critic CNN input:  [B, 5, 17, 11]
actor MLP input:   550
critic MLP input:  181
actor output:      29 action means
critic output:     1 value
disc output:       1 AMP score
```

Early training should not immediately become all `bad_orientation`
terminations. If it does, first check:

```text
Loss/amp/active
Policy/mean_noise_std
Train/mean_episode_length
Episode_Termination/bad_orientation
AMP/mean_style_reward
Loss/amp/disc_score
Loss/amp/disc_demo_score
```
