import os

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlSymmetryCfg
from legged_lab.rsl_rl import (
    RslRlPpoAmpAlgorithmCfg,
    RslRlAmpCfg,
    RslRlPpoActorCriticHeightScanCfg,
    RslRlPpoActorCriticSplitHeightScanCfg,
)
from legged_lab import LEGGED_LAB_ROOT_DIR
from legged_lab.tasks.locomotion.amp.mdp.symmetry import g1


def g1_hand_biased_disc_obs_weights() -> list[float]:
    """Weights for G1 AMP discriminator features.

    Observation layout per history step:
    root rot(6), root lin vel(3), root ang vel(3), joint pos(29),
    joint vel(29), key body pos(6 * 3).
    """
    weights = [1.0] * 88

    joint_pos_start = 12
    joint_vel_start = joint_pos_start + 29
    key_body_start = joint_vel_start + 29

    waist_joint_ids = range(12, 15)
    arm_joint_ids = range(15, 29)
    lower_body_joint_ids = range(0, 12)

    for joint_id in lower_body_joint_ids:
        weights[joint_pos_start + joint_id] = 0.6
        weights[joint_vel_start + joint_id] = 0.6

    for joint_id in waist_joint_ids:
        weights[joint_pos_start + joint_id] = 2.0
        weights[joint_vel_start + joint_id] = 2.5

    for joint_id in arm_joint_ids:
        weights[joint_pos_start + joint_id] = 3.0
        weights[joint_vel_start + joint_id] = 3.5

    # key bodies: ankles, wrists, shoulders in KEY_BODY_NAMES order.
    for key_id in (0, 1):
        start = key_body_start + key_id * 3
        weights[start : start + 3] = [0.6, 0.6, 0.6]
    for key_id in (2, 3):
        start = key_body_start + key_id * 3
        weights[start : start + 3] = [4.0, 4.0, 4.0]
    for key_id in (4, 5):
        start = key_body_start + key_id * 3
        weights[start : start + 3] = [3.0, 3.0, 3.0]

    return weights


@configclass
class G1RslRlOnPolicyRunnerAmpCfg(RslRlOnPolicyRunnerCfg):
    class_name = "AMPRunner"
    num_steps_per_env = 24
    max_iterations = 50000
    save_interval = 200
    experiment_name = "g1_amp"
    obs_groups = {
        "policy": ["policy"], 
        "critic": ["critic"], 
        "discriminator": ["disc"],
        "discriminator_demonstration": ["disc_demo"]
    }
    # policy = RslRlPpoActorCriticRecurrentCfg(
    #     init_noise_std=1.0,
    #     actor_hidden_dims=[512, 256, 128],
    #     critic_hidden_dims=[512, 256, 128],
    #     actor_obs_normalization=False,
    #     critic_obs_normalization=False,
    #     activation="elu",
    #     rnn_type="lstm",
    #     rnn_hidden_dim=64,
    #     rnn_num_layers=1
    # )
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        activation="elu",
    )
    algorithm = RslRlPpoAmpAlgorithmCfg(
        class_name="PPOAMP",
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        amp_cfg=RslRlAmpCfg(
            disc_obs_buffer_size=100,
            grad_penalty_scale=10.0,
            disc_trunk_weight_decay=1.0e-4,
            disc_linear_weight_decay=1.0e-2,
            disc_learning_rate=1.0e-4,
            disc_max_grad_norm=1.0,
            amp_discriminator=RslRlAmpCfg.AMPDiscriminatorCfg(
                hidden_dims=[1024, 512],
                activation="elu",
                style_reward_scale=5.0,
                task_style_lerp=0.4
            ),
            loss_type="LSGAN"
        ),
        symmetry_cfg=RslRlSymmetryCfg(
            use_data_augmentation=True, data_augmentation_func=g1.compute_symmetric_states,
            use_mirror_loss=True, mirror_loss_coeff=0.1,
        )
    )


@configclass
class G1MotionBricksRslRlOnPolicyRunnerAmpCfg(G1RslRlOnPolicyRunnerAmpCfg):
    experiment_name = "g1_motionbricks_amp"
    logger = "wandb"
    wandb_project = "g1_motionbricks_amp"


@configclass
class G1HeightScanRslRlOnPolicyRunnerAmpCfg(G1RslRlOnPolicyRunnerAmpCfg):
    experiment_name = "g1_amp_height_scan"
    logger = "wandb"
    wandb_project = "g1_amp_height_scan"
    policy = RslRlPpoActorCriticHeightScanCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        activation="elu",
        height_scan_shape=(17, 11),
        critic_height_scan_history_length=5,
    )


@configclass
class G1MotionBricksHeightScanRslRlOnPolicyRunnerAmpCfg(G1RslRlOnPolicyRunnerAmpCfg):
    experiment_name = "g1_motionbricks_amp_height_scan"
    logger = "wandb"
    wandb_project = "g1_motionbricks_amp_height_scan"
    policy = RslRlPpoActorCriticHeightScanCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        activation="elu",
        height_scan_shape=(17, 11),
        critic_height_scan_history_length=5,
    )


@configclass
class G1MotionBricksSoftAmpRslRlOnPolicyRunnerAmpCfg(G1MotionBricksRslRlOnPolicyRunnerAmpCfg):
    experiment_name = "g1_motionbricks_amp_soft_disc"
    wandb_project = "g1_motionbricks_amp_soft_disc"
    algorithm = RslRlPpoAmpAlgorithmCfg(
        class_name="PPOAMP",
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=3.0e-5,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        amp_cfg=RslRlAmpCfg(
            disc_obs_buffer_size=256,
            grad_penalty_scale=10.0,
            disc_trunk_weight_decay=1.0e-4,
            disc_linear_weight_decay=1.0e-2,
            disc_learning_rate=3.0e-5,
            disc_max_grad_norm=1.0,
            amp_discriminator=RslRlAmpCfg.AMPDiscriminatorCfg(
                hidden_dims=[512, 256],
                activation="elu",
                style_reward_scale=5.0,
                task_style_lerp=0.4,
            ),
            loss_type="LSGAN",
        ),
        symmetry_cfg=RslRlSymmetryCfg(
            use_data_augmentation=True,
            data_augmentation_func=g1.compute_symmetric_states,
            use_mirror_loss=True,
            mirror_loss_coeff=0.1,
        ),
    )


@configclass
class G1MotionBricksStyleHandsRslRlOnPolicyRunnerAmpCfg(G1MotionBricksRslRlOnPolicyRunnerAmpCfg):
    experiment_name = "g1_motionbricks_amp_style_hands"
    wandb_project = "g1_motionbricks_amp_style_hands"
    algorithm = RslRlPpoAmpAlgorithmCfg(
        class_name="PPOAMP",
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.1,
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-5,
        schedule="fixed",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.005,
        max_grad_norm=0.5,
        amp_cfg=RslRlAmpCfg(
            disc_obs_buffer_size=512,
            grad_penalty_scale=10.0,
            disc_trunk_weight_decay=1.0e-4,
            disc_linear_weight_decay=1.0e-2,
            disc_learning_rate=1.0e-5,
            disc_max_grad_norm=0.5,
            amp_discriminator=RslRlAmpCfg.AMPDiscriminatorCfg(
                hidden_dims=[512, 256],
                activation="elu",
                style_reward_scale=8.0,
                task_style_lerp=0.0,
                disc_obs_feature_weights=g1_hand_biased_disc_obs_weights(),
            ),
            loss_type="LSGAN",
        ),
        symmetry_cfg=RslRlSymmetryCfg(
            use_data_augmentation=True,
            data_augmentation_func=g1.compute_symmetric_states,
            use_mirror_loss=True,
            mirror_loss_coeff=0.1,
        ),
    )


@configclass
class G1MixedSoftAmpRslRlOnPolicyRunnerAmpCfg(G1MotionBricksSoftAmpRslRlOnPolicyRunnerAmpCfg):
    experiment_name = "g1_mixed_amp_soft_disc"
    wandb_project = "g1_mixed_amp_soft_disc"


@configclass
class G1MixedVerySoftAmpRslRlOnPolicyRunnerAmpCfg(G1MotionBricksSoftAmpRslRlOnPolicyRunnerAmpCfg):
    experiment_name = "g1_mixed_amp_very_soft_disc"
    wandb_project = "g1_mixed_amp_very_soft_disc"
    algorithm = RslRlPpoAmpAlgorithmCfg(
        class_name="PPOAMP",
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=3.0e-5,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        amp_cfg=RslRlAmpCfg(
            disc_obs_buffer_size=256,
            grad_penalty_scale=10.0,
            disc_trunk_weight_decay=1.0e-4,
            disc_linear_weight_decay=1.0e-2,
            disc_learning_rate=1.0e-5,
            disc_max_grad_norm=1.0,
            amp_discriminator=RslRlAmpCfg.AMPDiscriminatorCfg(
                hidden_dims=[512, 256],
                activation="elu",
                style_reward_scale=3.0,
                task_style_lerp=0.7,
            ),
            loss_type="LSGAN",
        ),
        symmetry_cfg=RslRlSymmetryCfg(
            use_data_augmentation=True,
            data_augmentation_func=g1.compute_symmetric_states,
            use_mirror_loss=True,
            mirror_loss_coeff=0.1,
        ),
    )


@configclass
class G1MixedVelocityTunedAmpRslRlOnPolicyRunnerAmpCfg(G1MotionBricksSoftAmpRslRlOnPolicyRunnerAmpCfg):
    experiment_name = "g1_mixed_amp_velocity_tuned"
    wandb_project = "g1_mixed_amp_velocity_tuned"
    algorithm = RslRlPpoAmpAlgorithmCfg(
        class_name="PPOAMP",
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=3.0e-5,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        amp_cfg=RslRlAmpCfg(
            disc_obs_buffer_size=256,
            grad_penalty_scale=10.0,
            disc_trunk_weight_decay=1.0e-4,
            disc_linear_weight_decay=1.0e-2,
            disc_learning_rate=1.0e-5,
            disc_max_grad_norm=1.0,
            amp_discriminator=RslRlAmpCfg.AMPDiscriminatorCfg(
                hidden_dims=[512, 256],
                activation="elu",
                style_reward_scale=1.0,
                task_style_lerp=0.85,
            ),
            loss_type="LSGAN",
        ),
        symmetry_cfg=RslRlSymmetryCfg(
            use_data_augmentation=True,
            data_augmentation_func=g1.compute_symmetric_states,
            use_mirror_loss=True,
            mirror_loss_coeff=0.1,
        ),
    )


def g1_mixed_stand_scaled_amp_algorithm_cfg(
    min_mean_episode_length: float = 0.0,
    learning_rate: float = 3.0e-5,
    disc_learning_rate: float = 3.0e-5,
) -> RslRlPpoAmpAlgorithmCfg:
    return RslRlPpoAmpAlgorithmCfg(
        class_name="PPOAMP",
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=learning_rate,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        amp_cfg=RslRlAmpCfg(
            disc_obs_buffer_size=100,
            grad_penalty_scale=10.0,
            disc_trunk_weight_decay=1.0e-4,
            disc_linear_weight_decay=1.0e-2,
            disc_learning_rate=disc_learning_rate,
            disc_max_grad_norm=1.0,
            min_mean_episode_length=min_mean_episode_length,
            amp_discriminator=RslRlAmpCfg.AMPDiscriminatorCfg(
                hidden_dims=[1024, 512],
                activation="elu",
                style_reward_scale=5.0,
                task_style_lerp=0.4,
            ),
            loss_type="LSGAN",
        ),
        symmetry_cfg=RslRlSymmetryCfg(
            use_data_augmentation=True,
            data_augmentation_func=g1.compute_symmetric_states,
            use_mirror_loss=True,
            mirror_loss_coeff=0.1,
        ),
    )


@configclass
class G1MixedStandScaledAmpRslRlOnPolicyRunnerAmpCfg(G1MotionBricksRslRlOnPolicyRunnerAmpCfg):
    experiment_name = "g1_mixed_amp_stand_scaled"
    wandb_project = "g1_mixed_amp_stand_scaled"
    algorithm = g1_mixed_stand_scaled_amp_algorithm_cfg()


@configclass
class G1MixedHistoryAmpRslRlOnPolicyRunnerAmpCfg(G1MixedStandScaledAmpRslRlOnPolicyRunnerAmpCfg):
    experiment_name = "g1_mixed_amp_history"
    wandb_project = "g1_mixed_amp_history"
    algorithm = g1_mixed_stand_scaled_amp_algorithm_cfg(
        min_mean_episode_length=float(os.environ.get("LEGGED_LAB_AMP_MIN_MEAN_EPISODE_LENGTH", "0.0")),
        learning_rate=1.0e-4,
        disc_learning_rate=1.0e-4,
    )
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        activation="elu",
    )


@configclass
class G1MixedHeightScanAmpRslRlOnPolicyRunnerAmpCfg(G1MixedStandScaledAmpRslRlOnPolicyRunnerAmpCfg):
    experiment_name = "g1_mixed_amp_height_scan"
    wandb_project = "g1_mixed_amp_height_scan"
    algorithm = g1_mixed_stand_scaled_amp_algorithm_cfg(
        min_mean_episode_length=float(os.environ.get("LEGGED_LAB_AMP_MIN_MEAN_EPISODE_LENGTH", "0.0")),
        learning_rate=1.0e-4,
        disc_learning_rate=1.0e-4,
    )
    policy = RslRlPpoActorCriticHeightScanCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        activation="elu",
        height_scan_shape=(17, 11),
        critic_height_scan_history_length=5,
    )


@configclass
class G1SplitPolicyHeightScanAmpRslRlOnPolicyRunnerAmpCfg(G1MixedHeightScanAmpRslRlOnPolicyRunnerAmpCfg):
    experiment_name = "g1_split_policy_heightscan"
    wandb_project = "g1_split_policy_heightscan"
    algorithm = g1_mixed_stand_scaled_amp_algorithm_cfg(
        min_mean_episode_length=float(os.environ.get("LEGGED_LAB_AMP_MIN_MEAN_EPISODE_LENGTH", "0.0")),
        learning_rate=1.0e-4,
        disc_learning_rate=1.0e-4,
    )
    algorithm.class_name = "PPOAMPSplit"
    algorithm.split_reward_cfg = {
        "strict": True,
        "lower": {
            "track_lin_vel_xy_exp": 1.0,
            "track_ang_vel_z_exp": 1.0,
            "feet_air_time": 1.0,
            "feet_slide": 1.0,
            "flat_orientation_l2": 1.0,
            "ang_vel_xy_l2": 1.0,
            "lin_vel_z_l2": 1.0,
            "root_height_below_target": 1.0,
            "dof_torques_l2_lower": 1.0,
            "dof_acc_l2_lower": 1.0,
            "action_rate_l2_lower": 1.0,
            "dof_pos_limits": 1.0,
            "joint_deviation_lower_body": 1.0,
            "termination_penalty": 1.0,
        },
        "upper": {
            "joint_deviation_arms": 1.0,
            "dof_torques_l2_upper": 1.0,
            "dof_acc_l2_upper": 1.0,
            "action_rate_l2_upper": 1.0,
            "dof_pos_limits_upper": 1.0,
            "flat_orientation_l2": 0.25,
            "ang_vel_xy_l2": 0.5,
            "lin_vel_z_l2": 0.25,
            "root_height_below_target": 0.25,
            "termination_penalty": 0.5,
        },
    }
    policy = RslRlPpoActorCriticSplitHeightScanCfg(
        init_noise_std=1.0,
        lower_init_noise_std=1.0,
        upper_init_noise_std=0.6,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        activation="elu",
        height_scan_shape=(17, 11),
        critic_height_scan_history_length=5,
    )
