import os
import math
from pathlib import Path
from dataclasses import MISSING

import joblib
import numpy as np
import torch
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise
from isaaclab.utils.noise import NoiseCfg

import legged_lab.tasks.locomotion.amp.mdp as mdp
from legged_lab.tasks.locomotion.amp.amp_env_cfg import LocomotionAmpEnvCfg
from legged_lab import LEGGED_LAB_ROOT_DIR
from legged_lab.sensors import RayCasterArrayCfg

##
# Pre-defined configs
##
from legged_lab.assets.unitree import UNITREE_G1_29DOF_CFG

# The order must align with the retarget config file scripts/tools/retarget/config/g1_29dof.yaml
KEY_BODY_NAMES = [
    "left_ankle_roll_link", 
    "right_ankle_roll_link",
    "left_wrist_yaw_link",
    "right_wrist_yaw_link",
    "left_shoulder_roll_link",
    "right_shoulder_roll_link",
] # if changed here and symmetry is enabled, remember to update amp.mdp.symmetry.g1 as well!
ANIMATION_TERM_NAME = "animation"
AMP_NUM_STEPS = 4
POLICY_PROPRIO_HISTORY_LENGTH = 5
CRITIC_HEIGHT_SCAN_HISTORY_LENGTH = 5
PACKED_ACTOR_FRAME_DIM = 114
HEIGHT_SCAN_BODY_NAME = "torso_link"
HEIGHT_SCAN_SENSOR_NAME = "height_scanner"
HEIGHT_SCAN_OFFSET = 0.5
HEIGHT_SCAN_RESOLUTION = 0.1
HEIGHT_SCAN_SIZE = (1.6, 1.0)
HEIGHT_SCAN_NOISE = 0.01
STAND_REWARD_MIN_MEAN_EPISODE_LENGTH = 950.0
STAND_REWARD_MIN_LEARNING_ITERATION = 40000.0
G1_MIN_ROOT_HEIGHT = 0.2


def packed_actor_uniform_noise(data: torch.Tensor, cfg: "PackedActorUniformNoiseCfg") -> torch.Tensor:
    """Apply old per-term actor observation noise to one packed actor frame."""
    noise_scales = data.new_tensor(
        [cfg.base_ang_vel] * 3
        + [cfg.projected_gravity] * 3
        + [0.0] * 3
        + [cfg.joint_pos] * 29
        + [cfg.joint_vel] * 29
        + [0.0] * 29
        + [0.0] * 18
    )
    if noise_scales.numel() != PACKED_ACTOR_FRAME_DIM:
        raise RuntimeError(f"Packed actor noise dim {noise_scales.numel()} != {PACKED_ACTOR_FRAME_DIM}.")
    return data + (torch.rand_like(data) * 2.0 - 1.0) * noise_scales


@configclass
class PackedActorUniformNoiseCfg(NoiseCfg):
    """Serializable config for packed actor observation noise."""

    func = packed_actor_uniform_noise

    base_ang_vel: float = 0.35
    projected_gravity: float = 0.05
    joint_pos: float = 0.03
    joint_vel: float = 1.75


def enable_policy_proprio_history(cfg: LocomotionAmpEnvCfg, history_length: int = POLICY_PROPRIO_HISTORY_LENGTH):
    """Use history for proprioceptive policy terms while keeping command and height map current."""
    cfg.observations.policy.history_length = None
    history_terms = ("base_ang_vel", "projected_gravity", "joint_pos", "joint_vel", "actions")
    current_terms = ("velocity_commands", "key_body_pos_b")

    for term_name in history_terms:
        term_cfg = getattr(cfg.observations.policy, term_name)
        term_cfg.history_length = history_length
        term_cfg.flatten_history_dim = True

    for term_name in current_terms:
        term_cfg = getattr(cfg.observations.policy, term_name)
        term_cfg.history_length = 0
        term_cfg.flatten_history_dim = True


def enable_packed_actor_policy_history(cfg: LocomotionAmpEnvCfg, history_length: int = POLICY_PROPRIO_HISTORY_LENGTH):
    """Stack the full concatenated actor observation as full history frames."""
    cfg.observations.policy.history_length = None
    cfg.observations.policy.flatten_history_dim = True

    policy_terms = (
        "base_ang_vel",
        "projected_gravity",
        "velocity_commands",
        "joint_pos",
        "joint_vel",
        "actions",
        "key_body_pos_b",
    )
    for term_name in policy_terms:
        setattr(cfg.observations.policy, term_name, None)

    cfg.observations.policy.packed_actor_obs = ObsTerm(
        func=mdp.packed_actor_obs,
        params={
            "command_name": "base_velocity",
            "asset_cfg": SceneEntityCfg("robot", body_names=KEY_BODY_NAMES, preserve_order=True),
        },
        noise=PackedActorUniformNoiseCfg(),
    )
    cfg.observations.policy.packed_actor_obs.history_length = history_length
    cfg.observations.policy.packed_actor_obs.flatten_history_dim = True


def enable_current_critic_observations(cfg: LocomotionAmpEnvCfg):
    """Use only the current privileged critic observation, matching FALCON critic history length 1."""
    cfg.observations.critic.history_length = 1
    cfg.observations.critic.flatten_history_dim = True

    critic_terms = (
        "base_lin_vel",
        "base_ang_vel",
        "projected_gravity",
        "velocity_commands",
        "joint_pos",
        "joint_vel",
        "actions",
        "key_body_pos_b",
    )
    for term_name in critic_terms:
        term_cfg = getattr(cfg.observations.critic, term_name)
        term_cfg.history_length = 0
        term_cfg.flatten_history_dim = True


def enable_critic_height_scan_history(cfg: LocomotionAmpEnvCfg):
    """Keep privileged critic state current while allowing height scan to own its history."""
    cfg.observations.critic.history_length = None
    current_terms = (
        "base_lin_vel",
        "base_ang_vel",
        "projected_gravity",
        "velocity_commands",
        "joint_pos",
        "joint_vel",
        "actions",
        "key_body_pos_b",
    )

    for term_name in current_terms:
        term_cfg = getattr(cfg.observations.critic, term_name)
        term_cfg.history_length = 0
        term_cfg.flatten_history_dim = True


def gate_stand_scaled_rewards_after_mean_episode_length(
    cfg: LocomotionAmpEnvCfg,
    min_mean_episode_length: float = STAND_REWARD_MIN_MEAN_EPISODE_LENGTH,
    min_learning_iteration: float | None = None,
):
    """Delay stand-specific shaping until locomotion is reasonably stable."""
    for term_name in ("joint_deviation_hip", "joint_deviation_arms", "joint_deviation_waist"):
        term_cfg = getattr(cfg.rewards, term_name)
        term_cfg.func = mdp.command_scaled_joint_deviation_l1_after_mean_episode_length
        term_cfg.params["min_mean_episode_length"] = min_mean_episode_length
        if min_learning_iteration is not None:
            term_cfg.params["min_learning_iteration"] = min_learning_iteration

    cfg.rewards.low_command_motion.func = mdp.low_command_motion_l2_after_mean_episode_length
    cfg.rewards.low_command_motion.params["min_mean_episode_length"] = min_mean_episode_length
    if min_learning_iteration is not None:
        cfg.rewards.low_command_motion.params["min_learning_iteration"] = min_learning_iteration

    cfg.rewards.root_height_below_target.func = mdp.root_height_below_target_l2_after_mean_episode_length
    cfg.rewards.root_height_below_target.params["min_mean_episode_length"] = min_mean_episode_length
    if min_learning_iteration is not None:
        cfg.rewards.root_height_below_target.params["min_learning_iteration"] = min_learning_iteration


def enable_height_scan_observations(
    cfg: LocomotionAmpEnvCfg,
    body_name: str = HEIGHT_SCAN_BODY_NAME,
    offset: float = HEIGHT_SCAN_OFFSET,
    resolution: float = HEIGHT_SCAN_RESOLUTION,
    size: tuple[float, float] = HEIGHT_SCAN_SIZE,
    noise: float | None = None,
    debug_vis: bool = False,
    critic_history_length: int = CRITIC_HEIGHT_SCAN_HISTORY_LENGTH,
    policy_proprio_history_length: int | None = POLICY_PROPRIO_HISTORY_LENGTH,
):
    """Attach a terrain height scanner and expose it to policy and critic observations."""
    if policy_proprio_history_length is not None:
        enable_policy_proprio_history(cfg, policy_proprio_history_length)
    enable_critic_height_scan_history(cfg)
    if noise is None:
        noise = float(os.environ.get("LEGGED_LAB_HEIGHT_SCAN_NOISE", str(HEIGHT_SCAN_NOISE)))
    cfg.scene.height_scanner = RayCasterArrayCfg(
        prim_path="{ENV_REGEX_NS}/Robot/" + body_name,
        offset=RayCasterArrayCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        attach_yaw_only=True,
        pattern_cfg=patterns.GridPatternCfg(resolution=resolution, size=size),
        debug_vis=debug_vis,
        mesh_prim_paths=["/World/ground"],
        update_period=cfg.sim.dt * cfg.decimation,
    )
    sensor_cfg = SceneEntityCfg(HEIGHT_SCAN_SENSOR_NAME)
    cfg.observations.policy.height_scan = ObsTerm(
        func=mdp.height_scan,
        params={"sensor_cfg": sensor_cfg, "offset": offset},
        noise=Unoise(n_min=-noise, n_max=noise),
    )
    cfg.observations.policy.height_scan.history_length = 0
    cfg.observations.critic.height_scan = ObsTerm(
        func=mdp.height_scan,
        params={"sensor_cfg": sensor_cfg, "offset": offset},
    )
    cfg.observations.critic.height_scan.history_length = critic_history_length
    cfg.observations.critic.height_scan.flatten_history_dim = True

@configclass
class G1AmpRewards():
    """Reward terms for the MDP."""
    # -- task
    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_exp, weight=1.0, params={"command_name": "base_velocity", "std": math.sqrt(0.25)}
    )
    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_exp, weight=1.0, params={"command_name": "base_velocity", "std": math.sqrt(0.25)}
    )

    # -- penalties
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-1.0)
    lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=-0.2)
    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)
    dof_torques_l2 = RewTerm(func=mdp.joint_torques_l2, weight=-2.0e-6)
    dof_acc_l2 = RewTerm(func=mdp.joint_acc_l2, weight=-1.0e-7)
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.005)
    dof_pos_limits = RewTerm(
        func=mdp.joint_pos_limits,
        weight=-1.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_ankle_pitch_joint", ".*_ankle_roll_joint"])},
    )
    
    joint_deviation_hip = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.1,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_yaw_joint", ".*_hip_roll_joint"])},
    )
    joint_deviation_arms = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.05,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
                    ".*_shoulder_.*_joint",
                    ".*_elbow_joint",
                    ".*_wrist_.*_joint",
                ],
            )
        },
    )
    joint_deviation_waist = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.1,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names="waist_.*_joint")},
    )
    
    feet_air_time = RewTerm(
        func=mdp.feet_air_time_positive_biped,
        weight=0.5,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
            "threshold": 0.4,
        },
    )
    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=-0.1,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_ankle_roll_link"),
        },
    )
    
    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-200.0)


@configclass
class G1StandScaledAmpRewards(G1AmpRewards):
    """G1 AMP rewards with command-scaled posture and explicit stop/height shaping."""

    joint_deviation_hip = RewTerm(
        func=mdp.command_scaled_joint_deviation_l1,
        weight=-0.1,
        params={
            "command_name": "base_velocity",
            "sigma": 0.5,
            "min_scale": 0.25,
            "angular_scale": 0.5,
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_yaw_joint", ".*_hip_roll_joint"]),
        },
    )
    joint_deviation_arms = RewTerm(
        func=mdp.command_scaled_joint_deviation_l1,
        weight=-0.05,
        params={
            "command_name": "base_velocity",
            "sigma": 0.5,
            "min_scale": 0.5,
            "angular_scale": 0.5,
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
                    ".*_shoulder_.*_joint",
                    ".*_elbow_joint",
                    ".*_wrist_.*_joint",
                ],
            ),
        },
    )
    joint_deviation_waist = RewTerm(
        func=mdp.command_scaled_joint_deviation_l1,
        weight=-0.1,
        params={
            "command_name": "base_velocity",
            "sigma": 0.5,
            "min_scale": 0.3,
            "angular_scale": 0.5,
            "asset_cfg": SceneEntityCfg("robot", joint_names="waist_.*_joint"),
        },
    )
    low_command_motion = RewTerm(
        func=mdp.low_command_motion_l2,
        weight=-2.0,
        params={
            "command_name": "base_velocity",
            "command_threshold": 0.1,
            "angular_command_scale": 0.5,
            "yaw_weight": 0.5,
        },
    )
    root_height_below_target = RewTerm(
        func=mdp.root_height_below_target_l2,
        weight=-20.0,
        params={"target_height": 0.75},
    )


@configclass
class G1AmpEnvCfg(LocomotionAmpEnvCfg):
    """Configuration for the G1 AMP environment."""
    
    rewards: G1AmpRewards = G1AmpRewards()
    
    def __post_init__(self):
        super().__post_init__()
        
        self.scene.robot = UNITREE_G1_29DOF_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        # ------------------------------------------------------
        # motion data
        # ------------------------------------------------------
        self.motion_data.motion_dataset.motion_data_dir = os.path.join(
            LEGGED_LAB_ROOT_DIR, "data", "MotionData", "g1_29dof", "amp", "walk_and_run"
        )
        self.motion_data.motion_dataset.motion_data_weights = {
            "B10_-__Walk_turn_left_45_stageii": 1.0,
            "B11_-__Walk_turn_left_135_stageii": 1.0,
            "B13_-__Walk_turn_right_90_stageii": 1.0,
            "B14_-__Walk_turn_right_45_t2_stageii": 1.0,
            "B15_-__Walk_turn_around_stageii": 1.0,
            "B22_-__side_step_left_stageii": 1.0,
            "B23_-__side_step_right_stageii": 1.0,
            "B4_-_Stand_to_Walk_backwards_stageii": 1.0,
            "B9_-__Walk_turn_left_90_stageii": 1.0,
            "C11_-_run_turn_left_90_stageii": 1.0,
            "C12_-_run_turn_left_45_stageii": 1.0,
            "C13_-_run_turn_left_135_stageii": 1.0,
            "C14_-_run_turn_right_90_stageii": 1.0,
            "C15_-_run_turn_right_45_stageii": 1.0,
            "C16_-_run_turn_right_135_stageii": 1.0,
            "C17_-_run_change_direction_stageii": 1.0,
            "C1_-_stand_to_run_stageii": 1.0,
            "C3_-_run_stageii": 1.0,
            "C4_-_run_to_walk_a_stageii": 1.0,
            "C5_-_walk_to_run_stageii": 1.0,
            "C6_-_stand_to_run_backwards_stageii": 1.0,
            "C8_-_run_backwards_to_stand_stageii": 1.0,
            "C9_-_run_backwards_turn_run_forward_stageii": 1.0,
            "Walk_B10_-_Walk_turn_left_45_stageii": 1.0,
            "Walk_B13_-_Walk_turn_right_45_stageii": 1.0,
            "Walk_B15_-_Walk_turn_around_stageii": 1.0,
            "Walk_B16_-_Walk_turn_change_stageii": 1.0,
            "Walk_B22_-_Side_step_left_stageii": 1.0,
            "Walk_B23_-_Side_step_right_stageii": 1.0,
            "Walk_B4_-_Stand_to_Walk_Back_stageii": 1.0,
        }

        # ------------------------------------------------------
        # animation
        # ------------------------------------------------------
        self.animation.animation.num_steps_to_use = AMP_NUM_STEPS

        # -----------------------------------------------------
        # Observations
        # -----------------------------------------------------
        
        # policy observations
        
        self.observations.policy.key_body_pos_b.params = {
            "asset_cfg": SceneEntityCfg(
                name="robot", 
                body_names=KEY_BODY_NAMES, 
                preserve_order=True
            )
        }
        
        # critic observations
        
        self.observations.critic.key_body_pos_b.params = {
            "asset_cfg": SceneEntityCfg(
                name="robot", 
                body_names=KEY_BODY_NAMES, 
                preserve_order=True
            )
        }
        
        # discriminator observations
        
        self.observations.disc.key_body_pos_b.params = {
            "asset_cfg": SceneEntityCfg(
                name="robot", 
                body_names=KEY_BODY_NAMES, 
                preserve_order=True
            )
        }
        self.observations.disc.history_length = AMP_NUM_STEPS
        
        # discriminator demostration observations
        
        self.observations.disc_demo.ref_root_local_rot_tan_norm.params["animation"] = ANIMATION_TERM_NAME
        self.observations.disc_demo.ref_root_lin_vel_b.params["animation"] = ANIMATION_TERM_NAME
        self.observations.disc_demo.ref_root_ang_vel_b.params["animation"] = ANIMATION_TERM_NAME
        self.observations.disc_demo.ref_joint_pos.params["animation"] = ANIMATION_TERM_NAME
        self.observations.disc_demo.ref_joint_vel.params["animation"] = ANIMATION_TERM_NAME
        self.observations.disc_demo.ref_key_body_pos_b.params["animation"] = ANIMATION_TERM_NAME

        # ------------------------------------------------------
        # Events
        # ------------------------------------------------------
        self.events.add_base_mass.params["asset_cfg"].body_names = "torso_link"
        self.events.base_external_force_torque.params["asset_cfg"].body_names = ["torso_link"]
        self.events.reset_from_ref.params = {
            "animation": ANIMATION_TERM_NAME,
            "height_offset": 0.1
        }
        
        # ------------------------------------------------------
        # Rewards
        # ------------------------------------------------------
        
        # ------------------------------------------------------
        # Commands
        # ------------------------------------------------------
        self.commands.base_velocity.ranges.lin_vel_x = (-0.5, 3.0)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.5, 0.5)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)
        self.commands.base_velocity.ranges.heading = (-math.pi, math.pi)
        
        # ------------------------------------------------------
        # Curriculum
        # ------------------------------------------------------
        self.curriculum.lin_vel_cmd_levels = None
        self.curriculum.ang_vel_cmd_levels = None
        
        # ------------------------------------------------------
        # terminations
        # ------------------------------------------------------
        self.terminations.base_contact = None
        # self.terminations.base_contact = DoneTerm(
        #     func=mdp.illegal_contact,
        #     params={
        #         "sensor_cfg": SceneEntityCfg(
        #             "contact_forces",
        #             body_names=[
        #                 "pelvis",
        #                 "waist_yaw_link",
        #                 "waist_roll_link",
        #                 "torso_link",
        #                 ".*_shoulder_.*_link",
        #                 ".*_elbow_link",
        #             ],
        #         ),
        #         "threshold": 1.0,
        #     },
        # )
        self.terminations.base_height.params["minimum_height"] = float(
            os.environ.get("LEGGED_LAB_G1_MIN_ROOT_HEIGHT", str(G1_MIN_ROOT_HEIGHT))
        )


@configclass
class G1AmpEnvCfg_PLAY(G1AmpEnvCfg):
    
    def __post_init__(self):
        super().__post_init__()
        
        self.scene.num_envs = 48 
        self.scene.env_spacing = 2.5
        
        self.commands.base_velocity.ranges.lin_vel_x = (0.5, 3.0)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.5, 0.5)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)
        self.commands.base_velocity.ranges.heading = (0.0, 0.0)
        
        self.events.reset_from_ref = None


@configclass
class G1AmpHeightScanEnvCfg(G1AmpEnvCfg):
    """G1 AMP environment with a flattened terrain height scan in actor and critic observations."""

    def __post_init__(self):
        super().__post_init__()
        enable_height_scan_observations(self)


@configclass
class G1AmpHeightScanEnvCfg_PLAY(G1AmpHeightScanEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.scene.num_envs = 48
        self.scene.env_spacing = 2.5
        self.commands.base_velocity.ranges.lin_vel_x = (0.5, 3.0)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.5, 0.5)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)
        self.commands.base_velocity.ranges.heading = (0.0, 0.0)
        self.events.reset_from_ref = None


@configclass
class G1MotionBricksAmpEnvCfg(G1AmpEnvCfg):
    """G1 AMP environment trained from converted MotionBricks motion priors."""

    def __post_init__(self):
        super().__post_init__()

        motion_data_dir = Path(
            os.environ.get(
                "LEGGED_LAB_MOTIONBRICKS_G1_DIR",
                "~/Documents/shared_datasets/motionbricks/motionbricks_sonic_grid_walk_dense/legged_lab_g1",
            )
        ).expanduser()
        if not motion_data_dir.exists():
            raise FileNotFoundError(
                f"Converted MotionBricks data directory does not exist: {motion_data_dir}. "
                "Run scripts/tools/motionbricks/convert_motionbricks_to_legged_lab.py first, "
                "or set LEGGED_LAB_MOTIONBRICKS_G1_DIR to the converted .pkl directory."
            )

        motion_names = sorted(path.stem for path in motion_data_dir.glob("*.pkl"))
        if not motion_names:
            raise FileNotFoundError(
                f"No converted MotionBricks .pkl files found in: {motion_data_dir}. "
                "Run scripts/tools/motionbricks/convert_motionbricks_to_legged_lab.py first."
            )

        self.motion_data.motion_dataset.motion_data_dir = str(motion_data_dir)
        self.motion_data.motion_dataset.motion_data_weights = {name: 1.0 for name in motion_names}

        self.commands.base_velocity.ranges.lin_vel_x = (-1.0, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (-1.0, 1.0)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)


def _make_still_motion(source_path: Path, output_path: Path, num_frames: int):
    motion = joblib.load(source_path)
    root_pos = np.asarray(motion["root_pos"], dtype=np.float32)
    root_rot = np.asarray(motion["root_rot"], dtype=np.float32)
    dof_pos = np.asarray(motion["dof_pos"], dtype=np.float32)
    key_body_pos = np.asarray(motion["key_body_pos"], dtype=np.float32)

    if len(root_pos) > 1:
        root_speed = np.linalg.norm(np.diff(root_pos[:, :2], axis=0), axis=1)
        frame_id = int(np.argmin(root_speed))
    else:
        frame_id = 0

    still_motion = {
        "fps": float(motion["fps"]),
        "root_pos": np.repeat(root_pos[frame_id : frame_id + 1], num_frames, axis=0),
        "root_rot": np.repeat(root_rot[frame_id : frame_id + 1], num_frames, axis=0),
        "dof_pos": np.repeat(dof_pos[frame_id : frame_id + 1], num_frames, axis=0),
        "key_body_pos": np.repeat(key_body_pos[frame_id : frame_id + 1], num_frames, axis=0),
        "loop_mode": 1,
        "vx": 0.0,
        "vy": 0.0,
        "omega": 0.0,
        "speed": 0.0,
    }
    joblib.dump(still_motion, output_path)


def _ensure_motionbricks_with_still_dir(
    motionbricks_motion_dir: Path,
    still_motion_dir: Path,
    still_ratio: float,
    still_count: int,
    still_frames: int,
) -> dict[str, float]:
    """Create a symlink/cached directory with MotionBricks clips plus generated still clips."""
    if not 0.0 <= still_ratio < 1.0:
        raise ValueError(f"Still ratio must be in [0, 1), got {still_ratio}.")
    if not motionbricks_motion_dir.exists():
        raise FileNotFoundError(f"MotionBricks motion directory does not exist: {motionbricks_motion_dir}")

    motion_files = sorted(motionbricks_motion_dir.glob("*.pkl"))
    if not motion_files:
        raise FileNotFoundError(f"No MotionBricks .pkl files found in: {motionbricks_motion_dir}")

    still_motion_dir.mkdir(parents=True, exist_ok=True)
    motion_weights = {}

    original_total_weight = 1.0 - still_ratio
    original_per_motion_weight = original_total_weight / len(motion_files)
    for source_path in motion_files:
        link_name = f"motion__{source_path.name}"
        link_path = still_motion_dir / link_name
        if link_path.exists() or link_path.is_symlink():
            if link_path.resolve() != source_path.resolve():
                link_path.unlink()
                link_path.symlink_to(source_path)
        else:
            link_path.symlink_to(source_path)
        motion_weights[link_path.stem] = original_per_motion_weight

    if still_ratio > 0.0:
        still_sources = motion_files[: max(1, min(still_count, len(motion_files)))]
        still_per_motion_weight = still_ratio / len(still_sources)
        for source_path in still_sources:
            still_path = still_motion_dir / f"still__{source_path.stem}.pkl"
            if not still_path.exists():
                _make_still_motion(source_path, still_path, still_frames)
            motion_weights[still_path.stem] = still_per_motion_weight

    return motion_weights


@configclass
class G1MotionBricksStyleHandsAmpEnvCfg(G1MotionBricksAmpEnvCfg):
    """MotionBricks-only AMP task with generated still references for style refinement."""

    def __post_init__(self):
        super().__post_init__()

        still_ratio = float(os.environ.get("LEGGED_LAB_STYLE_HANDS_STILL_RATIO", "0.2"))
        still_count = int(os.environ.get("LEGGED_LAB_STYLE_HANDS_STILL_COUNT", "32"))
        still_frames = int(os.environ.get("LEGGED_LAB_STYLE_HANDS_STILL_FRAMES", "120"))
        motionbricks_motion_dir = Path(self.motion_data.motion_dataset.motion_data_dir).expanduser()
        still_motion_dir = Path(
            os.environ.get(
                "LEGGED_LAB_STYLE_HANDS_MOTION_DIR",
                "~/Documents/shared_datasets/motionbricks/motionbricks_sonic_grid_walk_dense/legged_lab_g1_style_hands",
            )
        ).expanduser()

        self.motion_data.motion_dataset.motion_data_dir = str(still_motion_dir)
        self.motion_data.motion_dataset.motion_data_weights = _ensure_motionbricks_with_still_dir(
            motionbricks_motion_dir=motionbricks_motion_dir,
            still_motion_dir=still_motion_dir,
            still_ratio=still_ratio,
            still_count=still_count,
            still_frames=still_frames,
        )
        self.commands.base_velocity.rel_standing_envs = still_ratio


def _set_weighted_motion_dataset(
    cfg: LocomotionAmpEnvCfg,
    motion_data_dir: Path,
    motion_names_by_prefix: dict[str, list[str]],
    dataset_weights: dict[str, float],
):
    """Assign one motion directory while preserving dataset-level sampling weights."""
    motion_weights = {}
    for prefix, motion_names in motion_names_by_prefix.items():
        if not motion_names:
            continue
        dataset_weight = dataset_weights[prefix]
        per_motion_weight = dataset_weight / len(motion_names)
        motion_weights.update({name: per_motion_weight for name in motion_names})

    if not motion_weights:
        raise FileNotFoundError(f"No motion files found for mixed dataset directory: {motion_data_dir}")

    cfg.motion_data.motion_dataset.motion_data_dir = str(motion_data_dir)
    cfg.motion_data.motion_dataset.motion_data_weights = motion_weights


def _ensure_mixed_g1_motion_dir(
    default_motion_dir: Path,
    motionbricks_motion_dir: Path,
    mixed_motion_dir: Path,
):
    """Create a symlink-only mixed motion directory with prefixed names."""
    mixed_motion_dir.mkdir(parents=True, exist_ok=True)
    sources = {
        "default": default_motion_dir,
        "motionbricks": motionbricks_motion_dir,
    }
    motion_names_by_prefix = {prefix: [] for prefix in sources}

    for prefix, source_dir in sources.items():
        if not source_dir.exists():
            raise FileNotFoundError(f"Motion source directory does not exist: {source_dir}")
        motion_files = sorted(source_dir.glob("*.pkl"))
        if not motion_files:
            raise FileNotFoundError(f"No .pkl motion files found in: {source_dir}")

        for source_path in motion_files:
            link_name = f"{prefix}__{source_path.name}"
            link_path = mixed_motion_dir / link_name
            if link_path.exists() or link_path.is_symlink():
                if link_path.resolve() == source_path.resolve():
                    pass
                else:
                    link_path.unlink()
                    link_path.symlink_to(source_path)
            else:
                link_path.symlink_to(source_path)
            motion_names_by_prefix[prefix].append(link_path.stem)

    return motion_names_by_prefix


@configclass
class G1MixedAmpEnvCfg(G1AmpEnvCfg):
    """G1 AMP environment trained from a weighted mix of default AMP and MotionBricks priors."""

    def __post_init__(self):
        super().__post_init__()

        default_motion_dir = Path(
            os.environ.get(
                "LEGGED_LAB_DEFAULT_G1_AMP_DIR",
                os.path.join(LEGGED_LAB_ROOT_DIR, "data", "MotionData", "g1_29dof", "amp", "walk_and_run"),
            )
        ).expanduser()
        motionbricks_motion_dir = Path(
            os.environ.get(
                "LEGGED_LAB_MOTIONBRICKS_G1_DIR",
                "~/Documents/shared_datasets/motionbricks/motionbricks_sonic_grid_walk_dense/legged_lab_g1",
            )
        ).expanduser()
        mixed_motion_dir = Path(
            os.environ.get(
                "LEGGED_LAB_MIXED_G1_AMP_DIR",
                "~/Documents/shared_datasets/legged_lab_g1_mixed_default_motionbricks",
            )
        ).expanduser()

        default_weight = float(os.environ.get("LEGGED_LAB_MIXED_G1_DEFAULT_WEIGHT", "0.2"))
        motionbricks_weight = float(os.environ.get("LEGGED_LAB_MIXED_G1_MOTIONBRICKS_WEIGHT", "0.8"))
        total_weight = default_weight + motionbricks_weight
        if total_weight <= 0.0:
            raise ValueError("Mixed G1 dataset weights must sum to a positive value.")
        dataset_weights = {
            "default": default_weight / total_weight,
            "motionbricks": motionbricks_weight / total_weight,
        }

        motion_names_by_prefix = _ensure_mixed_g1_motion_dir(
            default_motion_dir=default_motion_dir,
            motionbricks_motion_dir=motionbricks_motion_dir,
            mixed_motion_dir=mixed_motion_dir,
        )
        _set_weighted_motion_dataset(
            cfg=self,
            motion_data_dir=mixed_motion_dir,
            motion_names_by_prefix=motion_names_by_prefix,
            dataset_weights=dataset_weights,
        )

        self.commands.base_velocity.ranges.lin_vel_x = (-1.0, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (-1.0, 1.0)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)


@configclass
class G1MixedVelocityTunedAmpEnvCfg(G1MixedAmpEnvCfg):
    """Mixed G1 AMP environment with reward weights biased toward command tracking."""

    def __post_init__(self):
        super().__post_init__()

        self.rewards.track_lin_vel_xy_exp.weight = 1.25
        self.rewards.track_ang_vel_z_exp.weight = 2.0
        self.rewards.feet_air_time.weight = 0.3


@configclass
class G1MixedStandScaledAmpEnvCfg(G1MixedAmpEnvCfg):
    """Mixed G1 AMP task with fixed standing commands and command-scaled posture rewards."""

    rewards: G1StandScaledAmpRewards = G1StandScaledAmpRewards()

    def __post_init__(self):
        super().__post_init__()

        self.commands.base_velocity.rel_standing_envs = float(
            os.environ.get("LEGGED_LAB_STAND_SCALED_STANDING_RATIO", "0.15")
        )
        self.rewards.root_height_below_target.params["target_height"] = float(
            os.environ.get("LEGGED_LAB_STAND_SCALED_ROOT_HEIGHT", "0.75")
        )


@configclass
class G1MixedAmpHistoryEnvCfg(G1MixedStandScaledAmpEnvCfg):
    """Mixed G1 AMP task with full actor observation history and no terrain perception."""

    def __post_init__(self):
        super().__post_init__()
        enable_packed_actor_policy_history(self)
        enable_current_critic_observations(self)
        gate_stand_scaled_rewards_after_mean_episode_length(
            self,
            min_mean_episode_length=float(
                os.environ.get(
                    "LEGGED_LAB_STAND_REWARD_MIN_MEAN_EPISODE_LENGTH",
                    str(STAND_REWARD_MIN_MEAN_EPISODE_LENGTH),
                )
            ),
        )


@configclass
class G1MixedAmpHeightScanEnvCfg(G1MixedStandScaledAmpEnvCfg):
    """Mixed G1 AMP environment with stand-scaled rewards and RPL-style terrain height scans."""

    def __post_init__(self):
        super().__post_init__()
        enable_packed_actor_policy_history(self)
        enable_height_scan_observations(self, policy_proprio_history_length=None)
        gate_stand_scaled_rewards_after_mean_episode_length(
            self,
            min_mean_episode_length=0.0,
            min_learning_iteration=STAND_REWARD_MIN_LEARNING_ITERATION,
        )


@configclass
class G1MixedAmpHeightScanNoHistoryEnvCfg(G1MixedStandScaledAmpEnvCfg):
    """Mixed G1 AMP height-scan task with current-state actor observations only."""

    def __post_init__(self):
        super().__post_init__()
        enable_height_scan_observations(self, policy_proprio_history_length=None)
        gate_stand_scaled_rewards_after_mean_episode_length(
            self,
            min_mean_episode_length=float(
                os.environ.get(
                    "LEGGED_LAB_STAND_REWARD_MIN_MEAN_EPISODE_LENGTH",
                    str(STAND_REWARD_MIN_MEAN_EPISODE_LENGTH),
                )
            ),
        )


@configclass
class G1MixedAmpEnvCfg_PLAY(G1MixedAmpEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.scene.num_envs = 48
        self.scene.env_spacing = 2.5


@configclass
class G1MixedAmpHeightScanEnvCfg_PLAY(G1MixedAmpHeightScanEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.scene.num_envs = 48
        self.scene.env_spacing = 2.5


@configclass
class G1MotionBricksAmpEnvCfg_PLAY(G1MotionBricksAmpEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.scene.num_envs = 48
        self.scene.env_spacing = 2.5


@configclass
class G1MotionBricksAmpHeightScanEnvCfg(G1MotionBricksAmpEnvCfg):
    """MotionBricks G1 AMP environment with a flattened terrain height scan."""

    def __post_init__(self):
        super().__post_init__()
        enable_height_scan_observations(self)


@configclass
class G1MotionBricksAmpHeightScanEnvCfg_PLAY(G1MotionBricksAmpHeightScanEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.scene.num_envs = 48
        self.scene.env_spacing = 2.5
