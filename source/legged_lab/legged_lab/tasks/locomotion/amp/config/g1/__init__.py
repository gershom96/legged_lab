import gymnasium as gym

from . import agents

from legged_lab.envs import ManagerBasedAmpEnv

gym.register(
    id="LeggedLab-Isaac-AMP-G1-v0",
    entry_point="legged_lab.envs:ManagerBasedAmpEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.g1_amp_env_cfg:G1AmpEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1RslRlOnPolicyRunnerAmpCfg",
    },
)

gym.register(
    id="LeggedLab-Isaac-AMP-G1-Play-v0",
    entry_point="legged_lab.envs:ManagerBasedAmpEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.g1_amp_env_cfg:G1AmpEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1RslRlOnPolicyRunnerAmpCfg",
    },
)

gym.register(
    id="LeggedLab-Isaac-AMP-G1-HeightScan-v0",
    entry_point="legged_lab.envs:ManagerBasedAmpEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.g1_amp_env_cfg:G1AmpHeightScanEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1HeightScanRslRlOnPolicyRunnerAmpCfg",
    },
)

gym.register(
    id="LeggedLab-Isaac-AMP-G1-HeightScan-Play-v0",
    entry_point="legged_lab.envs:ManagerBasedAmpEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.g1_amp_env_cfg:G1AmpHeightScanEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1HeightScanRslRlOnPolicyRunnerAmpCfg",
    },
)

gym.register(
    id="LeggedLab-Isaac-AMP-G1-MotionBricks-v0",
    entry_point="legged_lab.envs:ManagerBasedAmpEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.g1_amp_env_cfg:G1MotionBricksAmpEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1MotionBricksRslRlOnPolicyRunnerAmpCfg",
    },
)

gym.register(
    id="LeggedLab-Isaac-AMP-G1-MotionBricks-HeightScan-v0",
    entry_point="legged_lab.envs:ManagerBasedAmpEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.g1_amp_env_cfg:G1MotionBricksAmpHeightScanEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1MotionBricksHeightScanRslRlOnPolicyRunnerAmpCfg",
    },
)

gym.register(
    id="LeggedLab-Isaac-AMP-G1-MotionBricks-SoftDisc-v0",
    entry_point="legged_lab.envs:ManagerBasedAmpEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.g1_amp_env_cfg:G1MotionBricksAmpEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1MotionBricksSoftAmpRslRlOnPolicyRunnerAmpCfg",
    },
)

gym.register(
    id="LeggedLab-Isaac-AMP-G1-MotionBricks-HeightScan-Play-v0",
    entry_point="legged_lab.envs:ManagerBasedAmpEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.g1_amp_env_cfg:G1MotionBricksAmpHeightScanEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1MotionBricksHeightScanRslRlOnPolicyRunnerAmpCfg",
    },
)

gym.register(
    id="LeggedLab-Isaac-AMP-G1-MotionBricks-Play-v0",
    entry_point="legged_lab.envs:ManagerBasedAmpEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.g1_amp_env_cfg:G1MotionBricksAmpEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1MotionBricksRslRlOnPolicyRunnerAmpCfg",
    },
)
