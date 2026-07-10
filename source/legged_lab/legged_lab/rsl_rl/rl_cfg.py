from __future__ import annotations

from dataclasses import MISSING

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg, RslRlOnPolicyRunnerCfg
from .amp_cfg import RslRlAmpCfg

#########################
# Policy configurations #
#########################

@configclass
class RslRlPpoActorCriticConv2dCfg(RslRlPpoActorCriticCfg):
    """Configuration for the PPO actor-critic networks with convolutional layers."""

    class_name: str = "ActorCriticConv2d"
    """The policy class name. Default is ActorCriticConv2d."""

    conv_layers_params: list[dict] = [
        {"out_channels": 4, "kernel_size": 3, "stride": 2},
        {"out_channels": 8, "kernel_size": 3, "stride": 2},
        {"out_channels": 16, "kernel_size": 3, "stride": 2},
    ]
    """List of convolutional layer parameters for the convolutional network."""

    conv_linear_output_size: int = 16
    """Output size of the linear layer after the convolutional features are flattened."""


@configclass
class RslRlPpoActorCriticHeightScanCfg(RslRlPpoActorCriticCfg):
    """Configuration for actor-critic networks with flattened height-scan encoders."""

    class_name: str = "ActorCriticHeightScan"
    """The policy class name."""

    height_scan_shape: tuple[int, int] = (17, 11)
    """Shape used to reshape the flattened height scan into [C, H, W]."""

    critic_height_scan_history_length: int = 5
    """Number of height-scan frames stacked as critic CNN input channels."""

    height_scan_ordering: str = "xy"
    """GridPattern ordering used by the flattened height scan."""

    actor_cnn_cfg: dict = {
        "output_channels": [16, 32, 64],
        "kernel_size": [3, 3, 3],
        "stride": [1, 2, 2],
        "padding": "zeros",
        "global_pool": "avg",
        "flatten": True,
    }
    """CNN configuration for the actor height-scan encoder."""

    critic_cnn_cfg: dict = {
        "output_channels": [16, 32, 64],
        "kernel_size": [3, 3, 3],
        "stride": [1, 2, 2],
        "padding": "zeros",
        "global_pool": "avg",
        "flatten": True,
    }
    """CNN configuration for the critic height-scan encoder."""


@configclass
class RslRlPpoActorCriticSplitHeightScanCfg(RslRlPpoActorCriticHeightScanCfg):
    """Height-scan actor-critic with separate lower-body and upper-body actor heads."""

    class_name: str = "ActorCriticSplitHeightScan"
    """The policy class name."""

    lower_action_indices: tuple[int, ...] = (
        0,
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        10,
        13,
        14,
        17,
        18,
    )
    """G1 lower-body plus waist action indices in Isaac Lab action order."""

    upper_action_indices: tuple[int, ...] = (
        11,
        12,
        15,
        16,
        19,
        20,
        21,
        22,
        23,
        24,
        25,
        26,
        27,
        28,
    )
    """G1 upper-body action indices in Isaac Lab action order."""

    lower_init_noise_std: float = 1.0
    """Initial Gaussian std for the lower-body actor head."""

    upper_init_noise_std: float = 0.6
    """Initial Gaussian std for the upper-body actor head."""

    num_value_outputs: int = 2
    """Number of critic value outputs. Split-policy PPO uses lower and upper value streams."""

############################
# Algorithm configurations #
############################


@configclass
class RslRlPpoAmpAlgorithmCfg(RslRlPpoAlgorithmCfg):
    """Configuration for the AMP algorithm."""
    class_name: str = "PPOAmp"
    """The algorithm class name. Default is PPOAmp."""

    amp_cfg: RslRlAmpCfg = RslRlAmpCfg()
    """Configuration for the AMP (Adversarial Motion Priors) in the training."""

    split_reward_cfg: dict | None = None
    """Optional lower/upper reward routing configuration for split-policy AMP."""


#########################
# Runner configurations #
#########################
