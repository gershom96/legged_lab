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


#########################
# Runner configurations #
#########################
