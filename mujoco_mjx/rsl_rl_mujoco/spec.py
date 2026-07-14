from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class HeightScanSpec:
    enabled: bool = True
    size: tuple[float, float] = (1.6, 1.0)
    resolution: float = 0.1
    offset: float = 0.5
    noise: float = 0.01
    critic_history_length: int = 5

    def __post_init__(self) -> None:
        if not self.enabled:
            raise ValueError("The perception-enabled MuJoCo task requires observations.height_scan=true")
        if len(self.size) != 2 or min(self.size) <= 0.0:
            raise ValueError(f"height-scan size must contain two positive values, got {self.size}")
        if self.resolution <= 0.0:
            raise ValueError(f"height-scan resolution must be positive, got {self.resolution}")
        if self.noise < 0.0:
            raise ValueError(f"height-scan noise must be non-negative, got {self.noise}")
        if self.critic_history_length <= 0:
            raise ValueError("critic height-scan history length must be positive")

    @property
    def shape(self) -> tuple[int, int]:
        return (
            int(round(self.size[0] / self.resolution)) + 1,
            int(round(self.size[1] / self.resolution)) + 1,
        )

    @property
    def dim(self) -> int:
        x_points, y_points = self.shape
        return x_points * y_points


@dataclass(frozen=True)
class DomainRandomizationSpec:
    """Optional MuJoCo physics randomization and interval pushes."""

    enabled: bool = False
    static_friction_range: tuple[float, float] = (0.3, 1.6)
    dynamic_friction_range: tuple[float, float] = (0.3, 1.2)
    material_num_buckets: int = 64
    base_mass_add_range: tuple[float, float] = (-3.0, 3.0)
    com_range: tuple[float, float] = (-0.03, 0.03)
    limb_mass_scale_range: tuple[float, float] = (0.8, 1.2)
    actuator_gain_scale_range: tuple[float, float] = (0.8, 1.2)
    armature_scale_range: tuple[float, float] = (0.8, 1.2)
    push_interval_s: tuple[float, float] = (10.0, 20.0)
    push_linear_velocity_range: tuple[float, float] = (-0.5, 0.5)
    push_yaw_velocity_range: tuple[float, float] = (-1.0, 1.0)

    def __post_init__(self) -> None:
        ranges = (
            self.static_friction_range,
            self.dynamic_friction_range,
            self.base_mass_add_range,
            self.com_range,
            self.limb_mass_scale_range,
            self.actuator_gain_scale_range,
            self.armature_scale_range,
            self.push_interval_s,
            self.push_linear_velocity_range,
            self.push_yaw_velocity_range,
        )
        if any(len(value) != 2 or value[0] > value[1] for value in ranges):
            raise ValueError("domain-randomization ranges must be ordered pairs")
        if self.push_interval_s[0] <= 0.0:
            raise ValueError("domain-randomization push interval must be positive")
        if self.material_num_buckets <= 0:
            raise ValueError("domain-randomization material_num_buckets must be positive")


@dataclass(frozen=True)
class TerrainCurriculumSpec:
    enabled: bool = False
    safe_half_extent: float = 20.0
    start_stage: int = 0
    current_probability: float = 0.70
    previous_probability: float = 0.20
    easier_probability: float = 0.10
    promotion_window: int = 100
    min_stage_iterations: int = 400
    min_episode_length: float = 900.0
    min_lin_tracking: float = 0.60
    min_ang_tracking: float = 0.20
    max_termination_rate: float = 0.05

    def __post_init__(self) -> None:
        if self.safe_half_extent <= 0.0:
            raise ValueError("terrain curriculum safe_half_extent must be positive")
        if not 0 <= self.start_stage <= 5:
            raise ValueError("terrain curriculum start_stage must be in [0, 5]")
        probabilities = (
            self.current_probability,
            self.previous_probability,
            self.easier_probability,
        )
        if min(probabilities) < 0.0 or abs(sum(probabilities) - 1.0) > 1.0e-6:
            raise ValueError("terrain curriculum sampling probabilities must be non-negative and sum to one")
        if self.promotion_window <= 0 or self.min_stage_iterations < 0:
            raise ValueError("terrain curriculum promotion window/dwell settings are invalid")
        if self.min_episode_length <= 0.0:
            raise ValueError("terrain curriculum min_episode_length must be positive")
        if not 0.0 <= self.max_termination_rate <= 1.0:
            raise ValueError("terrain curriculum max_termination_rate must be in [0, 1]")


@dataclass(frozen=True)
class TerrainSpec:
    type: str = "flat"
    size: tuple[float, float] = (45.0, 45.0)
    horizontal_scale: float = 0.1
    difficulty: float = 0.0
    seed: int = 0
    center_platform_radius: float = 1.0
    curriculum: TerrainCurriculumSpec = field(default_factory=TerrainCurriculumSpec)
    ean_root: Path | None = None
    ean_scene_config: Path | None = None
    wfc_output_dir: Path | None = None

    def __post_init__(self) -> None:
        if len(self.size) != 2 or min(self.size) <= 0.0:
            raise ValueError(f"terrain size must contain two positive values, got {self.size}")
        if self.horizontal_scale <= 0.0:
            raise ValueError(f"terrain horizontal scale must be positive, got {self.horizontal_scale}")
        if not 0.0 <= self.difficulty <= 1.0:
            raise ValueError(f"terrain difficulty must be in [0, 1], got {self.difficulty}")
        if self.center_platform_radius < 0.0:
            raise ValueError("terrain center platform radius must be non-negative")
        if self.curriculum.enabled and self.type != "curriculum":
            raise ValueError("terrain curriculum requires terrain.type='curriculum'")
        if self.curriculum.enabled and 2.0 * self.curriculum.safe_half_extent >= min(self.size):
            raise ValueError("terrain curriculum safe area must leave a non-zero flat patch border")


@dataclass(frozen=True)
class MotionSpec:
    directory: Path
    default_weight: float = 0.2
    motionbricks_weight: float = 0.8
    max_files: int = 0
    seed: int = 0

    def __post_init__(self) -> None:
        if self.default_weight < 0.0 or self.motionbricks_weight < 0.0:
            raise ValueError("motion dataset weights must be non-negative")
        if self.default_weight + self.motionbricks_weight <= 0.0:
            raise ValueError("motion dataset weights must sum to a positive value")
        if self.max_files < 0:
            raise ValueError("motion.max_files must be zero or positive")


@dataclass(frozen=True)
class MujocoRslRlEnvSpec:
    robot_xml: Path
    motion: MotionSpec
    terrain: TerrainSpec = field(default_factory=TerrainSpec)
    height_scan: HeightScanSpec = field(default_factory=HeightScanSpec)
    domain_randomization: DomainRandomizationSpec = field(default_factory=DomainRandomizationSpec)
    num_envs: int = 4096
    device: str = "cuda:0"
    sim_dt: float = 0.005
    decimation: int = 4
    episode_length_s: float = 20.0
    action_dim: int = 29
    actor_history_length: int = 5
    discriminator_history_length: int = 4
    observation_layout: str = "isaac_scan_first_v2"
    split_policy: bool = True
    upper_body_amp: bool = True
    mjx_impl: str = "warp"
    naconmax: int = 1024
    contacts_per_env: int = 32
    njmax: int = 2048
    seed: int = 42
    command_resampling_time_s: float = 10.0
    standing_env_ratio: float = 0.15
    command_lin_vel_x: tuple[float, float] = (-1.0, 1.0)
    command_lin_vel_y: tuple[float, float] = (-1.0, 1.0)
    command_ang_vel_z: tuple[float, float] = (-1.0, 1.0)
    min_root_height: float = 0.2
    max_tilt_radians: float = 1.0471975511965976
    root_height_reward_start_iteration: int = 40000

    def __post_init__(self) -> None:
        if self.num_envs <= 0:
            raise ValueError("env.num_envs must be positive")
        if self.device != "cpu" and not self.device.startswith("cuda"):
            raise ValueError(f"env.device must be 'cpu' or a CUDA device, got {self.device!r}")
        if self.sim_dt <= 0.0 or self.decimation <= 0:
            raise ValueError("env.sim_dt and env.decimation must be positive")
        if self.episode_length_s <= 0.0:
            raise ValueError("env.episode_length_s must be positive")
        if self.action_dim != 29:
            raise ValueError(f"The G1 adapter requires action_dim=29, got {self.action_dim}")
        if self.actor_history_length <= 0 or self.discriminator_history_length <= 0:
            raise ValueError("actor and discriminator history lengths must be positive")
        if self.observation_layout not in {"isaac_scan_first_v2", "legacy_native_history_first_v1"}:
            raise ValueError(
                "observation_layout must be 'isaac_scan_first_v2' or "
                "'legacy_native_history_first_v1'"
            )
        if not self.split_policy or not self.upper_body_amp:
            raise ValueError(
                "This launcher currently implements the verified split-policy, upper-body-AMP task only"
            )
        if self.naconmax <= 0 or self.contacts_per_env <= 0 or self.njmax <= 0:
            raise ValueError("Warp contact and constraint capacities must be positive")
        if self.command_resampling_time_s <= 0.0:
            raise ValueError("command resampling time must be positive")
        if not 0.0 <= self.standing_env_ratio <= 1.0:
            raise ValueError("standing_env_ratio must be in [0, 1]")
        if self.min_root_height <= 0.0 or not 0.0 < self.max_tilt_radians < 3.141592653589793:
            raise ValueError("termination height and tilt limits are invalid")
        if self.root_height_reward_start_iteration < 0:
            raise ValueError("root-height reward start iteration must be non-negative")

    @property
    def step_dt(self) -> float:
        return self.sim_dt * self.decimation

    @property
    def max_episode_length(self) -> int:
        return int(round(self.episode_length_s / self.step_dt))

    @property
    def warp_naconmax(self) -> int:
        """Global Warp contact pool sized for the complete environment batch."""
        return max(self.naconmax, self.contacts_per_env * self.num_envs)

    @classmethod
    def from_mapping(cls, payload: dict[str, Any], root: Path) -> "MujocoRslRlEnvSpec":
        env_cfg = dict(payload.get("env", {}))
        obs_cfg = dict(payload.get("observations", {}))
        terrain_cfg = dict(payload.get("terrain", {}))
        domain_randomization_cfg = dict(payload.get("domain_randomization", {}))
        curriculum_cfg = dict(terrain_cfg.pop("curriculum", {}))
        motion_cfg = dict(payload.get("motion", {}))

        def path_value(value: str | None) -> Path | None:
            if not value:
                return None
            path = Path(value).expanduser()
            return path if path.is_absolute() else (root / path).resolve()

        robot_xml = path_value(env_cfg.pop("robot_xml"))
        if robot_xml is None:
            raise ValueError("env.robot_xml is required")
        motion_dir = path_value(motion_cfg.pop("directory"))
        if motion_dir is None:
            raise ValueError("motion.directory is required")

        actor_history_length = int(obs_cfg.pop("actor_history_length", 5))
        discriminator_history_length = int(obs_cfg.pop("discriminator_history_length", 4))
        observation_layout = str(obs_cfg.pop("observation_layout", "isaac_scan_first_v2"))
        height_scan = HeightScanSpec(
            enabled=bool(obs_cfg.pop("height_scan", True)),
            size=tuple(obs_cfg.pop("height_scan_size", (1.6, 1.0))),
            resolution=float(obs_cfg.pop("height_scan_resolution", 0.1)),
            offset=float(obs_cfg.pop("height_scan_offset", 0.5)),
            noise=float(obs_cfg.pop("height_scan_noise", 0.01)),
            critic_history_length=int(obs_cfg.pop("critic_height_scan_history_length", 5)),
        )

        def pair(mapping: dict[str, Any], key: str, default: tuple[float, float]) -> tuple[float, float]:
            return tuple(float(value) for value in mapping.pop(key, default))  # type: ignore[return-value]

        domain_randomization = DomainRandomizationSpec(
            enabled=bool(domain_randomization_cfg.pop("enabled", False)),
            static_friction_range=pair(domain_randomization_cfg, "static_friction_range", (0.3, 1.6)),
            dynamic_friction_range=pair(domain_randomization_cfg, "dynamic_friction_range", (0.3, 1.2)),
            material_num_buckets=int(domain_randomization_cfg.pop("material_num_buckets", 64)),
            base_mass_add_range=pair(domain_randomization_cfg, "base_mass_add_range", (-3.0, 3.0)),
            com_range=pair(domain_randomization_cfg, "com_range", (-0.03, 0.03)),
            limb_mass_scale_range=pair(domain_randomization_cfg, "limb_mass_scale_range", (0.8, 1.2)),
            actuator_gain_scale_range=pair(domain_randomization_cfg, "actuator_gain_scale_range", (0.8, 1.2)),
            armature_scale_range=pair(domain_randomization_cfg, "armature_scale_range", (0.8, 1.2)),
            push_interval_s=pair(domain_randomization_cfg, "push_interval_s", (10.0, 20.0)),
            push_linear_velocity_range=pair(domain_randomization_cfg, "push_linear_velocity_range", (-0.5, 0.5)),
            push_yaw_velocity_range=pair(domain_randomization_cfg, "push_yaw_velocity_range", (-1.0, 1.0)),
        )
        terrain = TerrainSpec(
            type=str(terrain_cfg.pop("type", "flat")),
            size=tuple(terrain_cfg.pop("size", (45.0, 45.0))),
            horizontal_scale=float(terrain_cfg.pop("horizontal_scale", 0.1)),
            difficulty=float(terrain_cfg.pop("difficulty", 0.0)),
            seed=int(terrain_cfg.pop("seed", env_cfg.get("seed", 42))),
            center_platform_radius=float(terrain_cfg.pop("center_platform_radius", 1.0)),
            curriculum=TerrainCurriculumSpec(**curriculum_cfg),
            ean_root=path_value(terrain_cfg.pop("ean_root", None)),
            ean_scene_config=path_value(terrain_cfg.pop("ean_scene_config", None)),
            wfc_output_dir=path_value(terrain_cfg.pop("wfc_output_dir", None)),
        )
        motion = MotionSpec(directory=motion_dir, **motion_cfg)
        if terrain_cfg:
            raise ValueError(f"Unknown terrain configuration keys: {sorted(terrain_cfg)}")
        if obs_cfg:
            raise ValueError(f"Unknown observation configuration keys: {sorted(obs_cfg)}")
        if domain_randomization_cfg:
            raise ValueError(f"Unknown domain-randomization configuration keys: {sorted(domain_randomization_cfg)}")

        for key in ("use_mjx", "critic_history_length", "policy_layout"):
            env_cfg.pop(key, None)
        return cls(
            robot_xml=robot_xml,
            motion=motion,
            terrain=terrain,
            height_scan=height_scan,
            domain_randomization=domain_randomization,
            actor_history_length=actor_history_length,
            discriminator_history_length=discriminator_history_length,
            observation_layout=observation_layout,
            **env_cfg,
        )
