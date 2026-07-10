from __future__ import annotations

from isaaclab.terrains import TerrainGenerator
from isaaclab.terrains import TerrainGeneratorCfg
from isaaclab.utils import configclass


@configclass
class FiveStageTerrainGeneratorCfg(TerrainGeneratorCfg):
    """Terrain generator that maps terrain rows to semantic locomotion stages."""

    class_type: type = None

    stage_sub_terrain_names: tuple[tuple[str, ...], ...] = (
        ("flat", "random_rough_mild", "wave_mild"),
        ("random_rough", "discrete_obstacles", "boxes_low"),
        ("pyramid_slope", "pyramid_slope_inv", "wave"),
        ("pyramid_stairs", "pyramid_stairs_inv"),
        ("stepping_stones", "gap", "pit", "rails"),
    )
    """Sub-terrain names sampled in each terrain row."""

    stage_difficulty_ranges: tuple[tuple[float, float], ...] = (
        (0.0, 0.15),
        (0.15, 0.35),
        (0.35, 0.60),
        (0.45, 0.80),
        (0.55, 1.0),
    )
    """Difficulty range sampled for each terrain row."""

    def __post_init__(self):
        self.class_type = FiveStageTerrainGenerator


class FiveStageTerrainGenerator(TerrainGenerator):
    """IsaacLab terrain generator whose rows are ordered by locomotion skill family."""

    def _generate_curriculum_terrains(self):
        if len(self.cfg.stage_sub_terrain_names) != self.cfg.num_rows:
            raise ValueError(
                "FiveStageTerrainGeneratorCfg.stage_sub_terrain_names must have one entry per terrain row. "
                f"Got {len(self.cfg.stage_sub_terrain_names)} entries for num_rows={self.cfg.num_rows}."
            )
        if len(self.cfg.stage_difficulty_ranges) != self.cfg.num_rows:
            raise ValueError(
                "FiveStageTerrainGeneratorCfg.stage_difficulty_ranges must have one entry per terrain row. "
                f"Got {len(self.cfg.stage_difficulty_ranges)} entries for num_rows={self.cfg.num_rows}."
            )

        for sub_row in range(self.cfg.num_rows):
            stage_names = self.cfg.stage_sub_terrain_names[sub_row]
            missing = [name for name in stage_names if name not in self.cfg.sub_terrains]
            if missing:
                raise ValueError(f"Unknown sub-terrain names in stage row {sub_row}: {missing}.")

            lower, upper = self.cfg.stage_difficulty_ranges[sub_row]
            for sub_col in range(self.cfg.num_cols):
                terrain_name = stage_names[sub_col % len(stage_names)]
                terrain_cfg = self.cfg.sub_terrains[terrain_name]
                difficulty = float(self.np_rng.uniform(lower, upper))
                mesh, origin = self._get_terrain_mesh(difficulty, terrain_cfg)
                self._add_sub_terrain(mesh, origin, sub_row, sub_col, terrain_cfg)

    def __str__(self) -> str:
        msg = super().__str__()
        msg += "\n\tStage rows:"
        for row, names in enumerate(self.cfg.stage_sub_terrain_names):
            msg += f"\n\t  {row}: {', '.join(names)}"
        return msg
