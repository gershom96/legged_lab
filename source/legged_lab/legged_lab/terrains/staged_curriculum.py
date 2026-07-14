from __future__ import annotations

import numpy as np

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

    edge_blend_width: float = 2.5
    """Width of the flat, zero-height rim applied around every terrain tile in meters.

    The terrain board is continuous, so a tile may be adjacent to a different
    terrain family.  Making both tile boundaries reach ``z=0`` removes a
    discontinuity at their shared edge without adding overlapping colliders.
    Set to ``0.0`` only for deliberate seam-stress tests.
    """

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
                self._blend_tile_edges(mesh)
                self._add_sub_terrain(mesh, origin, sub_row, sub_col, terrain_cfg)

    def _blend_tile_edges(self, mesh) -> None:
        """Flatten a local terrain mesh to ground height in its outer rim.

        ``TerrainGenerator._get_terrain_mesh`` centers every sub-terrain around
        ``(0, 0)`` before this method is called.  All geometry in the outer rim
        is mapped to ``z=0``; the untouched interior starts after that rim.
        Adjacent tiles therefore share a real flat contact band rather than
        overlapping collision geometry.
        """
        width = float(self.cfg.edge_blend_width)
        if width <= 0.0:
            return

        half_size = np.asarray(self.cfg.size, dtype=np.float64) * 0.5
        if width * 2.0 >= float(np.min(half_size * 2.0)):
            raise ValueError(
                "edge_blend_width must be smaller than half the shortest terrain-tile dimension. "
                f"Got width={width} for size={self.cfg.size}."
            )

        vertices = np.asarray(mesh.vertices, dtype=np.float64).copy()
        edge_distance = np.minimum(half_size[0] - np.abs(vertices[:, 0]), half_size[1] - np.abs(vertices[:, 1]))
        vertices[edge_distance <= width, 2] = 0.0
        mesh.vertices = vertices
        # Flattening a closed box can collapse its vertical faces.  Clean those
        # zero-area faces before the mesh is handed to PhysX.
        mesh.merge_vertices()
        mesh.update_faces(mesh.nondegenerate_faces(height=1.0e-8))
        mesh.remove_unreferenced_vertices()
        if hasattr(mesh, "unique_faces"):
            mesh.update_faces(mesh.unique_faces())

    def __str__(self) -> str:
        msg = super().__str__()
        msg += "\n\tStage rows:"
        for row, names in enumerate(self.cfg.stage_sub_terrain_names):
            msg += f"\n\t  {row}: {', '.join(names)}"
        msg += f"\n\tTile edge blend width: {self.cfg.edge_blend_width} m"
        return msg
