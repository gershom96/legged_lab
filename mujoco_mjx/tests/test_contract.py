from __future__ import annotations

from pathlib import Path
import os
import sys
import tempfile
from types import SimpleNamespace
import unittest

import mujoco
import numpy as np
import yaml

os.environ.setdefault("JAX_PLATFORMS", "cpu")

REPO_ROOT = Path(__file__).resolve().parents[2]
for import_root in (REPO_ROOT, REPO_ROOT / "rsl_rl"):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from mujoco_mjx.rsl_rl_mujoco.constants import (  # noqa: E402
    EFFORT_LIMIT_ISAAC,
    ISAAC_INDICES_IN_MUJOCO_ORDER,
    ISAAC_JOINT_NAMES,
    MUJOCO_INDICES_IN_ISAAC_ORDER,
    MUJOCO_JOINT_NAMES,
)
from mujoco_mjx.rsl_rl_mujoco.model import build_mujoco_model  # noqa: E402
from mujoco_mjx.rsl_rl_mujoco.native_motion import (  # noqa: E402
    load_motion_data,
    sample_discriminator_observations,
    sample_discriminator_observations_numpy,
)
from mujoco_mjx.rsl_rl_mujoco.native_env import NativeJaxAmpEnv, RobotState  # noqa: E402
from mujoco_mjx.rsl_rl_mujoco.spec import (  # noqa: E402
    HeightScanSpec,
    MotionSpec,
    MujocoRslRlEnvSpec,
    TerrainCurriculumSpec,
    TerrainSpec,
)
from mujoco_mjx.rsl_rl_mujoco.native_symmetry import mirror_policy_observation  # noqa: E402
from mujoco_mjx.rsl_rl_mujoco.terrain import generate_terrain  # noqa: E402


MOTION_DIR = Path("/home/gershom/Documents/shared_datasets/legged_lab_g1_mixed_default_motionbricks")
ROBOT_XML = REPO_ROOT / "source/legged_lab/legged_lab/data/Robots/Unitree/g1_29dof/g1_29dof.xml"


class ContractTests(unittest.TestCase):
    def test_height_scan_uses_isaac_sensor_frame_convention(self) -> None:
        scan = HeightScanSpec()
        self.assertEqual(scan.offset, 0.5)

    def test_feet_slide_subtracts_root_velocity_like_isaac(self) -> None:
        import jax.numpy as jnp

        env = object.__new__(NativeJaxAmpEnv)
        env.jnp = jnp
        env.num_envs = 1
        robot = RobotState(
            root_pos=jnp.zeros((1, 3), dtype=jnp.float32),
            root_quat=jnp.asarray([[1.0, 0.0, 0.0, 0.0]], dtype=jnp.float32),
            root_lin_vel_w=jnp.asarray([[1.25, -0.5, 0.0]], dtype=jnp.float32),
            root_lin_vel_b=jnp.zeros((1, 3), dtype=jnp.float32),
            root_ang_vel_b=jnp.zeros((1, 3), dtype=jnp.float32),
            gravity_b=jnp.asarray([[0.0, 0.0, -1.0]], dtype=jnp.float32),
            height_scan_pos_w=jnp.zeros((1, 3), dtype=jnp.float32),
            height_scan_quat_w=jnp.asarray([[1.0, 0.0, 0.0, 0.0]], dtype=jnp.float32),
            joint_pos=jnp.zeros((1, 29), dtype=jnp.float32),
            joint_vel=jnp.zeros((1, 29), dtype=jnp.float32),
            key_body_pos_w=jnp.zeros((1, 6, 3), dtype=jnp.float32),
            key_body_pos_b=jnp.zeros((1, 6, 3), dtype=jnp.float32),
            foot_vel_w=jnp.asarray([[[1.25, -0.5, 0.0], [1.25, -0.5, 0.0]]], dtype=jnp.float32),
            joint_acc=jnp.zeros((1, 29), dtype=jnp.float32),
            torque=jnp.zeros((1, 29), dtype=jnp.float32),
            foot_contacts=jnp.asarray([[True, True]]),
        )
        self.assertAlmostEqual(float(env._feet_slide_reward(robot)[0]), 0.0, places=6)

    def test_elu_gradient_stays_finite_for_large_positive_inputs(self) -> None:
        import jax
        import jax.numpy as jnp

        from mujoco_mjx.rsl_rl_mujoco.native_model import elu

        value = jnp.asarray(100.0, dtype=jnp.float32)
        self.assertEqual(float(elu(value)), 100.0)
        self.assertEqual(float(jax.grad(elu)(value)), 1.0)

    def test_default_config_dimensions_and_joint_round_trip(self) -> None:
        with (REPO_ROOT / "mujoco_mjx/configs/g1_rsl_rl_mjx_amp.yaml").open(encoding="utf-8") as stream:
            payload = yaml.safe_load(stream)
        spec = MujocoRslRlEnvSpec.from_mapping(payload, REPO_ROOT)
        self.assertEqual(spec.height_scan.shape, (17, 11))
        self.assertEqual(spec.actor_history_length * 114 + spec.height_scan.dim, 757)
        self.assertEqual(117 + spec.height_scan.critic_history_length * spec.height_scan.dim, 1052)
        self.assertEqual(spec.warp_naconmax, 278528)
        self.assertFalse(spec.domain_randomization.enabled)
        self.assertEqual(spec.domain_randomization.static_friction_range, (0.3, 1.6))
        self.assertEqual(spec.domain_randomization.dynamic_friction_range, (0.3, 1.2))
        self.assertEqual(spec.domain_randomization.base_mass_add_range, (-3.0, 3.0))
        self.assertEqual(spec.domain_randomization.com_range, (-0.03, 0.03))
        self.assertEqual(spec.domain_randomization.limb_mass_scale_range, (0.8, 1.2))
        self.assertEqual(spec.domain_randomization.actuator_gain_scale_range, (0.8, 1.2))
        self.assertEqual(spec.domain_randomization.armature_scale_range, (0.8, 1.2))
        self.assertEqual(spec.domain_randomization.push_interval_s, (10.0, 20.0))

        small_spec = MujocoRslRlEnvSpec(
            robot_xml=ROBOT_XML,
            motion=MotionSpec(MOTION_DIR),
            num_envs=2,
            device="cpu",
            mjx_impl="jax",
        )
        self.assertEqual(small_spec.warp_naconmax, 1024)

        action_isaac = np.arange(29)
        action_mujoco = action_isaac[ISAAC_INDICES_IN_MUJOCO_ORDER]
        for mujoco_index, joint_name in enumerate(MUJOCO_JOINT_NAMES):
            self.assertEqual(action_mujoco[mujoco_index], ISAAC_JOINT_NAMES.index(joint_name))
        round_trip = action_mujoco[MUJOCO_INDICES_IN_ISAAC_ORDER]
        np.testing.assert_array_equal(round_trip, action_isaac)

    def test_builtin_terrains_are_finite_and_centered(self) -> None:
        for terrain_type in ("flat", "random_rough_mild", "wave", "pyramid_stairs", "gap"):
            with self.subTest(terrain=terrain_type):
                terrain = generate_terrain(
                    TerrainSpec(type=terrain_type, size=(8.0, 8.0), horizontal_scale=0.1, difficulty=0.5)
                )
                self.assertEqual(terrain.heights.shape, (81, 81))
                self.assertTrue(np.isfinite(terrain.heights).all())
                center = terrain.heights[35:46, 35:46]
                self.assertLessEqual(float(center.max() - center.min()), 1.0e-6)
                if terrain_type in {"stepping_stones", "gap"}:
                    self.assertGreaterEqual(float(center.min()), 0.0)

    def test_scan_surface_includes_standalone_box_collision_geometry(self) -> None:
        from mujoco_mjx.rsl_rl_mujoco.terrain import TerrainBox, TerrainData

        terrain = TerrainData(
            heights=np.zeros((5, 5), dtype=np.float32),
            size=(4.0, 4.0),
            horizontal_scale=1.0,
            terrain_type="test",
            boxes=(TerrainBox(center=(0.0, 0.0, 0.25), half_size=(0.5, 0.5, 0.25)),),
        )
        surface = terrain.scan_surface_heights()
        self.assertAlmostEqual(float(surface[2, 2]), 0.5)

    def test_curriculum_board_has_clean_borders_and_all_stages(self) -> None:
        terrain = generate_terrain(
            TerrainSpec(
                type="curriculum",
                size=(8.0, 8.0),
                horizontal_scale=0.2,
                center_platform_radius=0.5,
                curriculum=TerrainCurriculumSpec(enabled=True, safe_half_extent=3.0),
            )
        )
        self.assertEqual(terrain.heights.shape, (161, 161))
        self.assertEqual(len(terrain.patches), 15)
        self.assertEqual({patch.stage for patch in terrain.patches}, set(range(6)))
        self.assertEqual(terrain.size, (32.0, 32.0))
        self.assertEqual(len(terrain.boxes), 1)
        self.assertIsNotNone(terrain.collision_heights)
        self.assertAlmostEqual(terrain.boxes[0].center[2] + terrain.boxes[0].half_size[2], 0.0)
        patch_cells = 41
        for seam in range(0, terrain.heights.shape[0], patch_cells - 1):
            np.testing.assert_allclose(terrain.heights[seam, :], 0.0, atol=1.0e-6)
            np.testing.assert_allclose(terrain.heights[:, seam], 0.0, atol=1.0e-6)

    def test_curriculum_reset_preserves_assigned_stage(self) -> None:
        import jax
        import jax.numpy as jnp

        env = object.__new__(NativeJaxAmpEnv)
        env.jax = jax
        env.jnp = jnp
        env.num_envs = 12
        env.terrain = SimpleNamespace(patches=(object(),))
        env.stage_patch_counts = jnp.asarray([1, 2, 3, 1, 2, 3], dtype=jnp.int32)
        env.stage_patch_indices = jnp.asarray(
            [
                [0, 0, 0],
                [1, 2, 1],
                [3, 4, 5],
                [6, 6, 6],
                [7, 8, 7],
                [9, 10, 11],
            ],
            dtype=jnp.int32,
        )
        patch_stages = np.asarray([0, 1, 1, 2, 2, 2, 3, 4, 4, 5, 5, 5], dtype=np.int32)
        assigned_stage = jnp.asarray([0, 1, 1, 2, 2, 2, 3, 4, 4, 5, 5, 5], dtype=jnp.int32)
        patch_index = np.asarray(env._sample_patch_indices_for_stages(jax.random.PRNGKey(9), assigned_stage))
        np.testing.assert_array_equal(patch_stages[patch_index], np.asarray(assigned_stage))

    def test_curriculum_updates_only_completed_environment_targets(self) -> None:
        import jax.numpy as jnp

        env = object.__new__(NativeJaxAmpEnv)
        env.jnp = jnp
        env.terrain = SimpleNamespace(patches=(object(),))
        env.spec = SimpleNamespace(
            episode_length_s=20.0,
            terrain=SimpleNamespace(
                size=(45.0, 45.0),
                curriculum=TerrainCurriculumSpec(
                    enabled=True,
                    min_lin_tracking=0.6,
                    min_ang_tracking=0.2,
                ),
            ),
        )
        current = jnp.asarray([1, 3, 2], dtype=jnp.int32)
        start_xy = jnp.zeros((3, 2), dtype=jnp.float32)
        command = jnp.asarray(((1.0, 0.0, 0.0),) * 3, dtype=jnp.float32)
        root_pos = jnp.asarray(((14.0, 0.0, 0.0), (0.1, 0.0, 0.0), (14.0, 0.0, 0.0)))
        sums = jnp.asarray(((16.0, 8.0),) * 3, dtype=jnp.float32)

        updated = env._update_terrain_stage(
            current,
            start_xy,
            command,
            root_pos,
            jnp.asarray((100, 100, 100), dtype=jnp.int32),
            sums,
            jnp.asarray((False, True, False)),
            jnp.asarray((True, True, False)),
        )
        # First env advances; second is terminated and demotes; third has not reset.
        np.testing.assert_array_equal(np.asarray(updated), np.asarray((2, 2, 2), dtype=np.int32))

    def test_height_scan_symmetry_is_an_involution(self) -> None:
        import jax

        observation = jax.random.normal(jax.random.PRNGKey(4), (3, 757))
        mirrored = mirror_policy_observation(observation)
        restored = mirror_policy_observation(mirrored)
        np.testing.assert_allclose(np.asarray(restored), np.asarray(observation), atol=1.0e-6)

    def test_discriminator_normalizer_count_does_not_wrap_at_int32_limit(self) -> None:
        import jax.numpy as jnp

        from mujoco_mjx.rsl_rl_mujoco.native_model import update_normalizer

        normalizer = {
            "mean": jnp.zeros((1, 49), dtype=jnp.float32),
            "var": jnp.ones((1, 49), dtype=jnp.float32),
            "std": jnp.ones((1, 49), dtype=jnp.float32),
            "count": jnp.asarray(2.2e9, dtype=jnp.float32),
        }
        updated = update_normalizer(normalizer, jnp.ones((4096, 4, 49), dtype=jnp.float32))
        self.assertEqual(updated["count"].dtype, jnp.float32)
        self.assertGreater(float(updated["count"]), 2.0e9)
        self.assertTrue(bool(jnp.all(jnp.isfinite(updated["mean"]))))
        self.assertTrue(bool(jnp.all(jnp.isfinite(updated["var"]))))

    @unittest.skipUnless(MOTION_DIR.is_dir(), "mixed AMP motion dataset is not available")
    def test_upper_amp_demo_layout(self) -> None:
        import jax

        dataset, _ = load_motion_data(MotionSpec(MOTION_DIR, max_files=1))
        demo = sample_discriminator_observations(dataset, jax.random.PRNGKey(8), 8, 4, 0.02)
        self.assertEqual(tuple(demo.shape), (8, 4, 49))
        self.assertTrue(np.isfinite(np.asarray(demo)).all())
        host_demo = sample_discriminator_observations_numpy(dataset, np.random.default_rng(8), 8, 4, 0.02)
        self.assertEqual(host_demo.shape, (8, 4, 49))
        self.assertTrue(np.isfinite(host_demo).all())

    @unittest.skipUnless(MOTION_DIR.is_dir(), "mixed AMP motion dataset is not available")
    def test_motionbricks_files_are_converted_to_isaac_order(self) -> None:
        import joblib

        path = next(MOTION_DIR.glob("motionbricks__*.pkl"))
        motion = joblib.load(path)
        self.assertEqual(motion.get("source_joint_order"), "mujoco")
        self.assertEqual(motion.get("joint_order"), "isaaclab")

    @unittest.skipUnless(ROBOT_XML.is_file() and MOTION_DIR.is_dir(), "G1 assets are not available")
    def test_mujoco_hfield_offset_and_effort_limits(self) -> None:
        spec = MujocoRslRlEnvSpec(
            robot_xml=ROBOT_XML,
            motion=MotionSpec(MOTION_DIR, max_files=1),
            terrain=TerrainSpec(type="gap", size=(8.0, 8.0), difficulty=0.5),
            num_envs=1,
            device="cpu",
            mjx_impl="jax",
        )
        terrain = generate_terrain(spec.terrain)
        with tempfile.TemporaryDirectory() as directory:
            model, metadata, _ = build_mujoco_model(spec, terrain, Path(directory))
        terrain_id = metadata.terrain_geom_ids[0]
        self.assertEqual(model.body(metadata.height_scan_body_id).name, "torso_link")
        self.assertAlmostEqual(float(model.geom_pos[terrain_id, 2]), float(terrain.heights.min()), places=6)
        expected = EFFORT_LIMIT_ISAAC[ISAAC_INDICES_IN_MUJOCO_ORDER]
        actual = model.actuator_ctrlrange[metadata.actuator_indices, 1]
        np.testing.assert_allclose(actual, expected)
        actual_joint_limits = model.jnt_actfrcrange[:, 1]
        np.testing.assert_allclose(actual_joint_limits[1:], expected)
        for mujoco_index, actuator_index in enumerate(metadata.actuator_indices):
            joint_id = int(model.actuator_trnid[actuator_index, 0])
            joint_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
            self.assertEqual(joint_name, MUJOCO_JOINT_NAMES[mujoco_index])

    def test_ean_occupancy_scene_adapter(self) -> None:
        ean_root = Path("/home/gershom/Documents/gershom96/Embodiment-Aware-Nav")
        scene_config = ean_root / "configs/scene/random_clutter.yaml"
        if not scene_config.is_file():
            self.skipTest("Embodiment-Aware-Nav scene configs are not available")
        terrain = generate_terrain(
            TerrainSpec(type="ean", ean_root=ean_root, ean_scene_config=scene_config, horizontal_scale=0.1)
        )
        self.assertGreater(len(terrain.boxes), 0)
        self.assertIsNotNone(terrain.collision_heights)
        self.assertGreater(float(terrain.heights.max()), float(terrain.collision_heights.max()))


if __name__ == "__main__":
    unittest.main()
