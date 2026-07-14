from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from .constants import EFFORT_LIMIT_ISAAC, ISAAC_INDICES_IN_MUJOCO_ORDER, KEY_BODY_NAMES, MUJOCO_JOINT_NAMES
from .spec import MujocoRslRlEnvSpec
from .terrain import TerrainData


@dataclass(frozen=True)
class ModelMetadata:
    joint_qpos_indices: np.ndarray
    joint_qvel_indices: np.ndarray
    actuator_indices: np.ndarray
    key_body_ids: np.ndarray
    height_scan_body_id: int
    foot_body_ids: np.ndarray
    foot_geom_ids: tuple[np.ndarray, np.ndarray]
    terrain_geom_ids: np.ndarray
    joint_limits_mujoco: np.ndarray
    root_spawn_height: float
    robot_geom_ids: np.ndarray
    base_mass_body_id: int
    com_body_ids: np.ndarray
    limb_body_ids: np.ndarray


def build_mujoco_model(
    spec: MujocoRslRlEnvSpec,
    terrain: TerrainData,
    output_dir: Path,
) -> tuple[mujoco.MjModel, ModelMetadata, Path]:
    source = spec.robot_xml.expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    output_dir.mkdir(parents=True, exist_ok=True)
    tree = ET.parse(source)
    root = tree.getroot()
    root.set("model", f"g1_rsl_rl_{terrain.terrain_type}")
    compiler = root.find("compiler")
    if compiler is None:
        compiler = ET.SubElement(root, "compiler")
    compiler.set("meshdir", str((source.parent / "meshes").resolve()))

    option = root.find("option")
    if option is None:
        option = ET.Element("option")
        root.insert(1, option)
    option.set("timestep", f"{spec.sim_dt:.8g}")
    option.set("gravity", "0 0 -9.81")
    option.set("integrator", "implicitfast")

    asset = root.find("asset")
    worldbody = root.find("worldbody")
    if asset is None or worldbody is None:
        raise ValueError(f"G1 XML {source} must contain asset and worldbody elements")
    ET.SubElement(
        asset,
        "material",
        {"name": "terrain_material", "rgba": "0.32 0.36 0.31 1"},
    )

    if spec.mjx_impl == "jax":
        _simplify_robot_collisions_for_jax(worldbody)

    terrain_geom_names: list[str] = []

    if terrain.mesh_path is not None:
        ET.SubElement(asset, "mesh", {"name": "terrain_mesh", "file": str(terrain.mesh_path.resolve())})
        ET.SubElement(
            worldbody,
            "geom",
            {
                "name": "terrain_visual",
                "type": "mesh",
                "mesh": "terrain_mesh",
                "material": "terrain_material",
                "contype": "0",
                "conaffinity": "0",
            },
        )
        collision_paths = terrain.collision_mesh_paths or (terrain.mesh_path,)
        for index, collision_path in enumerate(collision_paths):
            mesh_name = f"terrain_collision_mesh_{index:05d}"
            geom_name = f"terrain_collision_{index:05d}"
            ET.SubElement(asset, "mesh", {"name": mesh_name, "file": str(collision_path.resolve())})
            ET.SubElement(
                worldbody,
                "geom",
                {
                    "name": geom_name,
                    "type": "mesh",
                    "mesh": mesh_name,
                    "friction": "1.0 0.02 0.002",
                    "contype": "1",
                    "conaffinity": "1",
                },
            )
            terrain_geom_names.append(geom_name)
    elif np.allclose(terrain.heights, terrain.heights.flat[0]):
        ET.SubElement(
            worldbody,
            "geom",
            {
                "name": "terrain",
                "type": "plane",
                "pos": f"0 0 {float(terrain.heights.flat[0]):.8g}",
                "size": f"{terrain.size[0] / 2.0:.8g} {terrain.size[1] / 2.0:.8g} 0.05",
                "material": "terrain_material",
                "friction": "1.0 0.02 0.002",
                "contype": "1",
                "conaffinity": "1",
            },
        )
        terrain_geom_names.append("terrain")
    else:
        collision_heights = terrain.heights if terrain.collision_heights is None else terrain.collision_heights
        minimum = float(collision_heights.min())
        maximum = float(collision_heights.max())
        vertical_range = max(maximum - minimum, 1.0e-4)
        base = max(1.0e-3, -minimum)
        ET.SubElement(
            asset,
            "hfield",
            {
                "name": "terrain_hfield",
                "nrow": str(terrain.heights.shape[1]),
                "ncol": str(terrain.heights.shape[0]),
                "size": f"{terrain.size[0] / 2.0:.8g} {terrain.size[1] / 2.0:.8g} {vertical_range:.8g} {base:.8g}",
            },
        )
        ET.SubElement(
            worldbody,
            "geom",
            {
                "name": "terrain",
                "type": "hfield",
                "hfield": "terrain_hfield",
                "pos": f"0 0 {minimum:.8g}",
                "material": "terrain_material",
                "friction": "1.0 0.02 0.002",
                "contype": "1",
                "conaffinity": "1",
            },
        )
        terrain_geom_names.append("terrain")

    for index, box in enumerate(terrain.boxes):
        name = f"terrain_box_{index:05d}"
        ET.SubElement(
            worldbody,
            "geom",
            {
                "name": name,
                "type": "box",
                "pos": " ".join(f"{value:.8g}" for value in box.center),
                "size": " ".join(f"{value:.8g}" for value in box.half_size),
                "material": "terrain_material",
                "friction": "1.0 0.02 0.002",
                "contype": "1",
                "conaffinity": "1",
            },
        )
        terrain_geom_names.append(name)

    actuator = root.find("actuator")
    if actuator is None:
        raise ValueError(f"G1 XML {source} does not contain an actuator section")
    # EFFORT_LIMIT_ISAAC is indexed in Isaac order; reorder it before matching XML joints.
    effort_mujoco = EFFORT_LIMIT_ISAAC[ISAAC_INDICES_IN_MUJOCO_ORDER]
    effort_by_joint = dict(zip(MUJOCO_JOINT_NAMES, effort_mujoco.tolist(), strict=True))
    for joint in root.iter("joint"):
        joint_name = joint.get("name")
        if joint_name in effort_by_joint:
            limit = effort_by_joint[joint_name]
            joint.set("actuatorfrcrange", f"{-limit:.8g} {limit:.8g}")
            joint.set("actuatorfrclimited", "true")
    for motor in actuator.findall("motor"):
        joint_name = motor.get("joint")
        if joint_name in effort_by_joint:
            limit = effort_by_joint[joint_name]
            motor.set("ctrlrange", f"{-limit:.8g} {limit:.8g}")
            motor.set("ctrllimited", "true")

    ET.SubElement(worldbody, "light", {"pos": "0 0 4", "dir": "0 0 -1", "directional": "true"})
    xml_path = output_dir / "g1_scene.xml"
    ET.indent(tree, space="  ")
    tree.write(xml_path, encoding="utf-8", xml_declaration=True)
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    if model.nhfield:
        collision_heights = terrain.heights if terrain.collision_heights is None else terrain.collision_heights
        minimum = float(collision_heights.min())
        vertical_range = max(float(collision_heights.max()) - minimum, 1.0e-4)
        normalized = np.clip((collision_heights - minimum) / vertical_range, 0.0, 1.0)
        model.hfield_data[:] = normalized.T.reshape(-1)

    qpos_indices: list[int] = []
    qvel_indices: list[int] = []
    actuator_indices: list[int] = []
    limits: list[np.ndarray] = []
    for name in MUJOCO_JOINT_NAMES:
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        actuator_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name.removesuffix("_joint"))
        if joint_id < 0 or actuator_id < 0:
            raise ValueError(f"MuJoCo G1 model is missing joint/actuator for {name}")
        qpos_indices.append(int(model.jnt_qposadr[joint_id]))
        qvel_indices.append(int(model.jnt_dofadr[joint_id]))
        actuator_indices.append(actuator_id)
        limits.append(np.asarray(model.jnt_range[joint_id], dtype=np.float32))
    key_body_ids = np.asarray(
        [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name) for name in KEY_BODY_NAMES],
        dtype=np.int32,
    )
    if np.any(key_body_ids < 0):
        raise ValueError("MuJoCo G1 model is missing one or more AMP key bodies")
    height_scan_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "torso_link")
    if height_scan_body_id < 0:
        raise ValueError("MuJoCo G1 model is missing the torso_link height-scan body")
    foot_geom_ids = tuple(
        np.flatnonzero((model.geom_bodyid == body_id) & (model.geom_contype != 0)).astype(np.int32)
        for body_id in key_body_ids[:2]
    )
    if any(ids.size == 0 for ids in foot_geom_ids):
        raise ValueError("No collision-enabled foot geoms were found in the MuJoCo model")
    terrain_geom_ids = np.asarray(
        [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name) for name in terrain_geom_names],
        dtype=np.int32,
    )
    if terrain_geom_ids.size == 0 or np.any(terrain_geom_ids < 0):
        raise ValueError("One or more generated terrain geoms are missing from the MuJoCo model")
    robot_geom_ids = np.flatnonzero((model.geom_bodyid != 0) & (model.geom_contype != 0)).astype(np.int32)
    if robot_geom_ids.size == 0:
        raise ValueError("No collision-enabled robot geoms were found in the MuJoCo model")
    # Isaac's add_base_mass event explicitly targets torso_link, rather than
    # the free-joint body (pelvis) that owns the floating base.
    base_mass_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "torso_link")
    if base_mass_body_id < 0:
        raise ValueError("MuJoCo G1 model is missing torso_link for base-mass randomization")
    com_body_ids = np.asarray(
        [
            body_id
            for name in ("torso_link", "pelvis")
            if (body_id := mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)) >= 0
        ],
        dtype=np.int32,
    )
    limb_body_ids = np.asarray(
        [
            body_id
            for body_id in range(1, model.nbody)
            if (name := mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id))
            and (name.startswith("left_") or name.startswith("right_"))
            and name.endswith("_link")
        ],
        dtype=np.int32,
    )
    root_spawn_height = 0.8 + terrain.height_at(*terrain.spawn_xy)
    metadata = ModelMetadata(
        joint_qpos_indices=np.asarray(qpos_indices, dtype=np.int32),
        joint_qvel_indices=np.asarray(qvel_indices, dtype=np.int32),
        actuator_indices=np.asarray(actuator_indices, dtype=np.int32),
        key_body_ids=key_body_ids,
        height_scan_body_id=int(height_scan_body_id),
        foot_body_ids=key_body_ids[:2].copy(),
        foot_geom_ids=(foot_geom_ids[0], foot_geom_ids[1]),
        terrain_geom_ids=terrain_geom_ids,
        joint_limits_mujoco=np.stack(limits),
        root_spawn_height=root_spawn_height,
        robot_geom_ids=robot_geom_ids,
        base_mass_body_id=int(base_mass_body_id),
        com_body_ids=com_body_ids,
        limb_body_ids=limb_body_ids,
    )
    return model, metadata, xml_path


def _simplify_robot_collisions_for_jax(worldbody: ET.Element) -> None:
    """Keep foot spheres only; MJX-JAX lacks the G1 mesh collision pairs.

    This mode is intended for portable CPU smoke tests. Full training uses
    MJX-Warp and retains the robot's original collision geometry.
    """

    foot_bodies = {"left_ankle_roll_link", "right_ankle_roll_link"}

    def visit(body: ET.Element) -> None:
        preserve_primitives = body.get("name") in foot_bodies
        for geom_index, geom in enumerate(body.findall("geom")):
            if preserve_primitives and geom.get("type", "sphere") == "sphere":
                geom.set("name", geom.get("name", f"{body.get('name')}_contact_{geom_index}"))
                geom.set("contype", "1")
                geom.set("conaffinity", "1")
            else:
                geom.set("contype", "0")
                geom.set("conaffinity", "0")
        for child in body.findall("body"):
            visit(child)

    for body in worldbody.findall("body"):
        visit(body)
