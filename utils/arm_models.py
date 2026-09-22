"""The arm embodiments this pipeline can drive, described as data.

Everything that differs between the bimanual YAM rig and the bimanual UR5e rig lives here: the MuJoCo model
used for FK/IK, the joint box, a home pose, how each arm's base is mounted, the gripper geometry, the capsule
model for arm-to-arm clearance, and the names in the bimanual URDF the visualizer/workspace tool read.
The rest of the package only ever asks the selected `ArmModel`; nothing else hardcodes a robot.

Frames
------
The model-facing "arm base frame" of an arm is gravity-aligned (+x forward, +y left, +z up) with its origin at
that arm's base mount, exactly like the reference YAM trials. For the YAM both bases are upright, so this is
the MJCF frame itself. The UR5e bases on the skild bimanual stand are pitched +/-45 deg outward, so FK/IK run
in the UR's own base frame and `mount_rot[arm]` rotates the result into the gravity-aligned arm frame.
`base_pos[arm]` places the arm frames in the shared URDF root frame; `arm_offset(arm)` converts between them
(the simulator keeps its objects in the left arm's frame).

Select an embodiment with `embodiment_name` in the YAML (`yam_arms` | `ur5e_arms`) or `--embodiment` on the CLI.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

from utils.config import (
    ARMS,
    ASSETS_DIR,
    DEFAULT_SYSTEM_PROMPT,
    DEFAULT_TILT_NOTE,
    DEFAULT_YAM_XML,
    REFERENCE_HOME_JOINTS,
    REPO_ROOT,
    URDF_JOINT_LOWER,
    URDF_JOINT_UPPER,
    Bounds,
)

Vec3 = Tuple[float, float, float]


def _rpy_matrix(rpy: Vec3) -> np.ndarray:
    """URDF rpy (fixed-axis roll, pitch, yaw) -> rotation matrix, R = Rz(yaw) Ry(pitch) Rx(roll)."""
    from scipy.spatial.transform import Rotation

    return Rotation.from_euler("xyz", list(rpy)).as_matrix()


@dataclass(frozen=True)
class Capsule:
    """A capsule between two MJCF body origins (or 'base'/'flange'), in the arm's own frame."""

    name: str
    body_a: str
    body_b: str
    radius_m: float


@dataclass(frozen=True)
class ArmModel:
    name: str                                   # the `embodiment_name`, also the registry key
    description: str
    mjcf_path: str                              # single-arm MuJoCo model in the arm's own (URDF) base frame
    grasp_site: str                             # MJCF site between the fingertips (z = tool axis, y = jaw axis)
    flange_site: str                            # MJCF site at the tool flange the gripper is bolted to
    joint_names: Tuple[str, ...]                # the 6 arm joints, MJCF order = wire order of the robot server
    joint_lower: Tuple[float, ...]              # exact URDF limits (rad)
    joint_upper: Tuple[float, ...]
    home_joints: Dict[str, Tuple[float, ...]]   # per arm
    base_rpy: Dict[str, Vec3]                   # URDF rpy of each base mount in the shared root frame
    base_pos: Dict[str, Vec3]                   # URDF xyz of each base mount in the shared root frame
    grasp_offset_m: float                       # flange -> grasp point along the tool axis
    jaw_max_opening_m: float                    # gripper = 1.0
    housing_length_m: float                     # wide part of the gripper along the tool axis, from the flange
    housing_radius_m: float
    neck_length_m: float                        # tapered part before the fingers
    neck_radius_m: float
    finger_radius_m: float
    base_column: Tuple[float, float]            # (height, radius) of the base pedestal capsule from the base origin
    capsules: Tuple[Capsule, ...]               # arm links; the gripper capsules are derived from the numbers above
    urdf_path: str                              # bimanual URDF (visualizer, workspace tool, tests)
    urdf_root_link: str
    urdf_joint_templates: Tuple[str, ...]       # `{arm}` placeholders, one per arm joint, in joint order
    urdf_link_templates: Tuple[str, ...]        # `{arm}` placeholders, base first, then the chain to the last joint
    mjcf_body_for_link: Dict[str, str]          # URDF link template -> MJCF body (for meshes/frames)
    system_prompt_path: str
    tilt_note_path: str
    default_bounds: Bounds
    prompt_cache_key: str
    robot_server_hint: str                      # how to start the gello robot server for this rig
    wrist_cam_offset_flange: Vec3 = (-0.07, 0.0, 0.06)   # visualizer only: camera position in the flange frame
    mesh_dirs: Tuple[str, ...] = ()             # candidate directories holding the visual meshes (first that exists)
    proportions: Dict[str, float] = field(default_factory=dict)   # informational (prompt text lives in docs/)

    # ------------------------------------------------------------------ frames
    def mount_rot(self, arm: str) -> np.ndarray:
        """Rotation taking the MJCF (URDF base) frame of `arm` into its gravity-aligned arm frame."""
        return _rpy_matrix(self.base_rpy[arm])

    def arm_offset(self, arm: str) -> np.ndarray:
        """Position of `arm`'s frame origin expressed in the LEFT arm's frame (the simulator's world frame)."""
        return np.asarray(self.base_pos[arm], dtype=float) - np.asarray(self.base_pos["left"], dtype=float)

    def base_distance_m(self) -> float:
        return float(np.linalg.norm(self.arm_offset("right")))

    def side_sign(self, arm: str) -> float:
        """+1 if `arm` sits on the +y side of the other arm (its own left is 'away'), else -1."""
        other = "right" if arm == "left" else "left"
        d = self.arm_offset(arm)[1] - self.arm_offset(other)[1]
        return 1.0 if d >= 0 else -1.0

    # ------------------------------------------------------------------- urdf
    def urdf_joint_names(self, arm: str) -> Tuple[str, ...]:
        return tuple(t.format(arm=arm) for t in self.urdf_joint_templates)

    def urdf_link_names(self, arm: str) -> Tuple[str, ...]:
        return tuple(t.format(arm=arm) for t in self.urdf_link_templates)

    def urdf_base_link(self, arm: str) -> str:
        return self.urdf_link_templates[0].format(arm=arm)

    def mesh_dir(self) -> Optional[Path]:
        for d in self.mesh_dirs:
            p = Path(d).expanduser()
            if p.exists():
                return p
        return None

    def home_q14(self, gripper: float = 1.0) -> np.ndarray:
        q = np.zeros(14)
        q[0:6] = self.home_joints["left"]
        q[7:13] = self.home_joints["right"]
        q[6] = q[13] = gripper
        return q


# ---------------------------------------------------------------------------
# bimanual YAM (i2rt YAM v2, skild station skild-yam-8)
# ---------------------------------------------------------------------------
YAM_ARMS = ArmModel(
    name="yam_arms",
    description="Two i2rt YAM 6-DoF arms with the compact parallel-jaw gripper, bases upright 0.61 m apart",
    mjcf_path=DEFAULT_YAM_XML,
    grasp_site="grasp_site",
    flange_site="tcp_site",
    joint_names=tuple(f"joint{k}" for k in range(1, 7)),
    joint_lower=URDF_JOINT_LOWER,
    joint_upper=URDF_JOINT_UPPER,
    home_joints={"left": REFERENCE_HOME_JOINTS, "right": REFERENCE_HOME_JOINTS},
    base_rpy={"left": (0.0, 0.0, 0.0), "right": (0.0, 0.0, 0.0)},
    base_pos={"left": (0.0, 0.0, 0.0), "right": (0.0, -0.61, 0.0)},     # skild_yam_v2.urdf right_base_joint
    grasp_offset_m=0.1347,                                              # tcp_site -> grasp_site (yam.xml)
    jaw_max_opening_m=0.095,
    housing_length_m=0.06, housing_radius_m=0.05,
    neck_length_m=0.03, neck_radius_m=0.03,
    finger_radius_m=0.008,
    base_column=(0.11, 0.06),
    capsules=(
        Capsule("upper arm", "link_2", "link_3", 0.045),
        Capsule("forearm", "link_3", "link_4", 0.04),
        Capsule("forearm", "link_4", "link_5", 0.04),
        Capsule("wrist", "link_5", "link_6", 0.045),
    ),
    urdf_path=str(REPO_ROOT / "configs" / "skild_yam_v2.urdf"),
    urdf_root_link="bimanual_yam",
    urdf_joint_templates=tuple(f"{{arm}}_joint{k}" for k in range(1, 7)),
    urdf_link_templates=("{arm}_base_link",) + tuple(f"{{arm}}_link_{k}" for k in range(1, 7)),
    mjcf_body_for_link={"{arm}_base_link": "base_link", **{f"{{arm}}_link_{k}": f"link_{k}" for k in range(1, 7)}},
    system_prompt_path=DEFAULT_SYSTEM_PROMPT,
    tilt_note_path=DEFAULT_TILT_NOTE,
    default_bounds=Bounds(),
    prompt_cache_key="astra-yam",
    robot_server_hint="bash ~/bimanual_manipulation/launch_yam_node.sh   (launch_nodes.py --robot=bimanual_yam)",
    wrist_cam_offset_flange=(-0.07, 0.0, 0.06),
    mesh_dirs=("$GELLO_SOFTWARE_PATH/third_party/robot_models/yam",),   # resolved by the visualizer
    proportions={"upper_arm_m": 0.26, "forearm_m": 0.25, "wrist_to_grasp_m": 0.25, "reach_m": 0.76},
)

# ---------------------------------------------------------------------------
# bimanual UR5e + Robotiq 2F-85 on the skild stand (configs/skild_ur5e.urdf)
# ---------------------------------------------------------------------------
UR5E_URDF = str(REPO_ROOT / "configs" / "skild_ur5e.urdf")
UR5E_JOINT_NAMES = ("shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
                    "wrist_1_joint", "wrist_2_joint", "wrist_3_joint")
# Exact revolute limits of skild_ur5e.urdf (utils.workspace.urdf_joint_limits reads them back in a test).
UR5E_JOINT_LOWER = (-6.28319, -6.28319, -3.1415, -6.28319, -6.28319, -6.28319)
UR5E_JOINT_UPPER = (6.28319, 6.28319, 3.1415, 6.28319, 6.28319, 6.28319)
# Home: grasp point 0.35 m ahead of the base line, 0.05 m inboard of each base, 0.20 m above the table
# (root z = 0, i.e. 0.087 m above the base mounts), fingers pointing straight down with the jaws opening along
# the root y axis. Elbow up and outboard, every joint at least 0.94 rad inside its range; the right pose mirrors
# the left one exactly. Solved with utils.kinematics on the compact MJCF (tests/test_ur5e.py re-checks it).
UR5E_HOME_LEFT = (0.5897, -1.3318, 2.2059, -1.7514, -1.1666, -1.1288)
UR5E_HOME_RIGHT = (-0.5897, -1.8098, -2.2059, -1.3901, 1.1666, 1.1288)
UR5E_HOME = {"left": UR5E_HOME_LEFT, "right": UR5E_HOME_RIGHT}

UR5E_ARMS = ArmModel(
    name="ur5e_arms",
    description="Two Universal Robots UR5e arms with Robotiq 2F-85 grippers on the skild stand, bases 0.50 m apart "
                "and pitched 45 deg outward",
    mjcf_path=str(ASSETS_DIR / "ur5e.xml"),
    grasp_site="grasp_site",
    flange_site="tcp_site",
    joint_names=UR5E_JOINT_NAMES,
    joint_lower=UR5E_JOINT_LOWER,
    joint_upper=UR5E_JOINT_UPPER,
    home_joints=UR5E_HOME,
    # skild_ur5e.urdf left_base_joint / right_base_joint origins (root link `bimanual_ur5e`)
    base_rpy={"left": (0.0, -0.785398, -1.57079367), "right": (0.0, 0.7853982, -1.57079898)},
    base_pos={"left": (0.0, 0.25, 0.113), "right": (0.0, -0.25, 0.113)},
    grasp_offset_m=0.166,          # adapter (0.016) + robotiq base -> tcp (0.15), see *_tcp_joint in the URDF
    jaw_max_opening_m=0.085,       # Robotiq 2F-85
    housing_length_m=0.09, housing_radius_m=0.045,
    neck_length_m=0.03, neck_radius_m=0.03,
    finger_radius_m=0.012,
    base_column=(0.163, 0.075),
    capsules=(
        Capsule("shoulder", "shoulder_link", "upper_arm_link", 0.06),
        Capsule("upper arm", "upper_arm_link", "forearm_link", 0.055),
        Capsule("forearm", "forearm_link", "wrist_1_link", 0.045),
        Capsule("wrist", "wrist_1_link", "wrist_2_link", 0.045),
        Capsule("wrist", "wrist_2_link", "wrist_3_link", 0.045),
        Capsule("wrist", "wrist_3_link", "flange", 0.045),
    ),
    urdf_path=UR5E_URDF,
    urdf_root_link="bimanual_ur5e",
    urdf_joint_templates=tuple(f"{{arm}}_{j}" for j in UR5E_JOINT_NAMES),
    urdf_link_templates=("{arm}_base", "{arm}_shoulder_link", "{arm}_upper_arm_link", "{arm}_forearm_link",
                         "{arm}_wrist_1_link", "{arm}_wrist_2_link", "{arm}_wrist_3_link"),
    mjcf_body_for_link={"{arm}_base": "base", "{arm}_shoulder_link": "shoulder_link",
                        "{arm}_upper_arm_link": "upper_arm_link", "{arm}_forearm_link": "forearm_link",
                        "{arm}_wrist_1_link": "wrist_1_link", "{arm}_wrist_2_link": "wrist_2_link",
                        "{arm}_wrist_3_link": "wrist_3_link"},
    system_prompt_path=str(REPO_ROOT / "docs" / "SYSTEM_PROMPT_UR5E.md"),
    tilt_note_path=str(REPO_ROOT / "docs" / "TILT_NOTE_UR5E.md"),
    # Axis-aligned envelope of the grasp point over the URDF joint box in the gravity-aligned arm frame, union
    # of both (mirrored) arms, rounded outward to the millimetre: `python -m utils workspace --embodiment
    # ur5e_arms --arm left|right` (200000 FK samples each, max reach 1.268 m). Rotations: the full range the
    # reported extrinsic xyz Euler angles can represent. A bounding box, not the reachable set.
    default_bounds=Bounds(x=(-1.095, 1.083), y=(-1.21, 1.213), z=(-0.989, 1.213),
                          yaw=(-3.1416, 3.1416), pitch=(-1.5708, 1.5708), roll=(-3.1416, 3.1416)),
    prompt_cache_key="astra-ur5e",
    robot_server_hint="cd $GELLO_SOFTWARE_PATH && python experiments/launch_nodes.py --robot=bimanual_ur",
    wrist_cam_offset_flange=(-0.07, 0.0, 0.06),
    mesh_dirs=(
        # the URDF's `assets/` and `robotiq_meshes/` folders live next to the training-data copy of the URDF
        "/home/skild/bimanual_manipulation/skild-gello-vlm-deploy/act-torchtitan/datapods/robots/urdfs/skild_ur5e",
    ),
    proportions={"upper_arm_m": 0.425, "forearm_m": 0.392, "wrist_to_grasp_m": 0.49, "reach_m": 1.2},
)

ARM_MODELS: Dict[str, ArmModel] = {m.name: m for m in (YAM_ARMS, UR5E_ARMS)}
DEFAULT_ARM_MODEL = YAM_ARMS.name


def get_arm_model(name: Optional[str]) -> ArmModel:
    key = name or DEFAULT_ARM_MODEL
    try:
        return ARM_MODELS[key]
    except KeyError:
        raise KeyError(f"unknown embodiment '{name}'; choose one of {', '.join(sorted(ARM_MODELS))}") from None


__all__ = ["ArmModel", "Capsule", "ARM_MODELS", "DEFAULT_ARM_MODEL", "YAM_ARMS", "UR5E_ARMS", "get_arm_model", "ARMS"]
