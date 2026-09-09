"""Derive the robot's own limits from the URDF instead of hand-picking them.

* `urdf_joint_limits()` reads the exact revolute limits out of `skild_yam_v2.urdf`.
* `reachable_envelope()` sweeps that joint box with forward kinematics and reports the axis-aligned envelope of
  the grasp point plus the achievable tool orientation relative to a reference (home) pose.

The envelope is a bounding box, not the reachable set: points inside it may still be unreachable, and the gateway
rejects those per target via IK. Use `python -m astra_yam workspace` to print a YAML block for `configs/`.

Orientation note: the interface reports tool rotation as extrinsic xyz Euler angles (roll about base +x, pitch
about base +y, yaw about base +z) of `R_now @ R_start^T`. That convention puts pitch in [-pi/2, pi/2] by
construction, so +/-1.5708 rad is the full representable pitch range regardless of the joint limits; yaw and roll
span the full +/-pi.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import numpy as np

from astra_yam.config import DEFAULT_YAM_XML, REFERENCE_HOME_JOINTS
from astra_yam.kinematics import ArmKinematics, relative_ypr

# Full representable range of the reported Euler angles (a property of the convention, not of the joints).
EULER_YAW_LIMIT = float(np.pi)
EULER_PITCH_LIMIT = float(np.pi / 2)
EULER_ROLL_LIMIT = float(np.pi)


@dataclass
class JointLimits:
    lower: np.ndarray            # (6,) rad
    upper: np.ndarray            # (6,) rad
    names: Tuple[str, ...]
    source: str

    def as_lists(self) -> Tuple[list, list]:
        return [round(float(v), 5) for v in self.lower], [round(float(v), 5) for v in self.upper]


def urdf_joint_limits(urdf_path: str, arm: str = "left") -> JointLimits:
    """Exact revolute joint limits of one arm, straight out of the URDF (no rounding, no estimation)."""
    import yourdfpy

    urdf = yourdfpy.URDF.load(str(urdf_path), load_meshes=False, build_scene_graph=True)
    names = tuple(f"{arm}_joint{k}" for k in range(1, 7))
    missing = [n for n in names if n not in urdf.joint_map]
    if missing:
        raise KeyError(f"{urdf_path} has no joints {missing}; found {sorted(urdf.joint_map)}")
    lower = np.array([urdf.joint_map[n].limit.lower for n in names], dtype=float)
    upper = np.array([urdf.joint_map[n].limit.upper for n in names], dtype=float)
    return JointLimits(lower, upper, names, str(urdf_path))


@dataclass
class Envelope:
    """Axis-aligned bounds of the grasp point and of the tool orientation relative to `reference_joints`."""

    x: Tuple[float, float]
    y: Tuple[float, float]
    z: Tuple[float, float]
    yaw: Tuple[float, float]
    pitch: Tuple[float, float]
    roll: Tuple[float, float]
    max_reach_m: float
    samples: int

    def as_dict(self) -> Dict[str, Tuple[float, float]]:
        return {"x": self.x, "y": self.y, "z": self.z, "yaw": self.yaw, "pitch": self.pitch, "roll": self.roll}

    def yaml_block(self, decimals: int = 3) -> str:
        def rng(pair: Tuple[float, float]) -> str:
            lo = _round_out(pair[0], decimals, down=True)
            hi = _round_out(pair[1], decimals, down=False)
            return f"[{lo}, {hi}]"

        return "\n".join(
            [
                "bounds:",
                f"  x: {rng(self.x)}",
                f"  y: {rng(self.y)}",
                f"  z: {rng(self.z)}",
                f"  yaw: {rng(self.yaw)}",
                f"  pitch: {rng(self.pitch)}",
                f"  roll: {rng(self.roll)}",
                "  gripper: [0.0, 1.0]",
            ]
        )


def _round_out(value: float, decimals: int, down: bool) -> float:
    """Round away from zero-width: lower bounds down, upper bounds up, so nothing reachable is excluded."""
    scale = 10.0 ** decimals
    return float(np.floor(value * scale) / scale) if down else float(np.ceil(value * scale) / scale)


def reachable_envelope(
    joint_limits: JointLimits,
    reference_joints: Sequence[float] = REFERENCE_HOME_JOINTS,
    samples: int = 200_000,
    seed: int = 0,
    yam_xml_path: str = DEFAULT_YAM_XML,
    kin: Optional[ArmKinematics] = None,
) -> Envelope:
    """FK sweep of the joint box: random samples (fixed seed) plus all 64 corners, so results are reproducible."""
    kin = kin or ArmKinematics(yam_xml_path, joint_lower=joint_limits.lower, joint_upper=joint_limits.upper,
                               limit_margin=0.0)
    rng = np.random.default_rng(seed)
    q = rng.uniform(joint_limits.lower, joint_limits.upper, size=(int(samples), 6))
    corners = np.array(np.meshgrid(*[[lo, hi] for lo, hi in zip(joint_limits.lower, joint_limits.upper)],
                                   indexing="ij")).reshape(6, -1).T
    q = np.vstack([q, corners])
    _, rot_ref = kin.fk(np.asarray(reference_joints, dtype=float))
    pos = np.empty((len(q), 3))
    ypr = np.empty((len(q), 3))
    for i, qi in enumerate(q):
        p, r = kin.fk(qi)
        pos[i] = p
        ypr[i] = relative_ypr(r, rot_ref)
    return Envelope(
        x=(float(pos[:, 0].min()), float(pos[:, 0].max())),
        y=(float(pos[:, 1].min()), float(pos[:, 1].max())),
        z=(float(pos[:, 2].min()), float(pos[:, 2].max())),
        yaw=(float(ypr[:, 0].min()), float(ypr[:, 0].max())),
        pitch=(float(ypr[:, 1].min()), float(ypr[:, 1].max())),
        roll=(float(ypr[:, 2].min()), float(ypr[:, 2].max())),
        max_reach_m=float(np.linalg.norm(pos, axis=1).max()),
        samples=len(q),
    )


def full_rotation_bounds() -> Dict[str, Tuple[float, float]]:
    """The convention's full angle ranges, rounded to the 4 decimals the tool descriptions print."""
    return {
        "yaw": (-round(EULER_YAW_LIMIT, 4), round(EULER_YAW_LIMIT, 4)),
        "pitch": (-round(EULER_PITCH_LIMIT, 4), round(EULER_PITCH_LIMIT, 4)),
        "roll": (-round(EULER_ROLL_LIMIT, 4), round(EULER_ROLL_LIMIT, 4)),
    }


def report(urdf_path: str, samples: int = 200_000, arm: str = "left",
           reference_joints: Sequence[float] = REFERENCE_HOME_JOINTS) -> str:
    jl = urdf_joint_limits(urdf_path, arm)
    env = reachable_envelope(jl, reference_joints=reference_joints, samples=samples)
    lines = [f"URDF: {Path(jl.source).resolve()}", "", "revolute joint limits (rad):"]
    for name, lo, hi in zip(jl.names, jl.lower, jl.upper):
        lines.append(f"  {name:14s} [{lo:+.7f}, {hi:+.7f}]   span {np.degrees(hi - lo):7.2f} deg")
    lines += [
        "",
        f"grasp-point envelope from {env.samples} FK samples of that joint box:",
        f"  x: [{env.x[0]:+.4f}, {env.x[1]:+.4f}] m",
        f"  y: [{env.y[0]:+.4f}, {env.y[1]:+.4f}] m",
        f"  z: [{env.z[0]:+.4f}, {env.z[1]:+.4f}] m      (max reach {env.max_reach_m:.4f} m)",
        "",
        "tool orientation relative to the reference pose (extrinsic xyz Euler):",
        f"  yaw:   [{env.yaw[0]:+.4f}, {env.yaw[1]:+.4f}] rad",
        f"  pitch: [{env.pitch[0]:+.4f}, {env.pitch[1]:+.4f}] rad   (convention caps pitch at +/-{EULER_PITCH_LIMIT:.4f})",
        f"  roll:  [{env.roll[0]:+.4f}, {env.roll[1]:+.4f}] rad",
        "",
        "YAML block (positions rounded outward to the millimetre, rotations at the convention's full range):",
        "",
    ]
    block = env.yaml_block()
    rot = full_rotation_bounds()
    block = "\n".join(
        line if not any(line.strip().startswith(f"{k}:") for k in rot)
        else f"  {line.strip().split(':')[0]}: [{rot[line.strip().split(':')[0]][0]}, {rot[line.strip().split(':')[0]][1]}]"
        for line in block.splitlines()
    )
    lines.append(block)
    lines += ["", "joint limits block:", "",
              f"robot:\n  joint_lower: {jl.as_lists()[0]}\n  joint_upper: {jl.as_lists()[1]}"]
    return "\n".join(lines)
