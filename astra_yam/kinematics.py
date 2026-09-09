"""Forward / inverse kinematics for one YAM arm (MuJoCo), plus the eef-state angle conventions.

Frames
------
* The MuJoCo world frame of `yam.xml` is the arm's base frame: +x forward, +y left, +z up.
* `grasp_site` sits between the fingertips (13.47 cm past the flange `tcp_site`) - this is the
  "grasp point" the Astra embodiment notes refer to.
* yaw / pitch / roll reported to the model are relative to the trial start orientation:
      R_rel = R_now @ R_start^T  ->  extrinsic 'xyz' Euler angles (roll about base x, pitch about base y,
      yaw about base z, positive yaw = counter-clockwise seen from above).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from astra_yam.config import DEFAULT_YAM_XML, YAM_JOINT_LOWER, YAM_JOINT_UPPER


@dataclass
class IKResult:
    q: np.ndarray
    pos_err_m: float
    ori_err_rad: float
    converged: bool
    iterations: int


class ArmKinematics:
    """FK/IK for a single 6-DoF YAM arm. One instance can serve both (identical) arms."""

    def __init__(
        self,
        xml_path: str = DEFAULT_YAM_XML,
        site: str = "grasp_site",
        joint_lower: Optional[Sequence[float]] = None,
        joint_upper: Optional[Sequence[float]] = None,
        limit_margin: float = 0.0,
    ):
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        self.site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, site)
        if self.site_id < 0:
            raise ValueError(f"site '{site}' not found in {xml_path}")
        self.n = 6
        if self.model.nq < self.n:
            raise ValueError(f"model has {self.model.nq} joints, expected >= {self.n}")
        lower = np.asarray(joint_lower if joint_lower is not None else YAM_JOINT_LOWER, dtype=float)
        upper = np.asarray(joint_upper if joint_upper is not None else YAM_JOINT_UPPER, dtype=float)
        # never exceed the model's own joint ranges either
        model_lower = self.model.jnt_range[: self.n, 0]
        model_upper = self.model.jnt_range[: self.n, 1]
        self.lower = np.maximum(lower, model_lower) + limit_margin
        self.upper = np.minimum(upper, model_upper) - limit_margin
        self._jacp = np.zeros((3, self.model.nv))
        self._jacr = np.zeros((3, self.model.nv))

    # ------------------------------------------------------------------ FK
    def fk(self, q: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Return (position(3), rotation(3x3)) of the grasp site in the base frame."""
        q = np.asarray(q, dtype=float)
        if q.shape[0] != self.n:
            raise ValueError(f"expected {self.n} joint values, got {q.shape[0]}")
        self.data.qpos[:] = 0.0
        self.data.qpos[: self.n] = q
        mujoco.mj_forward(self.model, self.data)
        pos = self.data.site_xpos[self.site_id].copy()
        rot = self.data.site_xmat[self.site_id].reshape(3, 3).copy()
        return pos, rot

    def link_origins(self, q: np.ndarray) -> Dict[str, np.ndarray]:
        """Positions of the link origins (link_1..link_6), the flange/tcp and the grasp point in the base frame.

        Runs FK; the compact yam.xml bodies share the URDF link frames.
        """
        pos, rot = self.fk(q)
        out: Dict[str, np.ndarray] = {"base": np.zeros(3)}
        for i in range(1, 7):
            bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, f"link_{i}")
            if bid >= 0:
                out[f"link_{i}"] = self.data.xpos[bid].copy()
        out["grasp"] = pos
        out["tool_axis"] = rot[:, 2].copy()
        return out

    def within_limits(self, q: np.ndarray, tol: float = 1e-9) -> bool:
        q = np.asarray(q, dtype=float)
        return bool(np.all(q >= self.lower - tol) and np.all(q <= self.upper + tol))

    def clip(self, q: np.ndarray) -> np.ndarray:
        return np.clip(np.asarray(q, dtype=float), self.lower, self.upper)

    # ------------------------------------------------------------------ IK
    def ik(
        self,
        target_pos: np.ndarray,
        target_rot: np.ndarray,
        q_seed: np.ndarray,
        max_iters: int = 100,
        pos_tol: float = 1e-3,
        ori_tol: float = 5e-3,
        damping: float = 0.02,
        ori_weight: float = 0.5,
        max_step: float = 0.2,
    ) -> IKResult:
        """Damped-least-squares IK on the full 6-D pose, respecting joint limits by clipping.

        Seeded from `q_seed`; for tightly spaced waypoints this converges in a handful of iterations.
        """
        target_pos = np.asarray(target_pos, dtype=float)
        target_rot = np.asarray(target_rot, dtype=float)
        q = self.clip(q_seed)
        eye = np.eye(6)
        pos_err = ori_err = float("inf")
        for it in range(max_iters):
            pos, rot = self.fk(q)
            e_p = target_pos - pos
            e_r = Rotation.from_matrix(target_rot @ rot.T).as_rotvec()
            pos_err = float(np.linalg.norm(e_p))
            ori_err = float(np.linalg.norm(e_r))
            if pos_err < pos_tol and ori_err < ori_tol:
                return IKResult(q, pos_err, ori_err, True, it)
            mujoco.mj_jacSite(self.model, self.data, self._jacp, self._jacr, self.site_id)
            J = np.vstack([self._jacp[:, : self.n], ori_weight * self._jacr[:, : self.n]])
            e = np.concatenate([e_p, ori_weight * e_r])
            # adaptive damping: heavier when far from the target for stability, lighter near it
            lam = damping * (1.0 + 5.0 * min(pos_err, 0.2))
            dq = J.T @ np.linalg.solve(J @ J.T + (lam ** 2) * eye, e)
            scale = np.max(np.abs(dq)) / max_step
            if scale > 1.0:
                dq /= scale
            q = self.clip(q + dq)
        # final evaluation after the last update
        pos, rot = self.fk(q)
        pos_err = float(np.linalg.norm(target_pos - pos))
        ori_err = float(np.linalg.norm(Rotation.from_matrix(target_rot @ rot.T).as_rotvec()))
        return IKResult(q, pos_err, ori_err, pos_err < pos_tol and ori_err < ori_tol, max_iters)


# ---------------------------------------------------------------------- angles
def relative_ypr(rot_now: np.ndarray, rot_start: np.ndarray) -> np.ndarray:
    """(yaw, pitch, roll) of rot_now relative to rot_start, as extrinsic base-frame rotations."""
    rel = np.asarray(rot_now) @ np.asarray(rot_start).T
    roll, pitch, yaw = Rotation.from_matrix(rel).as_euler("xyz")
    return np.array([yaw, pitch, roll], dtype=float)


def rotation_from_ypr(rot_start: np.ndarray, yaw: float, pitch: float = 0.0, roll: float = 0.0) -> np.ndarray:
    """Inverse of relative_ypr: R = Rz(yaw) Ry(pitch) Rx(roll) R_start."""
    rel = Rotation.from_euler("xyz", [roll, pitch, yaw]).as_matrix()
    return rel @ np.asarray(rot_start)


def unwrap_angle(angle: float, reference: float) -> float:
    """Shift `angle` by multiples of 2*pi so that it is closest to `reference`."""
    return float(angle + 2.0 * np.pi * np.round((reference - angle) / (2.0 * np.pi)))
