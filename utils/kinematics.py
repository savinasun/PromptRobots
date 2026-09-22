"""Forward / inverse kinematics for one arm (MuJoCo), plus the eef-state angle conventions.

Frames
------
* FK/IK run on the arm model's single-arm MJCF (`yam.xml`, `ur5e.xml`) in that model's own base frame. The
  `ArmModel` may mount each arm differently (the UR5e bases are pitched 45 deg outward); `mount_rot[arm]`
  turns the MJCF result into the gravity-aligned arm frame the rest of the pipeline reasons in: +x forward,
  +y left, +z up, origin at the arm's base. For the YAM the mount is the identity and the `arm` argument
  makes no difference; pass it anyway so the same call sites serve both rigs.
* `grasp_site` sits between the fingertips - this is the "grasp point" the embodiment notes refer to. Its
  z axis is the tool axis, its y axis the jaw-opening axis.
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

from utils.arm_models import ArmModel, get_arm_model
from utils.config import ARMS, PipelineConfig


@dataclass
class IKResult:
    q: np.ndarray
    pos_err_m: float
    ori_err_rad: float
    converged: bool
    iterations: int


class ArmKinematics:
    """FK/IK for a single 6-DoF arm. One instance serves both arms of a rig (they share the MJCF; only the base
    mount differs, see `ArmModel`)."""

    def __init__(
        self,
        xml_path: Optional[str] = None,
        site: Optional[str] = None,
        joint_lower: Optional[Sequence[float]] = None,
        joint_upper: Optional[Sequence[float]] = None,
        limit_margin: float = 0.0,
        model: Optional[ArmModel] = None,
    ):
        self.spec: ArmModel = model or get_arm_model(None)
        xml_path = xml_path or self.spec.mjcf_path
        site = site or self.spec.grasp_site
        self.model = mujoco.MjModel.from_xml_path(str(xml_path))
        self.data = mujoco.MjData(self.model)
        self.site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, site)
        if self.site_id < 0:
            raise ValueError(f"site '{site}' not found in {xml_path}")
        self.n = 6
        if self.model.nq < self.n:
            raise ValueError(f"model has {self.model.nq} joints, expected >= {self.n}")
        lower = np.asarray(joint_lower if joint_lower is not None else self.spec.joint_lower, dtype=float)
        upper = np.asarray(joint_upper if joint_upper is not None else self.spec.joint_upper, dtype=float)
        # never exceed the model's own joint ranges either
        model_lower = self.model.jnt_range[: self.n, 0]
        model_upper = self.model.jnt_range[: self.n, 1]
        self.lower = np.maximum(lower, model_lower) + limit_margin
        self.upper = np.minimum(upper, model_upper) - limit_margin
        self._jacp = np.zeros((3, self.model.nv))
        self._jacr = np.zeros((3, self.model.nv))
        self.mount_rot: Dict[str, np.ndarray] = {arm: self.spec.mount_rot(arm) for arm in ARMS}
        self._identity_mount = all(np.allclose(R, np.eye(3)) for R in self.mount_rot.values())
        self._body_ids: Dict[str, int] = {}
        for i in range(self.model.nbody):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, i)
            if name:
                self._body_ids[name] = i

    @classmethod
    def from_config(cls, cfg: PipelineConfig, limit_margin: Optional[float] = None,
                    site: Optional[str] = None) -> "ArmKinematics":
        """The kinematics every pipeline entry point should use: the configured embodiment, MJCF and joint box."""
        margin = cfg.motion.joint_limit_margin_rad if limit_margin is None else limit_margin
        return cls(cfg.robot.mjcf_path, site=site, joint_lower=cfg.robot.joint_lower, joint_upper=cfg.robot.joint_upper,
                   limit_margin=margin, model=get_arm_model(cfg.embodiment_name))

    # --------------------------------------------------------------- frames
    def _mount(self, arm: Optional[str]) -> Optional[np.ndarray]:
        if arm is None or self._identity_mount:
            return None
        return self.mount_rot[arm]

    def arm_offset(self, arm: str) -> np.ndarray:
        """Origin of `arm`'s frame in the left arm's frame (the simulator/visualizer world frame)."""
        return self.spec.arm_offset(arm)

    @property
    def grasp_offset_m(self) -> float:
        return self.spec.grasp_offset_m

    @property
    def jaw_max_opening_m(self) -> float:
        return self.spec.jaw_max_opening_m

    # ------------------------------------------------------------------ FK
    def _forward(self, q: np.ndarray) -> None:
        q = np.asarray(q, dtype=float)
        if q.shape[0] != self.n:
            raise ValueError(f"expected {self.n} joint values, got {q.shape[0]}")
        self.data.qpos[:] = 0.0
        self.data.qpos[: self.n] = q
        mujoco.mj_forward(self.model, self.data)

    def fk(self, q: np.ndarray, arm: Optional[str] = None) -> Tuple[np.ndarray, np.ndarray]:
        """Return (position(3), rotation(3x3)) of the grasp site in the arm frame (see module docstring)."""
        self._forward(q)
        pos = self.data.site_xpos[self.site_id].copy()
        rot = self.data.site_xmat[self.site_id].reshape(3, 3).copy()
        R = self._mount(arm)
        if R is not None:
            pos, rot = R @ pos, R @ rot
        return pos, rot

    def link_origins(self, q: np.ndarray, arm: Optional[str] = None) -> Dict[str, np.ndarray]:
        """Positions of every MJCF body origin plus 'base', 'grasp' and the tool axis, in the arm frame.

        Runs FK; the compact MJCF bodies share the URDF link frames, so the arm model's capsule spec can name them.
        """
        pos, rot = self.fk(q, arm)
        R = self._mount(arm)
        out: Dict[str, np.ndarray] = {"base": np.zeros(3)}
        for name, bid in self._body_ids.items():
            p = self.data.xpos[bid].copy()
            out[name] = p if R is None else R @ p
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
        arm: Optional[str] = None,
    ) -> IKResult:
        """Damped-least-squares IK on the full 6-D pose, respecting joint limits by clipping.

        Seeded from `q_seed`; for tightly spaced waypoints this converges in a handful of iterations. The target
        is given in the arm frame; it is mapped into the MJCF frame once, so the solver itself is mount-agnostic.
        """
        target_pos = np.asarray(target_pos, dtype=float)
        target_rot = np.asarray(target_rot, dtype=float)
        R = self._mount(arm)
        if R is not None:
            target_pos, target_rot = R.T @ target_pos, R.T @ target_rot
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
