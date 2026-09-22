"""Arm-to-arm clearance check with capsules (used by the gateway when both arms share a workspace).

Each arm is approximated by capsules along its kinematic chain plus the gripper housing and the fingers. Which
links, with which radii, comes from the selected `ArmModel` (utils/arm_models.py); for the YAM that is
    base column, upper arm (link_2->link_3), forearm (link_3->link_4->link_5), wrist (link_5->link_6),
    housing (flange -> 6 cm along the tool axis, radius 5 cm), neck, fingers (-> grasp point, radius 0.8 cm).
The right arm is offset by the rig's base placement. `clearance()` returns the smallest surface-to-surface
distance between any left capsule and any right capsule (negative = overlap) and the pair of parts involved.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from utils.config import LEFT_SLICE, RIGHT_SLICE
from utils.kinematics import ArmKinematics

ARM_SLICES = {"left": LEFT_SLICE, "right": RIGHT_SLICE}
BASE_DISTANCE_M = 0.61       # YAM default; `clearance()` reads the real value from the kinematics' arm model
GRASP_OFFSET_M = 0.1347      # YAM default, idem
LINK_MARGIN_M = 0.02         # required clearance when an arm link or the wide housing is involved
TIP_MARGIN_M = 0.005         # required clearance between fingers/neck of the two tools (cooperative grasps)


@dataclass
class Capsule:
    name: str
    a: np.ndarray
    b: np.ndarray
    r: float
    margin: float = LINK_MARGIN_M


def arm_capsules(kin: ArmKinematics, q6: np.ndarray, offset: np.ndarray, arm: Optional[str] = None) -> List[Capsule]:
    """Capsules of one arm in the left arm's frame (`offset` = that arm's base position in it)."""
    spec = kin.spec
    o = kin.link_origins(q6, arm)
    z = o["tool_axis"]
    flange = o["grasp"] - spec.grasp_offset_m * z
    o["flange"] = flange
    housing_end = flange + spec.housing_length_m * z
    neck_end = housing_end + spec.neck_length_m * z
    height, radius = spec.base_column
    up = kin.mount_rot[arm][:, 2] if arm is not None else np.array([0.0, 0.0, 1.0])
    caps = [Capsule("base", o["base"], o["base"] + height * up, radius)]
    for c in spec.capsules:
        caps.append(Capsule(c.name, o[c.body_a], o[c.body_b], c.radius_m))
    caps += [
        Capsule("gripper housing", flange, housing_end, spec.housing_radius_m),
        Capsule("gripper neck", housing_end, neck_end, spec.neck_radius_m, TIP_MARGIN_M),
        Capsule("fingers", neck_end, o["grasp"], spec.finger_radius_m, TIP_MARGIN_M),
    ]
    for c in caps:
        c.a = c.a + offset
        c.b = c.b + offset
    return caps


def segment_distance(p1: np.ndarray, q1: np.ndarray, p2: np.ndarray, q2: np.ndarray) -> float:
    """Minimum distance between segments p1-q1 and p2-q2 (Ericson, Real-Time Collision Detection 5.1.9)."""
    d1, d2, r = q1 - p1, q2 - p2, p1 - p2
    a, e, f = d1 @ d1, d2 @ d2, d2 @ r
    eps = 1e-12
    if a <= eps and e <= eps:
        return float(np.linalg.norm(p1 - p2))
    if a <= eps:
        s, t = 0.0, float(np.clip(f / e, 0.0, 1.0))
    else:
        c = d1 @ r
        if e <= eps:
            t, s = 0.0, float(np.clip(-c / a, 0.0, 1.0))
        else:
            b = d1 @ d2
            denom = a * e - b * b
            s = float(np.clip((b * f - c * e) / denom, 0.0, 1.0)) if denom > eps else 0.0
            t = (b * s + f) / e
            if t < 0.0:
                t, s = 0.0, float(np.clip(-c / a, 0.0, 1.0))
            elif t > 1.0:
                t, s = 1.0, float(np.clip((b - c) / a, 0.0, 1.0))
    return float(np.linalg.norm((p1 + d1 * s) - (p2 + d2 * t)))


@dataclass
class ClearanceResult:
    clearance_m: float          # surface-to-surface distance of the most critical pair (negative = overlap)
    margin_m: float             # clearance required for that pair
    left_part: str
    right_part: str

    @property
    def deficit_m(self) -> float:
        """How far below the requirement the critical pair is (<= 0 means acceptable)."""
        return self.margin_m - self.clearance_m

    def __iter__(self):
        yield self.clearance_m
        yield self.left_part
        yield self.right_part


def clearance(kin: ArmKinematics, q14: np.ndarray, base_distance: Optional[float] = None,
              link_margin: float = LINK_MARGIN_M, tip_margin: float = TIP_MARGIN_M) -> ClearanceResult:
    """Most critical capsule pair between the two arms, judged against per-part clearance requirements.

    A pair of finger/neck capsules only needs `tip_margin` (both tools may work on one object); any pair
    involving an arm link or the wide housing needs `link_margin`. The right arm sits where the rig's arm
    model puts it; `base_distance` overrides that with a plain -y offset (YAM-style rigs only).
    """
    q14 = np.asarray(q14)
    right_offset = kin.arm_offset("right") if base_distance is None else np.array([0.0, -base_distance, 0.0])
    left = arm_capsules(kin, q14[ARM_SLICES["left"]], np.zeros(3), "left")
    right = arm_capsules(kin, q14[ARM_SLICES["right"]], right_offset, "right")
    best: ClearanceResult = ClearanceResult(float("inf"), link_margin, "", "")
    for lc in left:
        for rc in right:
            if lc.name == "base" and rc.name == "base":
                continue
            d = segment_distance(lc.a, lc.b, rc.a, rc.b) - lc.r - rc.r
            req = tip_margin if (lc.margin == TIP_MARGIN_M and rc.margin == TIP_MARGIN_M) else link_margin
            if req - d > best.deficit_m:
                best = ClearanceResult(d, req, lc.name, rc.name)
    return best
