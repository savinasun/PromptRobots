"""Arm-to-arm clearance check with capsules (used by the gateway when both arms share a workspace).

Each arm is approximated by capsules along its kinematic chain plus the gripper housing and the fingers:
    base column, upper arm (link_2->link_3), forearm (link_3->link_4->link_5), wrist (link_5->link_6),
    housing (flange -> 9 cm along the tool axis, radius 5 cm), fingers (housing end -> grasp point, radius 1.5 cm).
The right arm is offset by the base distance. `clearance()` returns the smallest surface-to-surface distance between
any left capsule and any right capsule (negative = overlap) and the pair of parts involved.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

from astra_yam.embodiment import ARM_SLICES
from astra_yam.kinematics import ArmKinematics

BASE_DISTANCE_M = 0.61
GRASP_OFFSET_M = 0.1347
HOUSING_LENGTH_M = 0.06      # wide part of the gripper housing along the tool axis (link_6 mesh: r ~ 5 cm)
NECK_LENGTH_M = 0.03         # tapered part before the fingers (r ~ 3 cm)
LINK_MARGIN_M = 0.02         # required clearance when an arm link or the wide housing is involved
TIP_MARGIN_M = 0.005         # required clearance between fingers/neck of the two tools (cooperative grasps)


@dataclass
class Capsule:
    name: str
    a: np.ndarray
    b: np.ndarray
    r: float
    margin: float = LINK_MARGIN_M


def arm_capsules(kin: ArmKinematics, q6: np.ndarray, offset: np.ndarray) -> List[Capsule]:
    o = kin.link_origins(q6)
    z = o["tool_axis"]
    flange = o["grasp"] - GRASP_OFFSET_M * z
    housing_end = flange + HOUSING_LENGTH_M * z
    neck_end = housing_end + NECK_LENGTH_M * z
    caps = [
        Capsule("base", o["base"], o["base"] + np.array([0, 0, 0.11]), 0.06),
        Capsule("upper arm", o["link_2"], o["link_3"], 0.045),
        Capsule("forearm", o["link_3"], o["link_4"], 0.04),
        Capsule("forearm", o["link_4"], o["link_5"], 0.04),
        Capsule("wrist", o["link_5"], o["link_6"], 0.045),
        Capsule("gripper housing", flange, housing_end, 0.05),
        Capsule("gripper neck", housing_end, neck_end, 0.03, TIP_MARGIN_M),
        Capsule("fingers", neck_end, o["grasp"], 0.008, TIP_MARGIN_M),
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


def clearance(kin: ArmKinematics, q14: np.ndarray, base_distance: float = BASE_DISTANCE_M,
              link_margin: float = LINK_MARGIN_M, tip_margin: float = TIP_MARGIN_M) -> ClearanceResult:
    """Most critical capsule pair between the two arms, judged against per-part clearance requirements.

    A pair of finger/neck capsules only needs `tip_margin` (both tools may work on one object); any pair
    involving an arm link or the wide housing needs `link_margin`.
    """
    left = arm_capsules(kin, np.asarray(q14)[ARM_SLICES["left"]], np.zeros(3))
    right = arm_capsules(kin, np.asarray(q14)[ARM_SLICES["right"]], np.array([0.0, -base_distance, 0.0]))
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
