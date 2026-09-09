"""Kinematic simulator for dry runs without hardware.

* SimYamRobot     - implements the RobotBackend protocol; joints track commands instantly.
* SimObject/SimWorld - boxes and cylinders on a table (presets: blocks, kitchen, empty) with grasp/release
                    emulation so pick-and-place tasks can be completed in simulation.
* SimCameraSource - renders schematic 'top_cam' / 'left_cam' / 'right_cam' images of the world.
* serve_sim_zmq   - serves SimYamRobot over the gello ZMQ REQ/REP pickle protocol (port 6001) so the
                    real `ZmqYamRobot` client path can be exercised end to end.
"""
from __future__ import annotations

import pickle
import threading
import time
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from astra_yam.config import (
    ARMS,
    LEFT_GRIPPER,
    LEFT_SLICE,
    NUM_DOFS,
    RIGHT_GRIPPER,
    RIGHT_SLICE,
    YAM_JOINT_LOWER,
    YAM_JOINT_UPPER,
)
from astra_yam.kinematics import ArmKinematics

ARM_SLICES = {"left": LEFT_SLICE, "right": RIGHT_SLICE}
ARM_GRIPPER = {"left": LEFT_GRIPPER, "right": RIGHT_GRIPPER}
BASE_DISTANCE = 0.61  # right base is 0.61 m to the right (-y) of the left base (hw v2.x)
GRIPPER_MAX_OPENING_M = 0.095
GRASP_XY_TOLERANCE_M = 0.015
GRASP_Z_TOLERANCE_M = 0.01
JAW_HALF_THICKNESS_M = 0.012     # outer half-width of the closed fingertips (pushing surface)
RELEASE_MARGIN_M = 0.004         # a held object is released once the jaws open this much wider than it


class SimYamRobot:
    """Ideal position-controlled bimanual YAM. Thread-safe."""

    def __init__(self, initial_q: Optional[np.ndarray] = None, world: Optional["SimWorld"] = None,
                 on_command: Optional[Callable[[np.ndarray], None]] = None):
        self._lock = threading.Lock()
        q = np.zeros(NUM_DOFS) if initial_q is None else np.asarray(initial_q, dtype=float).copy()
        self._q = q
        self.world = world
        self.on_command = on_command          # e.g. a visualizer's update_robot
        self.command_log: List[np.ndarray] = []
        self._lower = np.array(list(YAM_JOINT_LOWER) + [0.0] + list(YAM_JOINT_LOWER) + [0.0])
        self._upper = np.array(list(YAM_JOINT_UPPER) + [1.0] + list(YAM_JOINT_UPPER) + [1.0])

    def num_dofs(self) -> int:
        return NUM_DOFS

    def get_joint_positions(self) -> np.ndarray:
        with self._lock:
            q = self._q.copy()
        if self.world is not None:
            # a held object stops the jaws from closing further
            for arm in ARMS:
                floor = self.world.gripper_floor(arm)
                if floor is not None:
                    q[ARM_GRIPPER[arm]] = max(q[ARM_GRIPPER[arm]], floor)
        return q

    def command_joint_positions(self, q14: np.ndarray) -> None:
        q = np.clip(np.asarray(q14, dtype=float).reshape(-1), self._lower, self._upper)
        if q.shape[0] != NUM_DOFS:
            raise ValueError(f"expected {NUM_DOFS} values, got {q.shape[0]}")
        with self._lock:
            self._q = q
            self.command_log.append(q.copy())
        if self.world is not None:
            self.world.update(q)
        if self.on_command is not None:
            self.on_command(q)

    def get_observations(self) -> Dict[str, np.ndarray]:
        q = self.get_joint_positions()
        return {"joint_positions": q, "joint_velocities": np.zeros(NUM_DOFS), "ee_pos_quat": np.zeros(14)}

    def get_joint_state(self) -> np.ndarray:
        return self.get_joint_positions()

    def command_joint_state(self, joint_state: np.ndarray) -> None:
        """gello Robot-protocol name (used by gello.zmq_core.robot_node.ZMQServerRobot)."""
        self.command_joint_positions(joint_state)

    def close(self) -> None:
        pass


# ---------------------------------------------------------------------------
# Objects and world
# ---------------------------------------------------------------------------
class SimObject:
    """A box (size = (sx, sy, sz)) or a cylinder (size = (radius, height)); `pos` is the center."""

    def __init__(self, name: str, pos: Sequence[float], size: Sequence[float], color_bgr: Tuple[int, int, int],
                 shape: str = "box"):
        if shape not in ("box", "cylinder"):
            raise ValueError(f"unknown shape {shape}")
        self.name = name
        self.shape = shape
        self.pos = np.asarray(pos, dtype=float)              # LEFT arm base frame
        self.size = tuple(float(v) for v in size)
        self.color = tuple(int(c) for c in color_bgr)        # BGR for the schematic renderer
        self.held_by: Optional[str] = None
        self.grasp_offset = np.zeros(3)
        self.initial_pos = self.pos.copy()
        self.rot = np.eye(3)                  # orientation; held boxes/cylinders follow the tool's rotation
        self._tool_rot0: Optional[np.ndarray] = None
        self._rot0: Optional[np.ndarray] = None

    @property
    def tilt_rad(self) -> float:
        """Angle between the object's own up axis and vertical."""
        return float(np.arccos(np.clip(self.rot[2, 2], -1.0, 1.0)))

    def top_center(self) -> np.ndarray:
        return self.pos + self.rot @ np.array([0.0, 0.0, self.height / 2])

    @property
    def height(self) -> float:
        return self.size[2] if self.shape == "box" else self.size[1]

    @property
    def grasp_width(self) -> float:
        return min(self.size[0], self.size[1]) if self.shape == "box" else 2 * self.size[0]

    def width_along(self, jaw_axis: np.ndarray) -> float:
        """Extent of the object between two jaws closing along `jaw_axis` (world frame, horizontal part)."""
        if self.shape != "box":
            return 2 * self.size[0]
        a = np.asarray(jaw_axis, dtype=float)[:2]
        n = np.linalg.norm(a)
        if n < 1e-6:
            return self.grasp_width
        a = a / n
        return float(abs(a[0]) * self.size[0] + abs(a[1]) * self.size[1])

    @property
    def footprint_radius(self) -> float:
        return max(self.size[0], self.size[1]) / 2 if self.shape == "box" else self.size[0]

    @property
    def graspable(self) -> bool:
        return self.grasp_width <= GRIPPER_MAX_OPENING_M

    @property
    def color_rgb(self) -> Tuple[int, int, int]:
        b, g, r = self.color
        return (r, g, b)

    def status(self) -> dict:
        return {"pos": [round(float(v), 4) for v in self.pos], "held_by": self.held_by, "shape": self.shape,
                "size": list(self.size)}


class SimCan(SimObject):
    """Open-top can of powder. Tilting it past POUR_START_RAD lets powder out; what leaves the can lands in the
    target bowl if the opening is above the bowl's footprint, otherwise it is spilled. The amount poured is a
    static function of the largest tilt reached (kinematic sim, no flow dynamics)."""

    POUR_START_RAD = np.radians(50.0)
    POUR_FULL_RAD = np.radians(95.0)

    def __init__(self, name: str, pos: Sequence[float], radius: float, height: float, color_bgr: Tuple[int, int, int],
                 target: str):
        super().__init__(name, pos, (radius, height), color_bgr, "cylinder")
        self.target = target
        self.poured_in_bowl = 0.0
        self.spilled = 0.0

    @property
    def content(self) -> float:
        return max(0.0, 1.0 - self.poured_in_bowl - self.spilled)

    def status(self) -> dict:
        d = super().status()
        d.update({"tilt_deg": round(float(np.degrees(self.tilt_rad)), 1), "poured_in_bowl": round(self.poured_in_bowl, 3),
                  "spilled": round(self.spilled, 3), "content": round(self.content, 3)})
        return d


class SimCase(SimObject):
    """AirPods Pro (2nd gen) charging case reconstructed from Apple's dimensions: 60.6 x 45.2 x 21.7 mm, 16 mm lid,
    hinge along the top rear edge. Lies flat on a support (lid face up, hinge at the far edge). When lifted by its
    width it pivots to hang upright with the lid on top; the other gripper can then pinch the lid across its
    21.7 mm depth (jaws along x) and swing it open about the hinge.
    """

    W, D, H = 0.0606, 0.0452, 0.0217        # width (y when flat), standing height (x when flat), thickness (z when flat)
    LID = 0.016                             # lid height when standing
    OPEN_LATCH_RAD = 1.0                    # released above this angle the lid stays open, below it snaps shut
    MAX_OPEN_RAD = 1.9

    def __init__(self, name: str, pos: Sequence[float], color_bgr: Tuple[int, int, int] = (240, 240, 240)):
        super().__init__(name, pos, (self.D, self.W, self.H), color_bgr, "box")
        self.hanging = False
        self.lid_angle = 0.0
        self.lid_holder: Optional[str] = None
        self._lid_v0: Optional[np.ndarray] = None

    # geometry ------------------------------------------------------------
    @property
    def height(self) -> float:
        return self.D if self.hanging else self.H

    @property
    def footprint_radius(self) -> float:
        return max(self.W, self.H) / 2 if self.hanging else max(self.D, self.W) / 2

    def width_along(self, jaw_axis: np.ndarray) -> float:
        if not self.hanging:
            return super().width_along(jaw_axis)
        a = np.asarray(jaw_axis, dtype=float)[:2]
        n = np.linalg.norm(a)
        if n < 1e-6:
            return self.H
        a = a / n
        return float(abs(a[0]) * self.H + abs(a[1]) * self.W)

    def body_center(self) -> np.ndarray:
        """Center of the body part (case minus lid)."""
        if self.hanging:
            return self.pos + np.array([0.0, 0.0, -self.LID / 2])
        return self.pos + np.array([-self.LID / 2, 0.0, 0.0])

    def lid_center(self) -> np.ndarray:
        """Lid center in the closed configuration (opening is drawn as a rotation about the hinge)."""
        if self.hanging:
            return self.pos + np.array([0.0, 0.0, self.D / 2 - self.LID / 2])
        return self.pos + np.array([self.D / 2 - self.LID / 2, 0.0, 0.0])

    def lid_dims(self) -> Tuple[float, float, float]:
        return (self.H, self.W, self.LID) if self.hanging else (self.LID, self.W, self.H)

    def hinge(self) -> np.ndarray:
        """Point on the hinge line (top rear edge when hanging; far top edge when flat)."""
        if self.hanging:
            return self.pos + np.array([-self.H / 2, 0.0, self.D / 2])
        return self.pos + np.array([self.D / 2, 0.0, self.H / 2])

    def lid_free_edge(self) -> np.ndarray:
        """Mid-point of the lid's free (front, bottom) edge in the closed configuration."""
        if self.hanging:
            return self.pos + np.array([self.H / 2, 0.0, self.D / 2 - self.LID])
        return self.pos + np.array([self.D / 2 - self.LID, 0.0, self.H / 2])

    @property
    def is_open(self) -> bool:
        return self.lid_angle >= self.OPEN_LATCH_RAD

    def status(self) -> dict:
        d = super().status()
        d.update({"hanging": self.hanging, "lid_angle_deg": round(float(np.degrees(self.lid_angle)), 1),
                  "lid_open": self.is_open, "lid_holder": self.lid_holder})
        return d


def _airpods(table_z: float) -> List[SimObject]:
    """The real layout of 2026-09-08: case on the round wooden dish in front of the left arm, cup and plate nearby."""
    dish_h = 0.045
    return [
        SimObject("wooden dish", [0.38, -0.05, table_z + dish_h / 2], (0.07, dish_h), (110, 150, 190), "cylinder"),
        SimCase("airpods case", [0.38, -0.05, table_z + dish_h + SimCase.H / 2]),
        SimObject("teal cup", [0.30, -0.22, table_z + 0.05], (0.035, 0.10), (130, 140, 60), "cylinder"),
        SimObject("white plate", [0.30, 0.14, table_z + 0.0075], (0.10, 0.015), (235, 235, 235), "cylinder"),
    ]


def _blocks(table_z: float) -> List[SimObject]:
    return [
        SimObject("blue block", [0.33, 0.02, table_z + 0.015], (0.03, 0.03, 0.03), (220, 80, 40)),
        SimObject("green block", [0.33, -0.09, table_z + 0.015], (0.03, 0.03, 0.03), (40, 170, 60)),
    ]


def _kitchen(table_z: float) -> List[SimObject]:
    """Stand-ins for items.jpg, laid out for goals_kitchen.txt.

    Under the reference bounds the two arms have no shared workspace (left arm: y in [-0.25, 0.25]; right arm:
    y in [-0.86, -0.36] in the left frame), so cup/mug/bowl/white plate sit in the left zone and the green/blue
    plates in the right zone. Footprints do not overlap.
    """
    return [
        SimObject("teal cup", [0.30, -0.16, table_z + 0.05], (0.035, 0.10), (130, 140, 60), "cylinder"),
        SimObject("white mug", [0.30, 0.10, table_z + 0.045], (0.04, 0.09), (245, 245, 245), "cylinder"),
        SimObject("orange bowl", [0.42, 0.02, table_z + 0.025], (0.06, 0.05), (60, 140, 230), "cylinder"),
        SimObject("white plate", [0.44, -0.18, table_z + 0.0075], (0.10, 0.015), (235, 235, 235), "cylinder"),
        SimObject("green plate", [0.35, -0.51, table_z + 0.0075], (0.09, 0.015), (150, 200, 150), "cylinder"),
        SimObject("blue plate", [0.42, -0.73, table_z + 0.0075], (0.11, 0.015), (200, 130, 90), "cylinder"),
    ]


def _chili(table_z: float) -> List[SimObject]:
    """Chili powder can (6 cm x 10 cm, open top) and the orange bowl in the left arm's zone, with distractors."""
    return [
        SimObject("orange bowl", [0.42, 0.02, table_z + 0.025], (0.06, 0.05), (60, 140, 230), "cylinder"),
        SimCan("chili powder can", [0.30, -0.15, table_z + 0.05], 0.03, 0.10, (40, 40, 200), target="orange bowl"),
        SimObject("teal cup", [0.30, 0.16, table_z + 0.05], (0.035, 0.10), (130, 140, 60), "cylinder"),
        SimObject("white plate", [0.44, -0.20, table_z + 0.0075], (0.10, 0.015), (235, 235, 235), "cylinder"),
    ]


SCENES: Dict[str, Callable[[float], List[SimObject]]] = {"blocks": _blocks, "kitchen": _kitchen, "airpods": _airpods,
                                                          "chili": _chili, "empty": lambda z: []}


class SimWorld:
    """Objects live in the left arm's base frame; the right arm's frame is offset by -BASE_DISTANCE in y."""

    def __init__(self, kin: Optional[ArmKinematics] = None, table_z: float = 0.0, scene: str = "blocks",
                 objects: Optional[List[SimObject]] = None):
        self.kin = kin or ArmKinematics()
        self.table_z = table_z
        if objects is None:
            if scene not in SCENES:
                raise ValueError(f"unknown scene '{scene}'; choose from {sorted(SCENES)}")
            objects = SCENES[scene](table_z)
        self.scene = scene
        self.objects: Dict[str, SimObject] = {o.name: o for o in objects}
        self.grasp_points: Dict[str, np.ndarray] = {arm: np.zeros(3) for arm in ARMS}
        self.grasp_rots: Dict[str, np.ndarray] = {arm: np.eye(3) for arm in ARMS}
        self.grippers: Dict[str, float] = {arm: 1.0 for arm in ARMS}
        self._prev_openings: Dict[str, float] = {arm: GRIPPER_MAX_OPENING_M for arm in ARMS}
        self._prev_grasp_points: Dict[str, Optional[np.ndarray]] = {arm: None for arm in ARMS}
        self._contacts: set = set()
        self._straddled: set = set()        # (arm, object): object sits between the open jaws
        self._lid_straddled: set = set()    # (arm, case): lid sits between the open jaws
        self.events: List[str] = []
        self._lock = threading.Lock()

    @staticmethod
    def arm_offset(arm: str) -> np.ndarray:
        return np.zeros(3) if arm == "left" else np.array([0.0, -BASE_DISTANCE, 0.0])

    def gripper_floor(self, arm: str) -> Optional[float]:
        jaw = self.grasp_rots[arm][:, 1]
        for obj in self.objects.values():
            if obj.held_by == arm:
                return min(1.0, obj.width_along(jaw) / GRIPPER_MAX_OPENING_M)
            if isinstance(obj, SimCase) and obj.lid_holder == arm:
                return min(1.0, obj.H / GRIPPER_MAX_OPENING_M)
        return None

    def held_object(self, arm: str) -> Optional[SimObject]:
        held = next((o for o in self.objects.values() if o.held_by == arm), None)
        if held is None:
            held = next((o for o in self.objects.values() if isinstance(o, SimCase) and o.lid_holder == arm), None)
        return held

    def set_object_position(self, name: str, pos: Sequence[float]) -> None:
        with self._lock:
            obj = self.objects[name]
            p = np.asarray(pos, dtype=float)
            p[2] = max(p[2], self.table_z + obj.height / 2)
            obj.pos = p

    def reset_objects(self) -> None:
        with self._lock:
            for obj in self.objects.values():
                obj.pos = obj.initial_pos.copy()
                obj.held_by = None
            self.events.append("objects reset")

    @staticmethod
    def _z_overlap(gp: np.ndarray, obj: SimObject) -> bool:
        bottom, top = obj.pos[2] - obj.height / 2, obj.pos[2] + obj.height / 2
        return bottom - GRASP_Z_TOLERANCE_M <= gp[2] <= top + GRASP_Z_TOLERANCE_M

    @staticmethod
    def _straddles(opening_m: float, d_xy: float, obj: SimObject, width: Optional[float] = None) -> bool:
        """Object between the fingers: jaws at least as wide as the object and its axis close to the grasp point."""
        width = obj.grasp_width if width is None else width
        if opening_m < width:
            return False
        return d_xy <= max(0.0, opening_m / 2 - obj.footprint_radius) + GRASP_XY_TOLERANCE_M

    def update(self, q14: np.ndarray) -> None:
        """Contact model (kinematic):
        * a grasp happens only when the jaws close over an object they straddled while open;
        * a held object follows the grasp point and is released once the jaws open wider than it;
        * otherwise a tool that overlaps an object at its height pushes it away (closed fingertips
          push like a 2.4 cm wide pusher; open jaws that do not straddle the object push with their outer width).
        """
        q14 = np.asarray(q14, dtype=float)
        with self._lock:
            openings = {}
            for arm in ARMS:
                pos, rot = self.kin.fk(q14[ARM_SLICES[arm]])
                self.grasp_points[arm] = pos + self.arm_offset(arm)
                self.grasp_rots[arm] = rot
                self.grippers[arm] = float(q14[ARM_GRIPPER[arm]])
                openings[arm] = float(np.clip(self.grippers[arm], 0.0, 1.0)) * GRIPPER_MAX_OPENING_M
            for obj in self.objects.values():
                if isinstance(obj, SimCase):
                    self._update_lid(obj, openings)
                if obj.held_by is not None:
                    arm = obj.held_by
                    jaw = self.grasp_rots[arm][:, 1]
                    if openings[arm] > obj.width_along(jaw) + RELEASE_MARGIN_M:   # jaws opened past the object
                        obj.held_by = None
                        if isinstance(obj, SimCase):
                            obj.hanging = False
                            obj.lid_holder = None
                        obj.rot = np.eye(3)                                      # set down upright (simplification)
                        obj._tool_rot0 = obj._rot0 = None
                        self._settle(obj)
                        self.events.append(f"{obj.name} released by {arm}")
                    else:
                        if isinstance(obj, SimCase) or obj._tool_rot0 is None:
                            obj.pos = self.grasp_points[arm] + obj.grasp_offset
                        else:
                            rel = self.grasp_rots[arm] @ obj._tool_rot0.T          # tool rotation since the grasp
                            obj.rot = rel @ obj._rot0
                            obj.pos = self.grasp_points[arm] + rel @ obj.grasp_offset
                        obj.pos[2] = max(obj.pos[2], self.table_z + obj.height / 2)
                        if isinstance(obj, SimCan):
                            self._update_pour(obj)
                        if isinstance(obj, SimCase) and not obj.hanging:
                            support = self._support_height(obj)
                            if obj.pos[2] - obj.H / 2 > support + 0.03:      # lifted clear: pivots to hang lid-up
                                obj.hanging = True
                                obj.grasp_offset = np.array([0.0, 0.0, obj.LID / 2])    # pinched at the body center
                                obj.pos = self.grasp_points[arm] + obj.grasp_offset
                                self.events.append(f"{obj.name} pivoted upright (lid up)")
                    continue
                for arm in ARMS:
                    gp = self.grasp_points[arm]
                    key = (arm, obj.name)
                    if not self._z_overlap(gp, obj):
                        self._contacts.discard(key)
                        self._straddled.discard(key)
                        continue
                    d = float(np.linalg.norm(gp[:2] - obj.pos[:2]))
                    op, prev = openings[arm], self._prev_openings[arm]
                    width = obj.width_along(self.grasp_rots[arm][:, 1])
                    # remember while the open jaws are around the object; forget once the tool is clearly away
                    if width <= GRIPPER_MAX_OPENING_M and self._straddles(op, d, obj, width):
                        self._straddled.add(key)
                    elif d > GRIPPER_MAX_OPENING_M / 2 + obj.footprint_radius + GRASP_XY_TOLERANCE_M:
                        self._straddled.discard(key)
                    # the jaws had the object between them and now close below its width -> grasp
                    if key in self._straddled and prev >= width > op and self.held_object(arm) is None:
                        obj.held_by = arm
                        obj.grasp_offset = obj.pos - gp
                        obj._tool_rot0 = self.grasp_rots[arm].copy()
                        obj._rot0 = obj.rot.copy()
                        self._straddled.discard(key)
                        self.events.append(f"{obj.name} grasped by {arm}")
                        break
                    if key in self._straddled:
                        continue                                            # between the fingers, no contact
                    tool_half = op / 2 + JAW_HALF_THICKNESS_M
                    reach = obj.footprint_radius + tool_half
                    if d < reach - 1e-6:
                        direction = obj.pos[:2] - gp[:2]
                        if np.linalg.norm(direction) < 1e-6:
                            prev_gp = self._prev_grasp_points[arm]
                            direction = gp[:2] - prev_gp[:2] if prev_gp is not None else np.zeros(2)
                        if np.linalg.norm(direction) < 1e-6:
                            continue
                        direction = direction / np.linalg.norm(direction)
                        obj.pos[:2] = gp[:2] + direction * reach
                        if (arm, obj.name) not in self._contacts:
                            self._contacts.add((arm, obj.name))
                            self.events.append(f"{obj.name} pushed by {arm}")
                    else:
                        self._contacts.discard((arm, obj.name))
            self._prev_openings = openings
            self._prev_grasp_points = {arm: self.grasp_points[arm].copy() for arm in ARMS}

    def _update_pour(self, can: SimCan) -> None:
        """Powder leaves the can once it tilts past POUR_START; it lands in the bowl if the opening is over it."""
        frac = float(np.clip((can.tilt_rad - can.POUR_START_RAD) / (can.POUR_FULL_RAD - can.POUR_START_RAD), 0.0, 1.0))
        new_out = frac - (can.poured_in_bowl + can.spilled)
        if new_out <= 1e-6:
            return
        bowl = self.objects.get(can.target)
        opening = can.top_center()
        in_bowl = False
        if bowl is not None:
            over = np.linalg.norm(opening[:2] - bowl.pos[:2]) <= bowl.footprint_radius + 0.02
            above = opening[2] >= bowl.pos[2] + bowl.height / 2 - 0.02
            in_bowl = bool(over and above)
        if in_bowl:
            if can.poured_in_bowl == 0.0:
                self.events.append(f"{can.name} started pouring into {can.target}")
            can.poured_in_bowl += new_out
        else:
            if can.spilled == 0.0:
                self.events.append(f"{can.name} spilled outside the bowl")
            can.spilled += new_out

    def _support_height(self, obj: SimObject) -> float:
        """Top of whatever lies under the object's footprint (table or another object)."""
        z = self.table_z
        for other in self.objects.values():
            if other is obj or other.held_by is not None:
                continue
            if np.linalg.norm(other.pos[:2] - obj.pos[:2]) < other.footprint_radius + obj.footprint_radius * 0.5:
                z = max(z, other.pos[2] + other.height / 2)
        return z

    def _settle(self, obj: SimObject) -> None:
        """Drop onto the table, or onto another object whose footprint it overlaps."""
        obj.pos[2] = self._support_height(obj) + obj.height / 2

    def _update_lid(self, case: SimCase, openings: Dict[str, float]) -> None:
        """Lid pinch / swing / release by the arm that is not holding the case."""
        if case.lid_holder is not None:
            arm = case.lid_holder
            if openings[arm] > case.H + RELEASE_MARGIN_M:
                case.lid_holder = None
                case._lid_v0 = None
                if case.lid_angle < case.OPEN_LATCH_RAD:
                    case.lid_angle = 0.0
                    self.events.append(f"{case.name} lid released and snapped shut")
                else:
                    self.events.append(f"{case.name} lid released, stays open ({np.degrees(case.lid_angle):.0f} deg)")
                return
            # lid angle follows the pinching grasp point swinging about the hinge (x-z plane when hanging)
            hinge = case.hinge()
            v = self.grasp_points[arm] - hinge
            v0 = case._lid_v0
            if case.hanging:
                ang = np.arctan2(v[2], v[0]) - np.arctan2(v0[2], v0[0])   # free edge swinging up-and-back opens
            else:
                ang = np.arctan2(v[2], -v[0]) - np.arctan2(v0[2], -v0[0])
            ang = float((ang + np.pi) % (2 * np.pi) - np.pi)
            case.lid_angle = float(np.clip(ang, 0.0, case.MAX_OPEN_RAD))
            return
        if case.held_by is None or not case.hanging:
            self._lid_straddled = {k for k in self._lid_straddled if k[1] != case.name}
            return                                                   # lid interaction only on the held, upright case
        lid_c = case.lid_center()
        for arm in ARMS:
            if arm == case.held_by:
                continue
            key = (arm, case.name)
            gp = self.grasp_points[arm]
            jaw = self.grasp_rots[arm][:, 1]
            in_band = lid_c[2] - case.LID / 2 - GRASP_Z_TOLERANCE_M <= gp[2] <= lid_c[2] + case.LID / 2 + GRASP_Z_TOLERANCE_M
            if abs(jaw[0]) < 0.7 or not in_band:                     # jaws must close across the lid's depth (x)
                self._lid_straddled.discard(key)
                continue
            op, prev = openings[arm], self._prev_openings[arm]
            dx, dy = abs(gp[0] - lid_c[0]), abs(gp[1] - lid_c[1])
            if dy <= case.W / 2 and dx <= max(0.0, op / 2 - case.H / 2) + GRASP_XY_TOLERANCE_M:
                self._lid_straddled.add(key)                         # lid between the open jaws
            elif dx > GRIPPER_MAX_OPENING_M / 2 + case.H or dy > case.W / 2 + 0.02:
                self._lid_straddled.discard(key)
            if key in self._lid_straddled and prev >= case.H > op and self.held_object(arm) is None:
                case.lid_holder = arm
                case._lid_v0 = gp - case.hinge()
                self._lid_straddled.discard(key)
                self.events.append(f"{case.name} lid pinched by {arm}")

    def status(self) -> Dict[str, dict]:
        return {name: o.status() for name, o in self.objects.items()}


class SimCameraSource:
    """Schematic renderings: a top view of the left-frame workspace and one 'wrist' view per arm."""

    W, H = 640, 480

    def __init__(self, world: SimWorld, names: Optional[List[str]] = None, jpeg_quality: int = 85):
        self.world = world
        self.names = names or ["top_cam", "left_cam", "right_cam"]
        self.quality = jpeg_quality

    # left-frame x in [0, 0.6] -> image rows (x forward = up); y in [-0.95, 0.35] -> cols (y left = image left)
    def _top_px(self, p: np.ndarray):
        col = int((0.35 - p[1]) / 1.3 * self.W)
        row = int(self.H - p[0] / 0.6 * self.H)
        return col, row

    def _draw_object(self, img, obj: SimObject, center, px_per_m: float) -> None:
        c, r = center
        if obj.shape == "cylinder":
            rad = max(3, int(obj.size[0] * px_per_m))
            cv2.circle(img, (c, r), rad, obj.color, -1)
        else:
            sx, sy = max(4, int(obj.size[0] * px_per_m)), max(4, int(obj.size[1] * px_per_m))
            cv2.rectangle(img, (c - sy // 2, r - sx // 2), (c + sy // 2, r + sx // 2), obj.color, -1)

    def _render_top(self) -> np.ndarray:
        img = np.full((self.H, self.W, 3), (235, 225, 210), np.uint8)
        cv2.putText(img, "top_cam (sim schematic, left-arm frame)", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (60, 60, 60), 1)
        px_per_m = self.W / 1.3
        for arm in ARMS:
            base = self.world.arm_offset(arm)
            c, r = self._top_px(base)
            cv2.rectangle(img, (c - 12, r - 6), (c + 12, r + 6), (90, 90, 90), -1)
            cv2.putText(img, f"{arm} base", (c - 30, r + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (60, 60, 60), 1)
        for obj in self.world.objects.values():
            c, r = self._top_px(obj.pos)
            self._draw_object(img, obj, (c, r), px_per_m)
            s = int(obj.footprint_radius * px_per_m)
            cv2.putText(img, f"{obj.name} z={obj.pos[2]:.2f}", (c + s + 2, r), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (30, 30, 30), 1)
        for arm in ARMS:
            gp = self.world.grasp_points[arm]
            c, r = self._top_px(gp)
            radius = int(6 + 10 * self.world.grippers[arm])
            cv2.circle(img, (c, r), radius, (0, 0, 0), 2)
            cv2.putText(img, f"{arm} grasp z={gp[2]:.3f} g={self.world.grippers[arm]:.2f}", (c + radius + 4, r - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
        return img

    def _render_wrist(self, arm: str) -> np.ndarray:
        img = np.full((self.H, self.W, 3), (200, 205, 215), np.uint8)
        gp = self.world.grasp_points[arm]
        g = self.world.grippers[arm]
        height = max(gp[2] - self.world.table_z, 0.02)
        scale = 0.12 / height * 600  # px per meter, grows as the gripper approaches the table
        cx, cy = self.W // 2, self.H // 2
        cv2.putText(img, f"{arm}_cam (sim schematic, looking down)", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (40, 40, 40), 1)
        for obj in self.world.objects.values():
            d = obj.pos - gp
            col = int(cx - d[1] * scale)  # +y (left) -> image left
            row = int(cy - d[0] * scale)  # +x (forward) -> image up
            self._draw_object(img, obj, (col, row), scale)
            s = int(obj.footprint_radius * scale)
            cv2.putText(img, obj.name, (col - s, row - s - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (20, 20, 20), 1)
        jaw = int(GRIPPER_MAX_OPENING_M * g * scale / 2)
        cv2.rectangle(img, (cx - jaw - 8, cy - 60), (cx - jaw, cy + 60), (30, 30, 30), -1)
        cv2.rectangle(img, (cx + jaw, cy - 60), (cx + jaw + 8, cy + 60), (30, 30, 30), -1)
        cv2.drawMarker(img, (cx, cy), (0, 0, 255), cv2.MARKER_CROSS, 20, 1)
        cv2.putText(img, f"height above table {height:.3f} m  gripper {g:.2f}", (10, self.H - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (40, 40, 40), 1)
        return img

    def read_jpeg_frames(self) -> Dict[str, bytes]:
        from astra_yam.cameras import encode_jpeg

        out = {}
        for name in self.names:
            if name.startswith("left"):
                img = self._render_wrist("left")
            elif name.startswith("right"):
                img = self._render_wrist("right")
            else:
                img = self._render_top()
            out[name] = encode_jpeg(img, self.quality)
        return out

    def close(self) -> None:
        pass


# ---------------------------------------------------------------------------
# ZMQ server speaking the gello robot_node protocol
# ---------------------------------------------------------------------------
def serve_sim_zmq(robot: SimYamRobot, host: str = "127.0.0.1", port: int = 6001, stop_event=None, verbose=True):
    import zmq

    ctx = zmq.Context()
    sock = ctx.socket(zmq.REP)
    sock.bind(f"tcp://{host}:{port}")
    sock.setsockopt(zmq.RCVTIMEO, 500)
    if verbose:
        print(f"[sim] SimYamRobot serving on tcp://{host}:{port} (gello ZMQ protocol)")
    try:
        while stop_event is None or not stop_event.is_set():
            try:
                req = pickle.loads(sock.recv())
            except zmq.Again:
                continue
            method = req.get("method")
            args = req.get("args", {}) or {}
            if method == "num_dofs":
                result = robot.num_dofs()
            elif method == "get_joint_state":
                result = robot.get_joint_state()
            elif method == "command_joint_state":
                result = robot.command_joint_positions(args["joint_state"])
            elif method == "get_observations":
                result = robot.get_observations()
            else:
                result = {"error": f"unsupported method {method}"}
            sock.send(pickle.dumps(result))
    finally:
        sock.close(0)
        ctx.term()
        if verbose:
            print("[sim] server stopped")


def start_sim_zmq_thread(robot: SimYamRobot, host="127.0.0.1", port=6001):
    stop = threading.Event()
    t = threading.Thread(target=serve_sim_zmq, args=(robot, host, port, stop, False), daemon=True)
    t.start()
    time.sleep(0.2)
    return t, stop
