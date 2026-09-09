"""Safety gateway between Astra's `move_to` calls and the robot.

A `move_to` packet is: validated (names, numbers, bounds, pinned axes) -> planned as a straight Cartesian
path at `cadence_hz` -> converted to joint waypoints by IK (seeded from the previous waypoint) -> checked for
reachability, joint limits and configuration jumps -> paced so no joint moves more than
`max_joint_step_rad` per waypoint -> checked against the waypoint budget -> streamed to the robot at
`control_hz` while tracking error is monitored. Unsafe packets are rejected before any motion; nothing is
clamped silently.
"""
from __future__ import annotations

import math
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, Optional, Tuple

import numpy as np

from astra_yam.collision import clearance as arm_clearance
from astra_yam.config import ARMS, ARM_DIMS, DIM_NAMES, NUM_DOFS, PipelineConfig
from astra_yam.embodiment import ARM_GRIPPER_INDEX, ARM_SLICES, ArmPose, arm_poses, eef_state_dict
from astra_yam.kinematics import ArmKinematics, rotation_from_ypr
from astra_yam.robot_interface import RobotBackend


class GatewayRejection(Exception):
    """A packet failed the safety checks. The message is returned to the model verbatim."""


@dataclass
class ArmGoal:
    pos: np.ndarray
    yaw: float
    pitch: float
    roll: float
    gripper: float


@dataclass
class MotionPlan:
    q_path: np.ndarray                      # (steps, 14) joint waypoints, start pose excluded
    cartesian_steps: int
    steps: int
    paced: bool
    start_q: np.ndarray
    goals: Dict[str, ArmGoal]
    resolved_targets: Dict[str, float] = field(default_factory=dict)
    min_clearance_m: Optional[float] = None
    detour: Optional[str] = None            # description of the path deviation taken to keep the arms apart


class ClearanceRejection(GatewayRejection):
    """The straight path would bring the arms too close (a detour may still be possible)."""


@dataclass
class ExecutionResult:
    ok: bool
    status: str                             # completed | aborted | timeout | estop
    steps_executed: int
    reason: Optional[str] = None
    max_tracking_err_rad: float = 0.0


def _sleep_until(t_target: float) -> None:
    while True:
        dt = t_target - time.perf_counter()
        if dt <= 0:
            return
        time.sleep(min(dt, 0.002))


def move_joint_space(
    robot: RobotBackend,
    q_target: np.ndarray,
    seconds: float,
    control_hz: float,
    realtime: bool = True,
    progress: Optional[Callable[[int, int], None]] = None,
) -> np.ndarray:
    """Linearly interpolate all 14 joints from the current pose to `q_target` (used for homing)."""
    q0 = robot.get_joint_positions()
    q_target = np.asarray(q_target, dtype=float)
    n = max(2, int(math.ceil(seconds * control_hz)))
    period = 1.0 / control_hz
    t_next = time.perf_counter()
    for i in range(1, n + 1):
        q = q0 + (q_target - q0) * (i / n)
        robot.command_joint_positions(q)
        if progress:
            progress(i, n)
        if realtime:
            t_next += period
            _sleep_until(t_next)
    return robot.get_joint_positions()


class SafetyGateway:
    def __init__(
        self,
        cfg: PipelineConfig,
        kin: ArmKinematics,
        robot: RobotBackend,
        start_rot: Dict[str, np.ndarray],
        realtime: bool = True,
    ):
        self.cfg = cfg
        self.kin = kin
        self.robot = robot
        self.start_rot = {arm: np.asarray(start_rot[arm], dtype=float) for arm in ARMS}
        self.realtime = realtime
        self.yaw_ref: Dict[str, float] = {arm: 0.0 for arm in ARMS}
        self.waypoints_executed = 0
        self.on_plan: Optional[Callable[["MotionPlan"], None]] = None   # called after checks, before motion
        # Emergency stop. Any thread (the viser button, a signal handler) may set it; `execute` checks it before
        # every joint command and stops the motion where it is, so the latency is one control tick.
        self.estop: Optional[threading.Event] = None
        # Last *commanded* gripper value per arm. A gripper holding an object stalls above its command
        # (e.g. measured 0.74 for a 7 cm cup, commanded 0.1); "unnamed dimensions hold their current value"
        # must therefore hold the command, not the measurement, or every later move would loosen the grip.
        q_now = self.robot.get_joint_positions()
        self.gripper_cmd: Dict[str, float] = {}
        for arm in ARMS:
            g = float(q_now[ARM_GRIPPER_INDEX[arm]])
            # A gripper measured well below fully open at start may be stalled on an object (trial continued with
            # --no-home). Commanding the stalled value would relax the grip, so command a bit tighter.
            self.gripper_cmd[arm] = 1.0 if g > 0.95 else max(0.0, g - cfg.motion.gripper_init_squeeze)

    # ------------------------------------------------------------------ state
    def read_state(self) -> Tuple[np.ndarray, Dict[str, ArmPose], Dict[str, float]]:
        q = self.robot.get_joint_positions()
        poses = arm_poses(q, self.kin, self.start_rot, self.yaw_ref)
        return q, poses, eef_state_dict(poses)

    # ------------------------------------------------------------- validation
    def validate_targets(self, targets) -> Dict[str, Dict[str, float]]:
        if not isinstance(targets, dict) or not targets:
            raise GatewayRejection("targets must be a non-empty object mapping dimension names to numbers")
        per_arm: Dict[str, Dict[str, float]] = {arm: {} for arm in ARMS}
        bounds = self.cfg.bounds
        for name, value in targets.items():
            if name not in DIM_NAMES:
                raise GatewayRejection(f"unknown dimension '{name}'; valid names: {', '.join(DIM_NAMES)}")
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise GatewayRejection(f"{name} must be a number, got {value!r}")
            v = float(value)
            if not math.isfinite(v):
                raise GatewayRejection(f"{name} must be finite, got {value!r}")
            arm, dim = name.split("_", 1)
            lo, hi = bounds.for_dim(dim)
            if bounds.is_pinned(dim):
                if abs(v - lo) > 1e-6:
                    raise GatewayRejection(f"{name} is pinned at {lo} on this rig and cannot be actuated (got {v})")
            elif v < lo or v > hi:
                raise GatewayRejection(f"{name}={v} is outside its bounds [{lo}, {hi}]; targets are not clamped")
            per_arm[arm][dim] = v
        return per_arm

    # --------------------------------------------------------------- planning
    def _steps_for(self, dist_m: float, dyaw: float, dtilt: float, dgrip: float) -> int:
        m = self.cfg.motion
        per_step_lin = m.linear_speed_mps / m.cadence_hz
        per_step_rot = m.yaw_speed_rps / m.cadence_hz
        per_step_grip = m.gripper_speed_per_s / m.cadence_hz
        return int(max(
            math.ceil(dist_m / per_step_lin - 1e-9),
            math.ceil(dyaw / per_step_rot - 1e-9),
            math.ceil(dtilt / per_step_rot - 1e-9),
            math.ceil(dgrip / per_step_grip - 1e-9),
            1,
        ))

    def plan(self, targets, q_start: Optional[np.ndarray] = None) -> MotionPlan:
        m = self.cfg.motion
        bounds = self.cfg.bounds
        per_arm = self.validate_targets(targets)
        q0 = np.asarray(q_start if q_start is not None else self.robot.get_joint_positions(), dtype=float)
        poses = arm_poses(q0, self.kin, self.start_rot, self.yaw_ref)

        goals: Dict[str, ArmGoal] = {}
        n_steps = 1
        resolved: Dict[str, float] = {}
        for arm in ARMS:
            p = poses[arm]
            t = per_arm[arm]
            goal_pos = np.array([t.get("x", p.pos[0]), t.get("y", p.pos[1]), t.get("z", p.pos[2])], dtype=float)
            goal_yaw = t.get("yaw", p.yaw)
            goal_pitch = bounds.for_dim("pitch")[0] if bounds.is_pinned("pitch") else t.get("pitch", p.pitch)
            goal_roll = bounds.for_dim("roll")[0] if bounds.is_pinned("roll") else t.get("roll", p.roll)
            grip_start = self.gripper_cmd[arm]
            goal_grip = t.get("gripper", grip_start)
            goals[arm] = ArmGoal(goal_pos, float(goal_yaw), float(goal_pitch), float(goal_roll), float(goal_grip))
            for dim, val in zip(ARM_DIMS, [*goal_pos, goal_yaw, goal_pitch, goal_roll, goal_grip]):
                resolved[f"{arm}_{dim}"] = float(val)
            dist = float(np.linalg.norm(goal_pos - p.pos))
            dtilt = max(abs(goal_pitch - p.pitch), abs(goal_roll - p.roll))
            n_steps = max(n_steps, self._steps_for(dist, abs(goal_yaw - p.yaw), dtilt, abs(goal_grip - grip_start)))

        # 1) endpoint reachability with a generous iteration budget
        for arm in ARMS:
            g = goals[arm]
            rot_goal = rotation_from_ypr(self.start_rot[arm], g.yaw, g.pitch, g.roll)
            res = self.kin.ik(g.pos, rot_goal, q0[ARM_SLICES[arm]], max_iters=300,
                              pos_tol=m.ik_pos_tol_m, ori_tol=m.ik_ori_tol_rad)
            if not res.converged:
                raise GatewayRejection(
                    f"target for the {arm} arm is unreachable: IK residual {res.pos_err_m * 1000:.1f} mm / "
                    f"{res.ori_err_rad:.3f} rad at x={g.pos[0]:.3f} y={g.pos[1]:.3f} z={g.pos[2]:.3f} yaw={g.yaw:.3f}"
                )

        # 2) straight-line Cartesian path -> joint waypoints (IK, limits, config flips, arm clearance)
        start_state = {arm: ArmGoal(poses[arm].pos.copy(), poses[arm].yaw, poses[arm].pitch, poses[arm].roll,
                                    self.gripper_cmd[arm]) for arm in ARMS}
        start_deficit = self._clearance_deficit(q0)
        detour = None
        try:
            path, min_clear = self._ik_segments(q0, [(start_state, goals)], start_deficit)
        except ClearanceRejection as first:
            if not m.detour_enabled:
                raise
            path = None
            for name, segments in self._detour_candidates(start_state, goals):
                try:
                    path, min_clear = self._ik_segments(q0, segments, start_deficit)
                    detour = name
                    break
                except GatewayRejection:
                    continue
            if path is None:
                raise GatewayRejection(str(first) + " (no over/side detour keeps the arms apart either)")
        n_steps = len(path)

        # 3) joint pacing: subdivide waypoints whose joint step is too large
        paced = []
        prev = q0.copy()
        arm_idx = np.r_[0:6, 7:13]
        for wp in path:
            delta = float(np.max(np.abs((wp - prev)[arm_idx])))
            k = int(math.ceil(delta / m.max_joint_step_rad)) if delta > m.max_joint_step_rad else 1
            for j in range(1, k + 1):
                paced.append(prev + (wp - prev) * (j / k))
            prev = wp
        q_path = np.asarray(paced)
        return MotionPlan(q_path=q_path, cartesian_steps=n_steps, steps=len(q_path), paced=len(q_path) > n_steps,
                          start_q=q0, goals=goals, resolved_targets=resolved, min_clearance_m=min_clear, detour=detour)

    # ------------------------------------------------------------ clearance
    def _clearance_deficit(self, q14: np.ndarray) -> Optional[float]:
        m = self.cfg.motion
        if m.arm_clearance_m <= 0:
            return None
        return arm_clearance(self.kin, q14, link_margin=m.arm_clearance_m, tip_margin=m.arm_tip_clearance_m).deficit_m

    def _ik_segments(self, q0: np.ndarray, segments, start_deficit: Optional[float]):
        """IK along consecutive straight Cartesian segments [(state_from, state_to), ...] for both arms.

        Returns (joint waypoints, min clearance). Raises ClearanceRejection / GatewayRejection.
        """
        m = self.cfg.motion
        path = []
        q_prev = q0.copy()
        min_clear = None
        total_steps = sum(self._segment_steps(a, b) for a, b in segments)
        done_steps = 0
        for seg_index, (frm, to) in enumerate(segments):
            n_steps = self._segment_steps(frm, to)
            for k in range(1, n_steps + 1):
                a = k / n_steps
                q_k = q_prev.copy()
                for arm in ARMS:
                    p, g = frm[arm], to[arm]
                    pos_k = p.pos + (g.pos - p.pos) * a
                    yaw_k = p.yaw + (g.yaw - p.yaw) * a
                    pitch_k = p.pitch + (g.pitch - p.pitch) * a
                    roll_k = p.roll + (g.roll - p.roll) * a
                    grip_k = p.gripper + (g.gripper - p.gripper) * a
                    rot_k = rotation_from_ypr(self.start_rot[arm], yaw_k, pitch_k, roll_k)
                    res = self.kin.ik(pos_k, rot_k, q_prev[ARM_SLICES[arm]], max_iters=60,
                                      pos_tol=m.ik_pos_tol_m, ori_tol=m.ik_ori_tol_rad)
                    if not res.converged:
                        raise GatewayRejection(
                            f"path for the {arm} arm leaves the reachable workspace at waypoint {done_steps + k}/{total_steps} "
                            f"(IK residual {res.pos_err_m * 1000:.1f} mm); choose a nearer intermediate target"
                        )
                    jump = float(np.max(np.abs(res.q - q_prev[ARM_SLICES[arm]])))
                    if jump > m.max_ik_joint_jump_rad:
                        raise GatewayRejection(
                            f"the {arm} arm would flip between joint configurations at waypoint {done_steps + k}/{total_steps} "
                            f"(joint jump {jump:.2f} rad); choose a nearer intermediate target"
                        )
                    if not self.kin.within_limits(res.q, tol=1e-6):
                        raise GatewayRejection(f"the {arm} arm would exceed a joint limit at waypoint {done_steps + k}/{total_steps}")
                    q_k[ARM_SLICES[arm]] = res.q
                    q_k[ARM_GRIPPER_INDEX[arm]] = min(1.0, max(0.0, grip_k))
                    dip = self._tool_dip_below_floor(pos_k, rot_k, grip_k)
                    if dip > 0.01:
                        raise GatewayRejection(
                            f"with that tilt the {arm} tool's jaw tips or housing would dip {dip * 100:.1f} cm below the "
                            f"workspace floor at waypoint {done_steps + k}/{total_steps}; use less pitch/roll or a higher z"
                        )
                if start_deficit is not None:
                    cr = arm_clearance(self.kin, q_k, link_margin=m.arm_clearance_m, tip_margin=m.arm_tip_clearance_m)
                    min_clear = cr.clearance_m if min_clear is None else min(min_clear, cr.clearance_m)
                    if cr.deficit_m > 0 and cr.deficit_m > start_deficit + 1e-6:
                        raise ClearanceRejection(
                            f"the two arms would come within {max(cr.clearance_m, 0.0) * 100:.1f} cm of each other at "
                            f"waypoint {done_steps + k}/{total_steps} (left {cr.left_part} vs right {cr.right_part}; "
                            f"{cr.margin_m * 100:.1f} cm required). Keep the tools apart, approach from another side, "
                            f"or move one arm away first"
                        )
                path.append(q_k)
                q_prev = q_k
            done_steps += n_steps
        return path, min_clear

    def _tool_dip_below_floor(self, grasp_pos: np.ndarray, rot: np.ndarray, gripper: float) -> float:
        """How far (m) the lowest tool point (jaw tips, housing underside) is below the z floor; <= 0 is fine."""
        floor = self.cfg.motion.tool_floor_z_m
        z_floor = self.cfg.bounds.for_dim("z")[0] if floor is None else float(floor)
        half = 0.095 * float(np.clip(gripper, 0.0, 1.0)) / 2
        tips = [grasp_pos + half * rot[:, 1], grasp_pos - half * rot[:, 1]]
        flange = grasp_pos - 0.1347 * rot[:, 2]
        lowest = min(tips[0][2], tips[1][2], flange[2] - 0.05)
        return float(z_floor - lowest)

    def _segment_steps(self, frm: Dict[str, ArmGoal], to: Dict[str, ArmGoal]) -> int:
        n = 1
        for arm in ARMS:
            p, g = frm[arm], to[arm]
            dist = float(np.linalg.norm(g.pos - p.pos))
            dtilt = max(abs(g.pitch - p.pitch), abs(g.roll - p.roll))
            n = max(n, self._steps_for(dist, abs(g.yaw - p.yaw), dtilt, abs(g.gripper - p.gripper)))
        return n

    def _detour_candidates(self, start: Dict[str, ArmGoal], goals: Dict[str, ArmGoal]):
        """Piecewise-linear alternatives for the moving arm(s): over the top, to the side, or both.

        The gripper and yaw stay at their start values until the final segment, so grasps/releases still happen
        at the goal.
        """
        m, b = self.cfg.motion, self.cfg.bounds
        moving = [arm for arm in ARMS if np.linalg.norm(goals[arm].pos - start[arm].pos) > 1e-4]
        if not moving:
            return []
        z_hi = min(b.for_dim("z")[1], max(max(start[a].pos[2], goals[a].pos[2]) for a in moving) + m.detour_rise_m)

        def hold(arm: str, pos: np.ndarray) -> ArmGoal:
            s = start[arm]
            return ArmGoal(np.asarray(pos, dtype=float), s.yaw, s.pitch, s.roll, s.gripper)

        def side_offset(arm: str) -> float:
            # away from the other arm: +y for the left arm (its left), -y for the right arm (its right)
            off = m.detour_side_m if arm == "left" else -m.detour_side_m
            lo, hi = b.for_dim("y")
            return float(np.clip(goals[arm].pos[1] + off, lo, hi) - goals[arm].pos[1])

        def states(fn) -> Dict[str, ArmGoal]:
            return {arm: (fn(arm) if arm in moving else start[arm]) for arm in ARMS}

        def yawed(arm: str, pos: np.ndarray) -> ArmGoal:
            s, g = start[arm], goals[arm]
            return ArmGoal(np.asarray(pos, dtype=float), g.yaw, g.pitch, g.roll, s.gripper)

        over1 = states(lambda a: hold(a, [start[a].pos[0], start[a].pos[1], z_hi]))
        over2 = states(lambda a: hold(a, [goals[a].pos[0], goals[a].pos[1], z_hi]))
        side1 = states(lambda a: hold(a, [goals[a].pos[0], goals[a].pos[1] + side_offset(a), max(start[a].pos[2], goals[a].pos[2])]))
        both1 = states(lambda a: hold(a, [start[a].pos[0], start[a].pos[1], z_hi]))
        both2 = states(lambda a: hold(a, [goals[a].pos[0], goals[a].pos[1] + side_offset(a), z_hi]))
        yaw0 = states(lambda a: yawed(a, start[a].pos))                       # rotate in place first
        yaw_over1 = states(lambda a: yawed(a, [start[a].pos[0], start[a].pos[1], z_hi]))
        yaw_over2 = states(lambda a: yawed(a, [goals[a].pos[0], goals[a].pos[1], z_hi]))
        candidates = [
            ("over the top", [(start, over1), (over1, over2), (over2, goals)]),
            ("to the side", [(start, side1), (side1, goals)]),
            ("over the top and to the side", [(start, both1), (both1, both2), (both2, goals)]),
        ]
        if any(abs(goals[a].yaw - start[a].yaw) > 1e-3 for a in moving):
            candidates += [
                ("yaw first, then straight", [(start, yaw0), (yaw0, goals)]),
                ("yaw first, then over the top", [(start, yaw0), (yaw0, yaw_over1), (yaw_over1, yaw_over2), (yaw_over2, goals)]),
            ]
        return candidates

    # -------------------------------------------------------------- execution
    def _emergency_hold(self, q_cmd: np.ndarray, executed: int, max_err: float) -> ExecutionResult:
        """Stop here: the arms hold their measured pose, the grippers keep their last command.

        Commanding the *measured* gripper would relax a grip that is stalled on an object (see `gripper_cmd`
        in `__init__`), so a held object stays held through an emergency stop.
        """
        q_hold = np.asarray(self.robot.get_joint_positions(), dtype=float).copy()
        for arm in ARMS:
            idx = ARM_GRIPPER_INDEX[arm]
            self.gripper_cmd[arm] = float(q_cmd[idx])
            q_hold[idx] = float(q_cmd[idx])
        self.robot.command_joint_positions(q_hold)
        return ExecutionResult(False, "estop", executed,
                               "emergency stop: the operator halted the motion; the arms hold position", max_err)

    def execute(self, plan: MotionPlan, deadline: Optional[float] = None) -> ExecutionResult:
        m = self.cfg.motion
        substeps = max(1, int(round(m.control_hz / m.cadence_hz)))
        period = 1.0 / m.cadence_hz / substeps
        arm_idx = np.r_[0:6, 7:13]
        q_prev = plan.start_q.copy()
        executed = 0
        max_err = 0.0
        t_next = time.perf_counter()
        n = len(plan.q_path)
        q_last = plan.start_q.copy()
        for i, wp in enumerate(plan.q_path):
            for s in range(1, substeps + 1):
                if self.estop is not None and self.estop.is_set():
                    return self._emergency_hold(q_last, executed, max_err)
                q_cmd = q_prev + (wp - q_prev) * (s / substeps)
                self.robot.command_joint_positions(q_cmd)
                q_last = q_cmd
                if self.realtime:
                    t_next += period
                    _sleep_until(t_next)
            q_prev = wp
            executed += 1
            self.waypoints_executed += 1
            last = i == n - 1
            if last or (i + 1) % max(1, m.tracking_check_every) == 0:
                q_meas = self.robot.get_joint_positions()
                err = float(np.max(np.abs((q_meas - wp)[arm_idx])))
                max_err = max(max_err, err)
                if err > m.tracking_abort_rad:
                    self.robot.command_joint_positions(q_meas)  # hold where we actually are
                    for arm in ARMS:
                        self.gripper_cmd[arm] = float(wp[ARM_GRIPPER_INDEX[arm]])
                    return ExecutionResult(False, "aborted", executed,
                                           f"tracking error {err:.3f} rad exceeds {m.tracking_abort_rad} rad; "
                                           f"motion stopped and position held", max_err)
            if deadline is not None and time.perf_counter() > deadline and not last:
                for arm in ARMS:
                    self.gripper_cmd[arm] = float(wp[ARM_GRIPPER_INDEX[arm]])
                return ExecutionResult(False, "timeout", executed, "trial time limit reached during motion", max_err)
        for arm in ARMS:
            self.yaw_ref[arm] = plan.goals[arm].yaw
            self.gripper_cmd[arm] = plan.goals[arm].gripper
        if self.realtime and m.settle_seconds > 0:
            time.sleep(m.settle_seconds)
        return ExecutionResult(True, "completed", executed, None, max_err)

    def move_to(self, targets, waypoints_remaining: int, deadline: Optional[float] = None
                ) -> Tuple[dict, Optional[MotionPlan], Optional[ExecutionResult]]:
        """Full gateway pass. Returns (payload for the model, plan, execution result)."""
        try:
            plan = self.plan(targets)
        except GatewayRejection as e:
            return {"ok": False, "status": "rejected", "reason": str(e)}, None, None
        if plan.steps > waypoints_remaining:
            return ({"ok": False, "status": "rejected",
                     "reason": f"this motion needs {plan.steps} waypoints but only {waypoints_remaining} remain in the session budget"},
                    plan, None)
        if self.on_plan is not None:
            try:
                self.on_plan(plan)
            except Exception as e:  # noqa: BLE001 - visualization must never block motion
                print(f"[gateway] on_plan hook failed: {e}")
        res = self.execute(plan, deadline)
        payload = {"ok": res.ok, "steps": res.steps_executed, "status": res.status, "cadence_hz": self.cfg.motion.cadence_hz}
        if plan.detour:
            payload["detour"] = (f"the straight path would have brought the arms too close; the motion went {plan.detour} "
                                 f"instead ({plan.cartesian_steps} waypoints)")
        if res.reason:
            payload["reason"] = res.reason
        return payload, plan, res

    def hold(self) -> np.ndarray:
        q = self.robot.get_joint_positions()
        self.robot.command_joint_positions(q)
        return q
