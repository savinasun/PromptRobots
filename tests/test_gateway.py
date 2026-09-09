import math
from dataclasses import replace

import numpy as np
import pytest

from astra_yam.config import ARMS, PipelineConfig, reference_bounds
from astra_yam.embodiment import ARM_SLICES
from astra_yam.gateway import GatewayRejection, SafetyGateway
from astra_yam.kinematics import ArmKinematics
from astra_yam.sim import SimYamRobot


def _bounds(**overrides):
    """Reference-trial bounds (narrow, tilt pinned, floor at z=0.03) - the default bounds are the whole envelope,
    where nothing is out of bounds and the tool-floor guard is inactive."""
    return replace(reference_bounds(), **overrides)


def _home_q(cfg):
    q = np.zeros(14)
    q[0:6] = cfg.robot.home_joints_left
    q[7:13] = cfg.robot.home_joints_right
    q[6] = q[13] = cfg.robot.home_gripper
    return q


@pytest.fixture
def setup():
    cfg = PipelineConfig()
    cfg.bounds = _bounds()
    kin = ArmKinematics(limit_margin=cfg.motion.joint_limit_margin_rad)
    robot = SimYamRobot(initial_q=_home_q(cfg))
    q0 = robot.get_joint_positions()
    start_rot = {arm: kin.fk(q0[ARM_SLICES[arm]])[1] for arm in ARMS}
    gw = SafetyGateway(cfg, kin, robot, start_rot, realtime=False)
    return cfg, kin, robot, gw


def test_reference_move_has_reference_step_count(setup):
    cfg, kin, robot, gw = setup
    plan = gw.plan({"left_x": 0.327, "left_z": 0.14})
    assert 45 <= plan.cartesian_steps <= 60           # reference transcript: 52 steps at 10 Hz
    assert plan.steps == plan.cartesian_steps and not plan.paced
    assert plan.q_path.shape == (plan.steps, 14)
    assert np.allclose(plan.q_path[:, 7:14], plan.start_q[7:14])   # right arm untouched


def test_execute_reaches_target(setup):
    cfg, kin, robot, gw = setup
    payload, plan, res = gw.move_to({"left_x": 0.327, "left_z": 0.14}, 3000)
    assert payload["ok"] and payload["status"] == "completed" and payload["cadence_hz"] == 10.0
    assert payload["steps"] == plan.steps
    q, poses, eef = gw.read_state()
    assert abs(eef["left_x"] - 0.327) < 0.003 and abs(eef["left_z"] - 0.14) < 0.003
    assert abs(eef["left_yaw"]) < 0.02 and abs(eef["left_pitch"]) < 0.02 and abs(eef["left_roll"]) < 0.02
    assert abs(eef["left_gripper"] - 1.0) < 1e-6
    assert gw.waypoints_executed == plan.steps


def test_out_of_bounds_and_pinned_are_rejected_not_clamped(setup):
    cfg, kin, robot, gw = setup
    for bad in ({"left_x": 0.60}, {"left_z": 0.01}, {"right_y": -0.3}, {"left_gripper": 1.2}):
        with pytest.raises(GatewayRejection, match="outside its bounds"):
            gw.validate_targets(bad)
    with pytest.raises(GatewayRejection, match="pinned"):
        gw.validate_targets({"left_pitch": 0.1})
    gw.validate_targets({"left_pitch": 0.0, "left_roll": 0})  # equal to the pinned value is fine
    with pytest.raises(GatewayRejection, match="unknown dimension"):
        gw.validate_targets({"left_q": 0.1})
    with pytest.raises(GatewayRejection, match="must be a number"):
        gw.validate_targets({"left_x": "0.3"})
    with pytest.raises(GatewayRejection, match="finite"):
        gw.validate_targets({"left_x": float("nan")})
    with pytest.raises(GatewayRejection):
        gw.validate_targets({})
    assert robot.command_log == []  # nothing moved


def test_unreachable_target_rejected_before_motion(setup):
    cfg, kin, robot, gw = setup
    cfg.bounds = _bounds(x=(0.15, 1.0))
    payload, plan, res = gw.move_to({"left_x": 0.95}, 3000)
    assert not payload["ok"] and payload["status"] == "rejected" and "unreachable" in payload["reason"]
    assert robot.command_log == []


def test_waypoint_budget_enforced(setup):
    cfg, kin, robot, gw = setup
    payload, plan, res = gw.move_to({"left_x": 0.327, "left_z": 0.14}, 5)
    assert not payload["ok"] and "waypoints" in payload["reason"]
    assert robot.command_log == []


def test_yaw_rotation_keeps_position(setup):
    cfg, kin, robot, gw = setup
    _, _, before = gw.read_state()
    payload, plan, res = gw.move_to({"left_yaw": 0.6}, 3000)
    assert payload["ok"]
    assert plan.cartesian_steps == math.ceil(0.6 / (cfg.motion.yaw_speed_rps / cfg.motion.cadence_hz))
    _, _, after = gw.read_state()
    assert abs(after["left_yaw"] - 0.6) < 0.02
    for d in ("left_x", "left_y", "left_z"):
        assert abs(after[d] - before[d]) < 0.003


def test_gripper_only_move(setup):
    cfg, kin, robot, gw = setup
    payload, plan, res = gw.move_to({"left_gripper": 0.2, "right_gripper": 0.5}, 3000)
    assert payload["ok"] and plan.cartesian_steps == 16     # 0.8 / (0.5 per s / 10 Hz)
    _, _, eef = gw.read_state()
    assert abs(eef["left_gripper"] - 0.2) < 1e-6 and abs(eef["right_gripper"] - 0.5) < 1e-6


def test_pacing_subdivides_large_joint_steps(setup):
    cfg, kin, robot, gw = setup
    cfg.motion.linear_speed_mps = 0.5   # 5 cm per waypoint -> joints move more than max_joint_step_rad
    plan = gw.plan({"left_x": 0.40, "left_z": 0.10})
    assert plan.paced and plan.steps > plan.cartesian_steps
    deltas = np.abs(np.diff(np.vstack([plan.start_q, plan.q_path]), axis=0))[:, np.r_[0:6, 7:13]]
    assert deltas.max() <= cfg.motion.max_joint_step_rad + 1e-9


def test_tracking_abort_holds_position(setup):
    cfg, kin, robot, gw = setup

    class StuckRobot(SimYamRobot):
        def get_joint_positions(self):
            q = super().get_joint_positions()
            q[1] += 0.5   # pretend joint 2 lags far behind the command
            return q

    stuck = StuckRobot(initial_q=robot.get_joint_positions())
    gw.robot = stuck
    payload, plan, res = gw.move_to({"left_x": 0.33}, 3000)
    assert not payload["ok"] and payload["status"] == "aborted" and "tracking error" in payload["reason"]
    assert payload["steps"] <= cfg.motion.tracking_check_every


def test_unnamed_gripper_holds_last_command_not_stalled_measurement(setup):
    cfg, kin, robot, gw = setup

    class StallingGripper(SimYamRobot):
        """Measured gripper never closes below 0.7 (an object between the jaws)."""
        def get_joint_positions(self):
            q = super().get_joint_positions()
            q[6] = max(q[6], 0.7)
            return q

    stalled = StallingGripper(initial_q=robot.get_joint_positions())
    gw.robot = stalled
    gw.gripper_cmd = {arm: 1.0 for arm in ARMS}
    assert gw.move_to({"left_gripper": 0.1}, 10 ** 6)[0]["ok"]
    assert stalled.command_log[-1][6] == pytest.approx(0.1)
    assert gw.move_to({"left_z": 0.25}, 10 ** 6)[0]["ok"]          # gripper not named
    assert stalled.command_log[-1][6] == pytest.approx(0.1)        # still commanded closed
    _, _, eef = gw.read_state()
    assert eef["left_gripper"] == pytest.approx(0.7)               # but reported as measured


def test_no_home_start_keeps_a_held_object_squeezed(setup):
    cfg, kin, robot, gw = setup
    q = robot.get_joint_positions()
    q[6] = 0.55                       # left gripper stalled on an object at start
    robot.command_joint_positions(q)
    gw2 = SafetyGateway(cfg, kin, robot, gw.start_rot, realtime=False)
    assert gw2.gripper_cmd["left"] == pytest.approx(0.40) and gw2.gripper_cmd["right"] == 1.0
    assert gw2.move_to({"left_z": 0.25}, 10 ** 6)[0]["ok"]
    assert robot.command_log[-1][6] == pytest.approx(0.40)          # carried with the squeeze, not relaxed to 0.55


def test_released_tilt_is_actuated_and_floor_guarded(setup):
    cfg, kin, robot, gw = setup
    cfg.bounds = _bounds(pitch=(-0.7, 0.7), roll=(-0.7, 0.7))
    payload, plan, res = gw.move_to({"left_pitch": 0.3, "left_roll": -0.2}, 10 ** 6)
    assert payload["ok"], payload
    _, _, eef = gw.read_state()
    assert abs(eef["left_pitch"] - 0.3) < 0.02 and abs(eef["left_roll"] + 0.2) < 0.02
    # a large roll at the floor height would put a jaw tip below the floor -> rejected
    assert gw.move_to({"left_z": 0.03, "left_pitch": 0.0, "left_roll": 0.0}, 10 ** 6)[0]["ok"]
    payload, plan, res = gw.move_to({"left_roll": 0.7}, 10 ** 6)
    assert not payload["ok"] and "below the workspace floor" in payload["reason"], payload
    # pinned by default: non-zero pitch is still rejected with the reference bounds
    cfg.bounds = _bounds()
    with pytest.raises(GatewayRejection, match="pinned"):
        gw.validate_targets({"left_pitch": 0.1})
