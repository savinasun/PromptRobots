import numpy as np
import pytest

from astra_yam.collision import clearance, segment_distance
from astra_yam.config import ARMS, Bounds, PipelineConfig, REFERENCE_HOME_JOINTS
from astra_yam.embodiment import ARM_SLICES
from astra_yam.gateway import SafetyGateway
from astra_yam.kinematics import ArmKinematics
from astra_yam.sim import SimYamRobot


def test_segment_distance_basics():
    assert segment_distance(np.zeros(3), np.array([1, 0, 0.]), np.array([0, 1, 0.]), np.array([1, 1, 0.])) == pytest.approx(1.0)
    assert segment_distance(np.zeros(3), np.array([1, 0, 0.]), np.array([0.5, 0, 1.]), np.array([0.5, 0, 2.])) == pytest.approx(1.0)
    assert segment_distance(np.zeros(3), np.zeros(3), np.array([3, 4, 0.]), np.array([3, 4, 0.])) == pytest.approx(5.0)
    assert segment_distance(np.zeros(3), np.array([1, 0, 0.]), np.array([0.5, -1, 0.]), np.array([0.5, 1, 0.])) == pytest.approx(0.0)


def _home():
    q = np.zeros(14)
    q[0:6] = q[7:13] = REFERENCE_HOME_JOINTS
    q[6] = q[13] = 1.0
    return q


def _gw(clearance_on=True, detour=True):
    kin = ArmKinematics(limit_margin=0.01)
    cfg = PipelineConfig()
    cfg.bounds = Bounds(y=(-0.40, 0.40))
    cfg.motion.detour_enabled = detour
    if not clearance_on:
        cfg.motion.arm_clearance_m = 0.0
    robot = SimYamRobot(initial_q=_home())
    start_rot = {arm: kin.fk(_home()[ARM_SLICES[arm]])[1] for arm in ARMS}
    return cfg, kin, robot, SafetyGateway(cfg, kin, robot, start_rot, realtime=False)


LEFT_HOLD = {"left_x": 0.35, "left_y": -0.31, "left_z": 0.12}


def test_real_collision_configuration_is_detected():
    cfg, kin, robot, gw = _gw()
    assert clearance(kin, _home()).clearance_m > 0.3                 # 61 cm apart at home
    # from the collision run: left holds the case at (0.358,-0.285,0.165); right descends from above
    assert gw.move_to({"left_x": 0.358, "left_y": -0.285, "left_z": 0.165}, 10 ** 6)[0]["ok"]
    assert gw.move_to({"right_x": 0.361, "right_y": 0.278, "right_z": 0.35}, 10 ** 6)[0]["ok"]
    payload, plan, res = gw.move_to({"right_z": 0.19}, 10 ** 6)     # the descent that collided
    assert not payload["ok"] and "arms would come within" in payload["reason"], payload
    assert clearance(kin, robot.get_joint_positions()).deficit_m <= 0   # nothing moved, still clear
    assert gw.move_to({"right_y": 0.10}, 10 ** 6)[0]["ok"]           # moving away toward its own side is fine


def test_detour_to_the_side_rescues_a_descend_and_yaw():
    cfg, kin, robot, gw = _gw()
    assert gw.move_to(LEFT_HOLD, 10 ** 6)[0]["ok"]
    assert gw.move_to({"right_x": 0.38, "right_y": 0.27, "right_z": 0.30}, 10 ** 6)[0]["ok"]
    payload, plan, res = gw.move_to({"right_z": 0.12, "right_yaw": 1.57}, 10 ** 6)
    assert payload["ok"], payload
    assert plan.detour is not None and "detour" in payload
    _, _, eef = gw.read_state()
    assert abs(eef["right_z"] - 0.12) < 0.005 and abs(eef["right_yaw"] - 1.57) < 0.02
    assert clearance(kin, robot.get_joint_positions()).deficit_m <= 1e-6


def test_without_detours_the_same_move_is_rejected():
    cfg, kin, robot, gw = _gw(detour=False)
    assert gw.move_to(LEFT_HOLD, 10 ** 6)[0]["ok"]
    assert gw.move_to({"right_x": 0.38, "right_y": 0.27, "right_z": 0.30}, 10 ** 6)[0]["ok"]
    payload, plan, res = gw.move_to({"right_z": 0.12, "right_yaw": 1.57}, 10 ** 6)
    assert not payload["ok"] and "arms would come within" in payload["reason"]


def test_yawed_right_tool_can_reach_the_held_object():
    """Cooperative pose from the scan: right fingertips 5 cm from the left grasp point, tool yawed +1.57."""
    cfg, kin, robot, gw = _gw()
    assert gw.move_to(LEFT_HOLD, 10 ** 6)[0]["ok"]
    payload, plan, res = gw.move_to({"right_x": 0.38, "right_y": 0.25, "right_z": 0.14, "right_yaw": 1.57}, 10 ** 6)
    assert payload["ok"], payload
    cr = clearance(kin, robot.get_joint_positions())
    assert cr.deficit_m <= 1e-6, cr
    # parallel tools that close do collide: same fingertips with yaw 0 would overlap the housings
    q = robot.get_joint_positions().copy()
    from astra_yam.kinematics import rotation_from_ypr
    r = kin.ik(np.array([0.38, 0.25, 0.14]), rotation_from_ypr(gw.start_rot["right"], 0.0), q[7:13], max_iters=300)
    q[7:13] = r.q
    assert clearance(kin, q).deficit_m > 0


def test_clearance_check_can_be_disabled():
    cfg, kin, robot, gw = _gw(clearance_on=False)
    assert gw.move_to({"left_x": 0.35, "left_y": -0.30, "left_z": 0.16}, 10 ** 6)[0]["ok"]
    assert gw.move_to({"right_x": 0.35, "right_y": 0.31, "right_z": 0.19}, 10 ** 6)[0]["ok"]
