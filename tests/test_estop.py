"""Emergency stop: the viser button halts a motion in flight and ends the session."""
import threading

import numpy as np
import pytest

from astra_yam.config import ARMS, PipelineConfig
from astra_yam.embodiment import ARM_GRIPPER_INDEX, ARM_SLICES
from astra_yam.gateway import SafetyGateway
from astra_yam.kinematics import ArmKinematics
from astra_yam.sim import SimYamRobot


@pytest.fixture
def gw():
    cfg = PipelineConfig()
    kin = ArmKinematics(limit_margin=cfg.motion.joint_limit_margin_rad)
    q = np.zeros(14)
    q[0:6] = q[7:13] = cfg.robot.home_joints_left
    q[6] = q[13] = 1.0
    robot = SimYamRobot(initial_q=q)
    start_rot = {a: kin.fk(q[ARM_SLICES[a]])[1] for a in ARMS}
    return cfg, robot, SafetyGateway(cfg, kin, robot, start_rot, realtime=False)


def test_estop_halts_the_motion_and_holds_position(gw):
    cfg, robot, gateway = gw
    gateway.estop = threading.Event()
    plan = gateway.plan({"left_x": 0.45, "left_gripper": 0.2})
    assert plan.steps > 20                                    # long enough to be interrupted part way

    commanded = []
    original = robot.command_joint_positions

    def recording_command(q):                                 # press the button a few commands in
        commanded.append(np.asarray(q, dtype=float).copy())
        if len(commanded) == 5:
            gateway.estop.set()
        return original(q)

    robot.command_joint_positions = recording_command
    res = gateway.execute(plan)
    assert not res.ok and res.status == "estop" and "hold position" in res.reason
    assert 0 <= res.steps_executed < plan.steps                # stopped part way, not at the end
    _, _, eef = gateway.read_state()
    assert abs(eef["left_x"] - 0.45) > 0.01                    # never reached the target

    # the last command holds the measured arm pose, with the gripper kept where it was commanded, so an
    # object already squeezed is not released by the stop
    hold, previous = commanded[-1], commanded[-2]
    arm_joints = np.r_[0:6, 7:13]
    assert np.allclose(hold[arm_joints], robot.get_joint_positions()[arm_joints], atol=1e-9)
    for arm in ARMS:
        idx = ARM_GRIPPER_INDEX[arm]
        assert hold[idx] == pytest.approx(previous[idx], abs=1e-9)
        assert gateway.gripper_cmd[arm] == pytest.approx(previous[idx], abs=1e-9)


def test_estop_payload_reaches_the_model_and_nothing_moves(gw):
    cfg, robot, gateway = gw
    gateway.estop = threading.Event()
    gateway.estop.set()                                        # already pressed: nothing may move
    before = robot.get_joint_positions().copy()
    payload, plan, res = gateway.move_to({"left_x": 0.40}, 10 ** 6)
    assert payload["ok"] is False and payload["status"] == "estop" and "emergency stop" in payload["reason"]
    assert payload["steps"] == 0
    arm_joints = np.r_[0:6, 7:13]
    assert np.allclose(robot.get_joint_positions()[arm_joints], before[arm_joints], atol=1e-9)


def test_no_estop_event_means_no_checks(gw):
    """The gateway is usable without an e-stop wired in (tests, the sim server, `check`)."""
    cfg, robot, gateway = gw
    assert gateway.estop is None
    assert gateway.move_to({"left_x": 0.32}, 10 ** 6)[0]["ok"]


def test_session_ends_on_estop(tmp_path):
    """Pressing the button between turns ends the trial instead of spending the rest of the budget."""
    from test_session_sim import _make

    script = [{"name": "move_to", "arguments": {"targets": {"left_z": 0.25}, "note": "n"}}] * 5
    cfg, runner, world, astra = _make(tmp_path, script=script)

    class Hooks:
        def on_astra_response(self, resp):
            runner.request_estop()          # as if the operator pressed EMERGENCY STOP during this turn

    runner.hooks = Hooks()
    outcome = runner.run("stop me")
    assert outcome.status == "estop" and "emergency stop" in outcome.reason
    assert outcome.llm_calls == 1           # the remaining four scripted calls never happen
