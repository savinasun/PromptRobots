import numpy as np
import pytest

from astra_yam.config import ARMS, Bounds, PipelineConfig, REFERENCE_HOME_JOINTS
from astra_yam.embodiment import ARM_SLICES
from astra_yam.gateway import SafetyGateway
from astra_yam.kinematics import ArmKinematics
from astra_yam.sim import SCENES, SimCan, SimWorld, SimYamRobot


def _setup(tilt=1.4):
    cfg = PipelineConfig()
    cfg.bounds = Bounds(pitch=(-tilt, tilt), roll=(-tilt, tilt))
    cfg.motion.linear_speed_mps = 0.05
    cfg.motion.yaw_speed_rps = 0.6
    kin = ArmKinematics(limit_margin=cfg.motion.joint_limit_margin_rad)
    world = SimWorld(kin, scene="chili")
    q0 = np.zeros(14)
    q0[0:6] = q0[7:13] = REFERENCE_HOME_JOINTS
    q0[6] = q0[13] = 1.0
    robot = SimYamRobot(initial_q=q0, world=world)
    world.update(q0)
    start_rot = {arm: kin.fk(q0[ARM_SLICES[arm]])[1] for arm in ARMS}
    return cfg, kin, world, robot, SafetyGateway(cfg, kin, robot, start_rot, realtime=False)


def test_chili_scene():
    assert "chili" in SCENES
    world = SimWorld(scene="chili")
    can = world.objects["chili powder can"]
    assert isinstance(can, SimCan) and can.graspable and can.content == 1.0 and can.tilt_rad == 0.0
    assert can.target == "orange bowl" and world.objects["orange bowl"].footprint_radius == 0.06


def test_pick_up_carry_and_pour_into_bowl():
    cfg, kin, world, robot, gw = _setup()
    can, bowl = world.objects["chili powder can"], world.objects["orange bowl"]
    x, y = float(can.pos[0]), float(can.pos[1])
    assert gw.move_to({"left_x": x, "left_y": y, "left_z": 0.16}, 10 ** 6)[0]["ok"]
    assert gw.move_to({"left_z": 0.06}, 10 ** 6)[0]["ok"]
    assert gw.move_to({"left_gripper": 0.2}, 10 ** 6)[0]["ok"]
    assert can.held_by == "left"
    assert gw.move_to({"left_z": 0.18}, 10 ** 6)[0]["ok"]
    assert can.tilt_rad < 0.01 and can.content == 1.0                     # carried upright
    # over the bowl, opening above the rim, then roll the tool: the can tilts and pours
    bx, by = float(bowl.pos[0]), float(bowl.pos[1])
    assert gw.move_to({"left_x": bx, "left_y": by + 0.02, "left_z": 0.17}, 10 ** 6)[0]["ok"]
    p = gw.move_to({"left_roll": -1.3}, 10 ** 6)[0]
    assert p["ok"], p
    assert np.degrees(can.tilt_rad) > 70, np.degrees(can.tilt_rad)
    assert can.poured_in_bowl > 0.3 and can.spilled == 0.0, can.status()
    assert "started pouring into orange bowl" in " ".join(world.events)
    # un-tilt: nothing comes back, nothing more leaves
    poured = can.poured_in_bowl
    assert gw.move_to({"left_roll": 0.0}, 10 ** 6)[0]["ok"]
    assert can.poured_in_bowl == pytest.approx(poured) and can.tilt_rad < 0.05


def test_pouring_away_from_the_bowl_spills():
    cfg, kin, world, robot, gw = _setup()
    can = world.objects["chili powder can"]
    x, y = float(can.pos[0]), float(can.pos[1])
    assert gw.move_to({"left_x": x, "left_y": y, "left_z": 0.16}, 10 ** 6)[0]["ok"]
    assert gw.move_to({"left_z": 0.06}, 10 ** 6)[0]["ok"]
    assert gw.move_to({"left_gripper": 0.2}, 10 ** 6)[0]["ok"]
    assert gw.move_to({"left_z": 0.20}, 10 ** 6)[0]["ok"]
    assert gw.move_to({"left_roll": 1.3}, 10 ** 6)[0]["ok"]               # tilted, but not over the bowl
    assert can.spilled > 0.2 and can.poured_in_bowl == 0.0
    assert gw.move_to({"left_roll": 0.0, "left_gripper": 1.0}, 10 ** 6)[0]["ok"]
    assert can.held_by is None and can.tilt_rad == 0.0                     # set down upright (simplified)
