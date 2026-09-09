import numpy as np
import pytest

from astra_yam.config import ARMS, PipelineConfig
from astra_yam.embodiment import ARM_SLICES
from astra_yam.gateway import SafetyGateway
from astra_yam.kinematics import ArmKinematics
from astra_yam.sim import GRIPPER_MAX_OPENING_M, SCENES, SimCameraSource, SimWorld, SimYamRobot


def _setup(scene):
    cfg = PipelineConfig()
    cfg.motion.linear_speed_mps = 0.05   # faster plans for tests
    kin = ArmKinematics(limit_margin=cfg.motion.joint_limit_margin_rad)
    world = SimWorld(kin, scene=scene)
    q0 = np.zeros(14)
    q0[0:6] = q0[7:13] = cfg.robot.home_joints_left
    q0[6] = q0[13] = 1.0
    robot = SimYamRobot(initial_q=q0, world=world)
    world.update(q0)
    start_rot = {arm: kin.fk(q0[ARM_SLICES[arm]])[1] for arm in ARMS}
    gw = SafetyGateway(cfg, kin, robot, start_rot, realtime=False)
    return cfg, kin, world, robot, gw


def test_scene_presets():
    assert set(SCENES) == {"blocks", "kitchen", "airpods", "chili", "empty"}
    kitchen = SimWorld(scene="kitchen")
    assert kitchen.objects["teal cup"].graspable and kitchen.objects["white mug"].graspable
    assert not kitchen.objects["white plate"].graspable          # 20 cm wide > 9.5 cm jaws
    assert abs(kitchen.objects["teal cup"].height - 0.10) < 1e-9
    assert SimWorld(scene="empty").objects == {}
    with pytest.raises(ValueError):
        SimWorld(scene="nope")


def test_cylinder_grasp_lift_release_and_gripper_floor():
    cfg, kin, world, robot, gw = _setup("kitchen")
    cup = world.objects["teal cup"]
    x, y = cup.pos[:2]
    assert gw.move_to({"left_x": float(x), "left_y": float(y), "left_z": 0.12}, 10 ** 6)[0]["ok"]
    assert gw.move_to({"left_z": 0.07}, 10 ** 6)[0]["ok"]              # mid-height of a 10 cm cup
    assert gw.move_to({"left_gripper": 0.1}, 10 ** 6)[0]["ok"]
    assert cup.held_by == "left"
    q = robot.get_joint_positions()
    assert abs(q[6] - 0.07 / GRIPPER_MAX_OPENING_M) < 1e-6              # jaws stop at the cup's diameter
    z_before = cup.pos[2]
    assert gw.move_to({"left_z": 0.20}, 10 ** 6)[0]["ok"]
    assert cup.pos[2] > z_before + 0.10                                   # lifted with the grasp offset preserved
    assert gw.move_to({"left_y": float(y) + 0.10}, 10 ** 6)[0]["ok"]   # to a free spot inside the left arm's bounds
    assert gw.move_to({"left_gripper": 1.0}, 10 ** 6)[0]["ok"]
    assert cup.held_by is None and abs(cup.pos[2] - cup.height / 2) < 1e-9   # dropped back onto the table
    assert abs(cup.pos[1] - (y + 0.10)) < 0.01
    events = " ".join(world.events)
    assert "teal cup grasped by left" in events and "teal cup released by left" in events


def test_schematic_cameras_render_cylinders():
    cfg, kin, world, robot, gw = _setup("kitchen")
    frames = SimCameraSource(world).read_jpeg_frames()
    assert set(frames) == {"top_cam", "left_cam", "right_cam"} and all(len(v) > 1000 for v in frames.values())


def test_reset_and_set_object_position():
    world = SimWorld(scene="blocks")
    blue = world.objects["blue block"]
    world.set_object_position("blue block", [0.4, 0.1, -0.5])
    assert blue.pos[2] == pytest.approx(blue.height / 2)                 # clamped onto the table
    world.reset_objects()
    assert np.allclose(blue.pos, blue.initial_pos)


def test_on_command_callback():
    seen = []
    robot = SimYamRobot(on_command=lambda q: seen.append(q.copy()))
    robot.command_joint_positions(np.zeros(14))
    assert len(seen) == 1


def test_release_over_another_object_stacks_on_it():
    cfg, kin, world, robot, gw = _setup("blocks")
    blue, green = world.objects["blue block"], world.objects["green block"]
    assert gw.move_to({"left_x": float(blue.pos[0]), "left_y": float(blue.pos[1]), "left_z": 0.03}, 10 ** 6)[0]["ok"]
    assert gw.move_to({"left_gripper": 0.2}, 10 ** 6)[0]["ok"] and blue.held_by == "left"
    assert gw.move_to({"left_z": 0.12}, 10 ** 6)[0]["ok"]
    assert gw.move_to({"left_y": float(green.pos[1])}, 10 ** 6)[0]["ok"]
    assert gw.move_to({"left_gripper": 1.0}, 10 ** 6)[0]["ok"]
    assert blue.held_by is None and blue.pos[2] == pytest.approx(green.pos[2] + 0.03)


def test_closed_gripper_pushes_cup_without_grasping():
    cfg, kin, world, robot, gw = _setup("kitchen")
    cup = world.objects["teal cup"]
    x, y = float(cup.pos[0]), float(cup.pos[1])
    assert gw.move_to({"left_gripper": 0.0}, 10 ** 6)[0]["ok"]
    assert gw.move_to({"left_x": x, "left_y": y - 0.08, "left_z": 0.20}, 10 ** 6)[0]["ok"]   # above, beside the cup
    assert gw.move_to({"left_z": 0.05}, 10 ** 6)[0]["ok"]                                    # descend next to it
    assert np.allclose(cup.pos[:2], [x, y])                                                  # not touched yet
    assert gw.move_to({"left_y": y + 0.02}, 10 ** 6)[0]["ok"]                                # sweep 10 cm to the left
    assert cup.held_by is None
    assert cup.pos[1] == pytest.approx(y + 0.02 + 0.035 + 0.012, abs=0.002)   # pushed ahead of the closed fingertips
    assert abs(cup.pos[0] - x) < 0.01 and abs(cup.pos[2] - 0.05) < 1e-9
    assert robot.get_joint_positions()[6] == pytest.approx(0.0) # readback stays closed while pushing
    assert any("pushed by left" in e for e in world.events) and not any("grasped" in e for e in world.events)


def test_closed_gripper_descending_on_cup_does_not_grasp():
    cfg, kin, world, robot, gw = _setup("kitchen")
    cup = world.objects["teal cup"]
    x, y = float(cup.pos[0]), float(cup.pos[1])
    assert gw.move_to({"left_gripper": 0.0}, 10 ** 6)[0]["ok"]
    assert gw.move_to({"left_x": x, "left_y": y, "left_z": 0.20}, 10 ** 6)[0]["ok"]
    assert gw.move_to({"left_z": 0.12}, 10 ** 6)[0]["ok"]         # closed fingertips come down onto the rim
    assert cup.held_by is None and not any("grasped" in e for e in world.events)


def test_open_jaws_straddle_then_close_grasps():
    cfg, kin, world, robot, gw = _setup("kitchen")
    cup = world.objects["teal cup"]
    x, y = float(cup.pos[0]), float(cup.pos[1])
    assert gw.move_to({"left_x": x, "left_y": y, "left_z": 0.15}, 10 ** 6)[0]["ok"]
    assert gw.move_to({"left_z": 0.06}, 10 ** 6)[0]["ok"]         # open jaws around the cup: no push
    assert np.allclose(cup.pos[:2], [x, y])
    assert gw.move_to({"left_gripper": 0.3}, 10 ** 6)[0]["ok"]
    assert cup.held_by == "left"
    assert gw.move_to({"left_gripper": 0.8}, 10 ** 6)[0]["ok"]    # 7.6 cm opening > 7 cm cup + 4 mm -> released
    assert cup.held_by is None


def test_cli_scene_choices_match_presets():
    from astra_yam.cli import build_parser
    parser = build_parser()
    run_parser = next(a for a in parser._subparsers._group_actions[0].choices.values() if a.prog.endswith(" run"))
    scene_action = next(a for a in run_parser._actions if "--scene" in a.option_strings)
    assert set(scene_action.choices) == set(SCENES)
