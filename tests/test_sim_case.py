import numpy as np
import pytest

from astra_yam.config import ARMS, Bounds, PipelineConfig, REFERENCE_HOME_JOINTS
from astra_yam.embodiment import ARM_SLICES
from astra_yam.gateway import SafetyGateway
from astra_yam.kinematics import ArmKinematics
from astra_yam.sim import SimCase, SimWorld, SimYamRobot


def _setup():
    cfg = PipelineConfig()
    cfg.bounds = Bounds(y=(-0.40, 0.40))
    cfg.motion.linear_speed_mps = 0.05
    kin = ArmKinematics(limit_margin=cfg.motion.joint_limit_margin_rad)
    world = SimWorld(kin, scene="airpods")
    q0 = np.zeros(14)
    q0[0:6] = q0[7:13] = REFERENCE_HOME_JOINTS
    q0[6] = q0[13] = 1.0
    robot = SimYamRobot(initial_q=q0, world=world)
    world.update(q0)
    start_rot = {arm: kin.fk(q0[ARM_SLICES[arm]])[1] for arm in ARMS}
    return cfg, kin, world, robot, SafetyGateway(cfg, kin, robot, start_rot, realtime=False)


def test_airpods_scene_geometry():
    world = SimWorld(scene="airpods")
    case = world.objects["airpods case"]
    assert isinstance(case, SimCase) and case.graspable and not case.hanging and case.lid_angle == 0.0
    dish = world.objects["wooden dish"]
    assert case.pos[2] == pytest.approx(dish.pos[2] + dish.height / 2 + case.H / 2)   # resting on the dish
    assert case.width_along(np.array([0, 1, 0])) == pytest.approx(case.W)               # jaws along y span the width
    assert case.width_along(np.array([1, 0, 0])) == pytest.approx(case.D)


def test_pick_up_pivots_upright_and_lid_can_be_opened_by_the_other_arm():
    cfg, kin, world, robot, gw = _setup()
    case = world.objects["airpods case"]
    x, y = float(case.pos[0]), float(case.pos[1])
    # left arm: open jaws around the case (across its 60.6 mm width), close, lift
    assert gw.move_to({"left_x": x, "left_y": y, "left_z": 0.15}, 10 ** 6)[0]["ok"]
    assert gw.move_to({"left_z": float(case.pos[2])}, 10 ** 6)[0]["ok"]
    assert gw.move_to({"left_gripper": 0.3}, 10 ** 6)[0]["ok"]
    assert case.held_by == "left" and not case.hanging
    assert robot.get_joint_positions()[6] == pytest.approx(case.W / 0.095, abs=0.01)   # stalled on the case width
    assert gw.move_to({"left_z": 0.16}, 10 ** 6)[0]["ok"]
    assert case.hanging and "pivoted upright" in " ".join(world.events)
    # carry it toward the midline so the right arm can reach
    assert gw.move_to({"left_x": 0.35, "left_y": -0.31}, 10 ** 6)[0]["ok"]
    grasp = world.grasp_points["left"]
    lid_c = case.lid_center()
    assert lid_c[2] - grasp[2] == pytest.approx(case.D / 2, abs=1e-6)                   # lid 2.3 cm above the pinch
    # right arm: jaws across the case depth (yaw ~ +1.57); pinch the lid near its right end, fingertips a little in
    # front of the case so the right gripper's neck clears the left gripper's housing (2 cm rule)
    right_target = {"right_x": float(lid_c[0]) + 0.03, "right_y": float(lid_c[1] - 0.03 + 0.61),
                    "right_z": float(lid_c[2]) + 0.02, "right_yaw": 1.57}
    p = gw.move_to(right_target, 10 ** 6)[0]
    assert p["ok"], p
    p = gw.move_to({"right_z": float(lid_c[2])}, 10 ** 6)[0]
    assert p["ok"], p
    p = gw.move_to({"right_gripper": 0.1}, 10 ** 6)[0]
    assert p["ok"], p
    assert case.lid_holder == "right", world.events
    # swing the lid up and over the hinge. The hinge is level with the seam, so lifting alone only cracks
    # the lid open; carrying the pinch back over the hinge is what swings it far enough to latch.
    p = gw.move_to({"right_z": float(lid_c[2]) + 0.07}, 10 ** 6)[0]
    assert p["ok"], p
    assert 0.5 < case.lid_angle < case.OPEN_LATCH_RAD and not case.is_open, np.degrees(case.lid_angle)
    p = gw.move_to({"right_z": float(lid_c[2]) + 0.10}, 10 ** 6)[0]
    assert p["ok"], p
    assert case.lid_angle > 1.0 and case.is_open, np.degrees(case.lid_angle)
    # from that height there is room to carry the pinch back over the hinge, which opens it the rest of the way
    p = gw.move_to({"right_x": float(lid_c[0]) - 0.02}, 10 ** 6)[0]
    assert p["ok"], p
    assert case.lid_angle > 1.4, np.degrees(case.lid_angle)
    # release: the lid stays open; the case is still held by the left arm
    assert gw.move_to({"right_gripper": 1.0}, 10 ** 6)[0]["ok"]
    assert case.lid_holder is None and case.is_open and case.held_by == "left"
    st = case.status()
    assert st["lid_open"] and st["hanging"]


def test_lid_snaps_shut_when_released_early():
    cfg, kin, world, robot, gw = _setup()
    case = world.objects["airpods case"]
    case.held_by = "left"
    case.hanging = True
    case.lid_holder = "right"
    case._lid_v0 = np.array([case.H / 2, 0.0, -case.LID / 2])
    case.lid_angle = 0.4
    openings = {"left": 0.03, "right": 0.05}                     # right jaws open wide -> release
    world._update_lid(case, openings)
    assert case.lid_holder is None and case.lid_angle == 0.0


def test_case_meshes_agree_with_the_parametric_case():
    """The Apple-derived meshes viser draws and SimCase's constants must describe one object."""
    from astra_yam.viser_ui import case_meshes

    meshes = case_meshes()
    assert meshes, "astra_yam/assets/airpods/case_*.ply missing - run scripts/build_airpods_asset.py"
    case = SimCase("airpods case", [0.0, 0.0, 0.0])
    case.hanging = True
    body = meshes["body"].vertices + case.body_center()          # drawn at body_center() / lid_center()
    lid = meshes["lid"].vertices + case.lid_center()
    lo = np.minimum(body.min(0), lid.min(0))
    hi = np.maximum(body.max(0), lid.max(0))
    assert hi - lo == pytest.approx([case.H, case.W, case.D], abs=2e-4)     # shut case fills the box
    assert lo == pytest.approx([-case.H / 2, -case.W / 2, -case.D / 2], abs=2e-4)
    assert body[:, 2].max() == pytest.approx(case.D / 2 - case.LID, abs=2e-4)   # seam == top of the body
    hinge = case.hinge()
    assert hinge[0] < 0 and hinge[2] < body[:, 2].max()           # behind the centre, below the seam
    assert lid[:, 2].min() < body[:, 2].max()                     # lid flange reaches down inside the body
