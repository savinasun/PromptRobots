"""The bimanual UR5e embodiment: model registry, compact MJCF vs. the URDF, config resolution, gateway and sim."""
import json

import numpy as np
import pytest

from utils.arm_models import ARM_MODELS, UR5E_ARMS, YAM_ARMS, get_arm_model
from utils.config import ARMS, REPO_ROOT, Bounds, PipelineConfig, load_config
from utils.embodiment import ARM_SLICES, build_policy_prompt, start_rotations
from utils.gateway import SafetyGateway
from utils.kinematics import ArmKinematics, rotation_from_ypr
from utils.sim import SimRobot, SimWorld, SimYamRobot

UR5E_YAML = str(REPO_ROOT / "configs" / "skild_ur5e.yaml")
JOINTS = ["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint", "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"]


@pytest.fixture(scope="module")
def kin():
    return ArmKinematics(model=UR5E_ARMS)


def _urdf():
    yourdfpy = pytest.importorskip("yourdfpy")
    import logging
    logging.disable(logging.WARNING)      # the gripper mimic joints are fixed in this URDF; yourdfpy warns per load
    return yourdfpy.URDF.load(UR5E_ARMS.urdf_path, load_meshes=False, build_scene_graph=True)


def _urdf_tcp(urdf, arm, q6):
    cfg = {f"{arm}_{j}": v for j, v in zip(JOINTS, q6)}
    urdf.update_cfg(np.array([cfg.get(n, 0.0) for n in urdf.actuated_joint_names]))
    return urdf.get_transform(f"{arm}_robotiq_85_tcp", UR5E_ARMS.urdf_root_link)


# ----------------------------------------------------------------------------- registry
def test_registry_has_both_rigs_with_shared_conventions():
    assert set(ARM_MODELS) == {"yam_arms", "ur5e_arms"}
    assert get_arm_model(None) is YAM_ARMS and get_arm_model("ur5e_arms") is UR5E_ARMS
    with pytest.raises(KeyError):
        get_arm_model("franka_arms")
    for m in ARM_MODELS.values():
        assert len(m.joint_lower) == len(m.joint_upper) == 6 and len(m.home_joints["left"]) == 6
        assert m.arm_offset("left").tolist() == [0.0, 0.0, 0.0]
        assert m.arm_offset("right")[1] < 0                  # the right arm is on the -y side of the left arm
        assert m.side_sign("left") == 1.0 and m.side_sign("right") == -1.0
    assert UR5E_ARMS.base_distance_m() == pytest.approx(0.5) and YAM_ARMS.base_distance_m() == pytest.approx(0.61)
    assert UR5E_ARMS.jaw_max_opening_m == pytest.approx(0.085)


def test_mount_transforms_match_the_urdf():
    urdf = _urdf()
    for arm in ARMS:
        T = urdf.get_transform(f"{arm}_base", UR5E_ARMS.urdf_root_link)
        assert np.allclose(T[:3, 3], UR5E_ARMS.base_pos[arm], atol=1e-9)
        assert np.allclose(T[:3, :3], UR5E_ARMS.mount_rot(arm), atol=1e-7)
    # the root frame is gravity aligned: both bases are pitched 45 deg outward, away from each other
    assert UR5E_ARMS.mount_rot("left")[:, 2] == pytest.approx([0, np.sin(np.pi / 4), np.cos(np.pi / 4)], abs=1e-5)
    assert UR5E_ARMS.mount_rot("right")[:, 2] == pytest.approx([0, -np.sin(np.pi / 4), np.cos(np.pi / 4)], abs=1e-5)


# ----------------------------------------------------------------------------- kinematics
def test_compact_mjcf_reproduces_the_urdf_tcp_for_both_mounted_arms(kin):
    urdf = _urdf()
    rng = np.random.default_rng(0)
    for _ in range(25):
        q = rng.uniform(-3.0, 3.0, 6)
        for arm in ARMS:
            pos, rot = kin.fk(q, arm)
            T = _urdf_tcp(urdf, arm, q)
            assert np.allclose(pos, T[:3, 3] - np.asarray(UR5E_ARMS.base_pos[arm]), atol=1e-9), arm
            assert np.allclose(rot[:, 2], T[:3, 2], atol=1e-9)          # tool axis
            assert np.allclose(rot[:, 1], T[:3, 0], atol=1e-9)          # jaw axis = the 2F-85 knuckle axis (tcp x)


def test_mjcf_joint_box_equals_the_urdf_limits(kin):
    from utils.workspace import urdf_joint_limits

    jl = urdf_joint_limits(UR5E_ARMS.urdf_path, "left", UR5E_ARMS)
    assert jl.names == UR5E_ARMS.urdf_joint_names("left")
    assert jl.lower == pytest.approx(UR5E_ARMS.joint_lower) and jl.upper == pytest.approx(UR5E_ARMS.joint_upper)
    assert kin.model.jnt_range[:6, 0] == pytest.approx(jl.lower, abs=1e-6)
    assert kin.model.jnt_range[:6, 1] == pytest.approx(jl.upper, abs=1e-6)
    right = urdf_joint_limits(UR5E_ARMS.urdf_path, "right", UR5E_ARMS)
    assert right.lower == pytest.approx(jl.lower)


def test_home_pose_is_level_forward_and_mirrored(kin):
    poses = {arm: kin.fk(np.array(UR5E_ARMS.home_joints[arm]), arm) for arm in ARMS}
    for arm, (pos, rot) in poses.items():
        assert np.allclose(rot[:, 2], [0, 0, -1], atol=1e-3), f"{arm} tool not vertical: {rot[:, 2]}"
        assert np.allclose(np.abs(rot[:, 1]), [0, 1, 0], atol=1e-3)     # jaws open along y
        assert pos[0] == pytest.approx(0.35, abs=1e-3) and pos[2] == pytest.approx(0.087, abs=1e-3)
        o = kin.link_origins(np.array(UR5E_ARMS.home_joints[arm]), arm)
        assert o["forearm_link"][2] > o["wrist_1_link"][2] > pos[2]      # elbow up
        q = np.array(UR5E_ARMS.home_joints[arm])
        assert np.all(q - kin.lower > 0.5) and np.all(kin.upper - q > 0.5)
    assert poses["left"][0][1] == pytest.approx(-0.05, abs=1e-3) and poses["right"][0][1] == pytest.approx(0.05, abs=1e-3)
    # in the shared root frame the two grasp points mirror across the stand's centre plane
    left_root = poses["left"][0] + np.asarray(UR5E_ARMS.base_pos["left"])
    right_root = poses["right"][0] + np.asarray(UR5E_ARMS.base_pos["right"])
    assert np.allclose(left_root * [1, -1, 1], right_root, atol=1e-3)


def test_ik_roundtrip_in_the_mounted_frames(kin):
    rng = np.random.default_rng(3)
    n_ok = 0
    for i in range(30):
        arm = ARMS[i % 2]
        q = rng.uniform(-2.5, 2.5, 6)
        pos, rot = kin.fk(q, arm)
        seed = q + rng.normal(0, 0.15, 6)
        res = kin.ik(pos, rot, seed, max_iters=200, pos_tol=1e-3, ori_tol=5e-3, arm=arm)
        if res.converged:
            n_ok += 1
            p2, r2 = kin.fk(res.q, arm)
            assert np.linalg.norm(p2 - pos) < 1e-3 and np.allclose(r2, rot, atol=1e-2)
    assert n_ok >= 26, n_ok


def test_default_bounds_contain_the_reach_envelope(kin):
    from utils.workspace import reachable_envelope, urdf_joint_limits

    jl = urdf_joint_limits(UR5E_ARMS.urdf_path, "left", UR5E_ARMS)
    b = UR5E_ARMS.default_bounds
    for arm in ARMS:
        env = reachable_envelope(jl, reference_joints=UR5E_ARMS.home_joints[arm], samples=3000, arm=arm, model=UR5E_ARMS)
        for dim in ("x", "y", "z"):
            lo, hi = b.for_dim(dim)
            lo_env, hi_env = getattr(env, dim)
            assert lo <= lo_env + 1e-9 and hi >= hi_env - 1e-9, (arm, dim, (lo, hi), (lo_env, hi_env))
        assert 1.15 < env.max_reach_m < 1.30


# ----------------------------------------------------------------------------- config
def test_config_resolves_ur5e_defaults_and_yaml_wins():
    cfg = PipelineConfig(embodiment_name="ur5e_arms")
    assert cfg.robot.mjcf_path == UR5E_ARMS.mjcf_path and cfg.robot.joint_lower == list(UR5E_ARMS.joint_lower)
    assert cfg.robot.home_joints_left == list(UR5E_ARMS.home_joints["left"])
    assert cfg.bounds == UR5E_ARMS.default_bounds and cfg.viz.urdf_path == UR5E_ARMS.urdf_path
    assert cfg.system_prompt_path.endswith("SYSTEM_PROMPT_UR5E.md") and cfg.tilt_note_path.endswith("TILT_NOTE_UR5E.md")
    assert cfg.astra.prompt_cache_key == "astra-ur5e" and cfg.sim.table_z == pytest.approx(-0.113)
    yam = PipelineConfig()
    assert yam.robot.mjcf_path == YAM_ARMS.mjcf_path and yam.bounds == Bounds() and yam.sim.table_z == 0.0
    assert yam.robot.joint_lower == list(YAM_ARMS.joint_lower) and yam.system_prompt_path.endswith("SYSTEM_PROMPT.md")

    station = load_config(UR5E_YAML)
    assert station.embodiment_name == "ur5e_arms"
    assert station.motion.tool_floor_z_m == pytest.approx(-0.113) and station.motion.max_joint_step_rad == 0.03
    assert station.bounds == UR5E_ARMS.default_bounds          # the YAML spells out the same envelope
    explicit = load_config(UR5E_YAML, {"bounds.x": [0.1, 0.6], "system_prompt_path": str(REPO_ROOT / "docs/SYSTEM_PROMPT.md")})
    assert explicit.bounds.x == (0.1, 0.6) and explicit.system_prompt_path.endswith("docs/SYSTEM_PROMPT.md")
    assert explicit.bounds.y == UR5E_ARMS.default_bounds.y          # unspecified dims follow the rig, not the YAM
    partial = load_config(None, {"embodiment_name": "ur5e_arms", "bounds.z": [-0.2, 0.6]})
    assert partial.bounds.z == (-0.2, 0.6) and partial.bounds.x == UR5E_ARMS.default_bounds.x


def test_cli_embodiment_flag_and_prompt():
    from utils.cli import _config_from_args, build_parser

    cfg = _config_from_args(build_parser().parse_args(["show-prompt", "--embodiment", "ur5e_arms", "--sim"]))
    assert cfg.embodiment_name == "ur5e_arms" and cfg.robot.backend == "sim"
    text = build_policy_prompt(cfg)
    assert "named 'ur5e_arms'" in text and "8.5 cm" in text and "UR5e" in text
    assert "fingers point straight down" in text                  # tilt note: pitch/roll are released by default
    assert "named 'yam_arms'" not in text
    yam = _config_from_args(build_parser().parse_args(["show-prompt", "--sim"]))
    assert "named 'yam_arms'" in build_policy_prompt(yam)


def test_planner_prompt_names_the_gripper_opening():
    from utils.planning import build_planner_prompt

    cfg = PipelineConfig(embodiment_name="ur5e_arms")
    cfg.planning.enabled = True
    text = build_planner_prompt(cfg)
    assert "about 8.5 cm" in text and "Embodiment name: ur5e_arms" in text
    assert "about 9.5 cm" in build_planner_prompt(PipelineConfig())


# ----------------------------------------------------------------------------- gateway + sim
def _ur5e_setup(**overrides):
    cfg = load_config(UR5E_YAML, {"robot.backend": "sim", "cameras.backend": "sim", "motion.linear_speed_mps": 0.05,
                                  **overrides})
    kin = ArmKinematics.from_config(cfg)
    world = SimWorld(kin, table_z=cfg.sim.table_z, scene="blocks")
    q0 = UR5E_ARMS.home_q14()
    robot = SimRobot(initial_q=q0, world=world)
    world.update(q0)
    gw = SafetyGateway(cfg, kin, robot, start_rotations(q0, kin), realtime=False)
    return cfg, kin, world, robot, gw


def test_sim_robot_clips_to_the_ur5e_joint_box():
    kin = ArmKinematics(model=UR5E_ARMS)
    robot = SimRobot(initial_q=UR5E_ARMS.home_q14(), world=SimWorld(kin))
    q = UR5E_ARMS.home_q14()
    q[2] = 3.0                              # inside the +/-3.1415 elbow range, outside the YAM's [0, 3.13]? no: 3.0 fits both
    q[0] = -4.0                             # outside the YAM range (-2.618), inside the UR5e's (+/-2pi)
    robot.command_joint_positions(q)
    assert robot.get_joint_positions()[0] == pytest.approx(-4.0)
    assert SimYamRobot is SimRobot


def test_gateway_moves_both_ur5e_arms_and_reports_level_state():
    cfg, kin, world, robot, gw = _ur5e_setup()
    q, poses, eef = gw.read_state()
    for arm in ARMS:
        assert abs(eef[f"{arm}_yaw"]) < 1e-9 and abs(eef[f"{arm}_pitch"]) < 1e-9 and abs(eef[f"{arm}_roll"]) < 1e-9
        assert eef[f"{arm}_x"] == pytest.approx(0.35, abs=1e-3) and eef[f"{arm}_z"] == pytest.approx(0.087, abs=1e-3)
    payload, plan, res = gw.move_to({"left_x": 0.42, "left_z": 0.0, "right_y": 0.12, "right_yaw": 0.6}, 10 ** 6)
    assert payload["ok"], payload
    _, poses, eef = gw.read_state()
    assert eef["left_x"] == pytest.approx(0.42, abs=2e-3) and eef["left_z"] == pytest.approx(0.0, abs=2e-3)
    assert eef["right_y"] == pytest.approx(0.12, abs=2e-3) and eef["right_yaw"] == pytest.approx(0.6, abs=2e-2)
    assert np.allclose(poses["left"].rot[:, 2], [0, 0, -1], atol=1e-2)        # still pointing down: z is level
    assert plan.min_clearance_m is not None and plan.min_clearance_m > 0.05


def test_gateway_tilt_guard_uses_the_robotiq_geometry():
    cfg, kin, world, robot, gw = _ur5e_setup()
    grasp, rot_home = kin.fk(np.array(UR5E_ARMS.home_joints["left"]), "left")
    rolled = rotation_from_ypr(rot_home, 0.0, 0.0, 1.5708)     # jaw axis vertical: one tip 4.25 cm below the grasp point
    low = np.array([grasp[0], grasp[1], cfg.motion.tool_floor_z_m + 0.02])
    dip = gw._tool_dip_below_floor(low, rolled, 1.0)
    assert dip == pytest.approx(0.0425 - 0.02, abs=0.005), dip
    assert gw._tool_dip_below_floor(low, rot_home, 1.0) < 0
    payload, plan, res = gw.move_to({"left_z": cfg.motion.tool_floor_z_m - 0.05}, 10 ** 6)
    assert not payload["ok"] and "floor" in payload["reason"], payload


def test_arm_clearance_is_evaluated_at_the_ur5e_base_offset():
    from utils.collision import clearance

    cfg, kin, world, robot, gw = _ur5e_setup()
    home = clearance(kin, UR5E_ARMS.home_q14())
    assert home.deficit_m < 0 and 0.1 < home.clearance_m < 0.5, home
    # drive the grasp points onto each other (in the shared root frame): the capsule model must flag it
    payload, plan, res = gw.move_to({"left_y": -0.25, "right_y": 0.25, "left_x": 0.4, "right_x": 0.4}, 10 ** 6)
    assert not payload["ok"] and ("come within" in payload["reason"] or "detour" in payload["reason"]), payload


def test_sim_world_uses_the_ur5e_frames_and_jaw_span():
    cfg, kin, world, robot, gw = _ur5e_setup()
    assert world.jaw_max_opening_m == pytest.approx(0.085)
    assert np.allclose(world.arm_offset("right"), [0.0, -0.5, 0.0])
    assert world.table_z == pytest.approx(-0.113)
    blue = world.objects["blue block"]
    assert blue.pos[2] == pytest.approx(world.table_z + blue.height / 2)
    # a pick in the UR5e frame: descend over the block, close, lift
    over = {"left_x": float(blue.pos[0]), "left_y": float(blue.pos[1]), "left_z": float(blue.pos[2]) + 0.12}
    assert gw.move_to(over, 10 ** 6)[0]["ok"]
    assert gw.move_to({"left_z": float(blue.pos[2])}, 10 ** 6)[0]["ok"]
    assert gw.move_to({"left_gripper": 0.0}, 10 ** 6)[0]["ok"]
    assert blue.held_by == "left"
    assert gw.move_to({"left_z": float(blue.pos[2]) + 0.15}, 10 ** 6)[0]["ok"]
    assert blue.held_by == "left" and blue.pos[2] > world.table_z + 0.1


def test_scripted_trial_runs_end_to_end_on_the_ur5e(tmp_path):
    from utils.astra_client import ScriptedAstraClient
    from utils.embodiment import build_tools
    from utils.session import TrialRunner
    from utils.sim import SimCameraSource

    cfg = load_config(UR5E_YAML, {"robot.backend": "sim", "cameras.backend": "sim", "astra.backend": "scripted",
                                  "astra.actions_only": False, "log_dir": str(tmp_path), "motion.linear_speed_mps": 0.05})
    kin = ArmKinematics.from_config(cfg)
    world = SimWorld(kin, table_z=cfg.sim.table_z)
    q0 = UR5E_ARMS.home_q14()
    robot = SimRobot(initial_q=q0, world=world)
    world.update(q0)
    script = [
        {"name": "move_to", "arguments": {"targets": {"left_x": 0.40, "left_z": 0.02}, "note": "down and forward"}},
        {"name": "move_to", "arguments": {"targets": {"right_gripper": 0.2}, "note": "close right"}},
        {"name": "done", "arguments": {"summary": "moved", "hindsight": "-"}},
    ]
    astra = ScriptedAstraClient(script=script, tools=build_tools(cfg.bounds, cfg.prompts_path, actions_only=False,
                                                                  max_waypoints=cfg.motion.max_waypoints_per_call),
                                actions_only=False)
    runner = TrialRunner(cfg, robot, SimCameraSource(world), kin, astra, realtime=False, sim_world=world, verbose=False)
    outcome = runner.run("stretch")
    assert outcome.status == "done", outcome
    q = robot.get_joint_positions()
    pos, _ = kin.fk(q[ARM_SLICES["left"]], "left")
    assert pos[0] == pytest.approx(0.40, abs=3e-3) and pos[2] == pytest.approx(0.02, abs=3e-3)
    assert q[13] == pytest.approx(0.2)
    transcript = next(tmp_path.rglob("transcript.txt")).read_text()
    assert "ur5e_arms" in transcript
