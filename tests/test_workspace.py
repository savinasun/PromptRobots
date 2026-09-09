import numpy as np
import pytest

from astra_yam.config import REPO_ROOT, YAM_JOINT_LOWER, YAM_JOINT_UPPER, load_config
from astra_yam.kinematics import ArmKinematics
from astra_yam.workspace import (
    EULER_PITCH_LIMIT,
    EULER_ROLL_LIMIT,
    EULER_YAW_LIMIT,
    full_rotation_bounds,
    reachable_envelope,
    urdf_joint_limits,
)

URDF = str(REPO_ROOT / "configs" / "skild_yam_v2.urdf")
STATION = str(REPO_ROOT / "configs" / "skild_yam_8.yaml")


def test_urdf_joint_limits_are_read_exactly():
    jl = urdf_joint_limits(URDF)
    assert jl.names == tuple(f"left_joint{k}" for k in range(1, 7))
    assert jl.lower == pytest.approx([-2.61799, 0.0, 0.0, -1.5708, -1.5708, -2.0944])
    assert jl.upper == pytest.approx([3.13, 3.65, 3.13, 1.5708, 1.5708, 2.0944])
    right = urdf_joint_limits(URDF, arm="right")
    assert right.lower == pytest.approx(jl.lower) and right.upper == pytest.approx(jl.upper)
    with pytest.raises(KeyError):
        urdf_joint_limits(URDF, arm="middle")


def test_mjcf_ranges_match_the_urdf():
    """The FK/IK model and the URDF must declare the same joint box, or the derived envelope is meaningless."""
    jl = urdf_joint_limits(URDF)
    kin = ArmKinematics(joint_lower=jl.lower, joint_upper=jl.upper, limit_margin=0.0)
    assert kin.model.jnt_range[:6, 0] == pytest.approx(jl.lower, abs=1e-6)
    assert kin.model.jnt_range[:6, 1] == pytest.approx(jl.upper, abs=1e-6)
    assert kin.lower == pytest.approx(jl.lower) and kin.upper == pytest.approx(jl.upper)


def test_gello_constants_are_inside_the_urdf_limits():
    """Safety invariant: the pipeline must never plan outside what the robot driver accepts."""
    jl = urdf_joint_limits(URDF)
    assert np.all(np.asarray(YAM_JOINT_LOWER) >= jl.lower - 1e-9)
    assert np.all(np.asarray(YAM_JOINT_UPPER) <= jl.upper + 1e-9)


def test_envelope_is_deterministic_and_matches_the_reach():
    jl = urdf_joint_limits(URDF)
    a = reachable_envelope(jl, samples=4000, seed=0)
    b = reachable_envelope(jl, samples=4000, seed=0)
    assert a.as_dict() == b.as_dict() and a.samples == 4000 + 64
    assert 0.80 < a.max_reach_m < 0.90                      # ~0.866 m from the URDF link lengths
    assert a.pitch[0] > -EULER_PITCH_LIMIT - 1e-9 and a.pitch[1] < EULER_PITCH_LIMIT + 1e-9
    block = a.yaml_block()
    assert block.startswith("bounds:") and "gripper: [0.0, 1.0]" in block


def test_station_config_bounds_cover_the_urdf_derived_envelope():
    """The YAML numbers must be the robot's own limits, not hand-picked: they have to contain the FK envelope."""
    cfg = load_config(STATION)
    jl = urdf_joint_limits(URDF)
    env = reachable_envelope(jl, reference_joints=cfg.robot.home_joints_left, samples=20000, seed=0)
    for dim in ("x", "y", "z"):
        lo_cfg, hi_cfg = cfg.bounds.for_dim(dim)
        lo_env, hi_env = getattr(env, dim)
        assert lo_cfg <= lo_env + 1e-9, f"{dim} lower bound {lo_cfg} cuts off reachable {lo_env}"
        assert hi_cfg >= hi_env - 1e-9, f"{dim} upper bound {hi_cfg} cuts off reachable {hi_env}"
        # and no arbitrary padding: no bound may sit outside the robot's own maximum reach
        assert max(abs(lo_cfg), abs(hi_cfg)) <= env.max_reach_m + 0.01, f"{dim} bound exceeds the reach envelope"
    rot = full_rotation_bounds()
    for dim in ("yaw", "pitch", "roll"):
        assert cfg.bounds.for_dim(dim) == pytest.approx(rot[dim]), dim
        assert not cfg.bounds.is_pinned(dim)
    assert cfg.robot.joint_lower == pytest.approx(jl.lower) and cfg.robot.joint_upper == pytest.approx(jl.upper)


def test_full_rotation_bounds_follow_the_euler_convention():
    rot = full_rotation_bounds()
    assert rot["yaw"] == (-round(EULER_YAW_LIMIT, 4), round(EULER_YAW_LIMIT, 4))
    assert rot["pitch"] == (-round(EULER_PITCH_LIMIT, 4), round(EULER_PITCH_LIMIT, 4))
    assert rot["roll"] == (-round(EULER_ROLL_LIMIT, 4), round(EULER_ROLL_LIMIT, 4))
    assert round(EULER_PITCH_LIMIT, 4) == 1.5708 and round(EULER_YAW_LIMIT, 4) == 3.1416


def test_tool_floor_defaults_to_the_z_bound_and_can_be_overridden():
    from astra_yam.config import ARMS, Bounds, PipelineConfig
    from astra_yam.embodiment import ARM_SLICES
    from astra_yam.gateway import SafetyGateway
    from astra_yam.sim import SimYamRobot

    cfg = PipelineConfig()
    cfg.bounds = Bounds(z=(-0.516, 0.864), pitch=(-1.5708, 1.5708), roll=(-3.1416, 3.1416))
    kin = ArmKinematics(limit_margin=cfg.motion.joint_limit_margin_rad)
    q = np.zeros(14)
    q[0:6] = q[7:13] = cfg.robot.home_joints_left
    q[6] = q[13] = 1.0
    robot = SimYamRobot(initial_q=q)
    gw = SafetyGateway(cfg, kin, robot, {a: kin.fk(q[ARM_SLICES[a]])[1] for a in ARMS}, realtime=False)
    from astra_yam.kinematics import rotation_from_ypr

    grasp, rot_home = kin.fk(np.asarray(cfg.robot.home_joints_left))
    rolled = rotation_from_ypr(rot_home, 0.0, 0.0, 1.5708)        # jaw axis vertical: one tip 4.75 cm below the grasp
    low = np.array([grasp[0], grasp[1], 0.02])
    assert gw._tool_dip_below_floor(low, rolled, 1.0) < 0         # wide z bound (-0.516) -> guard inactive
    cfg.motion.tool_floor_z_m = 0.0
    dip = gw._tool_dip_below_floor(low, rolled, 1.0)
    assert dip == pytest.approx(0.0475 - 0.02, abs=0.005), dip    # explicit table floor -> guard active
    assert gw._tool_dip_below_floor(low, rot_home, 1.0) < 0       # upright tool at the same height is fine
