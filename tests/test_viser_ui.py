import socket

import numpy as np
import pytest

viser = pytest.importorskip("viser")
yourdfpy = pytest.importorskip("yourdfpy")

from astra_yam.config import ARMS, REFERENCE_HOME_JOINTS, PipelineConfig  # noqa: E402
from astra_yam.embodiment import ARM_SLICES  # noqa: E402
from astra_yam.gateway import SafetyGateway  # noqa: E402
from astra_yam.kinematics import ArmKinematics  # noqa: E402
from astra_yam.sim import SimCameraSource, SimWorld, SimYamRobot  # noqa: E402
from astra_yam.viser_ui import LinkMeshes, MirroredRobot, ViserCameraSource, ViserVisualizer, look_at_wxyz, wxyz_from_matrix  # noqa: E402
from scipy.spatial.transform import Rotation  # noqa: E402


def _free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def _home():
    q = np.zeros(14)
    q[0:6] = q[7:13] = REFERENCE_HOME_JOINTS
    q[6] = q[13] = 1.0
    return q


@pytest.fixture
def viz_env():
    cfg = PipelineConfig()
    cfg.viz.host = "127.0.0.1"
    cfg.viz.port = _free_port()
    cfg.viz.enabled = True
    kin = ArmKinematics(limit_margin=cfg.motion.joint_limit_margin_rad)
    world = SimWorld(kin, scene="kitchen")
    robot = SimYamRobot(initial_q=_home(), world=world)
    world.update(_home())
    viz = ViserVisualizer(cfg, kin, world=world, mode="sim")
    robot.on_command = viz.update_robot
    yield cfg, kin, world, robot, viz
    viz.close()


def test_look_at_convention():
    wxyz = look_at_wxyz((0, 0, 1), (0, 0, 0))
    w, x, y, z = wxyz
    R = Rotation.from_quat([x, y, z, w]).as_matrix()
    assert np.allclose(R[:, 2], [0, 0, -1])          # +Z camera axis points at the target (forward)
    assert np.allclose(np.linalg.det(R), 1.0)
    R2 = np.eye(3)
    assert np.allclose(wxyz_from_matrix(R2), [1, 0, 0, 0])


def test_urdf_links_follow_mujoco_fk(viz_env):
    cfg, kin, world, robot, viz = viz_env
    viz.update_robot(_home(), force=True)
    tcp_mj = ArmKinematics(site="tcp_site").fk(np.array(REFERENCE_HOME_JOINTS))[0]
    for arm in ARMS:
        link6 = np.asarray(viz._link_frames[(arm, "link_6")].position)
        assert np.allclose(link6, tcp_mj, atol=1e-4), f"{arm}: {link6} vs {tcp_mj}"
        tool = np.asarray(viz._tool_frames[arm].position)
        assert np.allclose(tool, kin.fk(np.array(REFERENCE_HOME_JOINTS))[0], atol=1e-6)
    # jaws fully open at gripper = 1.0
    assert abs(viz._jaws["left"][0].position[1] - 0.0475) < 1e-9
    if viz.meshes is not None:
        assert set(viz.meshes.items) >= {"base_link", "link_1", "link_6"}


def test_plan_hooks_and_operator_queue(viz_env):
    cfg, kin, world, robot, viz = viz_env
    q0 = robot.get_joint_positions()
    start_rot = {arm: kin.fk(q0[ARM_SLICES[arm]])[1] for arm in ARMS}
    gw = SafetyGateway(cfg, kin, robot, start_rot, realtime=False)
    gw.on_plan = viz.on_plan
    payload, plan, res = gw.move_to({"left_z": 0.15}, 10 ** 6)
    assert payload["ok"] and len(viz._plan_handles) == 2          # path + goal marker for the moving arm
    viz.on_tool_call("move_to", {"targets": {"left_z": 0.15}, "note": "test"}, payload, plan)
    assert "completed" in viz._status["last_result"]
    viz.clear_plan()
    assert viz._plan_handles == []

    viz._on_stop()
    viz._fb_text.value = "watch the cup"
    viz._on_send()
    assert viz.poll_operator() == ["/stop", "watch the cup"] and viz.poll_operator() == []
    assert viz.wait_for_start(timeout=0.05, allow_stdin=False) is False
    viz._on_start()
    assert viz.wait_for_start(timeout=0.5, allow_stdin=False) is True


def test_camera_source_falls_back_without_browser(viz_env):
    cfg, kin, world, robot, viz = viz_env
    viz.update_robot(_home(), force=True)
    assert viz.render_cameras() is None                           # no browser connected
    src = ViserCameraSource(viz, fallback=SimCameraSource(world))
    frames = src.read_jpeg_frames()
    assert set(frames) == {"top_cam", "left_cam", "right_cam"} and src.last_source == "fallback"
    viz.on_observation(0, _home(), {}, frames, 3000)             # previews update without error
    for name in ("top_cam", "left_cam", "right_cam"):
        pos, wxyz, fov = viz.camera_pose(name)
        assert pos.shape == (3,) and abs(np.linalg.norm(wxyz) - 1) < 1e-6 and 0.5 < fov < 1.6


def test_object_gizmos_and_reset(viz_env):
    cfg, kin, world, robot, viz = viz_env
    viz._set_edit_objects(True)
    assert set(viz._gizmos) == set(world.objects)
    world.set_object_position("teal cup", [0.3, -0.2, 0.05])
    viz.update_objects()
    assert np.allclose(viz._object_handles["teal cup"][0].position, [0.3, -0.2, 0.05])
    viz._on_reset_objects()
    assert np.allclose(world.objects["teal cup"].pos, world.objects["teal cup"].initial_pos)
    viz._set_edit_objects(False)
    assert viz._gizmos == {}


def test_mirrored_robot_updates_viz(viz_env):
    cfg, kin, world, robot, viz = viz_env
    robot.on_command = None
    mirrored = MirroredRobot(robot, viz)
    q = _home()
    q[6] = 0.2
    mirrored.command_joint_positions(q)
    assert np.allclose(viz._last_q, q) and abs(viz._jaws["left"][0].position[1] - 0.0095) < 1e-9
    assert mirrored.num_dofs() == 14 and np.allclose(mirrored.get_joint_positions()[6], 0.2)


def _project(pos, wxyz, fov, aspect, point):
    """Pixel coordinates (u, v in [0,1]) of a world point for a viser/OpenCV camera; None if behind the camera."""
    w, x, y, z = wxyz
    R = Rotation.from_quat([x, y, z, w]).as_matrix()          # camera axes in world coords (columns)
    p_cam = R.T @ (np.asarray(point) - np.asarray(pos))        # world -> camera
    if p_cam[2] <= 0:
        return None
    fy = 0.5 / np.tan(fov / 2)
    fx = fy / aspect
    return 0.5 + fx * p_cam[0] / p_cam[2], 0.5 + fy * p_cam[1] / p_cam[2]


def test_camera_poses_see_the_workspace(viz_env):
    cfg, kin, world, robot, viz = viz_env
    q = _home()
    viz.update_robot(q, force=True)
    # top camera: table center and both arm bases inside the image, bases in the lower half (like the real top_cam)
    pos, wxyz, fov = viz.camera_pose("top_cam")
    for point, lower_half in (((0.35, -0.305, 0.0), False), ((0.0, 0.0, 0.0), True), ((0.0, -0.61, 0.0), True)):
        uv = _project(pos, wxyz, fov, 4 / 3, point)
        assert uv is not None and 0 <= uv[0] <= 1 and 0 <= uv[1] <= 1, (point, uv)
        if lower_half:
            assert uv[1] > 0.5
    # wrist cameras: the arm's own grasp point is in view and roughly centered horizontally
    for arm in ARMS:
        pos, wxyz, fov = viz.camera_pose(f"{arm}_cam", q)
        grasp = kin.fk(q[ARM_SLICES[arm]])[0] + (np.zeros(3) if arm == "left" else np.array([0, -0.61, 0]))
        uv = _project(pos, wxyz, fov, 4 / 3, grasp)
        assert uv is not None and 0.35 < uv[0] < 0.65 and 0.45 < uv[1] < 1.0, (arm, uv)   # centered, lower half
        ahead = grasp + 0.15 * kin.fk(q[ARM_SLICES[arm]])[1][:, 2]       # a point further along the tool axis
        uv2 = _project(pos, wxyz, fov, 4 / 3, ahead)
        assert uv2 is not None and 0 <= uv2[0] <= 1 and 0 <= uv2[1] <= 1


def test_estop_button_halts_motion_and_requests_stop(viz_env):
    cfg, kin, world, robot, viz = viz_env
    calls = []
    viz.on_estop = lambda: calls.append("pressed")
    viz._on_estop()
    assert calls == ["pressed"]                       # halts a motion already streaming
    assert viz.poll_operator() == ["/stop"]           # and ends the session
    assert viz._status["phase"] == "EMERGENCY STOP"

    def boom():
        raise RuntimeError("gateway gone")

    viz.on_estop = boom                               # a failing callback must not swallow the stop
    viz._on_estop()
    assert viz.poll_operator() == ["/stop"]

    viz.on_estop = None                               # unwired (e.g. no trial running yet)
    viz._on_estop()
    assert viz.poll_operator() == ["/stop"]


def test_render_hides_overlays_and_restores_them(viz_env, monkeypatch):
    cfg, kin, world, robot, viz = viz_env
    viz.update_robot(_home(), force=True)
    seen = {}

    class FakeClient:
        def get_render(self, h, w, **kw):
            seen["hidden_during_render"] = all(not getattr(x, "visible", True) for x in viz._overlay_handles())
            return np.zeros((h, w, 3), np.uint8)

    class FakeClient2(FakeClient):
        def get_render(self, h, w, **kw):
            seen["labels_during_render"] = len(viz._label_handles)
            return super().get_render(h, w, **kw)

    assert viz._label_handles == {}                                              # labels off by default
    viz.vc.show_labels = True
    viz.vc.label_settle_s = 0.0
    viz._create_labels()
    n_labels = len(viz._label_handles)
    assert n_labels == 2 + len(world.objects)
    monkeypatch.setattr(viz.server, "get_clients", lambda: {0: FakeClient2()})
    visible_before = [x for x in viz._overlay_handles() if getattr(x, "visible", True)]
    frames = viz.render_cameras()
    assert frames is not None and set(frames) == {"top_cam", "left_cam", "right_cam"}
    assert seen["hidden_during_render"] is True and seen["labels_during_render"] == 0
    assert all(getattr(x, "visible", True) for x in visible_before)             # restored afterwards
    # (the workspace-bounds box is not in that list: it starts hidden when the bounds span the whole envelope)
    assert len(viz._label_handles) == n_labels                                   # labels recreated
    viz.update_objects()                                                         # label positions follow objects


def test_objects_markdown_lists_scene(viz_env):
    cfg, kin, world, robot, viz = viz_env
    md = viz._objects_markdown()
    assert "teal cup" in md and "cylinder" in md and "white plate" in md


def test_rate_limited_pose_is_flushed_before_render(viz_env, monkeypatch):
    cfg, kin, world, robot, viz = viz_env
    q0 = _home()
    viz.update_robot(q0, force=True)
    q1 = q0.copy()
    q1[6] = 0.3
    viz.update_robot(q1)                                  # within the rate limit -> dropped, kept as pending
    assert np.allclose(viz._last_q, q0) and viz._pending_q is not None
    rendered_q = {}

    class FakeClient:
        def get_render(self, h, w, **kw):
            rendered_q["q"] = viz._last_q.copy()
            return np.zeros((h, w, 3), np.uint8)

    monkeypatch.setattr(viz.server, "get_clients", lambda: {0: FakeClient()})
    assert viz.render_cameras(["top_cam"]) is not None
    assert np.allclose(rendered_q["q"], q1) and viz._pending_q is None
    assert abs(viz._jaws["left"][0].position[1] - 0.3 * 0.095 / 2) < 1e-9
