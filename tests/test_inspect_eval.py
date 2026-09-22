"""Exercise the real framework and YAM gateway with offline policy transports."""
import json
import time
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("inspect_robots")

from inspect_robots import eval, read_eval_log
from inspect_robots.errors import EmbodimentFault, PolicyError, SafetyAbort
from inspect_robots.logging.json_log import JsonLogSink
from inspect_robots.scene import Scene
from inspect_robots.types import Action

from utils.astra_client import ScriptedAstraClient
from utils.config import REPO_ROOT, load_config
from utils.gateway import ExecutionResult
from utils.inspect_adapter import AstraInspectPolicy, GatewayTraceSink, PromptYamEmbodiment, Rot6dYamEmbodiment
from utils.inspect_eval import load_scenes, main, make_policy, make_task, policy_params, smoke_client


@pytest.fixture
def cfg():
    return load_config(str(REPO_ROOT / "configs/skild_yam_8.yaml"), {
        "robot.backend": "sim", "cameras.backend": "sim", "robot.home_at_start": False,
        "astra.backend": "scripted", "limits.max_llm_calls": 3,
    })


@pytest.fixture
def body(cfg):
    body = PromptYamEmbodiment(cfg, realtime=False)
    yield body
    body.close()


def evaluate(tmp_path, cfg, body, *, graded=False, grader=None, factory=smoke_client, epochs=1):
    policy = AstraInspectPolicy(cfg, client_factory=factory)
    sink = JsonLogSink(str(tmp_path))
    task = make_task([Scene("trial", "move the gripper")], cfg, epochs=epochs, graded=graded)
    (log,) = eval(task, policy, body, log_dir=str(tmp_path), grader=grader,
                  sinks=[sink, GatewayTraceSink(tmp_path)], store_frames=True)
    return log, read_eval_log(str(sink.path))


def test_framework_smoke_epochs_frames_transcripts_and_trace(tmp_path, cfg, body):
    log, saved = evaluate(tmp_path, cfg, body, epochs=2)
    assert log.status == saved.status == "success"
    assert saved.results.total_trials == 2
    assert saved.results.metrics == {"episode_length": 3.0}
    sample = saved.samples[0]
    assert sample.termination_reasons == ("done", "done")
    assert Path(saved.stats.frames_dir).is_dir()
    traces = []
    for metadata in sample.trial_metadata:
        assert metadata["astra_calls"] == 3
        trace = json.loads((tmp_path / metadata["gateway_trace"]).read_text())
        assert trace[0]["result"]["gateway"]["status"] == "completed"
        assert trace[0]["result"]["executed_joint_waypoints"]
        assert trace[-1]["result"]["gateway"]["status"] == "done"
        traces.append(trace)
    assert traces[0][0]["joint_pos"] == traces[1][0]["joint_pos"]
    for transcript in sample.policy_transcripts:
        text = json.dumps(transcript)
        assert "$blob:" in text
        assert "base64,/9j/" not in text
        assert transcript["messages"][-1]["type"] == "function_call_output"
        assert json.loads(transcript["messages"][-1]["output"])["status"] == "done"


@pytest.mark.parametrize("verdict,expected", [("success", 1.0), ("failure", 0.0), (None, 0.0)])
def test_existing_operator_scorer_uses_judgement_not_done(tmp_path, cfg, body, verdict, expected):
    class Grader:
        name = "fixture"

        def grade(self, record, scene):
            record.operator_judgement = verdict

    log, _ = evaluate(tmp_path, cfg, body, graded=True, grader=Grader())
    assert log.results.metrics["operator"] == expected
    assert log.samples[0].termination_reasons == ("done",)


def test_bad_astra_protocol_is_policy_error_and_cannot_move(cfg, body, monkeypatch):
    observation = body.reset(Scene("bad", "move"))
    policy = AstraInspectPolicy(cfg, client_factory=lambda *_: ScriptedAstraClient([
        {"name": "move_to", "arguments": {"targets": {"left_gripper": 0.0}}}]))
    policy.reset(Scene("bad", "move"))
    monkeypatch.setattr(body.robot, "command_joint_positions", lambda *_: pytest.fail("unexpected motion"))
    with pytest.raises(PolicyError, match="every declared dimension"):
        policy.act(observation)


@pytest.mark.parametrize("stop", [{"operation": "done"}, {"request_stop": True, "stop_reason": "give_up"},
                                {"request_stop": True, "stop_reason": "move_to"}])
def test_stop_actions_never_command_robot(body, monkeypatch, stop):
    obs = body.reset(Scene("stop", "stop"))
    monkeypatch.setattr(body.robot, "command_joint_positions", lambda *_: pytest.fail("unexpected motion"))
    result = body.step(Action(obs.state["eef_targets"], meta=stop))
    assert result.terminated
    with pytest.raises(EmbodimentFault, match="already ended"):
        body.step(Action(obs.state["eef_targets"]))


def test_out_of_bounds_is_rejected_before_motion(cfg, body, monkeypatch):
    cfg.limits.strict_gateway = True
    obs = body.reset(Scene("bad", "move"))
    data = obs.state["eef_targets"].copy()
    data[0] = 99.0
    monkeypatch.setattr(body.robot, "command_joint_positions", lambda *_: pytest.fail("unexpected motion"))
    result = body.step(Action(data, meta={"active_dims": ["left_x"]}))
    assert result.terminated
    assert result.info["gateway"]["status"] == "rejected"


@pytest.mark.parametrize("stage", ["before", "after_plan"])
def test_deadline_prevents_even_first_motion_after_expiry(body, monkeypatch, stage):
    obs = body.reset(Scene("time", "move"))
    data = obs.state["eef_targets"].copy()
    data[6] = 0.8
    if stage == "before":
        body.deadline = time.perf_counter() - 1
    else:
        original = body.gateway.plan

        def expired_plan(targets):
            plan = original(targets)
            body.deadline = time.perf_counter() - 1
            return plan

        monkeypatch.setattr(body.gateway, "plan", expired_plan)
    monkeypatch.setattr(body.robot, "command_joint_positions", lambda *_: pytest.fail("unexpected motion"))
    result = body.step(Action(data, meta={"active_dims": ["left_gripper"]}))
    assert result.truncated and result.termination_reason == "timeout"


def test_waypoint_budget_rejects_full_plan_before_motion(cfg, body, monkeypatch):
    cfg.limits.max_waypoints = 1
    obs = body.reset(Scene("budget", "close"))
    data = obs.state["eef_targets"].copy()
    data[6] = 0.0
    monkeypatch.setattr(body.robot, "command_joint_positions", lambda *_: pytest.fail("unexpected motion"))
    result = body.step(Action(data, meta={"active_dims": ["left_gripper"]}))
    assert result.info["gateway"]["status"] == "rejected"
    assert body.gateway.waypoints_executed == 0


def test_tracking_failure_halts_framework_instead_of_advancing(tmp_path, cfg, body, monkeypatch):
    monkeypatch.setattr("utils.gateway.SafetyGateway.execute", lambda *_, **__: ExecutionResult(
        False, "aborted", 1, "tracking failure"))
    log, _ = evaluate(tmp_path, cfg, body, epochs=2)
    assert log.status == "error"
    assert log.results.total_trials == 1
    assert log.samples[0].epochs == ({},)
    assert "tracking failure" in log.error


def test_hold_reference_keeps_commanded_gripper_and_measured_state(body):
    body.reset(Scene("hold", "hold"))
    body.gateway.gripper_cmd["left"] = 0.1
    obs = body._observe()
    assert obs.state["eef_targets"][6] == 0.1
    assert obs.state["joint_pos"][6] == 1.0
    assert obs.extra["measured_eef"]["left_gripper"] == 1.0


def test_station_ik_margin_and_finite_floor(cfg, body):
    assert np.isclose(body.kin.lower[0], cfg.robot.joint_lower[0] + cfg.motion.joint_limit_margin_rad)
    cfg.motion.tool_floor_z_m = float("nan")
    with pytest.raises(ValueError, match="floor height must be finite"):
        PromptYamEmbodiment(cfg, realtime=False)


def test_camera_dropout_fails_observation(body, monkeypatch):
    body.reset(Scene("cameras", "look"))
    monkeypatch.setattr(body.cameras, "read_jpeg_frames", lambda: {})
    with pytest.raises(EmbodimentFault, match="camera frames"):
        body._observe()


def test_astra_call_budget_makes_no_extra_request(cfg, body):
    cfg.limits.max_llm_calls = 1
    obs = body.reset(Scene("budget", "close"))
    policy = AstraInspectPolicy(cfg, client_factory=smoke_client)
    policy.reset(Scene("budget", "close"))
    result = body.step(policy.act(obs).actions[0])
    final = body.step(policy.act(result.observation).actions[0])
    assert final.truncated and final.termination_reason == "budget_exhausted"
    assert policy.calls == 1


def test_goal_loading_retains_station_setup():
    scenes = load_scenes(str(REPO_ROOT / "tasks/spatial/goals_spatial.txt"))
    assert len(scenes) == 10
    assert scenes[0].id == "SP01" and scenes[-1].id == "SP10"
    assert "Scene A" in scenes[0].metadata["setup_notes"]
    assert policy_params(["model=provider/example", "max_llm_calls=4", "wire_capture=false"]) == {
        "model": "provider/example", "max_llm_calls": 4, "wire_capture": False}


def test_existing_agent_plugin_can_replace_astra(tmp_path, cfg):
    pytest.importorskip("inspect_robots_agent")
    import httpx

    calls = []

    def reply(request):
        calls.append(json.loads(request.content))
        assert len(calls) <= 2
        name, arguments = (("move_to", {"targets": {"left_gripper": 0.8}, "note": "Close the left gripper."})
                           if len(calls) == 1 else ("done", {"summary": "Finished", "hindsight": "none"}))
        return httpx.Response(200, json={"choices": [{"message": {
            "role": "assistant", "content": None, "tool_calls": [{"id": f"call{len(calls)}",
            "type": "function", "function": {"name": name, "arguments": json.dumps(arguments)}}]}}]})

    policy = make_policy("agent", cfg, {"model": "test/agent", "base_url": "https://example.test/v1",
        "api_key_env": "TEST_KEY", "env": {"TEST_KEY": "dummy"}, "wire": "chat",
        "transport": httpx.MockTransport(reply), "wire_capture": False})
    body = Rot6dYamEmbodiment(cfg, realtime=False)
    try:
        (log,) = eval(make_task([Scene("swap", "close left gripper")], cfg, graded=False), policy, body,
                      log_dir=str(tmp_path), sinks=[JsonLogSink(str(tmp_path)), GatewayTraceSink(tmp_path)])
        assert np.isclose(body.robot.get_joint_positions()[6], 0.8)
    finally:
        body.close()
    assert log.status == "success", log.error
    assert len(calls) == 2
    assert log.samples[0].termination_reasons == ("done",)
    assert log.samples[0].trial_metadata[0]["gateway_trace"]
    assert "eef_targets" in json.dumps(calls[0])


def test_hardware_cli_rejects_unattended_runs_before_opening_devices(monkeypatch):
    monkeypatch.setattr("utils.inspect_eval.sys.stdin.isatty", lambda: False)
    monkeypatch.setattr("utils.inspect_adapter._make_robot_and_cameras",
                        lambda *_: pytest.fail("hardware accessed"))
    with pytest.raises(SystemExit, match="2"):
        main(["run", "--hardware", "--goal", "move", "--tool-floor-z", "0.0"])


def test_rot6d_preserves_nonzero_orientation_and_rejects_degenerate_columns(cfg, monkeypatch):
    from scipy.spatial.transform import Rotation
    body = Rot6dYamEmbodiment(cfg, realtime=False)
    try:
        obs = body.reset(Scene("rotation", "turn"))
        data = obs.state["eef_targets"].copy()
        rotation = Rotation.from_euler("xyz", [0.01, -0.02, 0.03]).as_matrix()
        data[3:9] = rotation[:, :2].T.reshape(-1)
        targets = []
        original = body.gateway.plan

        def capture(values):
            targets.append(values)
            return original(values)

        monkeypatch.setattr(body.gateway, "plan", capture)
        result = body.step(Action(data))
        assert result.info["gateway"]["status"] == "completed"
        assert np.allclose([targets[0]["left_roll"], targets[0]["left_pitch"], targets[0]["left_yaw"]],
                           [0.01, -0.02, 0.03])
        data[3:9] = 0
        monkeypatch.setattr(body.robot, "command_joint_positions", lambda *_: pytest.fail("unexpected motion"))
        with pytest.raises(SafetyAbort, match="rotation columns"):
            body.step(Action(data))
    finally:
        body.close()
