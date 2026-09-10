import json
import threading
from pathlib import Path

import cv2
import numpy as np
import pytest

from astra_yam.config import ReactiveConfig
from astra_yam.evaluation import Case, load_suite, verify
from astra_yam.reactive import EpisodeMemory, changed_fraction
from astra_yam.sim import SimWorld
from test_session_sim import _make


def last_request(outcome):
    path = sorted((Path(outcome.log_dir) / "requests").glob("request_*.json"))[-1]
    return json.loads(path.read_text())


def frames(x=40, noise=0):
    img = np.full((240, 320, 3), 180 + noise, np.uint8)
    cv2.circle(img, (x, 100), 25, (20, 150, 20), -1)
    return {"top_cam": cv2.imencode(".jpg", img)[1].tobytes()}


def test_scene_guard_detects_movement_without_state_oracle():
    cfg = ReactiveConfig()
    assert changed_fraction(frames(), frames(noise=2), cfg) < cfg.change_fraction
    assert changed_fraction(frames(), frames(150), cfg) > cfg.change_fraction
    with pytest.raises(RuntimeError, match="missing"):
        changed_fraction(frames(), {}, cfg)
    with pytest.raises(RuntimeError, match="invalid"):
        changed_fraction(frames(), {"top_cam": b"bad"}, cfg)


def test_stale_release_is_not_executed(tmp_path):
    script = [{"name": "move_to", "arguments": {"targets": {"left_gripper": 1}, "note": "release"}},
              {"name": "give_up", "arguments": {"reason": "test finished", "hindsight": ""}}]
    cfg, runner, world, astra = _make(tmp_path, script=script, **{"reactive.enabled": True,
                                                               "robot.home_at_start": False})
    current = [frames()]
    runner.cameras.read_jpeg_frames = lambda: current[0]
    class Hooks:
        def on_astra_response(self, response):
            if astra._n == 1:
                current[0] = frames(150)  # bowl moves while a release decision is pending
    runner.hooks = Hooks()
    outcome = runner.run("place case in green bowl")
    assert outcome.stale_actions == 1 and outcome.waypoints == 0
    events = [json.loads(line) for line in (Path(outcome.log_dir) / "transcript.jsonl").read_text().splitlines()]
    discarded = [e for e in events if e["kind"] == "tool_call" and e.get("result", {}).get("status") == "stale_observation"]
    assert len(discarded) == 1 and discarded[0]["arguments"]["targets"]["left_gripper"] == 1
    items = last_request(outcome)["input"]
    assert any(i.get("type") == "function_call_output" and "stale_observation" in i["output"] for i in items)
    assert "Observation sequence: 1" in json.dumps(items)
    assert len(list((Path(outcome.log_dir) / "frames").glob("*.jpg"))) == 2


def test_feedback_during_inference_invalidates_pending_move(tmp_path):
    cfg, runner, _, astra = _make(tmp_path, **{"reactive.enabled": True})
    class Operator:
        lines = []
        def poll(self):
            lines, self.lines = self.lines, []
            return lines
    operator = Operator()
    runner.operator = operator
    class Hooks:
        def on_astra_response(self, response):
            operator.lines = ["the bowl moved"] if astra._n == 1 else ["/stop"]
    runner.hooks = Hooks()
    outcome = runner.run("test")
    assert outcome.status == "operator_stop" and outcome.stale_actions == 1 and outcome.waypoints == 0


def test_long_trajectory_returns_partial_result_and_measured_pose(tmp_path):
    script = [{"name": "move_to", "arguments": {"targets": {"left_z": .35}, "note": "lift"}},
              {"name": "give_up", "arguments": {"reason": "test finished", "hindsight": ""}}]
    cfg, runner, _, astra = _make(tmp_path, script=script, **{"reactive.enabled": True,
                                                           "reactive.max_motion_seconds": 1})
    result = runner.run("test")
    assert result.observation_pauses == 1
    assert result.waypoints == 10 and result.waypoints_predicted > 10
    text = json.dumps(last_request(result))
    assert "observation_required" in text and "Observation sequence: 1" in text


def test_observe_preserves_unique_frames_and_episode_lessons(tmp_path):
    script = [{"name": "observe", "arguments": {"note": "look", "lesson": "the case slipped; align lower on the body"}},
              {"name": "observe", "arguments": {"note": "look again"}},
              {"name": "done", "arguments": {"summary": "test only", "hindsight": ""}}]
    _, runner, _, astra = _make(tmp_path, script=script, **{"reactive.enabled": True})
    outcome = runner.run("test")
    assert outcome.waypoints == 0 and outcome.llm_calls == 3
    assert len(list((Path(outcome.log_dir) / "frames").glob("*.jpg"))) == 9
    assert "the case slipped; align lower on the body" in json.dumps(last_request(outcome))
    # A subsequent trial must not receive learned notes from this one.
    runner.run("another task")
    assert runner._episode_memory.lessons == []


def test_memory_bounded_and_deduplicated():
    memory = EpisodeMemory()
    for i in range(10):
        memory.add(str(i))
    memory.add("9")
    assert memory.lessons == [str(i) for i in range(4, 10)]


def test_reactive_release_must_not_include_pose_targets(tmp_path):
    script = [{"name": "move_to", "arguments": {"targets": {"left_gripper": .2}, "note": "close"}},
              {"name": "move_to", "arguments": {"targets": {"left_gripper": 1, "left_z": .3}, "note": "unsafe release"}},
              {"name": "give_up", "arguments": {"reason": "test", "hindsight": ""}}]
    _, runner, _, _ = _make(tmp_path, script=script, **{"reactive.enabled": True})
    outcome = runner.run("test")
    assert outcome.rejections == 1
    assert "opening the gripper must be separate" in (Path(outcome.log_dir) / "transcript.txt").read_text()


def test_bowl_verifier_rejects_stale_position_and_rim_placement():
    world = SimWorld(scene="airpod_bowl")
    case = Case("bowl", "place", "airpod_bowl", "case_in_green_bowl", 0, "train")
    obj, bowl = world.objects["airpods case"], world.objects["green bowl"]
    obj.pos[:2] = bowl.pos[:2]
    world._settle(obj)
    assert verify(world, case)["success"]
    obj.held_by = "left"
    assert not verify(world, case)["success"]
    obj.held_by = None
    obj.pos[2] = bowl.pos[2] + bowl.height / 2 + obj.height / 2
    assert not verify(world, case)["success"]
    world._settle(obj)
    bowl.pos[1] += .15
    assert not verify(world, case)["success"]


def test_disturbance_schedule_is_loaded_and_frozen():
    cases = load_suite("configs/airpod_bowl_disturbances.yaml")
    assert cases[0].disturbances == ((5, .06, .04),)
    assert len(cases[1].disturbances) == 2


def test_scene_notification_pauses_inflight_motion(tmp_path):
    script = [{"name": "move_to", "arguments": {"targets": {"left_z": .35}, "note": "lift"}},
              {"name": "give_up", "arguments": {"reason": "test", "hindsight": ""}}]
    _, runner, _, _ = _make(tmp_path, script=script, **{"reactive.enabled": True, "robot.home_at_start": False})
    command = runner.robot.command_joint_positions
    sent = []
    def disturbed(q):
        sent.append(np.array(q))
        command(q)
        if len(sent) == 5:
            runner.request_reobserve()
    runner.robot.command_joint_positions = disturbed
    outcome = runner.run("test")
    assert outcome.observation_pauses == 1 and outcome.waypoints < 3
    assert outcome.status == "give_up"


def test_task_profile_selects_zero_shot_prompt_and_observe_tool():
    from astra_yam.cli import _config_from_args, build_parser
    from astra_yam.embodiment import build_tools
    cfg = _config_from_args(build_parser().parse_args(["run", "--sim", "--task-profile", "airpod-bowl", "--goal", "test"]))
    assert cfg.reactive.enabled and Path(cfg.policy_notes_path).name == "AIRPOD_BOWL.md"
    tools = build_tools(cfg.bounds, reactive=cfg.reactive.enabled)
    assert "observe" in {t["name"] for t in tools}
    assert "lesson" in tools[0]["parameters"]["properties"]


def test_release_disturbance_requires_held_case_and_fires_once():
    from types import SimpleNamespace
    from astra_yam.evaluation import BowlDisturbances
    world = SimWorld(scene="airpod_bowl")
    case = load_suite("configs/airpod_bowl_release_eval.yaml")[0]
    disturbances = BowlDisturbances(world, case)
    release = SimpleNamespace(function_calls=[SimpleNamespace(name="move_to", arguments={"targets": {"left_gripper": 1}})])
    disturbances.on_astra_response(release)
    assert disturbances.applied == 0
    world.objects["airpods case"].held_by = "left"
    world.grippers["left"] = .3
    before = world.objects["green bowl"].pos.copy()
    disturbances.on_astra_response(release)
    disturbances.on_astra_response(release)
    assert disturbances.applied == 1
    assert world.objects["green bowl"].pos[0] == pytest.approx(before[0] + .08)
