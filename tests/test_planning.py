"""Two-model planning: text handoff, isolated native histories, real simulated IK, and audit logs."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from utils.astra_client import DEFAULT_SIM_SCRIPT, OpenAIAstraClient, OpenRouterChatClient, ScriptedAstraClient
from utils.cli import _config_from_args, build_parser, cmd_run
from utils.config import AstraConfig, load_config
from utils.embodiment import build_policy_prompt
from utils.observation import count_images
from utils.planning import planner_config
from test_session_sim import _make


class RecordingClient(ScriptedAstraClient):
    def __init__(self, script, model):
        super().__init__(script=script, model=model)
        self.seen = []

    def create(self, items):
        self.seen.append(copy.deepcopy(items))
        response = super().create(items)
        response.usage = {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15}
        return response


def planning_runner(tmp_path, plans=None, moves=None, **overrides):
    cfg, runner, world, _ = _make(tmp_path, **{"robot.home_at_start": False, **overrides})
    cfg.planning.enabled = True
    runner.system_prompt = build_policy_prompt(cfg)
    planner = RecordingClient(plans if plans is not None else [
        {"text": "Scene: gripper is open. Close the left gripper slightly; keep its pose.",
         "reasoning": "Planner trace one."},
        {"text": "The goal is complete. Call done.", "reasoning": "Planner trace two."}], "system-2")
    motion = RecordingClient(moves if moves is not None else [
        {"name": "move_to", "arguments": {"targets": {"left_gripper": .9}, "note": "Follow planner."},
         "reasoning": "Motion trace one."},
        {"name": "done", "arguments": {"summary": "Complete.", "hindsight": "none"}}], "system-1")
    runner.planner, runner.astra = planner, motion
    return runner, planner, motion


def test_connected_histories_execution_and_logs(tmp_path):
    runner, planner, motion = planning_runner(tmp_path)
    outcome = runner.run("Close the left gripper slightly.")
    assert outcome.status == "done", outcome
    assert outcome.llm_calls == 4 and outcome.planner_calls == outcome.motion_calls == 2
    assert outcome.usage == {"input_tokens": 40, "output_tokens": 20, "total_tokens": 60}
    assert outcome.waypoints > 0
    assert count_images(planner.seen[0]) == 3
    assert all(count_images(items) == 0 for items in motion.seen)
    assert "SYSTEM 2 PLAN" in motion.seen[0][-1]["content"]
    assert "Close the left gripper" in motion.seen[0][-1]["content"]
    p2 = json.dumps(planner.seen[1])
    assert "Planner trace one." in p2 and "Motion trace one." not in p2
    assert "MOTION EXECUTION RESULT" in p2 and "completed" in p2
    assert "SYSTEM 1 PROPOSAL" in p2 and "left_gripper" in p2
    assert not any(it.get("type") == "function_call" for it in planner.seen[1])
    m2 = json.dumps(motion.seen[1])
    assert "Motion trace one." in m2 and "Planner trace one." not in m2
    assert any(it.get("type") == "function_call_output" for it in motion.seen[1])
    assert "Call done" in motion.seen[1][-1]["content"]

    log = Path(outcome.log_dir)
    for name in ("requests/planner_0000.json", "responses/planner_0000.json",
                 "requests/request_0001.json", "responses/response_0001.json",
                 "requests/planner_0002.json", "responses/response_0003.json",
                 "planner_history.json", "motion_history.json"):
        assert (log / name).is_file(), name
    trajectory = json.loads((log / "motion_0001.json").read_text())
    assert trajectory["planner_call_index"] == 0 and trajectory["motion_call_index"] == 1
    assert len(trajectory["plan"]["q_path"]) == trajectory["execution"]["steps_executed"]
    assert trajectory["result"]["ok"]
    text = (log / "transcript.txt").read_text()
    assert "Planner trace one." in text and "Motion trace one." in text
    assert "[plan]" in text
    final_planner = json.loads((log / "planner_history.json").read_text())
    assert "session_ended" in final_planner[-1]["content"]


def test_gateway_rejection_is_fed_to_planner(tmp_path):
    runner, planner, motion = planning_runner(tmp_path, moves=[
        {"name": "move_to", "arguments": {"targets": {"left_x": 99}, "note": "bad"}},
        {"name": "done", "arguments": {}}])
    outcome = runner.run("test rejection feedback")
    assert outcome.status == "done" and outcome.rejections == 1 and outcome.waypoints == 0
    assert "rejected" in json.dumps(planner.seen[1])
    assert "99" in json.dumps(planner.seen[1])
    assert json.loads((Path(outcome.log_dir) / "motion_0001.json").read_text())["plan"] is None


def test_complete_pick_and_place_through_ik(tmp_path):
    plans = [{"text": step["arguments"].get("note", "The goal is complete. Call done.")}
             for step in DEFAULT_SIM_SCRIPT]
    runner, planner, motion = planning_runner(tmp_path, plans=plans, moves=DEFAULT_SIM_SCRIPT)
    outcome = runner.run("Stack the blue block on the green block.")
    assert outcome.status == "done" and outcome.rejections == 0
    assert outcome.llm_calls == 2 * len(DEFAULT_SIM_SCRIPT) and outcome.waypoints > 100
    blue, green = runner.sim_world.objects["blue block"], runner.sim_world.objects["green block"]
    assert blue.held_by is None and abs(blue.pos[2] - green.pos[2] - .03) < 1e-6
    assert len(list(Path(outcome.log_dir).glob("motion_*.json"))) == len(DEFAULT_SIM_SCRIPT)  # includes history


def test_cli_planning_with_scripted_models(tmp_path):
    plans, moves = tmp_path / "plans.json", tmp_path / "moves.json"
    plans.write_text(json.dumps([{"text": "The test goal is complete. Call done."}]))
    moves.write_text(json.dumps([{"name": "done", "arguments": {}}]))
    args = build_parser().parse_args([
        "run", "--planning", "--sim", "--fast-sim", "--no-home", "--goal", "test CLI",
        "--planner-backend", "scripted", "--set", f"planning.planner.script_path={plans}",
        "--set", "astra.backend=scripted", "--script", str(moves), "--log-dir", str(tmp_path / "runs")])
    assert cmd_run(args) == 0
    # a finished run is filed under runs/succeeded/ or runs/recycled/, so search one level deeper
    summary = json.loads(next((tmp_path / "runs").glob("*/*/summary.json")).read_text())
    assert summary["status"] == "done" and summary["llm_calls"] == 2


@pytest.mark.parametrize("plan", [{"text": ""}, {"name": "move_to", "arguments": {"targets": {"left_x": .4}}}])
def test_invalid_planner_output_never_reaches_motion(tmp_path, plan):
    runner, planner, motion = planning_runner(tmp_path, plans=[plan])
    outcome = runner.run("invalid planner")
    assert outcome.status == "error" and "text only" in outcome.error
    assert outcome.llm_calls == 1 and outcome.planner_calls == 1 and outcome.motion_calls == 0
    assert not motion.seen and outcome.waypoints == 0
    assert outcome.usage["total_tokens"] == 15
    assert (Path(outcome.log_dir) / "responses/planner_0000.json").is_file()


def test_motion_failure_keeps_successful_plan_and_both_attempts(tmp_path):
    runner, planner, motion = planning_runner(tmp_path)
    def fail(_):
        raise RuntimeError("offline motion failure")
    motion.create = fail
    outcome = runner.run("failed motion")
    assert outcome.status == "error" and outcome.llm_calls == 2
    assert outcome.planner_calls == outcome.motion_calls == 1
    assert outcome.usage["total_tokens"] == 15
    log = Path(outcome.log_dir)
    assert "offline motion failure" in (log / "responses/response_0001.json").read_text()
    assert "Close the left gripper" in (log / "responses/planner_0000.json").read_text()


def test_planner_failure_is_logged(tmp_path):
    runner, planner, motion = planning_runner(tmp_path)
    def fail(_):
        raise RuntimeError("offline planner failure")
    planner.create = fail
    outcome = runner.run("failed plan")
    assert outcome.status == "error" and outcome.llm_calls == outcome.planner_calls == 1
    assert not motion.seen
    assert "offline planner failure" in (Path(outcome.log_dir) / "responses/planner_0000.json").read_text()


@pytest.mark.parametrize("budget,expected", [(0, 0), (1, 0), (2, 2), (3, 2)])
def test_budget_counts_both_models(tmp_path, budget, expected):
    runner, planner, motion = planning_runner(tmp_path, **{"limits.max_llm_calls": budget})
    outcome = runner.run("budget")
    assert outcome.status == "budget_exhausted" and outcome.llm_calls == expected
    assert len(planner.seen) == len(motion.seen) == expected // 2


def test_estop_during_planning_prevents_motion_inference(tmp_path):
    runner, planner, motion = planning_runner(tmp_path)
    original = planner.create
    def stop(items):
        response = original(items)
        runner.request_estop()
        return response
    planner.create = stop
    outcome = runner.run("stop")
    assert outcome.status == "estop" and outcome.llm_calls == 1 and not motion.seen


def test_fresh_run_has_no_previous_conversation(tmp_path):
    runner, planner, motion = planning_runner(tmp_path,
        plans=[{"text": "Unique first plan. Call done."}, {"text": "Unique second plan. Call done."}],
        moves=[{"name": "done", "arguments": {}}] * 2)
    assert runner.run("first goal").status == "done"
    assert runner.run("second goal").status == "done"
    assert "Unique first plan" not in json.dumps(planner.seen[1])
    assert "Unique first plan" not in json.dumps(motion.seen[1])
    assert "first goal" not in json.dumps(planner.seen[1])


def test_optional_motion_images_and_planner_image_window(tmp_path):
    runner, planner, motion = planning_runner(tmp_path)
    runner.cfg.planning.motion_images = True
    runner.cfg.planning.planner.image_history = 1
    assert runner.run("images").status == "done"
    assert count_images(motion.seen[0]) == 3
    assert count_images(planner.seen[1]) == 3
    assert "Planner trace one" in json.dumps(planner.seen[1])


def test_cli_defaults_and_text_only_planner_contract():
    args = build_parser().parse_args(["run", "--planning", "--goal", "test"])
    cfg = _config_from_args(args)
    assert cfg.planning.enabled and cfg.astra.backend == "openai" and cfg.astra.model == "gpt-6-astra"
    args = build_parser().parse_args(["run", "--goal", "test", "--planner-model", "custom/vision",
                                     "--planner-effort", "medium"])
    cfg = _config_from_args(args)
    assert cfg.planning.enabled and cfg.planning.planner.model == "custom/vision"
    assert cfg.planning.planner.reasoning_effort == "medium"
    cfg.planning.planner.actions_only = True
    cfg.planning.planner.tool_choice = "required"
    effective = planner_config(cfg)
    assert not effective.actions_only and effective.tool_choice == "none"
    cfg = load_config(None, {"planning.enabled": True, "planning.planner.model": "custom/vision"})
    assert cfg.planning.planner.reasoning_effort == "high"


@pytest.mark.parametrize("backend,cls", [("openai", OpenAIAstraClient), ("openrouter", OpenRouterChatClient)])
def test_planner_wire_has_no_tools(monkeypatch, backend, cls):
    import openai
    monkeypatch.setenv("OPENAI_API_KEY", "test")
    monkeypatch.setenv("OPENROUTER_API_KEY", "test")
    monkeypatch.setattr(openai, "OpenAI", lambda **_: SimpleNamespace())
    client = cls(AstraConfig(backend=backend, actions_only=False, tool_choice="none", reasoning_effort="high"), [])
    request = client.build_request_dict([{"role": "user", "content": "Plan."}])
    assert all(key not in request for key in ("tools", "tool_choice", "parallel_tool_calls"))
