import json
from pathlib import Path

import numpy as np
import pytest

from astra_yam.astra_client import DEFAULT_SIM_SCRIPT, ScriptedAstraClient
from astra_yam.config import PipelineConfig, load_config
from astra_yam.embodiment import build_tools
from astra_yam.kinematics import ArmKinematics
from astra_yam.session import OperatorInput, TrialRunner
from astra_yam.sim import SimCameraSource, SimWorld, SimYamRobot


def _make(tmp_path, script=None, **overrides):
    cfg = load_config(None, {"robot.backend": "sim", "cameras.backend": "sim", "astra.backend": "scripted",
                             "astra.actions_only": False,  # historical language-mode fixture
                             "log_dir": str(tmp_path), **overrides})
    kin = ArmKinematics(limit_margin=cfg.motion.joint_limit_margin_rad)
    world = SimWorld(kin)
    q0 = np.zeros(14)
    q0[0:6], q0[7:13] = cfg.robot.home_joints_left, cfg.robot.home_joints_right
    q0[6] = q0[13] = 1.0
    robot = SimYamRobot(initial_q=q0, world=world)
    world.update(q0)
    cams = SimCameraSource(world)
    astra = ScriptedAstraClient(script=script, tools=build_tools(
        cfg.bounds, cfg.prompts_path, reactive=cfg.reactive.enabled, actions_only=cfg.astra.actions_only),
        actions_only=cfg.astra.actions_only)
    runner = TrialRunner(cfg, robot, cams, kin, astra, realtime=False, sim_world=world, verbose=False)
    return cfg, runner, world, astra


def test_scripted_pick_and_place_completes(tmp_path):
    cfg, runner, world, astra = _make(tmp_path)
    outcome = runner.run("pick up blue and place ontop of green block")
    assert outcome.status == "done", outcome
    assert outcome.llm_calls == len(DEFAULT_SIM_SCRIPT) and outcome.waypoints > 100
    blue, green = world.objects["blue block"], world.objects["green block"]
    assert blue.held_by is None
    assert np.linalg.norm(blue.pos[:2] - green.pos[:2]) < 0.02
    assert abs(blue.pos[2] - (green.pos[2] + 0.03)) < 1e-6      # stacked on top
    assert "grasped by left" in " ".join(world.events) and "released by left" in " ".join(world.events)

    log = Path(outcome.log_dir)
    assert (log / "summary.json").exists() and (log / "notes.md").exists()
    frames = sorted((log / "frames").glob("*.jpg"))
    assert len(frames) == 3 * len(DEFAULT_SIM_SCRIPT)   # 3 cameras per observation, one obs per call
    req0 = json.loads((log / "requests" / "request_0000.json").read_text())
    assert req0["model"] == astra.model and req0["store"] is False and req0["include"] == ["reasoning.encrypted_content"]
    assert [i.get("role") for i in req0["input"]] == ["system", "user", "user"]
    assert req0["input"][1]["content"] == "Goal: pick up blue and place ontop of green block"
    imgs = [p for p in req0["input"][2]["content"] if p["type"] == "input_image"]
    assert len(imgs) == 3 and all(p["image_url"].startswith("data:image/jpeg;base64,$blob:") for p in imgs)
    # the last request carries the full conversation: function_call + function_call_output pairs
    last = sorted((log / "requests").glob("request_*.json"))[-1]
    items = json.loads(last.read_text())["input"]
    calls = [i for i in items if i.get("type") == "function_call"]
    outs = [i for i in items if i.get("type") == "function_call_output"]
    assert len(calls) == len(outs) == len(DEFAULT_SIM_SCRIPT) - 1
    assert all(json.loads(o["output"])["ok"] for o in outs)
    assert json.loads(outs[0]["output"]).keys() == {"ok", "steps", "status", "cadence_hz"}
    # waypoints-remaining bookkeeping in the observation text
    obs_texts = [i["content"][0]["text"] for i in items if i.get("role") == "user" and isinstance(i.get("content"), list)]
    remaining = [int(t.split("remaining in this session: ")[1].rstrip(".")) for t in obs_texts]
    assert remaining[0] == cfg.limits.max_waypoints and remaining == sorted(remaining, reverse=True)
    assert f"(step {cfg.limits.max_waypoints - remaining[-1]})" in items[-1]["content"][1]["text"]


def test_llm_budget_ends_session(tmp_path):
    script = [{"name": "move_to", "arguments": {"targets": {"left_gripper": 0.9}, "note": "n"}}] * 10
    cfg, runner, world, astra = _make(tmp_path, script=script, **{"limits.max_llm_calls": 3})
    outcome = runner.run("wiggle")
    assert outcome.status == "budget_exhausted" and outcome.llm_calls == 3


def test_time_limit_ends_session(tmp_path):
    script = [{"name": "move_to", "arguments": {"targets": {"left_gripper": 0.9}, "note": "n"}}] * 10
    cfg, runner, world, astra = _make(tmp_path, script=script, **{"limits.max_trial_seconds": 0.0})
    outcome = runner.run("wiggle")
    assert outcome.status == "timeout" and outcome.llm_calls == 0


def test_repeated_rejections_end_session(tmp_path):
    script = [{"name": "move_to", "arguments": {"targets": {"left_x": 0.9}, "note": "bad"}}] * 10
    cfg, runner, world, astra = _make(tmp_path, script=script, **{"limits.max_consecutive_rejections": 3})
    outcome = runner.run("bad moves")
    assert outcome.status == "too_many_rejections" and outcome.llm_calls == 3 and outcome.waypoints == 0


def test_strict_gateway_ends_on_first_rejection(tmp_path):
    script = [{"name": "move_to", "arguments": {"targets": {"left_pitch": 0.3}, "note": "bad"}}] * 3
    cfg, runner, world, astra = _make(tmp_path, script=script,
                                      **{"limits.strict_gateway": True, "bounds.pitch": [0.0, 0.0]})
    outcome = runner.run("bad move")
    assert outcome.status == "rejected_packet" and outcome.llm_calls == 1 and "pinned" in outcome.reason


def test_give_up_and_invalid_json_arguments(tmp_path):
    script = [{"name": "move_to", "arguments": "{not json"},
              {"name": "give_up", "arguments": {"reason": "cannot see the block", "hindsight": "top camera is mirrored"}}]
    cfg, runner, world, astra = _make(tmp_path, script=script)
    outcome = runner.run("impossible")
    assert outcome.status == "give_up" and outcome.reason == "cannot see the block" and outcome.hindsight == "top camera is mirrored"
    transcript = [json.loads(l) for l in (Path(outcome.log_dir) / "transcript.jsonl").read_text().splitlines()]
    bad = [e for e in transcript if e["kind"] == "tool_call" and e.get("result", {}).get("status") == "rejected"]
    assert bad and "not valid JSON" in bad[0]["result"]["reason"]


def test_operator_stop_via_feedback_file(tmp_path):
    fb = tmp_path / "feedback.txt"
    fb.write_text("")
    op = OperatorInput(use_stdin=False, file_path=str(fb))
    script = [{"name": "move_to", "arguments": {"targets": {"left_gripper": 0.9}, "note": "n"}}] * 5
    cfg, runner, world, astra = _make(tmp_path / "logs", script=script)
    runner.operator = op
    import time
    with open(fb, "a") as f:
        f.write("please be careful near the plate\n/stop\n")
    time.sleep(0.6)
    outcome = runner.run("wiggle")
    op.close()
    assert outcome.status == "operator_stop" and outcome.llm_calls == 0


def test_image_history_pruning_in_requests(tmp_path):
    script = [{"name": "move_to", "arguments": {"targets": {"left_gripper": 0.9 - 0.1 * i}, "note": "n"}} for i in range(4)]
    cfg, runner, world, astra = _make(tmp_path, script=script, **{"astra.image_history": 2, "limits.max_llm_calls": 4})
    outcome = runner.run("wiggle")
    last = sorted((Path(outcome.log_dir) / "requests").glob("request_*.json"))[-1]
    items = json.loads(last.read_text())["input"]
    n_img = sum(1 for i in items if isinstance(i.get("content"), list)
                for p in i["content"] if p.get("type") == "input_image")
    assert n_img == 2 * 3


def test_max_calls_is_announced_in_system_prompt(tmp_path):
    cfg, runner, world, astra = _make(tmp_path, **{"limits.max_llm_calls": 7})
    assert "budget of 7 LLM calls" in runner.system_prompt


class _RecordingHooks:
    def __init__(self):
        self.calls = []

    def on_trial_start(self, goal, q0):
        self.calls.append(("start", goal))

    def on_observation(self, step, q, eef, frames, remaining):
        self.calls.append(("obs", step, remaining))

    def on_plan(self, plan):
        self.calls.append(("plan", plan.steps))

    def on_tool_call(self, name, args, payload, plan):
        self.calls.append(("tool", name, payload["ok"]))
        raise RuntimeError("hooks must never break the trial")

    def on_end(self, outcome):
        self.calls.append(("end", outcome.status))


def test_hooks_are_called_and_failures_are_contained(tmp_path):
    hooks = _RecordingHooks()
    cfg, runner, world, astra = _make(tmp_path)
    runner.hooks = hooks
    outcome = runner.run("pick up blue and place ontop of green block")
    assert outcome.status == "done"
    kinds = [c[0] for c in hooks.calls]
    assert kinds[0] == "start" and kinds[-1] == "end"
    assert kinds.count("plan") == len(DEFAULT_SIM_SCRIPT) - 1          # one plan per executed move
    assert kinds.count("obs") == len(DEFAULT_SIM_SCRIPT)               # initial + one per move
    assert ("tool", "done", True) in hooks.calls
