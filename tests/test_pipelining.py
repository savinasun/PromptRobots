"""Overlapping the two models and buying more motion per inference call.

Profiling runs/recycled/20260921_155454_fail: the System 2 planner was 79% of the wall clock, System 1 a
further 15% (almost all of it round trip, not generation), and the arm moved for 4% of the run. The three
things exercised here attack exactly that: a whole stroke per `move_to`, several System 1 turns per plan,
and System 1's request leaving while System 2 is still writing.
"""
import json
import threading
import time
from pathlib import Path

import numpy as np
import pytest

from utils.astra_client import AstraResponse, _extract_messages, _parse_function_calls
from utils.config import DIM_NAMES, load_config
from utils.embodiment import build_policy_prompt, build_tools
from utils.planning import extract_instruction
from test_session_sim import _make

FENCE = "NEXT INSTRUCTION\n{}\nEND INSTRUCTION\n\nSCENE\n{}\n\nPROGRESS\nstep {} of 4."


def fenced(instruction: str, scene: str = "the block sits left of the gripper.", step: int = 1) -> str:
    return FENCE.format(instruction, scene, step)


class FakePlanner:
    """A System 2 that streams: the instruction block first, then a slow tail of its own reasoning.

    `tail_s` is the point of the fixture. It is the window the motion request is supposed to leave in.
    """

    def __init__(self, messages, tail_s: float = 0.25, final_messages=None):
        self.messages = list(messages)
        self.final_messages = list(final_messages) if final_messages is not None else None
        self.tail_s = tail_s
        self.model = "system-2-fake"
        self.seen = []
        self.finished_at = []

    def build_request_dict(self, items):
        return {"model": self.model, "input": items}

    def _response(self, text):
        items = [{"type": "message", "id": f"msg_{len(self.seen)}", "role": "assistant",
                  "content": [{"type": "output_text", "text": text}], "status": "completed"}]
        return AstraResponse(output_items=items, function_calls=_parse_function_calls(items),
                             messages=_extract_messages(items), reasoning=[],
                             usage={"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
                             response_id=items[0]["id"], elapsed_s=0.0, model=self.model)

    def create(self, items):
        self.seen.append([dict(i) for i in items])
        text = self.messages[min(len(self.seen) - 1, len(self.messages) - 1)]
        self.finished_at.append(time.perf_counter())
        return self._response(text)

    def create_streamed(self, items, on_delta):
        self.seen.append([dict(i) for i in items])
        index = min(len(self.seen) - 1, len(self.messages) - 1)
        text = self.messages[index]
        on_delta("reset", "", "")
        head, _, tail = text.partition("END INSTRUCTION")
        on_delta("message", head + "END INSTRUCTION", "msg")
        time.sleep(self.tail_s)                 # System 2 keeps writing after the fence closes
        on_delta("message", tail, "msg")
        self.finished_at.append(time.perf_counter())
        # What finally comes back may differ from what streamed - that is what the guard is for.
        if self.final_messages is not None:
            text = self.final_messages[min(index, len(self.final_messages) - 1)]
        return self._response(text)


class TimedMotion:
    """A System 1 that records when each request started, so overlap is measurable rather than asserted."""

    def __init__(self, script, model="system-1-fake"):
        self.script = list(script)
        self.model = model
        self.started_at = []
        self.seen = []
        self.on_call = None         # fires before each answer, to inject events mid-run
        self._n = 0

    def build_request_dict(self, items):
        return {"model": self.model, "input": items}

    def create(self, items):
        self.started_at.append(time.perf_counter())
        self.seen.append([dict(i) for i in items])
        if self.on_call is not None:
            self.on_call(len(self.seen))
        step = self.script[self._n] if self._n < len(self.script) else {"name": "give_up", "arguments": {}}
        self._n += 1
        item = {"type": "function_call", "id": f"fc_{self._n}", "call_id": f"call_{self._n}",
                "name": step["name"], "arguments": json.dumps(step.get("arguments", {})), "status": "completed"}
        return AstraResponse(output_items=[item], function_calls=_parse_function_calls([item]), messages=[],
                             reasoning=[], usage={"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
                             response_id=item["id"], elapsed_s=0.0, model=self.model)


def hold_all(**changed):
    """One strict-schema waypoint: every dimension declared, null for the ones that hold."""
    return {d: changed.get(d) for d in DIM_NAMES}


def move(*waypoints):
    return {"name": "move_to", "arguments": {"waypoints": [hold_all(**w) for w in waypoints]}}


def planning_runner(tmp_path, plans, moves, tail_s=0.25, final_plans=None, **overrides):
    cfg, runner, world, _ = _make(tmp_path, **{
        "robot.home_at_start": False, "planning.enabled": True, "planning.planner.backend": "scripted",
        "motion.max_waypoints_per_call": 4, **overrides})
    runner.system_prompt = build_policy_prompt(cfg)
    runner.tools = build_tools(cfg.bounds, cfg.prompts_path, reactive=cfg.reactive.enabled,
                               actions_only=cfg.astra.actions_only,
                               max_waypoints=cfg.motion.max_waypoints_per_call)
    planner = FakePlanner(plans, tail_s=tail_s, final_messages=final_plans)
    motion = TimedMotion(moves)
    runner.planner, runner.astra = planner, motion
    return cfg, runner, planner, motion


def events(outcome, kind):
    lines = (Path(outcome.log_dir) / "transcript.jsonl").read_text().splitlines()
    return [e for e in (json.loads(l) for l in lines) if e["kind"] == kind]


# --------------------------------------------------------------------- fences
@pytest.mark.parametrize("text,expected", [
    ("NEXT INSTRUCTION\nPush left.\nEND INSTRUCTION\n\nSCENE\nclutter.", "Push left."),
    ("**NEXT INSTRUCTION**\nPush left.\n**END INSTRUCTION**", "Push left."),
    ("next instruction: Push left. end instruction", "Push left."),
    ("NEXT INSTRUCTION\nstill writing, no closing fence yet", None),
    ("NEXT INSTRUCTION\n\nEND INSTRUCTION", None),
    ("A planner that ignores the format entirely.", None),
])
def test_instruction_fence_extraction(text, expected):
    assert extract_instruction(text) == expected


def test_unfenced_planner_still_works_and_hands_over_its_whole_message(tmp_path):
    cfg, runner, planner, motion = planning_runner(
        tmp_path, plans=["No fences here. Close the left gripper."], moves=[move({"left_gripper": 0.6})], tail_s=0.0)
    outcome = runner.run("unfenced")
    assert outcome.waypoints > 0
    assert "No fences here" in motion.seen[0][-1]["content"]
    assert not events(outcome, "speculative_motion")     # nothing to speculate on without a closing fence


# ------------------------------------------------------- early motion dispatch
def test_motion_request_leaves_while_the_planner_is_still_writing(tmp_path):
    cfg, runner, planner, motion = planning_runner(
        tmp_path, plans=[fenced("Close the left gripper a little.")],
        moves=[move({"left_gripper": 0.6}), {"name": "done", "arguments": {}}], tail_s=0.4)
    outcome = runner.run("speculate")
    assert outcome.status == "done", outcome

    # The motion request started before System 2's stream ended: that gap is the latency that disappears.
    assert motion.started_at[0] < planner.finished_at[0]
    spec = events(outcome, "speculative_motion")
    assert len(spec) == 2 and spec[0]["hidden_s"] >= 0.3         # one per cycle: the move and the done
    assert "speculative motion dispatch: 2/2 reused" in (Path(outcome.log_dir) / "transcript.txt").read_text()


def test_speculation_is_discarded_when_the_final_instruction_differs(tmp_path):
    cfg, runner, planner, motion = planning_runner(
        tmp_path, plans=[fenced("Close the left gripper a little.")],
        final_plans=[fenced("Actually, open the left gripper instead.")],
        moves=[move({"left_gripper": 0.6}), {"name": "done", "arguments": {}}], tail_s=0.1)
    outcome = runner.run("revised")
    assert outcome.status == "done", outcome
    assert events(outcome, "speculative_discarded")
    assert not events(outcome, "speculative_motion")
    # seen[0] is the speculative request that was thrown away; the one the loop acted on is the re-request,
    # and it carries the instruction System 2 finally settled on.
    assert "Close the left gripper" in motion.seen[0][-1]["content"]
    assert "open the left gripper" in motion.seen[1][-1]["content"]


def test_speculation_off_keeps_the_cycle_sequential(tmp_path):
    cfg, runner, planner, motion = planning_runner(
        tmp_path, plans=[fenced("Close the left gripper a little.")],
        moves=[move({"left_gripper": 0.6}), {"name": "done", "arguments": {}}], tail_s=0.2,
        **{"planning.speculative_motion": False})
    outcome = runner.run("sequential")
    assert outcome.status == "done", outcome
    assert motion.started_at[0] > planner.finished_at[0]
    assert not events(outcome, "speculative_motion")


def test_system_1_receives_the_instruction_alone(tmp_path):
    cfg, runner, planner, motion = planning_runner(
        tmp_path, plans=[fenced("Close the left gripper a little.", scene="a secret scene note")],
        moves=[move({"left_gripper": 0.6}), {"name": "done", "arguments": {}}], tail_s=0.0)
    runner.run("scoped handoff")
    handed = motion.seen[0][-1]["content"]
    assert "Close the left gripper a little." in handed
    assert "a secret scene note" not in handed and "PROGRESS" not in handed
    # System 2 keeps its own full message in its own branch.
    assert "a secret scene note" in json.dumps(json.loads(
        (Path(runner.log_root).glob("*/*/planner_history.json").__next__()).read_text()))


# ----------------------------------------------- several motion turns per plan
def test_motion_calls_per_plan_amortizes_the_planner(tmp_path):
    cfg, runner, planner, motion = planning_runner(
        tmp_path, plans=[fenced("Close the left gripper in three small steps.")],
        moves=[move({"left_gripper": 0.8}), move({"left_gripper": 0.6}), move({"left_gripper": 0.4}),
               {"name": "done", "arguments": {}}],
        tail_s=0.0, **{"planning.motion_calls_per_plan": 3})
    outcome = runner.run("amortize")
    assert outcome.status == "done"
    # Four motion turns over two plans instead of four plans.
    assert outcome.motion_calls == 4 and outcome.planner_calls == 2
    cycles = [e["motion_in_cycle"] for e in events(outcome, "motion_request")]
    assert cycles == [1, 2, 3, 1]
    # Turns 2 and 3 are told they are continuing, and still get a fresh observation each.
    assert "Continuing under the same System 2 plan" in json.dumps(motion.seen[1])
    assert "Continuing under the same System 2 plan" not in json.dumps(motion.seen[0][-1])
    assert len(events(outcome, "observation")) == outcome.motion_calls


def test_a_rejected_action_recalls_the_planner_immediately(tmp_path):
    cfg, runner, planner, motion = planning_runner(
        tmp_path, plans=[fenced("Reach far left."), fenced("Close the left gripper instead.")],
        moves=[move({"left_x": 99.0}), move({"left_gripper": 0.6}), {"name": "done", "arguments": {}}],
        tail_s=0.0, **{"planning.motion_calls_per_plan": 3})
    outcome = runner.run("rejection recalls the planner")
    assert outcome.rejections == 1
    # Three motion turns would normally sit under one plan. The rejection ends that cycle after one turn,
    # so the second plan is written against the failure instead of two more turns being spent on a bad one.
    assert outcome.motion_calls == 3 and outcome.planner_calls == 2
    assert [e["motion_in_cycle"] for e in events(outcome, "motion_request")] == [1, 1, 2]
    assert "99" in json.dumps(planner.seen[1]), "System 2 is shown the rejection it has to plan around"


def test_operator_feedback_recalls_the_planner(tmp_path):
    cfg, runner, planner, motion = planning_runner(
        tmp_path, plans=[fenced("Close the left gripper.")] * 3,
        moves=[move({"left_gripper": 0.8}), move({"left_gripper": 0.6}), {"name": "done", "arguments": {}}],
        tail_s=0.0, **{"planning.motion_calls_per_plan": 5})
    from utils.session import OperatorInput
    operator = OperatorInput(use_stdin=False)
    runner.operator = operator
    # Guidance arrives after the first plan is already being carried out.
    motion.on_call = lambda n: operator._q.put("use the right arm instead") if n == 1 else None
    outcome = runner.run("feedback recalls the planner")
    assert outcome.motion_calls == 3
    # Five motion turns were allowed per plan; the new guidance still brings System 2 back for turn 2.
    assert outcome.planner_calls == 2, "new guidance is System 2's to weigh, not System 1's"
    assert "use the right arm instead" in json.dumps(planner.seen[1])
    # and System 2 weighs it *before* the next motion, not a turn later.
    assert "use the right arm instead" in json.dumps(motion.seen[1])


# ------------------------------------------------------------ waypoint strokes
def test_a_stroke_is_one_call_one_chunk_and_one_continuous_motion(tmp_path):
    cfg, runner, planner, motion = planning_runner(
        tmp_path, plans=[fenced("Lower, sweep right, lift clear.")],
        moves=[move({"left_z": 0.12}, {"left_y": -0.10}, {"left_z": 0.22}), {"name": "done", "arguments": {}}],
        tail_s=0.0)
    outcome = runner.run("stroke")
    assert outcome.status == "done", outcome
    chunks = events(outcome, "chunk")
    assert len(chunks) == 1 and chunks[0]["requested"] == 3
    call = events(outcome, "tool_call")[0]
    assert call["result"]["waypoints_requested"] == 3
    legs = call["plan"]["resolved_waypoints"]
    assert len(legs) == 3
    # Each leg inherits what the previous one left alone: the sweep keeps the lowered z.
    assert abs(legs[1]["left_z"] - 0.12) < 1e-9 and abs(legs[1]["left_y"] + 0.10) < 1e-9
    assert abs(legs[2]["left_y"] + 0.10) < 1e-9
    # One inference call bought all three legs of motion.
    assert chunks[0]["executed"] == chunks[0]["predicted"] > 100
    assert outcome.motion_calls == 2


def test_an_unreachable_leg_stops_the_whole_stroke_before_any_motion(tmp_path):
    cfg, runner, planner, motion = planning_runner(
        tmp_path, plans=[fenced("Lower then reach off the table.")],
        moves=[move({"left_z": 0.12}, {"left_x": 99.0}), {"name": "done", "arguments": {}}], tail_s=0.0)
    outcome = runner.run("bad leg")
    assert outcome.rejections == 1 and outcome.waypoints == 0
    assert events(outcome, "tool_call")[0]["result"]["status"] == "rejected"


def test_more_waypoints_than_allowed_are_refused_not_truncated(tmp_path):
    cfg, runner, planner, motion = planning_runner(
        tmp_path, plans=[fenced("Too many legs.")],
        moves=[move(*[{"left_z": 0.12 + 0.01 * i} for i in range(6)]), {"name": "done", "arguments": {}}],
        tail_s=0.0, **{"motion.max_waypoints_per_call": 4})
    outcome = runner.run("too many")
    assert outcome.status == "invalid_action_output"
    assert "at most 4 waypoints" in outcome.reason


def test_a_planner_that_cannot_stream_falls_back_instead_of_failing(tmp_path):
    """Speculation asks for a stream the trial did not previously need; refusing one must not end the run."""
    cfg, runner, planner, motion = planning_runner(
        tmp_path, plans=[fenced("Close the left gripper.")],
        moves=[move({"left_gripper": 0.6}), {"name": "done", "arguments": {}}], tail_s=0.0)
    calls = {"streamed": 0}

    def refuse(items, on_delta):
        calls["streamed"] += 1
        on_delta("reset", "", "")
        raise RuntimeError("this endpoint does not support streaming")

    planner.create_streamed = refuse
    outcome = runner.run("no streaming")
    assert outcome.status == "done", outcome
    assert outcome.planner_calls == 2 and outcome.waypoints > 0
    assert calls["streamed"] == 1, "streaming is abandoned for the rest of the run, not retried every cycle"
    assert events(outcome, "planner_stream_unavailable")
    assert not events(outcome, "speculative_motion")


# ------------------------------------------------- System 1 says nothing extra
def test_planning_puts_system_1_on_the_action_only_contract(tmp_path):
    # Explicitly asking for language mode is overridden while planning is on: System 2 owns the narration.
    cfg = load_config(None, {"planning.enabled": True, "astra.actions_only": False})
    assert cfg.astra.actions_only
    tools = build_tools(cfg.bounds, cfg.prompts_path, actions_only=cfg.astra.actions_only,
                        max_waypoints=cfg.motion.max_waypoints_per_call)
    move_to = next(t for t in tools if t["name"] == "move_to")
    assert "note" not in move_to["parameters"]["properties"]
    # done/give_up keep one reason: the trial's only record of why it ended.
    assert all(list(t["parameters"]["properties"]) == ["reason"]
               for t in tools if t["name"] in ("done", "give_up"))
    opted_out = load_config(None, {"planning.enabled": True, "planning.motion_actions_only": False,
                                   "astra.actions_only": False})
    assert not opted_out.astra.actions_only
    assert not load_config(None, {"astra.actions_only": False}).astra.actions_only   # planning off: untouched


def test_the_cli_puts_the_same_move_to_schema_on_the_wire_as_the_session_checks(tmp_path):
    """The client's tools are what the model actually sees; the session checks its own copy. One schema."""
    from utils.cli import _config_from_args, build_parser
    from utils.session import TrialRunner
    args = build_parser().parse_args(["run", "--planning", "--goal", "x", "--config",
                                      "configs/skild_yam_8_loop.yaml"])
    cfg = _config_from_args(args)
    assert cfg.motion.max_waypoints_per_call > 1
    wire = build_tools(cfg.bounds, cfg.prompts_path, reactive=cfg.reactive.enabled,
                       actions_only=cfg.astra.actions_only, max_waypoints=cfg.motion.max_waypoints_per_call)
    move_to = next(t for t in wire if t["name"] == "move_to")
    assert set(move_to["parameters"]["properties"]) == {"waypoints"}
    assert move_to["parameters"]["properties"]["waypoints"]["maxItems"] == cfg.motion.max_waypoints_per_call


def test_a_silent_give_up_is_impossible(tmp_path):
    """20260921_191247_fail: System 1 quit with a bare `{}` and the cause had to be dug out of the logs."""
    cfg, runner, planner, motion = planning_runner(
        tmp_path, plans=[fenced("Push the bottle left.")],
        moves=[{"name": "give_up", "arguments": {"reason": "the gateway rejected every approach to the pile"}}],
        tail_s=0.0)
    outcome = runner.run("give up with a reason")
    assert outcome.status == "give_up"
    assert outcome.reason == "the gateway rejected every approach to the pile"
    notes = (Path(outcome.log_dir) / "notes.md").read_text()
    assert "GIVE_UP: the gateway rejected every approach" in notes
    assert "the gateway rejected every approach" in (Path(outcome.log_dir) / "transcript.txt").read_text()


def test_the_configured_effort_survives_the_action_only_contract(tmp_path):
    """20260921_191247_fail: --effort high silently became effort=low once System 1 went actions-only."""
    import openai
    from utils.astra_client import OpenAIAstraClient
    from utils.cli import _config_from_args, build_parser
    from types import SimpleNamespace
    import os
    os.environ.setdefault("OPENAI_API_KEY", "test-only")
    cfg = _config_from_args(build_parser().parse_args(
        ["run", "--planning", "--goal", "x", "--config", "configs/skild_yam_8.yaml", "--effort", "high"]))
    assert cfg.astra.actions_only and cfg.astra.reasoning_effort == "high"
    real, openai.OpenAI = openai.OpenAI, lambda **_: SimpleNamespace()
    try:
        client = OpenAIAstraClient(cfg.astra, build_tools(
            cfg.bounds, cfg.prompts_path, actions_only=True,
            max_waypoints=cfg.motion.max_waypoints_per_call))
        assert client.build_request_dict([])["reasoning"]["effort"] == "high"
    finally:
        openai.OpenAI = real


def test_a_system_1_summary_never_reaches_the_robot(tmp_path):
    """The protocol check runs before dispatch, so prose from System 1 stops the trial with nothing moved."""
    cfg, runner, planner, motion = planning_runner(
        tmp_path, plans=[fenced("Close the left gripper.")],
        moves=[{"name": "move_to", "arguments": {"waypoints": [hold_all(left_gripper=0.6)],
                                                 "note": "The planner reports the bottle is ahead."}}],
        tail_s=0.0)
    outcome = runner.run("no narration")
    assert outcome.status == "invalid_action_output" and outcome.waypoints == 0
    assert "language fields are forbidden" in outcome.reason
