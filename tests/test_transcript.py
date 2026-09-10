"""Every run leaves a plain-text transcript that can be read start to finish, reasoning included."""
import json
from pathlib import Path

import pytest

from astra_yam.astra_client import AstraResponse, _extract_reasoning
from astra_yam.config import AstraConfig


def _run(tmp_path, script=None, **overrides):
    from test_session_sim import _make

    cfg, runner, world, astra = _make(tmp_path, script=script, **overrides)
    outcome = runner.run("pick up blue and place ontop of green block")
    text = (Path(outcome.log_dir) / "transcript.txt").read_text()
    return outcome, text


def test_transcript_has_the_whole_conversation(tmp_path):
    outcome, text = _run(tmp_path)
    assert outcome.status == "done"
    for section in ("--- TRIAL ", "--- SYSTEM PROMPT ", "--- GOAL ", "--- OBSERVATION step 0 ",
                    "--- ASTRA call 1 ", "--- OUTCOME "):
        assert section in text, section
    # the system prompt verbatim, so a transcript records the prompt that produced the run
    assert "You are controlling a real robot embodiment named 'yam_arms'" in text
    assert "Do not give up easily" in text
    # every turn: reasoning, the call, its note, and what the gateway did with it
    assert "[reasoning] The blue block is left of centre" in text
    assert '[move_to]   {"left_x": 0.33' in text
    assert "[note]      Scripted: moving above the blue block." in text
    assert '[gateway]   {"ok": true' in text
    assert "[done]      Scripted pick-and-place finished." in text
    assert "[hindsight] none" in text
    # observations arrive as text plus a pointer to the saved frame, never as pixels
    assert "state[eef_state]: left_x=" in text and "frames/step_00000_top_cam.jpg" in text
    assert "base64" not in text and "$blob" not in text
    assert text.count("--- ASTRA call ") == outcome.llm_calls


def test_transcript_records_rejections_and_missing_reasoning(tmp_path):
    script = [{"name": "move_to", "arguments": {"targets": {"left_x": 99.0}, "note": "out of bounds"}},
              {"name": "give_up", "arguments": {"reason": "cannot reach", "hindsight": "check bounds first"}}]
    outcome, text = _run(tmp_path, script=script)
    assert outcome.status == "give_up"
    assert "[reasoning] (not returned as text" in text        # no summary in this scripted step
    assert '"status": "rejected"' in text and "outside its bounds" in text
    assert "[give_up]   cannot reach" in text and "[hindsight] check bounds first" in text


def test_operator_feedback_appears_in_the_transcript(tmp_path):
    from astra_yam.session import OperatorInput
    from test_session_sim import _make

    cfg, runner, world, astra = _make(tmp_path)
    runner.operator = OperatorInput(use_stdin=False)
    runner.operator.add_source(lambda: ["keep the mug upright"])
    outcome = runner.run("g")
    text = (Path(outcome.log_dir) / "transcript.txt").read_text()
    assert "--- OPERATOR FEEDBACK " in text and "keep the mug upright" in text


def test_extract_reasoning_reads_summary_and_content():
    items = [
        {"type": "reasoning", "summary": [{"type": "summary_text", "text": " first thought "}],
         "encrypted_content": "gAAAA-opaque"},
        {"type": "reasoning", "content": [{"type": "reasoning_text", "text": "second thought"}]},
        {"type": "reasoning", "encrypted_content": "gAAAA-only-encrypted"},   # nothing readable
        {"type": "function_call", "name": "move_to", "arguments": "{}"},
    ]
    assert _extract_reasoning(items) == ["first thought", "second thought"]
    assert _extract_reasoning([]) == []
    assert AstraResponse(output_items=[], function_calls=[], messages=[]).reasoning == []


def test_reasoning_summary_is_requested_and_dropped_when_rejected(monkeypatch):
    """`reasoning.summary` is what makes the trace readable; a model that rejects it must not kill the trial."""
    import openai

    from astra_yam.astra_client import OpenAIAstraClient

    class FakeItem:
        def model_dump(self, **kw):
            return {"type": "reasoning", "summary": [{"type": "summary_text", "text": "thinking"}]}

    class FakeResponses:
        def __init__(self):
            self.kwargs = []

        def create(self, **kw):
            self.kwargs.append(kw)
            if len(self.kwargs) == 1:
                raise ValueError("Unsupported parameter: 'reasoning.summary' is not supported with this model")
            return type("R", (), {"output": [FakeItem()], "usage": None, "id": "resp_1", "model": "m"})()

    fake = FakeResponses()
    monkeypatch.setattr(openai, "OpenAI", lambda **kw: type("C", (), {"responses": fake})())
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    client = OpenAIAstraClient(AstraConfig(reasoning_effort="high"), tools=[])

    resp = client.create([{"role": "user", "content": "hi"}])
    assert fake.kwargs[0]["reasoning"] == {"effort": "high", "summary": "auto"}
    assert fake.kwargs[1]["reasoning"] == {"effort": "high"}     # retried without it
    assert resp.reasoning == ["thinking"]
    client.create([{"role": "user", "content": "hi"}])
    assert fake.kwargs[2]["reasoning"] == {"effort": "high"}     # and stays off

    with pytest.raises(ValueError):                              # unrelated errors still propagate
        fake.create = lambda **kw: (_ for _ in ()).throw(ValueError("rate limited"))
        client.create([{"role": "user", "content": "hi"}])


def test_chunk_counters_per_move_and_totals(tmp_path):
    outcome, text = _run(tmp_path, **{"limits.max_waypoints": 3000})
    assert "[chunk]     90 waypoints predicted (90 Cartesian at 10 Hz), 90 executed (9.0 s)" in text
    assert "[counters]  move_to calls 1, waypoints 90 executed / 90 predicted, 2910 of 3000 budget left" in text
    assert "mean chunk 90.0 (min 90, max 90)" in text

    table = text.split("--- ACTION CHUNKS ")[1]
    assert "call  cartesian  predicted  executed  paced  seconds  status" in table
    n_moves = outcome.llm_calls - 1                        # the last scripted call is `done`
    assert f"{n_moves} move_to calls (0 rejected, 0 paced)" in table
    assert f"{outcome.waypoints} waypoints executed of {outcome.waypoints_predicted} predicted" in table
    assert "executed chunk size: mean" in table

    # the same counters end up in the machine-readable outputs
    assert outcome.chunk_sizes and sum(outcome.chunk_sizes) == outcome.waypoints == outcome.waypoints_predicted
    assert len(outcome.chunk_sizes) == n_moves
    events = [json.loads(l) for l in (Path(outcome.log_dir) / "transcript.jsonl").read_text().splitlines()]
    assert [e for e in events if e["kind"] == "chunk"][0]["predicted"] == 90


def test_joint_pacing_is_visible_in_the_chunk_line(tmp_path):
    """At 30 cm/s the Cartesian path exceeds max_joint_step_rad per waypoint, so the gateway subdivides it."""
    script = [{"name": "move_to", "arguments": {"targets": {"left_x": 0.40, "left_z": 0.30}, "note": "fast"}},
              {"name": "done", "arguments": {"summary": "s", "hindsight": "none"}}]
    outcome, text = _run(tmp_path, script=script, **{"motion.linear_speed_mps": 0.3})
    assert "after joint pacing" in text and "paced" in text
    assert "(1 rejected, 0 paced)" not in text
    chunk = [line for line in text.splitlines() if line.startswith("[chunk]")][0]
    predicted = int(chunk.split()[1])
    assert predicted > 10 and outcome.waypoints_predicted == predicted


def test_rejected_and_halted_chunks_are_counted(tmp_path):
    script = [{"name": "move_to", "arguments": {"targets": {"left_x": 99.0}, "note": "far"}},
              {"name": "move_to", "arguments": {"targets": {"left_z": 0.30}, "note": "ok"}},
              {"name": "done", "arguments": {"summary": "s", "hindsight": "none"}}]
    outcome, text = _run(tmp_path, script=script)
    table = text.split("--- ACTION CHUNKS ")[1]
    assert "rejected" in table and "0 waypoints predicted (0 Cartesian" in text
    assert "2 move_to calls (1 rejected" in table
    assert outcome.chunk_sizes[0] == 0                     # the rejected call executed nothing


def test_estop_chunk_shows_fewer_executed_than_predicted(tmp_path):
    from test_session_sim import _make

    script = [{"name": "move_to", "arguments": {"targets": {"left_x": 0.40}, "note": "long move"}}] * 3
    cfg, runner, world, astra = _make(tmp_path, script=script)

    class Hooks:
        def on_astra_response(self, resp):
            runner.request_estop()

    runner.hooks = Hooks()
    outcome = runner.run("halt me")
    text = (Path(outcome.log_dir) / "transcript.txt").read_text()
    assert outcome.status == "estop"
    assert "estop" in text.split("--- ACTION CHUNKS ")[1]
    assert outcome.waypoints < outcome.waypoints_predicted


def test_note_only_transcript_is_astra_prose_and_nothing_else(tmp_path):
    outcome, _ = _run(tmp_path)
    notes = (Path(outcome.log_dir) / "transcript_notes.txt").read_text()
    lines = notes.splitlines()
    assert lines[0] == "# pick up blue and place ontop of green block"      # the goal, as a heading
    assert lines[1] == "  1. [call 1, step 0] Scripted: moving above the blue block."
    assert lines[2].startswith("  2. [call 2, step 90] ")                   # step = the observation it saw
    assert "[done, call 9] Scripted pick-and-place finished." in lines[-2]
    assert lines[-1] == "     hindsight: none"                              # aligned under its entry
    # one numbered entry per note plus the closing done/give_up, and none of the surrounding machinery
    assert len(lines) == 1 + outcome.llm_calls + 1
    for noise in ("state[joint_pos]", "state[eef_state]", "[reasoning]", "[gateway]", "[chunk]",
                  "SYSTEM PROMPT", "cadence_hz", "frames/", "base64"):
        assert noise not in notes, noise


def test_note_only_transcript_records_a_missing_note_and_a_give_up(tmp_path):
    script = [{"name": "move_to", "arguments": {"targets": {"left_z": 0.25}}},          # no note at all
              {"name": "give_up", "arguments": {"reason": "cannot see it", "hindsight": "clean the lens"}}]
    outcome, text = _run(tmp_path, script=script)
    notes = (Path(outcome.log_dir) / "transcript_notes.txt").read_text().splitlines()
    assert notes[1] == "  1. [call 1, step 0] (no note)"
    assert notes[2] == "  2. [give_up, call 2] cannot see it"
    assert notes[3] == "     hindsight: clean the lens"
    assert "note missing; every move must include a note" in text          # the model is told, too


def test_note_only_transcript_keeps_multi_line_notes_aligned(tmp_path):
    script = [{"name": "move_to", "arguments": {"targets": {"left_z": 0.25},
                                                "note": "First line of the note.\nSecond line of the note."}},
              {"name": "done", "arguments": {"summary": "s", "hindsight": ""}}]
    outcome, _ = _run(tmp_path, script=script)
    notes = (Path(outcome.log_dir) / "transcript_notes.txt").read_text().splitlines()
    assert notes[1] == "  1. [call 1, step 0] First line of the note."
    assert notes[2] == " " * len("  1. [call 1, step 0] ") + "Second line of the note."
    assert notes[-1] == "  2. [done, call 2] s"                            # empty hindsight adds no line
