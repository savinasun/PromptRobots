import json
from pathlib import Path

import pytest

from astra_yam.astra_client import OpenAIAstraClient
from astra_yam.cli import _config_from_args, build_parser
from astra_yam.config import AstraConfig, DIM_NAMES, PipelineConfig
from astra_yam.embodiment import build_policy_prompt, build_tools


def move(**targets):
    return {"name": "move_to", "arguments": {"targets": {d: targets.get(d) for d in DIM_NAMES}}}


def test_branch_defaults_and_explicit_language_opt_out():
    parser = build_parser()
    assert PipelineConfig().astra.actions_only
    assert _config_from_args(parser.parse_args(["show-prompt"])).astra.actions_only
    assert not _config_from_args(parser.parse_args(["show-prompt", "--language-output"])).astra.actions_only
    cfg = _config_from_args(parser.parse_args([
        "show-prompt", "--config", "configs/skild_yam_8.yaml", "--language-output"]))
    assert not cfg.astra.actions_only


def test_action_schema_contains_no_language_fields():
    tools = build_tools(PipelineConfig().bounds, reactive=True, actions_only=True)
    assert [t["name"] for t in tools] == ["move_to", "done", "give_up", "observe"]
    for tool in tools:
        assert tool["strict"] is True
        params = tool["parameters"]
        assert params["additionalProperties"] is False
        if tool["name"] != "move_to":
            assert params["properties"] == {} and params["required"] == []
    params = tools[0]["parameters"]
    assert list(params["properties"]) == params["required"] == ["targets"]
    targets = params["properties"]["targets"]
    assert targets["required"] == list(DIM_NAMES)
    assert targets["additionalProperties"] is False
    assert all(p == {"type": ["number", "null"]} for p in targets["properties"].values())


def test_actual_request_forces_low_effort_and_preserves_summary_setting(monkeypatch):
    import openai
    received = []
    class Responses:
        def create(self, **kwargs):
            received.append(kwargs)
            return type("Response", (), {"output": [], "usage": None, "id": "test", "model": "gpt-6-astra"})()
    monkeypatch.setenv("OPENAI_API_KEY", "test-only")
    monkeypatch.setattr(openai, "OpenAI", lambda **kwargs: type("Client", (), {"responses": Responses()})())
    cfg = AstraConfig(actions_only=True, reasoning_effort="high", reasoning_summary="detailed",
                      tool_choice="auto", parallel_tool_calls=True)
    client = OpenAIAstraClient(cfg, build_tools(PipelineConfig().bounds, actions_only=True))
    client.create([{"role": "user", "content": "test"}])
    request = received[0]
    assert request["reasoning"] == {"effort": "low", "summary": "detailed"}
    assert request["tool_choice"] == "required" and request["parallel_tool_calls"] is False
    assert request["include"] == ["reasoning.encrypted_content"]
    assert request["store"] is False
    client.cfg.reasoning_summary = None
    assert client.build_request_dict([])["reasoning"] == {"effort": "low"}

    def unsupported(**kwargs):
        received.append(kwargs)
        raise ValueError("unsupported reasoning effort")
    client._client.responses.create = unsupported
    with pytest.raises(ValueError, match="unsupported reasoning effort"):
        client.create([])
    assert len(received) == 2  # no silent retry with default or increased reasoning


def test_scripted_action_only_pick_place_keeps_gateway_and_history(tmp_path):
    from test_session_sim import _make
    _, runner, world, _ = _make(tmp_path, **{"astra.actions_only": True})
    outcome = runner.run("Stack the blue block on the green block.")
    assert outcome.status == "done" and outcome.rejections == 0
    assert outcome.waypoints > 100 and world.objects["blue block"].held_by is None
    assert abs(world.objects["blue block"].pos[2] - world.objects["green block"].pos[2] - .03) < 1e-6
    log = Path(outcome.log_dir)
    text = (log / "transcript.txt").read_text()
    assert "[note]" not in text and "[hindsight]" not in text and "[reasoning]" not in text
    assert "note missing" not in text
    assert outcome.hindsight is None and outcome.summary is None
    request = json.loads(sorted((log / "requests").glob("request_*.json"))[-1].read_text())
    assert any(i.get("type") == "function_call_output" for i in request["input"])
    assert not any(i.get("role") == "assistant" for i in request["input"])
    for item in request["input"]:
        if item.get("type") == "function_call":
            assert set(json.loads(item["arguments"])) <= {"targets"}


@pytest.mark.parametrize("call", [
    {"name": "move_to", "arguments": {**move(left_z=.2)["arguments"], "note": "moving up"}},
    {"name": "move_to", "arguments": {"targets": {"left_z": .2}}},  # incomplete strict schema
    move(left_z="0.2"), move(left_z=True), move(left_z=float("nan")), move(),
    {"name": "done", "arguments": {"summary": "finished"}},
    {"name": "observe", "arguments": {"lesson": "written memory"}},
    {"name": "move_to", "arguments": "{invalid"},
    {"name": "think", "arguments": {}},
])
def test_malformed_actions_stop_before_any_motion(tmp_path, call):
    from test_session_sim import _make
    _, runner, _, _ = _make(tmp_path, script=[call, move(left_z=.3)], **{
        "astra.actions_only": True, "reactive.enabled": True, "robot.home_at_start": False, "home_on_end": True})
    outcome = runner.run("test")
    assert outcome.status == "invalid_action_output"
    assert outcome.waypoints == 0 and outcome.llm_calls == 1 and outcome.rejections == 1


@pytest.mark.parametrize("extra", ["message", "second_action"])
def test_language_or_multiple_actions_cannot_authorize_first_motion(tmp_path, extra):
    from astra_yam.astra_client import _parse_function_calls
    from test_session_sim import _make
    _, runner, _, client = _make(tmp_path, script=[move(left_z=.3)], **{"astra.actions_only": True})
    create = client.create
    def invalid(items):
        response = create(items)
        if extra == "message":
            response.output_items.append({"type": "message", "content": [{"type": "output_text", "text": "Here is my plan"}]})
        else:
            response.output_items.append({"type": "function_call", "name": "done", "arguments": "{}", "call_id": "extra"})
            response.function_calls = _parse_function_calls(response.output_items)
        return response
    client.create = invalid
    outcome = runner.run("test")
    assert outcome.status == "invalid_action_output" and outcome.waypoints == 0


def test_reactive_actions_and_opaque_reasoning_without_written_memory(tmp_path):
    from test_session_sim import _make
    script = [{"name": "observe", "arguments": {}}, move(left_gripper=.8), {"name": "done", "arguments": {}}]
    cfg, runner, _, client = _make(tmp_path, script=script, **{"astra.actions_only": True, "reactive.enabled": True})
    create = client.create
    def with_opaque_metadata(items):
        response = create(items)
        response.output_items.insert(0, {"type": "reasoning", "id": "opaque", "summary": [], "encrypted_content": "opaque-test"})
        return response
    client.create = with_opaque_metadata
    outcome = runner.run("test")
    assert outcome.status == "done" and outcome.rejections == 0 and outcome.llm_calls == 3
    assert runner._episode_memory.lessons == []
    request = json.loads((Path(outcome.log_dir) / "requests/request_0002.json").read_text())
    assert any(i.get("encrypted_content") == "opaque-test" for i in request["input"])
    assert "Tentative lessons" not in json.dumps(request)
    system = build_policy_prompt(cfg)
    assert "Do not emit a thinking section" in system
    assert "must include a `note`" not in system
    assert "optional lesson field" not in system




@pytest.mark.parametrize("actions_only", [True, False])
def test_summaries_print_and_log_without_changing_actions(tmp_path, capsys, actions_only):
    from test_session_sim import _make
    args = {} if actions_only else {"summary": "finished", "hindsight": "none"}
    _, runner, _, _ = _make(tmp_path, script=[{
        "name": "done", "arguments": args, "reasoning": "The observed end state satisfies the goal."}],
        **{"astra.actions_only": actions_only, "astra.reasoning_effort": "low"})
    runner.verbose = True
    outcome = runner.run("test")
    assert outcome.status == "done" and outcome.rejections == 0
    assert "[reasoning summary] The observed end state" in capsys.readouterr().out
    assert "[reasoning] The observed end state" in (Path(outcome.log_dir) / "transcript.txt").read_text()
