"""Shadow model: the controlling model's answer is returned unchanged; the shadow's is only reported."""
import time

from utils.astra_client import AstraResponse, ScriptedAstraClient, _parse_function_calls
from utils.config import load_config
import json

from utils.shadow import ShadowClient, build_branch, describe_response, not_executed_payload


def scripted(name, steps):
    return ScriptedAstraClient(script=steps, model=name)


def test_primary_response_returned_and_shadow_kept_aside():
    primary = scripted("astra", [{"name": "move_to", "arguments": {"targets": {"left_z": 0.1}, "note": "Astra lifts."}}])
    shadow = scripted("qwen", [{"name": "move_to", "arguments": {"targets": {"left_x": 0.3}, "note": "Qwen reaches."}}])
    glm = scripted("glm", [{"name": "done", "arguments": {"summary": "GLM stops.", "hindsight": "none"}}])
    client = ShadowClient(primary, [shadow, glm])
    assert client.model == "astra" and client.shadow_models == ["qwen", "glm"]
    resp = client.create([{"role": "user", "content": "go"}])
    assert resp.function_calls[0].arguments["note"] == "Astra lifts."
    assert [s.model for s in client.last_shadows] == ["qwen", "glm"]
    assert all(s.error is None for s in client.last_shadows)
    assert client.last_shadows[0].response.function_calls[0].arguments["note"] == "Qwen reaches."
    assert client.last_shadows[1].response.function_calls[0].name == "done"
    assert len(primary.requests) == 1 and len(shadow.requests) == 1 and len(glm.requests) == 1
    assert client.build_request_dict([])["model"] == "astra"


def test_streamed_call_goes_to_primary_only_when_it_streams():
    primary = scripted("astra", [{"name": "done", "arguments": {"summary": "ok", "hindsight": "none"}}])
    shadow = scripted("qwen", [{"name": "give_up", "arguments": {"reason": "lost", "hindsight": "none"}}])
    deltas = []
    resp = ShadowClient(primary, [shadow]).create_streamed([], lambda *d: deltas.append(d))
    assert resp.function_calls[0].name == "done" and deltas == []      # scripted client has no streaming


def test_shadow_failure_and_timeout_never_break_the_trial():
    class Broken:
        model = "broken"

        def create(self, items):
            raise RuntimeError("boom")

    class Slow:
        model = "slow"

        def create(self, items):
            time.sleep(0.5)
            return None

    primary = scripted("astra", [{"name": "done", "arguments": {}}] * 2)
    client = ShadowClient(primary, [Broken(), Slow()], wait_timeout_s=0.05)
    assert client.create([]).function_calls[0].name == "done"
    broken, slow = client.last_shadows
    assert broken.error == "RuntimeError: boom" and broken.response is None
    assert "no answer within" in slow.error and slow.response is None


def response(name, arguments, messages=(), *more):
    calls = ([(name, arguments)] if name else []) + list(more)
    items = [{"type": "function_call", "call_id": f"c{i}", "name": n, "arguments": a} for i, (n, a) in enumerate(calls)]
    return AstraResponse(output_items=items, function_calls=_parse_function_calls(items), messages=list(messages))


def test_describe_response_lines():
    assert describe_response(None) == "(no response)"
    assert describe_response(response(None, "", ["I think", "we wait"])) == "(no tool call) I think we wait"
    assert describe_response(response("move_to", '{"targets": {"left_z": 0.1, "left_x": null}, "note": "Lift."}')) \
        == 'move_to {"left_z": 0.1} -- Lift.'
    assert describe_response(response("move_to", '{"targets": {"left_z": 0.1}}')) == 'move_to {"left_z": 0.1}'
    assert describe_response(response("done", '{"summary": "Placed.", "hindsight": "Approach lower."}')) \
        == "done -- Placed.\n  hindsight: Approach lower."
    assert describe_response(response("give_up", '{"reason": "stuck", "hindsight": "none"}')) == "give_up -- stuck"
    assert "not valid JSON" in describe_response(response("move_to", "{oops"))
    multi = response("move_to", '{"targets": {"right_x": 0.25}, "note": "Park right."}', (),
                     ("move_to", '{"targets": {"left_z": 0.03}, "note": "Descend."}'), ("done", '{"summary": "ok"}'))
    assert describe_response(multi) == ('move_to {"right_x": 0.25} -- Park right.\n'
                                        '  +2 more calls this turn (not executed): move_to {"left_z": 0.03} -- Descend.; '
                                        'done -- ok')


def test_station_yaml_loads_shadow(tmp_path):
    from utils.shadow import shadow_configs
    cfg = load_config("configs/skild_yam_8.yaml")
    # GPT-family models always go through OpenAI's own API, never OpenRouter; OpenRouter carries the
    # non-OpenAI shadows. Shadow mode itself is opt-in per run (--shadow-model), so the YAML leaves it off.
    assert cfg.astra.backend == "openai" and cfg.astra.model == "gpt-6-astra"
    assert not cfg.shadow.enabled and cfg.shadow.backend == "openrouter"
    assert cfg.shadow.model == "qwen/qwen3.8-max-0902" and cfg.shadow.api_key_env == "OPENROUTER_API_KEY"
    assert cfg.shadow.extra_models == [{"backend": "openai", "model": "gpt-6-astra"}]
    shadows = shadow_configs(cfg)
    assert [(c.backend, c.model, c.api_key_env) for c in shadows] == [
        ("openrouter", "qwen/qwen3.8-max-0902", "OPENROUTER_API_KEY"),
        ("openai", "gpt-6-astra", "OPENAI_API_KEY")]
    assert shadows[1].base_url is None and shadows[0].base_url == "https://openrouter.ai/api/v1"
    assert all(c.actions_only == cfg.astra.actions_only and c.reasoning_effort == "low" for c in shadows)
    cfg = load_config("configs/skild_yam_8.yaml", {"shadow.enabled": False, "shadow.model": "qwen/qwen3.8-flash"})
    assert not cfg.shadow.enabled and cfg.shadow.model == "qwen/qwen3.8-flash"


class Recording(ScriptedAstraClient):
    """Scripted client that also keeps the exact items of every request."""

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.seen = []

    def create(self, items):
        self.seen.append(list(items))
        return super().create(items)


def _fco(call_id, payload):
    return {"type": "function_call_output", "call_id": call_id, "output": json.dumps(payload)}


def test_shadow_sees_its_own_turns_not_the_primarys_notes():
    primary = Recording(model="astra", script=[
        {"name": "move_to", "arguments": {"targets": {"left_z": 0.1}, "note": "ASTRA SECRET 1"}, "reasoning": "astra thinks"},
        {"name": "move_to", "arguments": {"targets": {"left_z": 0.2}, "note": "ASTRA SECRET 2"}}])
    shadow = Recording(model="qwen", script=[
        {"name": "move_to", "arguments": {"targets": {"left_x": 0.3}, "note": "QWEN 1"}, "reasoning": "qwen thinks"},
        {"name": "done", "arguments": {"summary": "QWEN 2", "hindsight": "none"}}])
    client = ShadowClient(primary, [shadow])
    items = [{"role": "system", "content": "SYS"}, {"role": "user", "content": "GOAL"},
             {"role": "user", "content": [{"type": "input_text", "text": "OBS 1"}]}]
    resp1 = client.create(items)
    assert shadow.seen[0] == items                                  # first turn: identical shared prefix
    # the session appends the primary's output + gateway result + the next observation
    items = items + resp1.output_items + [_fco(resp1.function_calls[0].call_id, {"ok": True, "steps": 12}),
                                          {"role": "user", "content": [{"type": "input_text", "text": "OBS 2"}]}]
    client.create(items)
    branch = shadow.seen[1]
    flat = json.dumps(branch)
    assert "ASTRA SECRET" not in flat and "astra thinks" not in flat
    assert branch[:3] == items[:3] and branch[-1] == items[-1]
    own = client.last_shadows[0]
    kinds = [(it.get("type"), it.get("role")) for it in branch[3:-1]]
    assert kinds == [("reasoning", None), ("function_call", None), ("function_call_output", None)]
    assert branch[4]["name"] == "move_to" and "QWEN 1" in branch[4]["arguments"]
    fco = json.loads(branch[5]["output"])
    assert branch[5]["call_id"] == branch[4]["call_id"]
    assert fco == not_executed_payload('move_to {"left_z": 0.1}') and fco["status"] == "not_executed"
    assert "ASTRA SECRET" not in fco["reason"] and '{"left_z": 0.1}' in fco["reason"]
    # the primary's own request is untouched
    assert primary.seen[1] == items
    # the second proposal (done) is a proposal only: the trial went on and the shadow keeps being asked
    assert own.response.function_calls[0].name == "done"


def test_branch_handles_shadow_turn_without_action_and_extra_calls():
    from utils.shadow import _Turn
    items = [{"role": "system", "content": "S"},
             {"type": "function_call", "id": "fc_a", "call_id": "call_a", "name": "move_to", "arguments": "{}"},
             _fco("call_a", {"ok": True}),
             {"role": "user", "content": "OBS"}]
    # shadow errored that turn -> a user notice replaces the primary's turn
    turns = {"call_a": _Turn(output_items=[], call_ids=[], executed="move_to {}", error="boom")}
    branch = build_branch(items, turns)
    assert [b.get("role") for b in branch] == ["system", "user", "user"]
    assert "produced no action (boom)" in branch[1]["content"] and "move_to {}" in branch[1]["content"]
    # shadow emitted two calls -> first not executed, second rejected, so every call id gets an output
    out = [{"type": "function_call", "id": "x1", "call_id": "s1", "name": "move_to", "arguments": "{}"},
           {"type": "function_call", "id": "x2", "call_id": "s2", "name": "done", "arguments": "{}"}]
    turns = {"call_a": _Turn(output_items=out, call_ids=["s1", "s2"], executed="done", error=None)}
    branch = build_branch(items, turns)
    assert [b.get("call_id") for b in branch[1:5]] == ["s1", "s2", "s1", "s2"]
    assert json.loads(branch[3]["output"])["status"] == "not_executed"
    assert json.loads(branch[4]["output"])["status"] == "rejected"
    assert branch[-1] == items[-1]


def test_shared_history_mode_forwards_the_primary_request():
    primary = scripted("astra", [{"name": "done", "arguments": {}}])
    shadow = Recording(model="qwen", script=[{"name": "done", "arguments": {}}])
    items = [{"role": "user", "content": "x"}, {"type": "function_call", "id": "f", "call_id": "c", "name": "done",
                                                 "arguments": "{}"}, _fco("c", {"ok": True})]
    ShadowClient(primary, [shadow], independent_history=False).create(items)
    assert shadow.seen[0] == items
