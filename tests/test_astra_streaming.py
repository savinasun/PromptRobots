"""Offline regressions for display streaming; incomplete calls must never execute."""
from types import SimpleNamespace

import pytest

from astra_yam.astra_client import OpenAIAstraClient
from astra_yam.config import AstraConfig


class Stream:
    def __init__(self, events):
        self.events = events
        self.closed = False

    def __enter__(self):
        return iter(self.events)

    def __exit__(self, *_):
        self.closed = True


def make_client(monkeypatch, responses):
    import openai
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setattr(openai, "OpenAI", lambda **_: SimpleNamespace(responses=responses))
    return OpenAIAstraClient(AstraConfig(actions_only=False), [])


def completed():
    item = {"type": "function_call", "call_id": "c1", "name": "move_to",
            "arguments": '{"targets":{"left_z":0.2},"note":"Lift"}'}
    return SimpleNamespace(type="response.completed", response=SimpleNamespace(
        output=[SimpleNamespace(model_dump=lambda **_: item)], usage=None, id="r1", model="m"))


def test_streams_notes_and_summary_before_returning_complete_call(monkeypatch):
    events = [SimpleNamespace(type="response.reasoning_summary_text.delta", delta="Case held.", item_id="r1"),
              SimpleNamespace(type="response.function_call_arguments.delta", delta='{"note":"Lift', item_id="fc1"),
              completed()]
    stream = Stream(events)
    requests, deltas = [], []

    def create(**kwargs):
        requests.append(kwargs)
        return stream

    client = make_client(monkeypatch, SimpleNamespace(create=create))
    response = client.create_streamed([], lambda *delta: deltas.append(delta))
    assert requests[0]["stream"] is True
    assert requests[0]["store"] is False
    assert deltas == [("reset", "", ""), ("summary", "Case held.", "r1"), ("arguments", '{"note":"Lift', "fc1")]
    assert response.function_calls[0].arguments["targets"]["left_z"] == 0.2
    assert stream.closed


@pytest.mark.parametrize("terminal", [None, "response.incomplete", "response.failed", "error"])
def test_incomplete_stream_never_returns_an_executable_response(monkeypatch, terminal):
    events = [SimpleNamespace(type="response.function_call_arguments.delta", delta='{"targets":', item_id="fc1")]
    if terminal:
        events.append(SimpleNamespace(type=terminal))
    stream = Stream(events)
    client = make_client(monkeypatch, SimpleNamespace(create=lambda **_: stream))
    with pytest.raises(RuntimeError, match="Astra stream ended"):
        client.create_streamed([], lambda *_: None)
    assert stream.closed


def test_stream_retries_unsupported_summary_and_clears_preview(monkeypatch):
    requests, deltas = [], []
    stream = Stream([completed()])

    def create(**kwargs):
        requests.append(kwargs)
        if len(requests) == 1:
            raise ValueError("Unsupported parameter: reasoning.summary")
        return stream

    client = make_client(monkeypatch, SimpleNamespace(create=create))
    response = client.create_streamed([], lambda *delta: deltas.append(delta))
    assert len(response.function_calls) == 1
    assert deltas == [("reset", "", ""), ("reset", "", "")]
    assert requests[0]["reasoning"]["summary"] == "auto"
    assert "reasoning" not in requests[1]
    assert stream.closed


def test_runner_routes_stream_to_hooks_before_executing(tmp_path):
    from test_session_sim import _make
    script = [{"name": "move_to", "arguments": {"targets": {"left_z": 0.2}, "note": "Lift"}},
              {"name": "done", "arguments": {"summary": "Finished", "hindsight": ""}}]
    _, runner, _, astra = _make(tmp_path, script=script)
    order = []

    class Hooks:
        def on_astra_stream(self, kind, delta, item_id):
            order.append("stream")

        def on_astra_response(self, response):
            order.append("response")

        def on_plan(self, plan):
            order.append("plan")

    def create_streamed(items, on_delta):
        on_delta("arguments", '{"note":"Lift', "fc1")
        return astra.create(items)

    astra.create_streamed = create_streamed
    runner.hooks = Hooks()
    outcome = runner.run("Lift")
    assert outcome.status == "done" and outcome.waypoints > 0
    assert order == ["stream", "response", "plan", "stream", "response"]
