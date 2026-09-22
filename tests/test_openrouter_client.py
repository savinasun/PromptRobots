"""Offline regressions for the OpenRouter (Chat Completions) client: wire translation both ways + streaming."""
import json
from types import SimpleNamespace

import pytest

from utils import astra_client
from utils.astra_client import (OpenRouterChatClient, chat_message_to_items, items_to_chat_messages,
                                    make_astra_client, tools_to_chat_tools)
from utils.config import AstraConfig
from utils.observation import redact_images

IMG = "data:image/jpeg;base64,/9j/AAAA"


def test_items_to_chat_messages_round_trip():
    items = [
        {"role": "system", "content": "SYS"},
        {"role": "user", "content": [{"type": "input_text", "text": "goal"}]},
        {"role": "user", "content": [{"type": "input_text", "text": "obs"},
                                     {"type": "input_image", "image_url": IMG, "detail": "high"}]},
        {"type": "reasoning", "id": "rs_1", "summary": [{"type": "summary_text", "text": "think"}],
         "reasoning_details": [{"type": "reasoning.text", "text": "think"}]},
        {"type": "function_call", "id": "fc_1", "call_id": "call_1", "name": "move_to",
         "arguments": '{"targets": {}}'},
        {"type": "function_call_output", "call_id": "call_1", "output": '{"ok": true}'},
        {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "hello"}]},
        {"type": "function_call", "id": "fc_2", "call_id": "call_2", "name": "done", "arguments": "{}"},
    ]
    msgs = items_to_chat_messages(items)
    assert msgs[0] == {"role": "system", "content": "SYS"}
    assert msgs[1] == {"role": "user", "content": "goal"}                    # text-only parts collapse
    assert msgs[2]["content"] == [{"type": "text", "text": "obs"},
                                  {"type": "image_url", "image_url": {"url": IMG, "detail": "high"}}]
    assert msgs[3]["role"] == "assistant" and msgs[3]["content"] is None
    assert msgs[3]["reasoning_details"] == [{"type": "reasoning.text", "text": "think"}]
    assert msgs[3]["tool_calls"] == [{"id": "call_1", "type": "function",
                                      "function": {"name": "move_to", "arguments": '{"targets": {}}'}}]
    assert msgs[4] == {"role": "tool", "tool_call_id": "call_1", "content": '{"ok": true}'}
    assert msgs[5]["content"] == "hello" and msgs[5]["tool_calls"][0]["id"] == "call_2"
    assert len(msgs) == 6
    json.dumps(msgs)


def test_tools_translate_to_nested_function_form():
    tools = [{"type": "function", "name": "done", "strict": True, "description": "d",
              "parameters": {"type": "object", "properties": {}, "required": []}}]
    assert tools_to_chat_tools(tools) == [{"type": "function", "function": {
        "name": "done", "description": "d", "strict": True,
        "parameters": {"type": "object", "properties": {}, "required": []}}}]
    assert "strict" not in tools_to_chat_tools(tools, strict=False)[0]["function"]


def test_chat_message_to_items_keeps_reasoning_as_summary_only():
    msg = {"role": "assistant", "content": "", "reasoning": " plan ",
           "tool_calls": [{"id": "call_9", "type": "function",
                           "function": {"name": "move_to", "arguments": '{"targets":{"left_z":0.1}}'}}]}
    items = chat_message_to_items(msg, "r1")
    assert [it["type"] for it in items] == ["reasoning", "function_call"]
    assert items[0]["summary"] == [{"type": "summary_text", "text": "plan"}] and "content" not in items[0]
    assert items[1]["call_id"] == "call_9" and items[1]["name"] == "move_to"
    # empty content never becomes an assistant message (the action contract would reject it)
    assert not any(it["type"] == "message" for it in items)


def test_redaction_covers_chat_image_urls():
    req = {"messages": [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": IMG}}]}]}
    out = redact_images(req)
    assert out["messages"][0]["content"][0]["image_url"]["url"].startswith("data:image/jpeg;base64,$blob:")


def make_client(monkeypatch, completions, **cfg):
    import openai
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
    monkeypatch.setattr(openai, "OpenAI", lambda **_: SimpleNamespace(
        chat=SimpleNamespace(completions=completions), close=lambda: None))
    tools = [{"type": "function", "name": "move_to", "strict": True, "description": "",
              "parameters": {"type": "object", "properties": {}}}]
    return OpenRouterChatClient(AstraConfig(backend="openrouter", **cfg), tools)


def test_factory_and_request_shape(monkeypatch):
    import openai
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
    seen = {}
    monkeypatch.setattr(openai, "OpenAI", lambda **kw: seen.update(kw) or SimpleNamespace())
    client = make_astra_client(AstraConfig(backend="openrouter", reasoning_effort="max"), [])
    assert isinstance(client, OpenRouterChatClient)
    assert seen["base_url"] == "https://openrouter.ai/api/v1" and seen["api_key"] == "sk-or-test"
    req = client.build_request_dict([{"role": "user", "content": "hi"}])
    assert req["model"] == "qwen/qwen3.8-max-0902"
    assert req["tool_choice"] == "required" and req["parallel_tool_calls"] is False       # actions_only
    assert req["extra_body"] == {"reasoning": {"effort": "xhigh"}}
    assert "prompt_cache_key" not in req and "include" not in req and "store" not in req


class Stream:
    def __init__(self, chunks):
        self.chunks, self.closed = chunks, False

    def __enter__(self):
        return iter(self.chunks)

    def __exit__(self, *_):
        self.closed = True


def chunk(**kw):
    d = {"id": "cc1", "model": "qwen/x", "choices": []}
    d.update(kw)
    return SimpleNamespace(model_dump=lambda **_: d)


def test_streamed_call_returns_complete_tool_call(monkeypatch):
    chunks = [
        chunk(choices=[{"delta": {"reasoning": "Case held."}}]),
        chunk(choices=[{"delta": {"tool_calls": [{"index": 0, "id": "call_1",
                                                  "function": {"name": "move_to", "arguments": '{"targets":'}}]}}]),
        chunk(choices=[{"delta": {"tool_calls": [{"index": 0, "function": {"arguments": '{"left_z":0.2}}'}}]}}]),
        chunk(choices=[{"delta": {}, "finish_reason": "tool_calls"}]),
        chunk(usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15,
                     "prompt_tokens_details": {"cached_tokens": 4}}),
    ]
    stream, requests, deltas = Stream(chunks), [], []

    def create(**kwargs):
        requests.append(kwargs)
        return stream

    client = make_client(monkeypatch, SimpleNamespace(create=create), actions_only=False)
    resp = client.create_streamed([{"role": "user", "content": "go"}], lambda *d: deltas.append(d))
    assert requests[0]["stream"] is True and requests[0]["stream_options"] == {"include_usage": True}
    assert deltas == [("reset", "", ""), ("summary", "Case held.", "cc1"),
                      ("arguments", '{"targets":', "call_1"), ("arguments", '{"left_z":0.2}}', "call_1")]
    assert resp.function_calls[0].arguments == {"targets": {"left_z": 0.2}}
    assert resp.function_calls[0].call_id == "call_1"
    assert resp.reasoning == ["Case held."] and resp.messages == []
    assert resp.usage == {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15, "cached_tokens": 4}
    assert resp.model == "qwen/x" and stream.closed


def test_incomplete_stream_never_returns_an_executable_response(monkeypatch):
    chunks = [chunk(choices=[{"delta": {"tool_calls": [{"index": 0, "id": "c", "function": {"arguments": "{"}}]}}])]
    client = make_client(monkeypatch, SimpleNamespace(create=lambda **_: Stream(chunks)), actions_only=False)
    with pytest.raises(RuntimeError, match="finish_reason"):
        client.create_streamed([], lambda *_: None)


def test_unsupported_feature_400_retries_plain_once(monkeypatch):
    import openai
    calls = []

    def create(**kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            raise openai.BadRequestError("strict not supported", response=SimpleNamespace(status_code=400,
                                         headers={}, request=None), body=None)
        return SimpleNamespace(model_dump=lambda **_: {
            "id": "cc2", "model": "qwen/x", "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            "choices": [{"message": {"role": "assistant", "content": None, "tool_calls": [
                {"id": "call_1", "type": "function", "function": {"name": "move_to", "arguments": "{}"}}]},
                "finish_reason": "tool_calls"}]})

    client = make_client(monkeypatch, SimpleNamespace(create=create))
    resp = client.create([{"role": "user", "content": "go"}])
    assert len(calls) == 2
    assert "strict" in calls[0]["tools"][0]["function"] and "extra_body" in calls[0]
    assert "strict" not in calls[1]["tools"][0]["function"] and "extra_body" not in calls[1]
    assert resp.function_calls[0].name == "move_to" and resp.response_id == "cc2"
    assert astra_client._parse_function_calls(resp.output_items)[0].call_id == "call_1"


def test_open_targets_object_gets_declared_dimensions_on_the_chat_wire_only():
    from utils.config import DIM_NAMES, Bounds
    from utils.embodiment import build_tools
    tools = build_tools(Bounds(), actions_only=False)
    before = json.dumps(tools)
    chat = tools_to_chat_tools(tools)
    targets = chat[0]["function"]["parameters"]["properties"]["targets"]
    assert set(targets["properties"]) == set(DIM_NAMES) and targets["properties"]["left_x"] == {"type": "number"}
    assert "required" not in targets and targets["description"].startswith("Map of dimension name")
    assert json.dumps(tools) == before                                   # Responses tools untouched
    strict = build_tools(Bounds(), actions_only=True)                    # already declared: unchanged
    assert tools_to_chat_tools(strict)[0]["function"]["parameters"] == strict[0]["parameters"]


def test_corrupted_signature_retries_once_without_reasoning_details_only(monkeypatch):
    import openai
    calls = []
    history = [{"role": "user", "content": "go"},
               {"type": "reasoning", "id": "rs_1", "summary": [],
                "reasoning_details": [{"type": "reasoning.encrypted", "data": "sig", "id": "call_1"}]},
               {"type": "function_call", "id": "fc_1", "call_id": "call_1", "name": "move_to", "arguments": "{}"},
               {"type": "function_call_output", "call_id": "call_1", "output": "{}"},
               {"role": "user", "content": "next"}]

    def create(**kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            raise openai.BadRequestError("Provider returned error: Corrupted thought signature.", response=SimpleNamespace(
                status_code=400, headers={}, request=None), body=None)
        return SimpleNamespace(model_dump=lambda **_: {
            "id": "cc3", "model": "google/x", "provider": "Google AI Studio",
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            "choices": [{"message": {"role": "assistant", "content": None, "tool_calls": [
                {"id": "call_2", "type": "function", "function": {"name": "done", "arguments": "{}"}}]},
                "finish_reason": "tool_calls"}]})

    client = make_client(monkeypatch, SimpleNamespace(create=create))
    resp = client.create(history)
    assert len(calls) == 2
    assert "reasoning_details" in calls[0]["messages"][1] and "reasoning_details" not in calls[1]["messages"][1]
    assert calls[1]["messages"][1]["tool_calls"][0]["id"] == "call_1"           # history otherwise intact
    assert "strict" in calls[1]["tools"][0]["function"] and "extra_body" in calls[1]   # features kept
    assert resp.provider == "Google AI Studio" and resp.function_calls[0].name == "done"
    # the next request sends signatures again (the strip is per request, not sticky)
    client.create(history)
    assert "reasoning_details" in calls[2]["messages"][1]
