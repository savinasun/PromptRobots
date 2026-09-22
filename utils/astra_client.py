"""Model clients: OpenAI Responses API (gpt-6-astra), OpenRouter Chat Completions (Qwen etc.), scripted stand-in.

The session speaks the Responses-API item vocabulary (input_text / input_image / function_call /
function_call_output / reasoning) whatever the backend; the OpenRouter client translates on the wire.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Protocol, Tuple

from utils.config import REPO_ROOT, DIM_NAMES, AstraConfig

REASONING_INCLUDE = ["reasoning.encrypted_content"]


@dataclass
class FunctionCall:
    call_id: str
    name: str
    arguments: Optional[dict]
    raw_arguments: str
    parse_error: Optional[str] = None
    item_id: Optional[str] = None


@dataclass
class AstraResponse:
    output_items: List[dict]
    function_calls: List[FunctionCall]
    messages: List[str]
    reasoning: List[str] = field(default_factory=list)   # summary text of the reasoning items, when returned
    usage: Dict[str, int] = field(default_factory=dict)
    response_id: Optional[str] = None
    elapsed_s: float = 0.0
    model: str = ""
    provider: Optional[str] = None      # upstream provider that served the call, when the API reports it (OpenRouter)
    raw_response: Optional[dict] = None  # preserve provider fields for the per-call audit log


class AstraClient(Protocol):
    model: str

    def create(self, input_items: List[dict]) -> AstraResponse: ...
    def build_request_dict(self, input_items: List[dict]) -> dict: ...


def _parse_function_calls(items: List[dict]) -> List[FunctionCall]:
    calls = []
    for it in items:
        if it.get("type") != "function_call":
            continue
        raw = it.get("arguments") or ""
        args, err = None, None
        try:
            parsed = json.loads(raw) if raw else {}
            if not isinstance(parsed, dict):
                err = "arguments must be a JSON object"
            else:
                args = parsed
        except json.JSONDecodeError as e:
            err = str(e)
        calls.append(FunctionCall(call_id=it.get("call_id", ""), name=it.get("name", ""), arguments=args,
                                  raw_arguments=raw, parse_error=err, item_id=it.get("id")))
    return calls


def _extract_reasoning(items: List[dict]) -> List[str]:
    """Human-readable reasoning text from the response's reasoning items.

    `include=["reasoning.encrypted_content"]` only brings back the opaque blob the next request needs; the
    readable part is the summary the API adds when `reasoning.summary` is requested (some models also fill
    `content`). Both are collected here, in order, and an item with neither contributes nothing.
    """
    out: List[str] = []
    for it in items:
        if it.get("type") != "reasoning":
            continue
        for key in ("summary", "content"):
            for part in it.get(key) or []:
                text = part.get("text") if isinstance(part, dict) else str(part)
                if text and text.strip():
                    out.append(text.strip())
    return out


def _extract_messages(items: List[dict]) -> List[str]:
    texts = []
    for it in items:
        if it.get("type") == "message":
            for part in it.get("content", []) or []:
                if isinstance(part, dict) and part.get("type") == "output_text":
                    texts.append(part.get("text", ""))
    return texts


# Key files: a bare key (optionally `export NAME=...`) in <repo>/.secrets/<NAME> or ~/.secrets/<NAME>.
SECRET_DIRS = [REPO_ROOT / ".secrets", Path.home() / ".secrets"]


def _read_key_file(path: Path, env_name: str) -> Optional[str]:
    text = path.read_text().strip()
    if not text:
        return None
    for line in text.splitlines():
        line = line.strip()
        if f"{env_name}=" in line:                      # `export OPENAI_API_KEY=sk-...` / `OPENAI_API_KEY=sk-...`
            return line.split(f"{env_name}=", 1)[1].strip().strip('"').strip("'") or None
    return text.splitlines()[0].strip()


def find_api_key(env_name: str) -> Tuple[Optional[str], str]:
    """(key, source) - environment first, then <repo>/.env, then .secrets/<NAME> files."""
    key = os.environ.get(env_name)
    if key:
        return key.strip(), "environment"
    try:
        from dotenv import load_dotenv

        if load_dotenv(REPO_ROOT / ".env") and os.environ.get(env_name):
            return os.environ[env_name].strip(), str(REPO_ROOT / ".env")
    except ImportError:
        pass
    for d in SECRET_DIRS:
        candidate = Path(d) / env_name
        if candidate.is_file():
            key = _read_key_file(candidate, env_name)
            if key:
                return key, str(candidate)
    return None, ""


def load_api_key(env_name: str) -> str:
    key, _ = find_api_key(env_name)
    if not key:
        places = ", ".join(str(Path(d) / env_name) for d in SECRET_DIRS)
        raise RuntimeError(
            f"{env_name} not found. Export it, put `{env_name}=sk-...` in {REPO_ROOT / '.env'}, or write the key to "
            f"one of: {places}"
        )
    return key


class OpenAIAstraClient:
    def __init__(self, cfg: AstraConfig, tools: List[dict]):
        from openai import OpenAI

        self.cfg = cfg
        self.tools = tools
        self.model = cfg.model
        self._no_summary = False        # set if the endpoint rejects reasoning.summary (see `create`)
        self._no_cache_key = False      # ditto for prompt_cache_key
        self._client = OpenAI(api_key=load_api_key(cfg.api_key_env), base_url=cfg.base_url,
                              max_retries=cfg.max_retries, timeout=cfg.request_timeout_s)

    def _request_kwargs(self, input_items: List[dict]) -> dict:
        kwargs: Dict[str, Any] = {
            "model": self.cfg.model,
            "input": input_items,
            "tools": self.tools,
            "store": False,
            "include": REASONING_INCLUDE,
        }
        if self.cfg.tool_choice:
            kwargs["tool_choice"] = self.cfg.tool_choice
        if self.cfg.prompt_cache_key and not self._no_cache_key:
            kwargs["prompt_cache_key"] = self.cfg.prompt_cache_key
        if self.cfg.parallel_tool_calls is not None:
            kwargs["parallel_tool_calls"] = bool(self.cfg.parallel_tool_calls)
        reasoning: Dict[str, Any] = {}
        if self.cfg.reasoning_effort:
            reasoning["effort"] = self.cfg.reasoning_effort
        if self.cfg.reasoning_summary and not self._no_summary:
            reasoning["summary"] = self.cfg.reasoning_summary
        if reasoning:
            kwargs["reasoning"] = reasoning
        if self.cfg.actions_only:
            # Astra does not support effort="none", so strict action mode needs *an* effort - but it is a
            # floor, not a ceiling. Overriding a configured effort here silently discarded `--effort high`
            # for every planning run once System 1 moved onto this contract. `setdefault` matches what the
            # OpenRouter client already does (`effort or "low"`). Keep encrypted reasoning for stateless
            # tool continuity; summaries are optional diagnostic metadata.
            kwargs.setdefault("reasoning", {}).setdefault("effort", "low")
            kwargs["tool_choice"] = "required"
            kwargs["parallel_tool_calls"] = False
        if self.cfg.max_output_tokens:
            kwargs["max_output_tokens"] = int(self.cfg.max_output_tokens)
        if not self.tools and not self.cfg.actions_only:
            for key in ("tools", "tool_choice", "parallel_tool_calls"):
                kwargs.pop(key, None)
        return kwargs

    def build_request_dict(self, input_items: List[dict]) -> dict:
        return self._request_kwargs(input_items)

    def close(self) -> None:
        self._client.close()

    def create(self, input_items: List[dict]) -> AstraResponse:
        return self._create(input_items)

    def create_streamed(self, input_items: List[dict], on_delta: Callable[[str, str, str], None]) -> AstraResponse:
        """Stream display-only text; return executable calls only after a complete response."""
        return self._create(input_items, on_delta)

    def _send_request(self, input_items: List[dict], on_delta=None):
        kwargs = self._request_kwargs(input_items)
        if on_delta is None:
            return self._client.responses.create(**kwargs)
        on_delta("reset", "", "")
        kinds = {"response.function_call_arguments.delta": "arguments",
                 "response.reasoning_summary_text.delta": "summary",
                 "response.output_text.delta": "message"}
        with self._client.responses.create(**kwargs, stream=True) as stream:
            for event in stream:
                if event.type in kinds:
                    on_delta(kinds[event.type], event.delta, getattr(event, "item_id", ""))
                elif event.type == "response.completed":
                    return event.response
                elif event.type in ("response.failed", "response.incomplete", "error"):
                    raise RuntimeError(f"Astra stream ended with {event.type}")
        raise RuntimeError("Astra stream ended before response.completed")

    def _create(self, input_items: List[dict], on_delta=None) -> AstraResponse:
        t0 = time.perf_counter()
        try:
            resp = self._send_request(input_items, on_delta)
        except Exception as e:  # noqa: BLE001 - only one specific cause is handled, the rest re-raise
            if self._no_summary or not any(k in str(e).lower() for k in ("summary", "prompt_cache_key")):
                raise
            # The model does not accept reasoning.summary: drop it and keep the trial alive (no reasoning
            # text in the transcript from here on, only the encrypted blob and the token count).
            self._no_summary = True
            self._no_cache_key = True
            print(f"[astra] retrying without reasoning.summary / prompt_cache_key on {self.cfg.model}: "
                  f"{type(e).__name__}: {e}")
            resp = self._send_request(input_items, on_delta)
        elapsed = time.perf_counter() - t0
        items = [o.model_dump(exclude_none=True, mode="json") for o in resp.output]
        usage: Dict[str, int] = {}
        if getattr(resp, "usage", None) is not None:
            u = resp.usage
            usage = {
                "input_tokens": int(getattr(u, "input_tokens", 0) or 0),
                "output_tokens": int(getattr(u, "output_tokens", 0) or 0),
                "total_tokens": int(getattr(u, "total_tokens", 0) or 0),
            }
            details_in = getattr(u, "input_tokens_details", None)
            if details_in is not None:
                usage["cached_tokens"] = int(getattr(details_in, "cached_tokens", 0) or 0)
            details_out = getattr(u, "output_tokens_details", None)
            if details_out is not None:
                usage["reasoning_tokens"] = int(getattr(details_out, "reasoning_tokens", 0) or 0)
        return AstraResponse(output_items=items, function_calls=_parse_function_calls(items),
                             messages=_extract_messages(items), reasoning=_extract_reasoning(items), usage=usage,
                             response_id=getattr(resp, "id", None), elapsed_s=elapsed,
                             model=getattr(resp, "model", self.cfg.model) or self.cfg.model,
                             raw_response=resp.model_dump(exclude_none=True, mode="json")
                             if callable(getattr(resp, "model_dump", None)) else None)


# ---------------------------------------------------------------------------
# OpenRouter (OpenAI-compatible Chat Completions wire)
# ---------------------------------------------------------------------------
OPENROUTER_HEADERS = {"X-Title": "PromptRobots utils"}
# OpenRouter's unified reasoning effort levels; the Responses-API value "max" has no equivalent.
OPENROUTER_EFFORT = {"max": "xhigh"}


def _chat_content(content: Any, role: str) -> Any:
    """Responses content (string or input_text/input_image/output_text parts) -> Chat Completions content."""
    if content is None or isinstance(content, str):
        return content or ""
    parts: List[dict] = []
    for part in content:
        if not isinstance(part, dict):
            parts.append({"type": "text", "text": str(part)})
            continue
        kind = part.get("type")
        if kind in ("input_text", "output_text", "text"):
            parts.append({"type": "text", "text": part.get("text", "")})
        elif kind == "input_image":
            image: Dict[str, Any] = {"url": part.get("image_url", "")}
            if part.get("detail"):
                image["detail"] = part["detail"]
            parts.append({"type": "image_url", "image_url": image})
        elif kind == "image_url":
            parts.append(part)
        else:
            raise ValueError(f"cannot translate content part type {kind!r} for a chat message")
    if role == "assistant" or all(p["type"] == "text" for p in parts):
        return "\n".join(p["text"] for p in parts if p["type"] == "text")
    return parts


def items_to_chat_messages(items: List[dict], include_reasoning_details: bool = True) -> List[dict]:
    """Responses-API input items -> Chat Completions messages.

    Consecutive assistant output items (reasoning, message, function_call) become one assistant message
    with `tool_calls`; `function_call_output` becomes a `tool` message. Reasoning summaries are not sent back
    (they are not part of the wire); `reasoning_details` blocks returned by OpenRouter are echoed unchanged.
    """
    messages: List[dict] = []
    pending: Optional[dict] = None

    def flush() -> None:
        nonlocal pending
        if pending is not None and (pending.get("content") or pending.get("tool_calls")):
            messages.append(pending)
        pending = None

    def assistant() -> dict:
        nonlocal pending
        if pending is None:
            pending = {"role": "assistant", "content": ""}
        return pending

    for it in items:
        kind = it.get("type")
        if kind == "function_call":
            assistant().setdefault("tool_calls", []).append({
                "id": it.get("call_id") or it.get("id") or "", "type": "function",
                "function": {"name": it.get("name", ""), "arguments": it.get("arguments") or "{}"}})
        elif kind == "reasoning":
            msg = assistant()
            if include_reasoning_details and it.get("reasoning_details"):
                msg["reasoning_details"] = it["reasoning_details"]
        elif kind == "function_call_output":
            flush()
            output = it.get("output", "")
            messages.append({"role": "tool", "tool_call_id": it.get("call_id", ""),
                             "content": output if isinstance(output, str) else json.dumps(output)})
        elif kind == "message" or "role" in it:
            role = it.get("role") or ("assistant" if kind == "message" else "user")
            content = _chat_content(it.get("content"), role)
            if role == "assistant":
                msg = assistant()
                msg["content"] = (msg["content"] + "\n" if msg["content"] else "") + content
            else:
                flush()
                messages.append({"role": role, "content": content})
        else:
            raise ValueError(f"cannot translate item type {kind!r} to a chat message")
    flush()
    for msg in messages:
        if msg["role"] == "assistant" and msg.get("tool_calls") and not msg["content"]:
            msg["content"] = None
    return messages


def _declare_target_dims(schema: Any) -> Any:
    """Copy of `schema` where an open `targets` object lists every dimension as an optional number.

    Qwen (and other models with an XML-style native tool format) let the provider convert the call to JSON
    using the declared schema; a `targets` object with no `properties` then comes back as `"targets": ,`
    (invalid JSON) whenever the model fills it. Declaring the keys fixes that without changing the Responses
    request the reference trials use.
    """
    if not isinstance(schema, dict):
        return schema
    out = {k: _declare_target_dims(v) if k in ("properties", "items") or isinstance(v, dict) else v
           for k, v in schema.items()}
    for name, prop in list((out.get("properties") or {}).items()):
        if name == "targets" and isinstance(prop, dict) and prop.get("type") == "object" and not prop.get("properties"):
            out["properties"][name] = {**prop, "properties": {d: {"type": "number"} for d in DIM_NAMES}}
    return out


def tools_to_chat_tools(tools: List[dict], strict: bool = True) -> List[dict]:
    """Responses-API flat function tools -> Chat Completions nested `function` tools."""
    out = []
    for t in tools:
        if t.get("type") != "function":
            raise ValueError(f"unsupported tool type {t.get('type')!r} for the chat wire")
        fn: Dict[str, Any] = {"name": t["name"], "description": t.get("description", ""),
                              "parameters": _declare_target_dims(t.get("parameters") or {"type": "object", "properties": {}})}
        if strict and t.get("strict") is not None:
            fn["strict"] = bool(t["strict"])
        out.append({"type": "function", "function": fn})
    return out


def chat_message_to_items(message: dict, response_id: str) -> List[dict]:
    """One Chat Completions assistant message (as a dict) -> Responses-API output items.

    Reasoning text goes under `summary` only (the action contract rejects reasoning `content`), so the
    transcript and the strict action-only check behave exactly as with the Responses backend.
    """
    items: List[dict] = []
    reasoning_text = message.get("reasoning")
    details = message.get("reasoning_details")
    if (reasoning_text and reasoning_text.strip()) or details:
        item: Dict[str, Any] = {"type": "reasoning", "id": f"rs_{response_id}", "summary": []}
        if reasoning_text and reasoning_text.strip():
            item["summary"].append({"type": "summary_text", "text": reasoning_text.strip()})
        if details:
            item["reasoning_details"] = details
        items.append(item)
    content = message.get("content")
    if isinstance(content, list):
        content = "".join(p.get("text", "") for p in content if isinstance(p, dict))
    if content and content.strip():
        items.append({"type": "message", "id": f"msg_{response_id}", "role": "assistant", "status": "completed",
                      "content": [{"type": "output_text", "text": content}]})
    for i, tc in enumerate(message.get("tool_calls") or []):
        fn = tc.get("function") or {}
        items.append({"type": "function_call", "id": f"fc_{response_id}_{i}",
                      "call_id": tc.get("id") or f"call_{response_id}_{i}", "name": fn.get("name", ""),
                      "arguments": fn.get("arguments") or "{}", "status": "completed"})
    return items


def _chat_usage(usage: Optional[dict]) -> Dict[str, int]:
    if not usage:
        return {}
    out = {"input_tokens": int(usage.get("prompt_tokens") or 0),
           "output_tokens": int(usage.get("completion_tokens") or 0),
           "total_tokens": int(usage.get("total_tokens") or 0)}
    cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens")
    if cached is not None:
        out["cached_tokens"] = int(cached or 0)
    reasoning = (usage.get("completion_tokens_details") or {}).get("reasoning_tokens")
    if reasoning is not None:
        out["reasoning_tokens"] = int(reasoning or 0)
    return out


class OpenRouterChatClient:
    """Any tool-capable OpenRouter model (default qwen/qwen3.8-max-0902) behind the AstraClient protocol."""

    def __init__(self, cfg: AstraConfig, tools: List[dict]):
        from openai import OpenAI

        self.cfg = cfg
        self.tools = tools
        self.model = cfg.model
        self._plain = False          # set after a 4xx: drop strict schemas + reasoning controls and retry once
        self._client = OpenAI(api_key=load_api_key(cfg.api_key_env), base_url=cfg.base_url,
                              max_retries=cfg.max_retries, timeout=cfg.request_timeout_s,
                              default_headers=OPENROUTER_HEADERS)

    def _request_kwargs(self, input_items: List[dict], strip_signatures: bool = False) -> dict:
        cfg = self.cfg
        kwargs: Dict[str, Any] = {
            "model": cfg.model,
            "messages": items_to_chat_messages(input_items, include_reasoning_details=not strip_signatures),
            "tools": tools_to_chat_tools(self.tools, strict=not self._plain),
        }
        if cfg.tool_choice:
            kwargs["tool_choice"] = cfg.tool_choice
        if cfg.parallel_tool_calls is not None:
            kwargs["parallel_tool_calls"] = bool(cfg.parallel_tool_calls)
        effort = cfg.reasoning_effort
        if cfg.actions_only:
            kwargs["tool_choice"] = "required"
            kwargs["parallel_tool_calls"] = False
            effort = effort or "low"
        if cfg.max_output_tokens:
            kwargs["max_tokens"] = int(cfg.max_output_tokens)
        extra: Dict[str, Any] = {}
        if effort and not self._plain:
            extra["reasoning"] = {"effort": OPENROUTER_EFFORT.get(effort, effort)}
        if cfg.openrouter_providers:
            extra["provider"] = {"order": list(cfg.openrouter_providers), "allow_fallbacks": False}
        if extra:
            kwargs["extra_body"] = extra
        if not self.tools and not cfg.actions_only:
            for key in ("tools", "tool_choice", "parallel_tool_calls"):
                kwargs.pop(key, None)
        return kwargs

    def build_request_dict(self, input_items: List[dict]) -> dict:
        return self._request_kwargs(input_items)

    def close(self) -> None:
        self._client.close()

    def create(self, input_items: List[dict]) -> AstraResponse:
        return self._create(input_items)

    def create_streamed(self, input_items: List[dict], on_delta: Callable[[str, str, str], None]) -> AstraResponse:
        """Stream display-only text; return executable calls only after a complete response."""
        return self._create(input_items, on_delta)

    def _send_request(self, input_items: List[dict], on_delta=None,
                      strip_signatures: bool = False) -> Tuple[dict, dict, str, str, Optional[str]]:
        """-> (assistant message dict, usage dict, response id, model name, provider name)."""
        kwargs = self._request_kwargs(input_items, strip_signatures)
        if on_delta is None:
            resp = self._client.chat.completions.create(**kwargs).model_dump(exclude_none=True, mode="json")
            self._last_raw_response = resp
            choices = resp.get("choices") or []
            if not choices:
                raise RuntimeError(f"OpenRouter returned no choices: {resp.get('error') or resp}")
            return (choices[0].get("message") or {}, resp.get("usage") or {}, resp.get("id") or "",
                    resp.get("model") or self.cfg.model, resp.get("provider"))
        on_delta("reset", "", "")
        message: Dict[str, Any] = {"content": "", "reasoning": "", "tool_calls": []}
        self._last_raw_response = {"chunks": []}
        slots: Dict[int, dict] = {}
        usage, rid, model, provider, finished = {}, "", self.cfg.model, None, False
        with self._client.chat.completions.create(**kwargs, stream=True,
                                                  stream_options={"include_usage": True}) as stream:
            for chunk in stream:
                c = chunk.model_dump(exclude_none=True, mode="json")
                self._last_raw_response["chunks"].append(c)
                rid, model = c.get("id") or rid, c.get("model") or model
                provider = c.get("provider") or provider
                if c.get("usage"):
                    usage = c["usage"]
                if c.get("error"):
                    raise RuntimeError(f"OpenRouter stream error: {c['error']}")
                for choice in c.get("choices") or []:
                    delta = choice.get("delta") or {}
                    if delta.get("reasoning"):
                        message["reasoning"] += delta["reasoning"]
                        on_delta("summary", delta["reasoning"], rid)
                    if delta.get("content"):
                        message["content"] += delta["content"]
                        on_delta("message", delta["content"], rid)
                    for tc in delta.get("tool_calls") or []:
                        slot = slots.setdefault(int(tc.get("index", 0)), {"id": "", "name": "", "arguments": ""})
                        slot["id"] = tc.get("id") or slot["id"]
                        fn = tc.get("function") or {}
                        if fn.get("name"):
                            slot["name"] = fn["name"]
                        if fn.get("arguments"):
                            slot["arguments"] += fn["arguments"]
                            on_delta("arguments", fn["arguments"], slot["id"])
                    if choice.get("finish_reason"):
                        finished = True
        if not finished:
            raise RuntimeError("Astra stream ended before a finish_reason")
        message["tool_calls"] = [{"id": s["id"], "type": "function",
                                  "function": {"name": s["name"], "arguments": s["arguments"]}}
                                 for _, s in sorted(slots.items())]
        return message, usage, rid, model, provider

    def _create(self, input_items: List[dict], on_delta=None) -> AstraResponse:
        import openai

        t0 = time.perf_counter()
        strip_signatures = False
        while True:
            try:
                message, usage, rid, model, provider = self._send_request(input_items, on_delta, strip_signatures)
                break
            except openai.APIStatusError as e:
                retryable = e.status_code in (400, 404, 422)
                if retryable and not strip_signatures and "signature" in str(e).lower():
                    # Gemini's encrypted thought signatures are bound to the endpoint that produced them; after a
                    # provider failover the history's signatures are "corrupted". Resend this one request without
                    # them (the model simply re-thinks), keeping reasoning control and strict schemas.
                    strip_signatures = True
                    print(f"[astra] retrying without stored thought signatures on {self.cfg.model}: "
                          f"{type(e).__name__}: {e}")
                    continue
                if retryable and not self._plain:
                    # The provider rejected an optional feature (strict schemas / reasoning effort): drop them
                    # for the rest of the trial and retry.
                    self._plain = True
                    print(f"[astra] retrying without strict tool schemas / reasoning effort on {self.cfg.model}: "
                          f"{type(e).__name__}: {e}")
                    continue
                raise
        elapsed = time.perf_counter() - t0
        rid = rid or f"chatcmpl_{int(t0 * 1000)}"
        items = chat_message_to_items(message, rid)
        return AstraResponse(output_items=items, function_calls=_parse_function_calls(items),
                             messages=_extract_messages(items), reasoning=_extract_reasoning(items),
                             usage=_chat_usage(usage), response_id=rid, elapsed_s=elapsed,
                             model=model or self.cfg.model, provider=provider,
                             raw_response=getattr(self, "_last_raw_response", None))


# ---------------------------------------------------------------------------
# Scripted stand-in
# ---------------------------------------------------------------------------
DEFAULT_SIM_SCRIPT: List[dict] = [
    {"name": "move_to", "arguments": {"targets": {"left_x": 0.33, "left_y": 0.02, "left_z": 0.10},
                                      "note": "Scripted: moving above the blue block."},
     "reasoning": "The blue block is left of centre; approach from straight above so the jaws straddle it."},
    {"name": "move_to", "arguments": {"targets": {"left_z": 0.03},
                                      "note": "Scripted: descending onto the blue block."},
     "reasoning": "Descending to the table height reported by the last observation, gripper still open."},
    {"name": "move_to", "arguments": {"targets": {"left_gripper": 0.2},
                                      "note": "Scripted: closing the gripper."},
     "reasoning": "Closing to 0.2 rather than 0 so the block is squeezed, not crushed."},
    {"name": "move_to", "arguments": {"targets": {"left_z": 0.12},
                                      "note": "Scripted: lifting the block."}},
    {"name": "move_to", "arguments": {"targets": {"left_y": -0.09},
                                      "note": "Scripted: carrying it over the green block."}},
    {"name": "move_to", "arguments": {"targets": {"left_z": 0.05},
                                      "note": "Scripted: lowering onto the green block."}},
    {"name": "move_to", "arguments": {"targets": {"left_gripper": 1.0},
                                      "note": "Scripted: releasing."}},
    {"name": "move_to", "arguments": {"targets": {"left_z": 0.15},
                                      "note": "Scripted: retreating upward."}},
    {"name": "done", "arguments": {"summary": "Scripted pick-and-place finished.",
                                   "hindsight": "none"}},
]


class ScriptedAstraClient:
    """Replays a fixed list of tool calls; calls `give_up` once the script is exhausted."""

    def __init__(self, script: Optional[List[dict]] = None, tools: Optional[List[dict]] = None,
                 model: str = "scripted-astra", delay_s: float = 0.0, actions_only: bool = False):
        self.script = list(script if script is not None else DEFAULT_SIM_SCRIPT)
        self.actions_only = actions_only
        if script is None and actions_only:
            # Only adapt the built-in plumbing fixture. Explicit scripts retain
            # their exact output, including protocol violations for regression tests.
            self.script = [{"name": step["name"], "arguments": {
                "targets": {d: step["arguments"]["targets"].get(d) for d in DIM_NAMES}}
                if step["name"] == "move_to" else {}} for step in self.script]
        self.tools = tools or []
        self.model = model
        self.delay_s = delay_s
        self._n = 0
        self.requests: List[dict] = []

    @classmethod
    def from_file(cls, path: str, **kw) -> "ScriptedAstraClient":
        with open(path, "r") as f:
            data = json.load(f)
        if isinstance(data, dict) and "script" in data:
            data = data["script"]
        return cls(script=data, **kw)

    def build_request_dict(self, input_items: List[dict]) -> dict:
        request = {"model": self.model, "input": input_items, "tools": self.tools, "store": False,
                   "include": REASONING_INCLUDE}
        if self.actions_only:
            request.update(reasoning={"effort": "low"}, tool_choice="required", parallel_tool_calls=False)
        return request

    # `prompt_cache_key` is accepted by the Responses API; if this model rejects it the same one-shot
    # fallback as `reasoning.summary` applies (see `create`).

    def create(self, input_items: List[dict]) -> AstraResponse:
        if self.delay_s:
            time.sleep(self.delay_s)
        self.requests.append({"n_items": len(input_items)})
        if self._n < len(self.script):
            step = self.script[self._n]
        else:
            step = {"name": "give_up", "arguments": {} if self.actions_only else {
                "reason": "scripted client exhausted", "hindsight": "none"}}
        self._n += 1
        args = step.get("arguments", {})
        raw = args if isinstance(args, str) else json.dumps(args)
        item = {"type": "function_call", "id": f"fc_scripted_{self._n:04d}", "call_id": f"call_scripted_{self._n:04d}",
                "name": step.get("name", ""), "arguments": raw, "status": "completed"}
        items: List[dict] = []
        if step.get("reasoning"):
            items.append({"type": "reasoning", "id": f"rs_scripted_{self._n:04d}",
                          "summary": [{"type": "summary_text", "text": step["reasoning"]}]})
        if "text" in step:
            item = {"type": "message", "id": f"msg_scripted_{self._n:04d}", "role": "assistant",
                    "content": [{"type": "output_text", "text": step["text"]}], "status": "completed"}
        items.append(item)
        return AstraResponse(output_items=items, function_calls=_parse_function_calls(items), messages=_extract_messages(items),
                             reasoning=_extract_reasoning(items),
                             usage={"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}, response_id=item["id"],
                             elapsed_s=0.0, model=self.model)


def make_astra_client(cfg: AstraConfig, tools: List[dict]) -> AstraClient:
    if cfg.backend == "openai":
        return OpenAIAstraClient(cfg, tools)
    if cfg.backend == "openrouter":
        return OpenRouterChatClient(cfg, tools)
    if cfg.backend == "scripted":
        if cfg.script_path:
            return ScriptedAstraClient.from_file(cfg.script_path, tools=tools, actions_only=cfg.actions_only)
        return ScriptedAstraClient(tools=tools, actions_only=cfg.actions_only)
    raise ValueError(f"unknown astra backend '{cfg.backend}'")
