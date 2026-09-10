"""Astra (OpenAI Responses API) client + a scripted stand-in for offline tests."""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, Tuple

from astra_yam.config import REPO_ROOT, AstraConfig

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
        if self.cfg.max_output_tokens:
            kwargs["max_output_tokens"] = int(self.cfg.max_output_tokens)
        return kwargs

    def build_request_dict(self, input_items: List[dict]) -> dict:
        return self._request_kwargs(input_items)

    def close(self) -> None:
        self._client.close()

    def create(self, input_items: List[dict]) -> AstraResponse:
        t0 = time.perf_counter()
        try:
            resp = self._client.responses.create(**self._request_kwargs(input_items))
        except Exception as e:  # noqa: BLE001 - only one specific cause is handled, the rest re-raise
            if self._no_summary or not any(k in str(e).lower() for k in ("summary", "prompt_cache_key")):
                raise
            # The model does not accept reasoning.summary: drop it and keep the trial alive (no reasoning
            # text in the transcript from here on, only the encrypted blob and the token count).
            self._no_summary = True
            self._no_cache_key = True
            print(f"[astra] retrying without reasoning.summary / prompt_cache_key on {self.cfg.model}: "
                  f"{type(e).__name__}: {e}")
            resp = self._client.responses.create(**self._request_kwargs(input_items))
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
                             model=getattr(resp, "model", self.cfg.model) or self.cfg.model)


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
                 model: str = "scripted-astra", delay_s: float = 0.0):
        self.script = list(script if script is not None else DEFAULT_SIM_SCRIPT)
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
        return {"model": self.model, "input": input_items, "tools": self.tools, "store": False,
                "include": REASONING_INCLUDE}

    # `prompt_cache_key` is accepted by the Responses API; if this model rejects it the same one-shot
    # fallback as `reasoning.summary` applies (see `create`).

    def create(self, input_items: List[dict]) -> AstraResponse:
        if self.delay_s:
            time.sleep(self.delay_s)
        self.requests.append({"n_items": len(input_items)})
        if self._n < len(self.script):
            step = self.script[self._n]
        else:
            step = {"name": "give_up", "arguments": {"reason": "scripted client exhausted", "hindsight": "none"}}
        self._n += 1
        args = step.get("arguments", {})
        raw = args if isinstance(args, str) else json.dumps(args)
        item = {"type": "function_call", "id": f"fc_scripted_{self._n:04d}", "call_id": f"call_scripted_{self._n:04d}",
                "name": step["name"], "arguments": raw, "status": "completed"}
        items: List[dict] = []
        if step.get("reasoning"):
            items.append({"type": "reasoning", "id": f"rs_scripted_{self._n:04d}",
                          "summary": [{"type": "summary_text", "text": step["reasoning"]}]})
        items.append(item)
        return AstraResponse(output_items=items, function_calls=_parse_function_calls(items), messages=[],
                             reasoning=_extract_reasoning(items),
                             usage={"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}, response_id=item["id"],
                             elapsed_s=0.0, model=self.model)


def make_astra_client(cfg: AstraConfig, tools: List[dict]) -> AstraClient:
    if cfg.backend == "openai":
        return OpenAIAstraClient(cfg, tools)
    if cfg.backend == "scripted":
        if cfg.script_path:
            return ScriptedAstraClient.from_file(cfg.script_path, tools=tools)
        return ScriptedAstraClient(tools=tools)
    raise ValueError(f"unknown astra backend '{cfg.backend}'")
