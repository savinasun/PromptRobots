"""Shadow models: extra clients asked the same question every turn; their answers are shown, never executed.

Each shadow keeps its own conversation branch (`independent_history`): the system prompt, goal, observations
and operator messages are the controlling model's, but every model turn is the shadow's own earlier proposal
followed by an honest tool result saying it was not executed and which targets the robot followed instead.
Shadows therefore never read the controlling model's notes, nor each other's.
"""
from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence

from utils.astra_client import AstraClient, AstraResponse


@dataclass
class ShadowResult:
    model: str
    response: Optional[AstraResponse]
    error: Optional[str]
    elapsed_s: float


def describe_response(resp: Optional[AstraResponse]) -> str:
    """One readable entry per model: the chosen action, its arguments and the note/summary it wrote.

    Only the first tool call of a turn is ever executed; any further calls the model emitted in the same turn
    are listed after it so the comparison shows the whole plan the model proposed.
    """
    if resp is None:
        return "(no response)"
    if not resp.function_calls:
        text = " ".join(m.strip() for m in resp.messages if m.strip())
        return f"(no tool call) {text[:300]}" if text else "(empty response)"
    line = _describe_call(resp.function_calls[0])
    extra = resp.function_calls[1:]
    if extra:
        line += f"\n  +{len(extra)} more call{'s' if len(extra) > 1 else ''} this turn (not executed): " + "; ".join(
            _describe_call(fc).replace("\n  hindsight: ", " / hindsight: ") for fc in extra)
    return line


def _describe_call(fc) -> str:
    if fc.parse_error or not isinstance(fc.arguments, dict):
        return f"{fc.name} (arguments not valid JSON: {fc.parse_error})"
    args = dict(fc.arguments)
    if fc.name == "move_to":
        targets = args.pop("targets", None) or {}
        if isinstance(targets, dict):
            targets = {k: v for k, v in targets.items() if v is not None}
        note = args.pop("note", None)
        line = f"move_to {json.dumps(targets)}"
        return line + (f" -- {str(note).strip()}" if note else "")
    hindsight = args.pop("hindsight", None)
    text = args.pop("summary", None) or args.pop("reason", None)
    line = fc.name + (f" -- {str(text).strip()}" if text else "") + (f" {json.dumps(args)}" if args else "")
    if hindsight and str(hindsight).strip().lower() != "none":
        line += f"\n  hindsight: {str(hindsight).strip()}"
    return line


def shadow_configs(cfg) -> list:
    """One AstraConfig per shadow: `shadow.model` plus each `shadow.extra_models` entry.

    An entry is either a model ID (same backend and settings as `shadow`) or a mapping of AstraConfig fields,
    e.g. `{backend: openai, model: gpt-6-astra}`; a different backend re-resolves the key and endpoint defaults
    unless the entry sets them. Tools and language/action mode always follow the controlling `astra` config.
    """
    from dataclasses import replace

    out = []
    for entry in [cfg.shadow.model, *cfg.shadow.extra_models]:
        patch = {"model": entry} if isinstance(entry, str) else dict(entry)
        if "model" not in patch:
            raise ValueError(f"shadow entry needs a model: {entry!r}")
        if patch.get("backend", cfg.shadow.backend) != cfg.shadow.backend:
            patch.setdefault("api_key_env", None)
            patch.setdefault("base_url", None)
        out.append(replace(cfg.shadow, actions_only=cfg.astra.actions_only, **patch))
    return out


ASSISTANT_ITEM_TYPES = ("reasoning", "function_call", "message")


def _item_key(item: dict) -> Optional[str]:
    return item.get("call_id") or item.get("id")


def _is_model_output(item: dict) -> bool:
    kind = item.get("type")
    return kind in ASSISTANT_ITEM_TYPES and (kind != "message" or item.get("role") == "assistant")


def _targets_of(call) -> Optional[dict]:
    if call is None or not isinstance(call.arguments, dict):
        return None
    targets = call.arguments.get("targets")
    return {k: v for k, v in targets.items() if v is not None} if isinstance(targets, dict) else None


@dataclass
class _Turn:
    """What one shadow proposed for one controlling-model turn, and what the robot did instead."""
    output_items: List[dict]              # the shadow's own output items (reasoning / function_call / message)
    call_ids: List[str]                   # its function_call ids, in order
    executed: Optional[str]               # controlling model's action, numbers only ("move_to {...}" / "done")
    error: Optional[str]                  # shadow error/timeout for that turn, if any


def not_executed_payload(executed: Optional[str]) -> dict:
    return {"ok": False, "status": "not_executed",
            "reason": "comparison mode: your proposal was recorded, and a different controller's action was "
                      "executed on the robot instead" + (f": {executed}" if executed else "") +
                      ". This is expected and not a fault; do not give up because of it. Read the next observation "
                      "and keep proposing the single best next action toward the goal from the current state."}


def build_branch(items: List[dict], turns: Dict[str, _Turn]) -> List[dict]:
    """The shadow's view of `items`: shared items verbatim, controlling-model turns swapped for the shadow's own.

    A controlling-model turn is a run of assistant items followed by its function_call_output items. The
    shadow's stored output for that turn goes in its place, each of its own calls answered (first: not executed,
    the rest: ignored), and a turn where the shadow produced nothing becomes a short user notice.
    """
    branch: List[dict] = []
    primary_call_ids: set = set()
    pending: Optional[_Turn] = None
    i = 0
    while i < len(items):
        it = items[i]
        if _is_model_output(it):
            turn = turns.get(_item_key(it) or "")
            if it.get("type") == "function_call":
                primary_call_ids.add(it.get("call_id"))
            if turn is not None and turn is not pending:
                branch.extend(turn.output_items)
                pending = turn
            i += 1
            continue
        if it.get("type") == "function_call_output" and it.get("call_id") in primary_call_ids:
            if pending is not None:
                if pending.call_ids:
                    branch.append({"type": "function_call_output", "call_id": pending.call_ids[0],
                                   "output": json.dumps(not_executed_payload(pending.executed))})
                    for extra in pending.call_ids[1:]:
                        branch.append({"type": "function_call_output", "call_id": extra, "output": json.dumps(
                            {"ok": False, "status": "rejected", "reason": "only the first tool call of a turn is considered"})})
                else:
                    why = f" ({pending.error})" if pending.error else ""
                    branch.append({"role": "user", "content":
                                   f"Your previous turn produced no action{why}. The controlling model's action was "
                                   f"executed instead" + (f": {pending.executed}" if pending.executed else "") +
                                   ". Continue from the next observation."})
                pending = None
            i += 1
            continue
        branch.append(it)
        i += 1
    return branch


class ShadowClient:
    """AstraClient that forwards to `primary` and, in parallel, asks every shadow client the same question.

    The primary response is returned unchanged; the shadow answers are kept in `last_shadows` (one per shadow,
    in configuration order) for the session to report. A shadow failure or timeout never affects the trial.
    With `independent_history` each shadow sees its own past proposals instead of the controlling model's turns.
    """

    def __init__(self, primary: AstraClient, shadows: Sequence[AstraClient], wait_timeout_s: float = 60.0,
                 independent_history: bool = True):
        self.primary = primary
        self.shadows = list(shadows)
        self.wait_timeout_s = wait_timeout_s
        self.independent_history = independent_history
        self.model = primary.model
        self.shadow_models = [c.model for c in self.shadows]
        self.last_shadows: List[ShadowResult] = []
        self.last_shadow_requests: List[Optional[dict]] = []   # per shadow: the request it was sent this turn
        self._turns: List[Dict[str, _Turn]] = [{} for _ in self.shadows]   # per shadow: primary item key -> turn

    def build_request_dict(self, input_items: List[dict]) -> dict:
        return self.primary.build_request_dict(input_items)

    def close(self) -> None:
        for client in [self.primary, *self.shadows]:
            close = getattr(client, "close", None)
            if callable(close):
                close()

    def create(self, input_items: List[dict]) -> AstraResponse:
        return self._all(input_items, None)

    def create_streamed(self, input_items: List[dict], on_delta: Callable[[str, str, str], None]) -> AstraResponse:
        return self._all(input_items, on_delta)

    def shadow_items(self, k: int, input_items: List[dict]) -> List[dict]:
        return build_branch(input_items, self._turns[k]) if self.independent_history else list(input_items)

    def _all(self, input_items: List[dict], on_delta) -> AstraResponse:
        self.last_shadows = []
        results: List[dict] = [{} for _ in self.shadows]
        branches = [self.shadow_items(k, input_items) for k in range(len(self.shadows))]
        self.last_shadow_requests = []
        for client, branch in zip(self.shadows, branches):
            try:
                self.last_shadow_requests.append(client.build_request_dict(branch))
            except Exception as e:  # noqa: BLE001 - diagnostics only
                self.last_shadow_requests.append({"error": f"{type(e).__name__}: {e}"})

        def run_shadow(i: int) -> None:
            t0 = time.perf_counter()
            try:
                results[i]["response"] = self.shadows[i].create(branches[i])
            except Exception as e:  # noqa: BLE001 - shadows are diagnostics only
                results[i]["error"] = f"{type(e).__name__}: {e}"
            results[i]["elapsed_s"] = time.perf_counter() - t0

        threads = [threading.Thread(target=run_shadow, args=(i,), name=f"shadow-{c.model}", daemon=True)
                   for i, c in enumerate(self.shadows)]
        t_start = time.perf_counter()
        for th in threads:
            th.start()
        stream = getattr(self.primary, "create_streamed", None)
        if on_delta is not None and callable(stream):
            resp = stream(input_items, on_delta)
        else:
            resp = self.primary.create(input_items)
        deadline = time.perf_counter() + self.wait_timeout_s
        for client, th, result in zip(self.shadows, threads, results):
            th.join(timeout=max(0.0, deadline - time.perf_counter()))
            if th.is_alive():
                self.last_shadows.append(ShadowResult(
                    client.model, None, f"no answer within {self.wait_timeout_s:.0f}s after the controlling model",
                    time.perf_counter() - t_start))
            else:
                self.last_shadows.append(ShadowResult(client.model, result.get("response"), result.get("error"),
                                                      result.get("elapsed_s", 0.0)))
        self._record_turn(resp)
        return resp

    def _record_turn(self, resp: AstraResponse) -> None:
        """Remember each shadow's proposal under every id of the controlling model's output for this turn."""
        keys = [k for k in (_item_key(it) for it in resp.output_items if _is_model_output(it)) if k]
        if not keys:
            return
        first = resp.function_calls[0] if resp.function_calls else None
        if first is None:
            executed = None
        elif first.name == "move_to":
            executed = f"move_to {json.dumps(_targets_of(first) or {})}"
        else:
            executed = first.name
        for k, shadow in enumerate(self.last_shadows):
            r = shadow.response
            turn = _Turn(output_items=list(r.output_items) if r else [],
                         call_ids=[fc.call_id for fc in r.function_calls] if r else [],
                         executed=executed, error=shadow.error if not r else None)
            for key in keys:
                self._turns[k][key] = turn
