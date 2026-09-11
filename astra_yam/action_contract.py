"""Local enforcement of the robot policy's strict action-only output contract."""
from __future__ import annotations

import math

from astra_yam.config import DIM_NAMES


def action_output_error(response, reactive: bool) -> str | None:
    """Return a protocol error before any action is dispatched.

    Reasoning summaries and encrypted metadata are allowed as diagnostics.
    Assistant messages and unexpected tool arguments never authorize motion.
    """
    if response.messages:
        return "action-only response contained an assistant message"
    calls = [item for item in response.output_items if item.get("type") == "function_call"]
    if len(calls) != 1 or len(response.function_calls) != 1:
        return "action-only response must contain exactly one action"
    for item in response.output_items:
        if item.get("type") not in ("function_call", "reasoning"):
            return "action-only response contained a non-action output item"
        if item.get("type") == "reasoning" and item.get("content"):
            return "action-only response contained reasoning content outside a summary"
    call = response.function_calls[0]
    if call.parse_error or not isinstance(call.arguments, dict):
        return "action arguments must be a JSON object"
    if call.name == "move_to":
        if set(call.arguments) != {"targets"}:
            return "move_to accepts only targets; language fields are forbidden"
        targets = call.arguments["targets"]
        if not isinstance(targets, dict) or set(targets) != set(DIM_NAMES):
            return "targets must contain every declared dimension; use null to hold a dimension"
        values = [v for v in targets.values() if v is not None]
        if not values:
            return "move_to requires at least one numeric target"
        try:
            if any(type(v) not in (float, int) or not math.isfinite(v) for v in values):
                return "targets must be finite numbers or null"
        except OverflowError:
            return "target number is too large"
    elif call.name in ("done", "give_up") or (reactive and call.name == "observe"):
        if call.arguments:
            return f"{call.name} accepts an empty object only"
    else:
        return "response selected an unavailable action"
    return None
