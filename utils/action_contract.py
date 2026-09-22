"""Local enforcement of the robot policy's strict action-only output contract."""
from __future__ import annotations

import math

from utils.config import DIM_NAMES


def _targets_error(targets) -> str | None:
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
    return None


def action_output_error(response, reactive: bool, max_waypoints: int = 1) -> str | None:
    """Return a protocol error before any action is dispatched.

    Reasoning summaries and encrypted metadata are allowed as diagnostics.
    Assistant messages and unexpected tool arguments never authorize motion.

    `max_waypoints` > 1 is the sequence schema: `move_to` carries `waypoints`, and `targets` is no longer
    an accepted field. The two spellings are never both legal, so a model cannot mean one and send the other.
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
        sequence = max(1, int(max_waypoints)) > 1
        field = "waypoints" if sequence else "targets"
        if set(call.arguments) != {field}:
            return f"move_to accepts only {field}; language fields are forbidden"
        if not sequence:
            return _targets_error(call.arguments["targets"])
        waypoints = call.arguments["waypoints"]
        if not isinstance(waypoints, list) or not waypoints:
            return "waypoints must be a non-empty list of target objects"
        if len(waypoints) > int(max_waypoints):
            return f"at most {int(max_waypoints)} waypoints may be sent in one call"
        for index, wp in enumerate(waypoints, start=1):
            error = _targets_error(wp)
            if error:
                return f"waypoint {index}: {error}"
    elif call.name in ("done", "give_up"):
        # One short reason, and nothing else: a trial that ends must say why.
        if set(call.arguments) - {"reason"}:
            return f"{call.name} accepts only reason"
        reason = call.arguments.get("reason")
        if reason is not None and not isinstance(reason, str):
            return f"{call.name} reason must be a string"
    elif reactive and call.name == "observe":
        if call.arguments:
            return "observe accepts an empty object only"
    else:
        return "response selected an unavailable action"
    return None
