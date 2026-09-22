"""Connected System 2 text planning and System 1 motion conversations, scoped to one trial."""
from __future__ import annotations

import json
import re
from dataclasses import replace
from pathlib import Path
from typing import List, Optional

from utils.astra_client import AstraResponse
from utils.config import PipelineConfig
from utils.observation import prune_image_history, redact_images


# System 2 writes its instruction to System 1 first and its reasoning for the record afterwards, fenced by
# these markers. Putting the instruction first is what makes the early dispatch worth anything: the motion
# request can leave while the planner is still writing, and its time-to-first-token disappears under the
# planner's remaining tokens. The markers are matched leniently because models decorate headings.
INSTRUCTION_OPEN = "NEXT INSTRUCTION"
INSTRUCTION_CLOSE = "END INSTRUCTION"
_INSTRUCTION_RE = re.compile(
    re.escape(INSTRUCTION_OPEN) + r"[*_:#\s]*(.*?)[*_:#\s]*" + re.escape(INSTRUCTION_CLOSE),
    re.IGNORECASE | re.DOTALL)


def extract_instruction(text: str) -> Optional[str]:
    """The fenced instruction block, or None while it is absent or still being written.

    None is not a failure: a planner that ignores the format simply hands its whole message over and the
    cycle runs sequentially, exactly as it did before.
    """
    match = _INSTRUCTION_RE.search(text or "")
    if match is None:
        return None
    return match.group(1).strip().strip("*_# \n") or None


def planner_config(cfg: PipelineConfig):
    """Enforce the planner's text-only wire contract independently of motion settings."""
    return replace(cfg.planning.planner, actions_only=False, tool_choice="none", parallel_tool_calls=False)


def build_planner_prompt(cfg: PipelineConfig) -> str:
    from utils.arm_models import get_arm_model

    text = Path(cfg.planning.system_prompt_path).read_text().strip()
    # The shared planner prompt describes the YAM's 9.5 cm jaws; other grippers substitute their own opening.
    jaw_cm = get_arm_model(cfg.embodiment_name).jaw_max_opening_m * 100
    text = text.replace("about 9.5 cm", f"about {jaw_cm:g} cm")
    # No Cartesian bounds here: System 2 plans in words, and a table of numeric limits only invites the
    # coordinates the prompt forbids. System 1 still receives them in the move_to tool schema, and the
    # gateway rejects anything outside them regardless.
    per_plan = max(1, int(cfg.planning.motion_calls_per_plan))
    text += (f"\n\nEmbodiment name: {cfg.embodiment_name}."
             f"\nWaypoint cadence: {cfg.motion.cadence_hz:g} Hz.")
    if cfg.motion.max_waypoints_per_call > 1:
        text += (f"\nSystem 1 may send up to {int(cfg.motion.max_waypoints_per_call)} waypoints in one call, "
                 "executed as one continuous validated motion.")
    if per_plan > 1:
        text += (f"\nSystem 1 takes up to {per_plan} turns on each of your instructions before you are asked "
                 "again; a rejected action, a changed scene or new operator guidance brings you back sooner.")
    text += (f"\nTotal inference-call budget: {cfg.limits.max_llm_calls}; each cycle uses one call of yours "
             f"and up to {per_plan} of System 1's.")
    if cfg.policy_notes_path:
        text += "\n\nOperator policy notes:\n" + Path(cfg.policy_notes_path).read_text().strip()
    if cfg.reactive.enabled:
        text += (f"\nExecution pauses for fresh observations after at most {cfg.reactive.max_motion_seconds:g} "
                 "seconds of motion; stale actions can be rejected before execution.")
    return text


def motion_items(items: List[dict], include_images: bool) -> List[dict]:
    """Keep robot state and the entire text history; perception normally belongs to System 2."""
    if include_images:
        return items
    return [
        {**it, "content": [
            {"type": "input_text", "text": "[Camera image supplied to System 2; use its scene assessment.]"}
            if p.get("type") == "input_image" else p for p in it["content"]]}
        if isinstance(it.get("content"), list) and it.get("role") == "user" else it
        for it in items
    ]


def response_payload(resp: AstraResponse) -> dict:
    return {"output": resp.output_items, "messages": resp.messages, "reasoning": resp.reasoning,
            "usage": resp.usage, "response_id": resp.response_id, "elapsed_s": resp.elapsed_s,
            "model": resp.model, "provider": resp.provider, "raw_response": resp.raw_response}


class PlanningSession:
    """Preserve each model's native output only in its own branch.

    The planner sees motion proposals/results as labeled user reports, never as its own tool calls.
    Only its final text crosses to System 1. Opaque provider reasoning stays with its producing model.
    """

    def __init__(self, cfg: PipelineConfig, client, logger, on_delta=None):
        self.cfg, self.client, self.logger = cfg, client, logger
        # System 2 is the slowest step in the loop (a reasoning model writes thousands of tokens before its
        # first visible word); stream them so the operator UI shows progress instead of a frozen console.
        self.on_delta = on_delta
        self.items = [{"role": "system", "content": build_planner_prompt(cfg)}]
        self.cursor = 0
        self.round = 0
        self.calls = 0
        self.usage = {}
        self.last_call_index = None
        self.last_plan_text = ""        # System 2's whole message; only its instruction block crosses over
        self._no_stream = False         # set if this endpoint will not stream (see `request`)
        logger.text_section("SYSTEM 2 SYSTEM PROMPT", self.items[0]["content"], stamp=False)

    def sync(self, motion_history: List[dict]) -> None:
        for it in motion_history[self.cursor:]:
            kind = it.get("type")
            if it.get("role") == "system":
                continue
            if kind == "reasoning":
                # Foreign encrypted reasoning cannot be replayed by another provider/model.
                continue
            if kind in ("function_call", "function_call_output") or it.get("role") == "assistant":
                label = "MOTION EXECUTION RESULT" if kind == "function_call_output" else "SYSTEM 1 PROPOSAL"
                self.items.append({"role": "user", "content": f"{label}: {json.dumps(it)}"})
            else:
                self.items.append(it)
        self.cursor = len(motion_history)

    def plan_message(self, instruction: str) -> dict:
        """The one item System 1 receives from System 2. Built here so the speculative dispatch and the
        sequential path produce the identical request, which is what makes speculation safe to reuse."""
        return {"role": "user", "content": f"SYSTEM 2 PLAN (cycle {self.round}):\n{instruction}"}

    def _delta_handler(self, on_instruction):
        """Forward display deltas, and fire `on_instruction` the moment the instruction block closes."""
        if self.on_delta is None and on_instruction is None:
            return None
        state = {"text": "", "fired": False, "seen": False}

        def handler(kind: str, delta: str, item_id: str) -> None:
            state["seen"] = state["seen"] or kind != "reset"
            if self.on_delta is not None:
                self.on_delta(kind, delta, item_id)
            if on_instruction is None or state["fired"]:
                return
            if kind == "reset":
                state["text"] = ""
                return
            if kind != "message":
                return
            state["text"] += delta
            instruction = extract_instruction(state["text"])
            if instruction is not None:
                state["fired"] = True
                on_instruction(instruction)     # must not block: the provider stream is still open
        handler.state = state
        return handler

    def request(self, motion_history: List[dict], call_index: int, on_instruction=None) -> AstraResponse:
        self.sync(motion_history)
        self.round += 1
        self.last_call_index = call_index
        self.items.append({"role": "user", "content":
            f"Planning cycle {self.round}. {self.cfg.limits.max_llm_calls - call_index} inference calls remain "
            "including this planner call and the next motion call. Assess the latest feedback and give the next instruction."})
        request_items = prune_image_history(self.items, self.cfg.planning.planner.image_history, self.cfg.prompts_path)
        stem = f"planner_{call_index:04d}.json"
        self.logger.write_json(f"requests/{stem}", redact_images(self.client.build_request_dict(request_items)))
        self.logger.event("planner_request", call_index=call_index, cycle=self.round, model=self.client.model)
        self.calls += 1
        handler = self._delta_handler(on_instruction)
        stream = getattr(self.client, "create_streamed", None)
        streaming = handler is not None and callable(stream) and not self._no_stream
        try:
            resp = stream(request_items, handler) if streaming else self.client.create(request_items)
        except Exception as e:
            # Speculation asks for a stream the trial did not previously need. An endpoint that refuses
            # one must not cost the run its planner: fall back once, and stop streaming from here on.
            if not (streaming and not handler.state["seen"]):
                self.logger.write_json(f"responses/{stem}", {"error": f"{type(e).__name__}: {e}"})
                self.logger.event("planner_error", call_index=call_index, cycle=self.round, error=str(e))
                raise
            self._no_stream = True
            self.logger.event("planner_stream_unavailable", call_index=call_index, cycle=self.round,
                              error=f"{type(e).__name__}: {e}")
            try:
                resp = self.client.create(request_items)
            except Exception as retry:
                self.logger.write_json(f"responses/{stem}", {"error": f"{type(retry).__name__}: {retry}"})
                self.logger.event("planner_error", call_index=call_index, cycle=self.round, error=str(retry))
                raise
        self.items.extend(resp.output_items)
        for key, value in resp.usage.items():
            self.usage[key] = self.usage.get(key, 0) + value
        self.logger.write_json(f"responses/{stem}", response_payload(resp))
        self.logger.text_section(f"SYSTEM 2 call {call_index + 1} / cycle {self.round}",
                                 f"model: {resp.model}; {resp.elapsed_s:.1f}s; usage: {resp.usage}")
        for trace in resp.reasoning:
            self.logger.text_block(self.logger.indent(trace, "[planner reasoning] "))
        for message in resp.messages:
            self.logger.text_block(self.logger.indent(message, "[plan] "))
        self.logger.event("planner_response", call_index=call_index, cycle=self.round,
                          response_id=resp.response_id, model=resp.model, usage=resp.usage)
        return resp

    def handoff(self, resp: AstraResponse, motion_history: List[dict]) -> str:
        """Append what System 1 acts on and return it: the instruction block alone when the planner fenced
        one, otherwise its whole message. The planner's own scene notes stay in the planner's branch, so
        System 1 is never asked to read, weigh, or restate them."""
        text = "\n\n".join(resp.messages).strip()
        if (resp.function_calls or any(it.get("type") not in ("message", "reasoning") for it in resp.output_items)
                or not text):
            raise ValueError("System 2 must return nonempty text only, without tool calls")
        self.last_plan_text = text
        instruction = extract_instruction(text) or text
        motion_history.append(self.plan_message(instruction))
        self.cursor = len(motion_history)  # the planner already has its own native output, not this copy
        self.logger.event("planning_handoff", cycle=self.round, planner_call_index=self.last_call_index,
                          motion_call_index=self.last_call_index + 1, plan=instruction,
                          fenced=instruction is not text)
        return instruction

    def finish(self, motion_history: List[dict]) -> None:
        self.sync(motion_history)
        self.logger.write_json("planner_history.json", redact_images(self.items))
        self.logger.write_json("motion_history.json", redact_images(
            motion_items(motion_history, self.cfg.planning.motion_images)))
