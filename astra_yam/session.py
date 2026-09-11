"""The closed loop: observation -> Astra tool call -> gateway/robot -> observation ... until done/give_up
or a budget (LLM calls, waypoints, wall time) is exhausted."""
from __future__ import annotations

import datetime as dt
import json
import queue
import sys
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from astra_yam.astra_client import AstraClient
from astra_yam.cameras import CameraSource
from astra_yam.config import ARMS, NUM_DOFS, PipelineConfig, to_dict
from astra_yam.embodiment import ARM_SLICES, build_policy_prompt, build_tools
from astra_yam.gateway import SafetyGateway, move_joint_space
from astra_yam.kinematics import ArmKinematics
from astra_yam.logging_utils import TrialLogger
from astra_yam.observation import build_observation_item, prune_image_history, summarize_items
from astra_yam.prompts import prompt
from astra_yam.robot_interface import RobotBackend

STOP_COMMANDS = {"/stop", "stop", "/abort", "abort"}


@dataclass
class _Chunk:
    """One `move_to` call = one action chunk: what the gateway predicted, and what actually ran."""

    call: int
    cartesian: int          # waypoints of the straight Cartesian path at motion.cadence_hz
    predicted: int          # waypoints after joint pacing - what the plan was going to stream
    executed: int           # waypoints actually streamed (< predicted when a motion is cut short)
    paced: bool
    status: str


def _chunk_table(chunks: List[_Chunk], cadence_hz: float) -> str:
    """Per-call action-chunk sizes plus totals, for the end of the transcript."""
    if not chunks:
        return "no move_to calls in this trial"
    head = f"{'call':>4}  {'cartesian':>9}  {'predicted':>9}  {'executed':>8}  {'paced':>5}  {'seconds':>7}  status"
    rows = [head, "-" * len(head)]
    for c in chunks:
        rows.append(f"{c.call:>4}  {c.cartesian:>9}  {c.predicted:>9}  {c.executed:>8}  "
                    f"{'yes' if c.paced else 'no':>5}  {c.executed / max(cadence_hz, 1e-9):>7.1f}  {c.status}")
    sizes = [c.executed for c in chunks if c.executed]
    predicted, executed = sum(c.predicted for c in chunks), sum(c.executed for c in chunks)
    rejected = sum(1 for c in chunks if c.status == "rejected")
    paced = sum(1 for c in chunks if c.paced)
    rows.append("")
    rows.append(f"{len(chunks)} move_to calls ({rejected} rejected, {paced} paced), "
                f"{executed} waypoints executed of {predicted} predicted")
    if sizes:
        rows.append(f"executed chunk size: mean {sum(sizes) / len(sizes):.1f}, min {min(sizes)}, max {max(sizes)} "
                    f"waypoints ({sum(sizes) / len(sizes) / max(cadence_hz, 1e-9):.1f} s mean at {cadence_hz:g} Hz)")
    return "\n".join(rows)


def _time_table(spent: Dict[str, float], total: float, llm_calls: int, chunks: List["_Chunk"], motion) -> str:
    """Where the wall clock went - the two big terms are model latency and motion time."""
    moves = [c for c in chunks if c.executed]
    rows = [f"{'phase':<22}{'seconds':>9}{'share':>8}  per call"]
    rows.append("-" * len(rows[0]))
    for name, key, n in (("Astra (API + model)", "astra", llm_calls),
                         ("motion (gateway)", "motion", len(moves)),
                         ("observation capture", "observation", llm_calls)):
        s = spent.get(key, 0.0)
        per = f"{s / n:.2f} s x {n}" if n else "-"
        rows.append(f"{name:<22}{s:>9.1f}{100.0 * s / max(total, 1e-9):>7.0f}%  {per}")
    other = total - sum(spent.values())
    rows.append(f"{'everything else':<22}{other:>9.1f}{100.0 * other / max(total, 1e-9):>7.0f}%  "
                f"(homing, logging, planning)")
    rows.append(f"{'TOTAL':<22}{total:>9.1f}{100:>7.0f}%")
    rows.append("")
    rows.append(f"motion budget: {motion.linear_speed_mps * 100:g} cm/s, {motion.yaw_speed_rps:g} rad/s yaw, "
                f"gripper {motion.gripper_speed_per_s:g}/s, {motion.settle_seconds:g} s settle per move "
                f"(`--fast` raises these; every waypoint costs 1/{motion.cadence_hz:g} s)")
    return "\n".join(rows)


@dataclass
class TrialOutcome:
    status: str                 # done | give_up | budget_exhausted | waypoints_exhausted | timeout | aborted |
                                # estop |
                                # rejected_packet | too_many_rejections | operator_stop | error
    goal: str
    summary: Optional[str] = None
    hindsight: Optional[str] = None
    reason: Optional[str] = None
    llm_calls: int = 0
    waypoints: int = 0                              # waypoints executed
    waypoints_predicted: int = 0                    # waypoints the accepted plans were going to stream
    chunk_sizes: List[int] = field(default_factory=list)   # executed waypoints per move_to call
    elapsed_s: float = 0.0
    usage: Dict[str, int] = field(default_factory=dict)
    log_dir: Optional[str] = None
    rejections: int = 0
    error: Optional[str] = None
    stale_actions: int = 0
    observation_pauses: int = 0

    def as_dict(self) -> dict:
        return asdict(self)


class OperatorInput:
    """Non-blocking operator feedback lines from stdin (when interactive) and/or an appended-to text file."""

    def __init__(self, use_stdin: bool = True, file_path: Optional[str] = None):
        self._q: "queue.Queue[str]" = queue.Queue()
        self._stop = threading.Event()
        self._threads: List[threading.Thread] = []
        self.extra_sources: List[callable] = []       # callables returning List[str] (e.g. a GUI's poll_operator)
        if use_stdin and sys.stdin is not None and sys.stdin.isatty():
            t = threading.Thread(target=self._read_stdin, daemon=True)
            t.start()
            self._threads.append(t)
        if file_path:
            t = threading.Thread(target=self._tail_file, args=(file_path,), daemon=True)
            t.start()
            self._threads.append(t)

    def _read_stdin(self):
        while not self._stop.is_set():
            line = sys.stdin.readline()
            if not line:
                return
            line = line.strip()
            if line:
                self._q.put(line)

    def _tail_file(self, path: str):
        p = Path(path)
        p.touch(exist_ok=True)
        with open(p, "r") as f:
            f.seek(0, 2)
            while not self._stop.is_set():
                line = f.readline()
                if line:
                    line = line.strip()
                    if line:
                        self._q.put(line)
                else:
                    time.sleep(0.2)

    def add_source(self, fn) -> None:
        self.extra_sources.append(fn)

    def poll(self) -> List[str]:
        lines = []
        while True:
            try:
                lines.append(self._q.get_nowait())
            except queue.Empty:
                break
        for fn in self.extra_sources:
            try:
                lines.extend(fn())
            except Exception:  # noqa: BLE001
                pass
        return lines

    def close(self):
        self._stop.set()


def _function_call_output(call_id: str, payload: dict) -> dict:
    return {"type": "function_call_output", "call_id": call_id, "output": json.dumps(payload)}


def _accumulate(total: Dict[str, int], usage: Dict[str, int]) -> None:
    for k, v in (usage or {}).items():
        total[k] = total.get(k, 0) + int(v or 0)


class TrialRunner:
    def __init__(
        self,
        cfg: PipelineConfig,
        robot: RobotBackend,
        cameras: CameraSource,
        kin: ArmKinematics,
        astra: AstraClient,
        operator: Optional[OperatorInput] = None,
        realtime: bool = True,
        confirm: Optional[callable] = None,
        sim_world=None,
        log_root: Optional[str] = None,
        verbose: bool = True,
        hooks=None,
    ):
        """`hooks` may define any of: on_trial_start(goal, q0), on_observation(step, q, eef, frames, remaining),
        on_astra_call(i, max_calls, n_items, n_images), on_astra_stream(kind, delta, item_id),
        on_astra_response(resp), on_plan(plan),
        on_tool_call(name, args, payload, plan), on_end(outcome). Failures in hooks are logged, never fatal."""
        self.hooks = hooks
        # Emergency stop, shared with the gateway once a trial starts: set it from any thread (the viser
        # E-STOP button) to halt a motion in flight and end the session.
        self.estop = threading.Event()
        self.reobserve = threading.Event()
        self.cfg = cfg
        self.robot = robot
        self.cameras = cameras
        self.kin = kin
        self.astra = astra
        self.operator = operator
        self.realtime = realtime
        self.confirm = confirm
        self.sim_world = sim_world
        self.log_root = log_root or cfg.log_dir
        self.verbose = verbose
        self.tools = build_tools(cfg.bounds, cfg.prompts_path, reactive=cfg.reactive.enabled,
                                 actions_only=cfg.astra.actions_only)
        self.system_prompt = build_policy_prompt(cfg)

    # ------------------------------------------------------------------ utils
    def request_estop(self) -> None:
        """Halt the current motion and end the session. Safe to call from another thread."""
        self.estop.set()

    def request_reobserve(self) -> None:
        """UI/perception event: pause an in-flight move, invalidate the pending decision."""
        self.reobserve.set()

    def _prompt(self, key: str, **fmt) -> str:
        return prompt(key, self.cfg.prompts_path, **fmt)

    def _say(self, msg: str) -> None:
        if self.verbose:
            print(msg, flush=True)

    def _hook(self, name: str, *args) -> None:
        fn = getattr(self.hooks, name, None) if self.hooks is not None else None
        if fn is None:
            return
        try:
            fn(*args)
        except Exception as e:  # noqa: BLE001
            self._say(f"[hooks] {name} failed: {type(e).__name__}: {e}")

    def home_joints(self) -> np.ndarray:
        r = self.cfg.robot
        q = np.zeros(NUM_DOFS)
        q[0:6] = r.home_joints_left
        q[6] = r.home_gripper
        q[7:13] = r.home_joints_right
        q[13] = r.home_gripper
        return q

    def go_home(self, logger: Optional[TrialLogger] = None) -> np.ndarray:
        q_home = self.home_joints()
        q_now = self.robot.get_joint_positions()
        arm_idx = np.r_[0:6, 7:13]
        max_delta = float(np.max(np.abs((q_home - q_now)[arm_idx])))
        if max_delta > self.cfg.robot.max_homing_joint_delta_rad:
            raise RuntimeError(
                f"refusing to home: a joint would move {max_delta:.2f} rad (> {self.cfg.robot.max_homing_joint_delta_rad}). "
                f"Move the arms closer to the home pose first (current: {np.round(q_now, 3).tolist()})."
            )
        # preview the grasp-point path for the operator
        lows = []
        for a in np.linspace(0, 1, 20):
            q = q_now + (q_home - q_now) * a
            for arm in ARMS:
                pos, _ = self.kin.fk(q[ARM_SLICES[arm]])
                lows.append(pos[2])
        self._say(f"[home] moving both arms to the home pose over {self.cfg.motion.homing_seconds:.1f}s "
                  f"(max joint delta {max_delta:.2f} rad, lowest grasp-point z along the path {min(lows):.3f} m)")
        if self.confirm is not None and not self.confirm("Move the robot to the home pose now?"):
            raise RuntimeError("operator declined homing")
        q_final = move_joint_space(self.robot, q_home, self.cfg.motion.homing_seconds, self.cfg.motion.control_hz,
                                   realtime=self.realtime)
        if logger:
            logger.event("homed", joint_pos=q_final.tolist())
        return q_final

    def _observe(self, gateway: SafetyGateway, goal: str, remaining: int, step: int, logger: TrialLogger) -> dict:
        q, poses, eef = gateway.read_state()
        frames = self.cameras.read_jpeg_frames()
        self._last_frames = frames
        self._observation_sequence += 1
        paths = logger.save_frames(step, frames, sequence=self._observation_sequence if self.cfg.reactive.enabled else None)
        payload = {"step": step, "joint_pos": q.tolist(), "eef": eef, "frames": paths}
        if self.sim_world is not None:
            payload["sim_world"] = self.sim_world.status()
        logger.event("observation", **payload)
        self._hook("on_observation", step, q, eef, frames, remaining)
        item = build_observation_item(goal, q, eef, remaining, step, frames, self.cfg.cameras.detail,
                                      self.cfg.prompts_path)
        if self.cfg.reactive.enabled:
            item["content"][0]["text"] += "\n" + self._prompt(
                "session.action_context" if self.cfg.astra.actions_only else "session.reactive_context",
                sequence=self._observation_sequence,
                elapsed=logger.elapsed, lessons=json.dumps(self._episode_memory.lessons))
        text = "\n".join(p["text"] for p in item["content"] if p.get("type") == "input_text")
        logger.text_section(f"OBSERVATION step {step}", text)
        if paths:
            logger.text_block("images: " + ", ".join(f"{cam} -> {rel}" for cam, rel in paths.items()))
        return item

    # -------------------------------------------------------------------- run
    def run(self, goal: str) -> TrialOutcome:
        cfg = self.cfg
        lim = cfg.limits
        logger = TrialLogger(self.log_root, goal)
        self._say(f"[trial] goal: {goal}\n[trial] logging to {logger.dir}")
        logger.event("config", goal=goal, config=to_dict(cfg), model=getattr(self.astra, "model", "?"))
        model_name = getattr(self.astra, "model", "?")
        logger.text_section(f"TRIAL {logger.dir.name}", stamp=False,
                            body=f"goal:  {goal}\nmodel: {model_name}\nstarted: "
                                 f"{dt.datetime.now().isoformat(timespec='seconds')}")
        logger.text_section("SYSTEM PROMPT", self.system_prompt, stamp=False)
        logger.text_section("GOAL", goal, stamp=False)
        t_start = time.perf_counter()
        deadline = t_start + lim.max_trial_seconds
        outcome = TrialOutcome(status="error", goal=goal, log_dir=str(logger.dir))
        usage_total: Dict[str, int] = {}
        from astra_yam.reactive import EpisodeMemory
        self._episode_memory = EpisodeMemory()
        self._observation_sequence = -1
        self._last_frames = {}
        llm_calls = executed = total_rejections = 0
        chunks: List[_Chunk] = []
        gateway: Optional[SafetyGateway] = None
        try:
            if cfg.robot.home_at_start:
                q0 = self.go_home(logger)
            else:
                q0 = self.robot.get_joint_positions()
            start_rot = {arm: self.kin.fk(q0[ARM_SLICES[arm]])[1] for arm in ARMS}
            gateway = SafetyGateway(cfg, self.kin, self.robot, start_rot, realtime=self.realtime)
            gateway.on_plan = lambda plan: self._hook("on_plan", plan)
            gateway.estop = self.estop
            if cfg.reactive.enabled:
                gateway.reobserve = self.reobserve
            logger.event("trial_start", joint_pos=q0.tolist(), start_rot={a: start_rot[a].tolist() for a in ARMS})
            self._hook("on_trial_start", goal, q0)

            remaining = lim.max_waypoints
            executed = 0
            llm_calls = 0
            rejections = 0
            total_rejections = 0
            chunks: List[_Chunk] = []
            spent = {"astra": 0.0, "motion": 0.0, "observation": 0.0}   # wall clock per phase, for the transcript
            input_items: List[dict] = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": self._prompt("session.goal", goal=goal)},
                None,   # observation 0, filled in below so its capture time is measured like the others
            ]
            t_obs = time.perf_counter()
            input_items[2] = self._observe(gateway, goal, remaining, executed, logger)
            spent["observation"] += time.perf_counter() - t_obs
            status: Optional[str] = None

            while status is None:
                if self.estop.is_set():
                    status, outcome.reason = "estop", "emergency stop requested by the operator"
                    break
                if llm_calls >= lim.max_llm_calls:
                    status, outcome.reason = "budget_exhausted", f"LLM call budget of {lim.max_llm_calls} used"
                    break
                if time.perf_counter() > deadline:
                    status, outcome.reason = "timeout", f"trial time limit of {lim.max_trial_seconds:.0f}s reached"
                    break
                if remaining <= 0:
                    status, outcome.reason = "waypoints_exhausted", "gateway waypoint budget exhausted"
                    break
                if self.operator is not None:
                    for line in self.operator.poll():
                        if line.strip().lower() in STOP_COMMANDS:
                            status, outcome.reason = "operator_stop", "operator requested stop"
                            break
                        if cfg.reactive.enabled and line.strip().lower() in ("/reobserve", "/scene_changed"):
                            self.reobserve.set()
                            logger.event("scene_change_notification", source="operator")
                            continue
                        logger.event("operator_feedback", text=line)
                        logger.note(f"OPERATOR: {line}")
                        logger.text_section("OPERATOR FEEDBACK", line)
                        input_items.append({"role": "user",
                                            "content": self._prompt("session.operator_feedback", feedback=line)})
                    if status is not None:
                        break

                if cfg.reactive.enabled and self.reobserve.is_set():
                    self.reobserve.clear()
                    input_items.append(self._observe(gateway, goal, remaining, executed, logger))

                request_items = prune_image_history(input_items, cfg.astra.image_history, cfg.prompts_path)
                n_items, n_images = summarize_items(request_items)
                logger.log_request(llm_calls, self.astra.build_request_dict(request_items))
                self._say(f"[astra] call {llm_calls + 1}/{lim.max_llm_calls}: {n_items} items, {n_images} images ...")
                self._hook("on_astra_call", llm_calls + 1, lim.max_llm_calls, n_items, n_images)
                t_call = time.perf_counter()
                try:
                    stream = getattr(self.astra, "create_streamed", None)
                    if callable(stream) and callable(getattr(self.hooks, "on_astra_stream", None)):
                        resp = stream(request_items, lambda *delta: self._hook("on_astra_stream", *delta))
                    else:
                        resp = self.astra.create(request_items)
                    spent["astra"] += time.perf_counter() - t_call
                except Exception as e:  # noqa: BLE001 - SDK retries already exhausted
                    spent["astra"] += time.perf_counter() - t_call
                    status, outcome.error = "error", f"Astra request failed: {type(e).__name__}: {e}"
                    logger.event("astra_error", error=outcome.error)
                    break
                llm_calls += 1
                _accumulate(usage_total, resp.usage)
                logger.log_response(llm_calls - 1, {"output": resp.output_items, "usage": resp.usage,
                                                     "response_id": resp.response_id, "elapsed_s": resp.elapsed_s,
                                                     "model": resp.model})
                if cfg.astra.actions_only:
                    from astra_yam.action_contract import action_output_error
                    protocol_error = action_output_error(resp, cfg.reactive.enabled)
                    if protocol_error:
                        status, outcome.reason = "invalid_action_output", protocol_error
                        total_rejections += 1
                        logger.event("invalid_action_output", reason=protocol_error)
                        logger.text_section("ACTION OUTPUT REJECTED", protocol_error)
                        break
                self._say(f"[astra] {resp.elapsed_s:.1f}s, usage {resp.usage}")
                usage_line = ", ".join(f"{k}={v}" for k, v in (resp.usage or {}).items())
                n_in = int((resp.usage or {}).get("input_tokens") or 0)
                n_cached = int((resp.usage or {}).get("cached_tokens") or 0)
                cache_line = f", {100.0 * n_cached / n_in:.0f}% of input from cache" if n_in and n_cached else ""
                logger.text_section(f"ASTRA call {llm_calls}",
                                    f"({resp.elapsed_s:.1f}s; {n_images} images in this request{cache_line}; "
                                    f"{usage_line})")
                for trace in resp.reasoning:
                    logger.text_block(logger.indent(trace, "[reasoning] "))
                    self._say(f"[reasoning summary] {trace}")
                if not resp.reasoning and not cfg.astra.actions_only:
                    n_tok = (resp.usage or {}).get("reasoning_tokens")
                    logger.text_block(f"[reasoning] (not returned as text{f'; {n_tok} reasoning tokens' if n_tok else ''})")
                for msg in resp.messages:
                    logger.text_block(logger.indent(msg, "[message]   "))
                self._hook("on_astra_response", resp)
                input_items.extend(resp.output_items)

                # Guidance and scene changes arriving during inference invalidate
                # this response before ANY tool (including release/done) executes.
                feedback_changed = False
                if self.operator is not None:
                    for line in self.operator.poll():
                        if line.strip().lower() in STOP_COMMANDS:
                            status, outcome.reason = "operator_stop", "operator requested stop"
                            break
                        feedback_changed = True
                        if cfg.reactive.enabled and line.strip().lower() in ("/reobserve", "/scene_changed"):
                            self.reobserve.set()
                            logger.event("scene_change_notification", source="operator")
                        else:
                            logger.event("operator_feedback", text=line)
                            input_items.append({"role": "user", "content": self._prompt(
                                "session.operator_feedback", feedback=line)})
                if status is not None or time.perf_counter() > deadline:
                    if status is None:
                        status, outcome.reason = "timeout", "trial time limit reached during model inference"
                    for call in resp.function_calls:
                        input_items.append(_function_call_output(call.call_id, {"ok": False, "status": status}))
                    break
                if cfg.reactive.enabled:
                    from astra_yam.reactive import changed_fraction
                    fraction = changed_fraction(self._last_frames, self.cameras.read_jpeg_frames(), cfg.reactive)
                    if fraction >= cfg.reactive.change_fraction or self.reobserve.is_set() or feedback_changed:
                        self.reobserve.clear()
                        outcome.stale_actions += 1
                        for call in resp.function_calls:
                            payload = {"ok": False, "status": "stale_observation", "reason": self._prompt("session.scene_changed")}
                            input_items.append(_function_call_output(call.call_id, payload))
                            logger.event("tool_call", name=call.name, arguments=call.arguments, result=payload)
                        logger.event("stale_decision", changed_fraction=fraction, feedback_changed=feedback_changed)
                        input_items.append(self._observe(gateway, goal, remaining, executed, logger))
                        continue

                if not resp.function_calls:
                    text = " ".join(resp.messages)[:400]
                    logger.event("no_tool_call", text=text)
                    logger.note(f"(no tool call) {text}")
                    logger.text_block("[no tool call] a reminder was sent instead")
                    input_items.append({"role": "user",
                                        "content": self._prompt("session.reactive_reminder" if cfg.reactive.enabled
                                                                else "session.tool_call_reminder")})
                    continue

                fc = resp.function_calls[0]
                for extra in resp.function_calls[1:]:
                    input_items.append(_function_call_output(extra.call_id, {
                        "ok": False, "status": "rejected", "reason": "only the first tool call of a turn is executed"}))

                if fc.parse_error is not None:
                    payload = {"ok": False, "status": "rejected", "reason": f"arguments were not valid JSON: {fc.parse_error}"}
                    input_items.append(_function_call_output(fc.call_id, payload))
                    rejections += 1
                    total_rejections += 1
                    logger.event("tool_call", name=fc.name, arguments=fc.raw_arguments, result=payload)
                elif fc.name == "move_to":
                    args = fc.arguments or {}
                    targets, note = args.get("targets"), args.get("note")
                    if cfg.astra.actions_only:
                        targets = {name: value for name, value in targets.items() if value is not None}
                    logger.note(f"step {executed}: move_to {json.dumps(targets)}" +
                                (f"\n  note: {note}" if not cfg.astra.actions_only else ""))
                    self._say(f"[move_to] {targets}" + (f"\n[note] {note}" if not cfg.astra.actions_only else ""))
                    t_move = time.perf_counter()
                    payload, plan, res = gateway.move_to(targets, remaining, deadline)
                    spent["motion"] += time.perf_counter() - t_move
                    if note is None and not cfg.astra.actions_only:
                        payload["warning"] = "note missing; every move must include a note"
                    input_items.append(_function_call_output(fc.call_id, payload))
                    logger.event("tool_call", name="move_to", arguments=args, result=payload,
                                 plan={"steps": plan.steps, "cartesian_steps": plan.cartesian_steps, "paced": plan.paced,
                                       "resolved_targets": plan.resolved_targets,
                                       "min_clearance_m": plan.min_clearance_m} if plan else None,
                                 tracking_err=res.max_tracking_err_rad if res else None)
                    self._say(f"[gateway] {payload}")
                    logger.text_block(logger.indent(json.dumps(targets), "[move_to]   "))
                    if not cfg.astra.actions_only:
                        logger.text_block(logger.indent(note or "(missing)", "[note]      "))
                        logger.astra_note(note, f"call {llm_calls}, step {executed}")
                    logger.text_block(logger.indent(json.dumps(payload), "[gateway]   "))
                    chunk = _Chunk(call=llm_calls, cartesian=plan.cartesian_steps if plan else 0,
                                   predicted=plan.steps if plan else 0, executed=int(payload.get("steps", 0)),
                                   paced=bool(plan.paced) if plan else False, status=payload["status"])
                    chunks.append(chunk)
                    hz = cfg.motion.cadence_hz
                    pacing = f" -> {chunk.predicted} after joint pacing" if chunk.paced else ""
                    logger.text_block(f"[chunk]     {chunk.predicted} waypoints predicted "
                                      f"({chunk.cartesian} Cartesian at {hz:g} Hz{pacing}), "
                                      f"{chunk.executed} executed ({chunk.executed / max(hz, 1e-9):.1f} s)")
                    sizes = [c.executed for c in chunks if c.executed]
                    counters = (f"[counters]  move_to calls {len(chunks)}, waypoints "
                                f"{sum(c.executed for c in chunks)} executed / "
                                f"{sum(c.predicted for c in chunks)} predicted, "
                                f"{remaining - chunk.executed} of {lim.max_waypoints} budget left, "
                                f"{rejections} rejected in a row")
                    if sizes:
                        counters += (f", mean chunk {sum(sizes) / len(sizes):.1f} (min {min(sizes)}, "
                                     f"max {max(sizes)})")
                    logger.text_block(counters)
                    logger.event("chunk", **asdict(chunk))
                    self._hook("on_tool_call", "move_to", args, payload, plan)
                    if payload["ok"]:
                        remaining -= payload["steps"]
                        executed += payload["steps"]
                        rejections = 0
                        if cfg.reactive.enabled and not cfg.astra.actions_only:
                            self._episode_memory.add(args.get("lesson"))
                        if payload["status"] == "observation_required":
                            outcome.observation_pauses += 1
                            self.reobserve.clear()
                    elif payload["status"] == "rejected":
                        rejections += 1
                        total_rejections += 1
                        logger.note(f"  REJECTED: {payload['reason']}")
                        if lim.strict_gateway:
                            status, outcome.reason = "rejected_packet", payload["reason"]
                        elif rejections >= lim.max_consecutive_rejections:
                            status, outcome.reason = "too_many_rejections", \
                                f"{rejections} consecutive rejected packets (last: {payload['reason']})"
                    else:  # aborted / timeout during motion
                        remaining -= payload["steps"]
                        executed += payload["steps"]
                        status, outcome.reason = payload["status"], payload.get("reason")
                elif fc.name == "observe" and cfg.reactive.enabled:
                    gateway.hold()
                    if not cfg.astra.actions_only:
                        self._episode_memory.add((fc.arguments or {}).get("lesson"))
                    input_items.append(_function_call_output(fc.call_id, {"ok": True, "status": "observed", "steps": 0}))
                    logger.event("tool_call", name="observe", arguments=fc.arguments, result={"ok": True, "status": "observed"})
                    rejections = 0
                elif fc.name in ("done", "give_up"):
                    args = fc.arguments or {}
                    status = fc.name
                    outcome.summary = args.get("summary")
                    outcome.reason = args.get("reason")
                    outcome.hindsight = args.get("hindsight")
                    input_items.append(_function_call_output(fc.call_id, {"ok": True, "status": "session_ended"}))
                    logger.event("tool_call", name=fc.name, arguments=args)
                    if cfg.astra.actions_only:
                        logger.note(fc.name)
                        self._say(f"[{fc.name}]")
                        logger.text_block(f"[{fc.name}]")
                    else:
                        logger.note(f"{fc.name.upper()}: {outcome.summary or outcome.reason}\n  hindsight: {outcome.hindsight}")
                        self._say(f"[{fc.name}] {outcome.summary or outcome.reason}\n[hindsight] {outcome.hindsight}")
                        logger.text_block(logger.indent(outcome.summary or outcome.reason or "", f"[{fc.name}]" .ljust(12)))
                        logger.text_block(logger.indent(outcome.hindsight or "", "[hindsight] "))
                        logger.astra_note(outcome.summary or outcome.reason, f"{fc.name}, call {llm_calls}")
                        logger.astra_note_extra(outcome.hindsight, "hindsight")
                    self._hook("on_tool_call", fc.name, args, {"ok": True, "status": "session_ended"}, None)
                else:
                    payload = {"ok": False, "status": "rejected", "reason": f"unknown tool '{fc.name}'"}
                    input_items.append(_function_call_output(fc.call_id, payload))
                    rejections += 1
                    total_rejections += 1
                    logger.event("tool_call", name=fc.name, arguments=fc.arguments, result=payload)
                    logger.text_block(logger.indent(json.dumps(payload), "[rejected]  "))

                if status is None and rejections:
                    if lim.strict_gateway:
                        status, outcome.reason = "rejected_packet", payload["reason"]
                    elif rejections >= lim.max_consecutive_rejections:
                        status, outcome.reason = "too_many_rejections", f"{rejections} consecutive rejected packets"
                if status is None:
                    t_obs = time.perf_counter()
                    input_items.append(self._observe(gateway, goal, remaining, executed, logger))
                    spent["observation"] += time.perf_counter() - t_obs

            outcome.status = status or "error"
            outcome.llm_calls = llm_calls
            outcome.waypoints = executed
            outcome.waypoints_predicted = sum(c.predicted for c in chunks)
            outcome.chunk_sizes = [c.executed for c in chunks]
            outcome.rejections = total_rejections
            logger.text_section("ACTION CHUNKS", _chunk_table(chunks, cfg.motion.cadence_hz))
            logger.text_section("TIME", _time_table(spent, time.perf_counter() - t_start, llm_calls, chunks,
                                                    cfg.motion))
        except Exception as e:  # noqa: BLE001
            outcome.status = "error"
            outcome.error = f"{type(e).__name__}: {e}"
            logger.event("exception", error=outcome.error)
            self._say(f"[trial] ERROR: {outcome.error}")
        finally:
            outcome.llm_calls = llm_calls
            outcome.waypoints = executed
            outcome.rejections = total_rejections
            outcome.waypoints_predicted = sum(c.predicted for c in chunks)
            outcome.chunk_sizes = [c.executed for c in chunks]
            try:
                if gateway is not None:
                    gateway.hold()
                    if cfg.home_on_end and not self.estop.is_set() and outcome.status in ("done", "give_up"):
                        self._say("[trial] returning to the home pose")
                        move_joint_space(self.robot, self.home_joints(), cfg.motion.homing_seconds,
                                         cfg.motion.control_hz, realtime=self.realtime)
            except Exception as e:  # noqa: BLE001
                self._say(f"[trial] warning: could not hold/home at end: {e}")
            outcome.elapsed_s = round(time.perf_counter() - t_start, 2)
            outcome.usage = usage_total
            logger.finish(outcome.as_dict())
            self._hook("on_end", outcome)
            self._say(f"[trial] finished: {outcome.status} after {outcome.llm_calls} LLM calls, "
                      f"{outcome.waypoints} waypoints, {outcome.elapsed_s:.0f}s -> {logger.dir}")
        return outcome
