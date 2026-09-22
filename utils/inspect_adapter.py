"""Inspect Robots adapters using PromptRobots' Astra protocol and YAM gateway.

One framework step is one Cartesian tool call, possibly many joint waypoints.
Only the embodiment executes motion. Null targets retain the gateway's hold
semantics, including commanded (rather than measured) gripper closure.
"""
from __future__ import annotations

import copy
import hashlib
import json
import time
import uuid
from dataclasses import asdict, replace
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from inspect_robots.embodiment import EmbodimentBase, EmbodimentInfo
from inspect_robots.errors import EmbodimentFault, PolicyError, SafetyAbort
from inspect_robots.logging.sink import NullSink
from inspect_robots.policy import PolicyBase, PolicyConfig, PolicyInfo
from inspect_robots.spaces import ActionSemantics, Box, CameraSpec, ObservationSpace, StateField, StateSpec
from inspect_robots.types import Action, ActionChunk, Observation, StepResult

from utils.action_contract import action_output_error
from utils.astra_client import make_astra_client
from utils.cameras import encode_jpeg
from utils.cli import _make_robot_and_cameras
from utils.config import ARMS, DIM_NAMES, PipelineConfig, to_dict
from utils.embodiment import ARM_SLICES, build_policy_prompt, build_tools
from utils.gateway import GatewayRejection, SafetyGateway
from utils.kinematics import ArmKinematics
from utils.observation import build_observation_item, prune_image_history, redact_images


def action_space(cfg: PipelineConfig) -> Box:
    """Describe named Cartesian targets in each arm's own base frame."""
    bounds = [cfg.bounds.for_dim(name.split('_', 1)[1]) for name in DIM_NAMES]
    return Box(
        shape=(14,), low=np.array([b[0] for b in bounds]), high=np.array([b[1] for b in bounds]),
        semantics=ActionSemantics(
            control_mode="eef_abs_pose", rotation_repr="euler_xyz", gripper="continuous",
            frame="base", dim_labels=DIM_NAMES,
        ),
    )


OBSERVATION_SPACE = ObservationSpace(state=StateSpec(fields=(
    StateField("joint_pos", (14,), "rad+normalized"),
    StateField("eef_targets", (14,), "m+rad+normalized"),
)))


def observation_space(cfg):
    """Both station RealSenseCameraFast and SimCameraSource emit 640x480 RGB."""
    return replace(OBSERVATION_SPACE, cameras=tuple(
        CameraSpec(name=name, height=480, width=640) for name in cfg.cameras.names.values()))


class PromptYamEmbodiment(EmbodimentBase):
    """Execute Astra targets through the existing IK, clearance and tracking checks.

Hardware episodes begin at an operator-staged pose; this adapter never homes.
``prepare_scene`` must explicitly stage/confirm each hardware scene and repeat.
Resources are opened lazily on reset, so construction cannot move the robot.
"""

    def __init__(self, cfg: PipelineConfig, *, realtime=True, prepare_scene=None):
        if cfg.robot.backend != "sim" and prepare_scene is None:
            raise ValueError("hardware evaluation requires a prepare_scene callback")
        if cfg.robot.backend != "sim" and not realtime:
            raise ValueError("hardware motion must run in real time")
        if cfg.robot.home_at_start or cfg.home_on_end:
            raise ValueError("evaluation uses operator-staged starts; disable automatic homing")
        if cfg.reactive.enabled:
            raise ValueError("dynamic-scene execution is not supported by this eval adapter")
        if cfg.motion.tool_floor_z_m is not None and not np.isfinite(cfg.motion.tool_floor_z_m):
            raise ValueError("tool floor height must be finite")
        self.cfg = cfg
        self.realtime = realtime
        self.prepare_scene = prepare_scene
        self.robot = self.cameras = self.world = self.gateway = None
        self.kin = ArmKinematics.from_config(cfg)
        self.info = EmbodimentInfo(
            name=f"promptrobots_{cfg.embodiment_name}", action_space=action_space(cfg),
            observation_space=observation_space(cfg), is_simulated=cfg.robot.backend == "sim",
            capabilities=frozenset({"self_paced", "resettable"}),
            # A variable-duration Cartesian tool call has no fixed control_hz.
            docs="Each arm: x,y,z,yaw,pitch,roll,gripper. Metres/radians in its own base frame. "
                 "Extrinsic xyz Euler rotation relative to trial start, stored yaw,pitch,roll. "
                 f"Right arm frame origin in the left frame: {np.round(self.kin.arm_offset('right'), 3).tolist()} m. "
                 "Grippers: 0 closed, 1 open. "
                 "eef_targets reports measured poses with commanded gripper closure; "
                 "joint_pos contains measured grippers, which may stall on held objects. "
                 "meta.active_dims selects numeric targets; omitted dimensions hold. "
                 "Each step executes a complete gateway motion, not one control tick.",
        )

    def reset(self, scene, *, seed=None):
        """Stage a fresh scene and capture the orientation reference without homing."""
        if self.prepare_scene is not None:
            self.prepare_scene(scene)
        if self.robot is None:
            self.robot, self.cameras, self.world = _make_robot_and_cameras(self.cfg, self.kin)
        if self.world is not None:
            # The lightweight simulator has deterministic presets, not seeded physics.
            self.world.reset_objects()
            q0 = np.array([*self.cfg.robot.home_joints_left, self.cfg.robot.home_gripper,
                           *self.cfg.robot.home_joints_right, self.cfg.robot.home_gripper])
            self.robot.command_joint_positions(q0)
        self.scene = scene
        q = self.robot.get_joint_positions()
        start_rot = {arm: self.kin.fk(q[ARM_SLICES[arm]], arm)[1] for arm in ARMS}
        self.gateway = SafetyGateway(self.cfg, self.kin, self.robot, start_rot, self.realtime)
        self.deadline = time.perf_counter() + self.cfg.limits.max_trial_seconds
        self.rejections = 0
        self.feedback = None
        self.ended = False
        return self._observe()

    def _observe(self):
        q, _, eef = self.gateway.read_state()
        frames = self.cameras.read_jpeg_frames()
        if set(frames) != set(self.cfg.cameras.names.values()):
            raise EmbodimentFault("missing or unexpected YAM camera frames")
        images = {}
        for name, jpeg in frames.items():
            bgr = cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
            if bgr is None:
                raise EmbodimentFault(f"invalid JPEG from {name}")
            if bgr.shape != (480, 640, 3):
                raise EmbodimentFault(f"{name}: expected 640x480 RGB frame, got {bgr.shape}")
            images[name] = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        target_state = dict(eef)
        for arm in ARMS:
            target_state[f"{arm}_gripper"] = self.gateway.gripper_cmd[arm]
        return Observation(
            images=images, state={"joint_pos": q.copy(), "eef_targets": np.array(list(target_state.values()))},
            instruction=self.scene.instruction, state_time=time.time(),
            extra={"gateway_feedback": copy.deepcopy(self.feedback),
                   "waypoints_remaining": self.cfg.limits.max_waypoints - self.gateway.waypoints_executed,
                   "deadline": self.deadline, "measured_eef": eef},
        )

    def step(self, action):
        """Validate and execute only named numeric dimensions; return measured feedback."""
        if self.ended:
            raise EmbodimentFault("episode already ended; reset before another action")
        operation = action.meta.get("operation", "move_to")
        if action.meta.get("request_stop"):
            operation = str(action.meta.get("stop_reason", "policy_stop"))
        payload = {"ok": True, "status": operation}
        info = {}
        terminated = operation in {"done", "give_up"} or bool(action.meta.get("request_stop"))
        truncated = False
        if time.perf_counter() >= self.deadline:
            payload = {"ok": False, "status": "timeout"}
            terminated, truncated = False, True
        elif terminated:
            pass
        elif operation == "move_to":
            data = np.asarray(action.data)
            if data.shape != (14,) or not np.all(np.isfinite(data)):
                raise EmbodimentFault("expected a finite 14-dimensional Cartesian action")
            active = action.meta.get("active_dims", DIM_NAMES)
            if not active or any(name not in DIM_NAMES for name in active):
                raise EmbodimentFault("invalid active target dimensions")
            targets = {name: float(data[DIM_NAMES.index(name)]) for name in active}
            try:
                plan = self.gateway.plan(targets)
                remaining = self.cfg.limits.max_waypoints - self.gateway.waypoints_executed
                if plan.steps > remaining:
                    raise GatewayRejection(f"motion needs {plan.steps} waypoints; {remaining} remain")
            except GatewayRejection as exc:
                payload = {"ok": False, "status": "rejected", "reason": str(exc)}
                self.rejections += 1
                terminated = (self.cfg.limits.strict_gateway or
                              self.rejections >= self.cfg.limits.max_consecutive_rejections)
            else:
                # Planning and inference may consume the entire time budget.
                if time.perf_counter() >= self.deadline:
                    payload = {"ok": False, "status": "timeout"}
                    truncated = True
                else:
                    try:
                        result = self.gateway.execute(plan, deadline=self.deadline)
                    except BaseException:
                        self.gateway.hold()
                        raise
                    payload = asdict(result)
                    info = {"planned_waypoints": plan.steps,
                            "executed_joint_waypoints": plan.q_path[:result.steps_executed].tolist(),
                            "detour": plan.detour}
                    self.rejections = 0
                    if result.status in {"aborted", "estop"}:
                        self.gateway.hold()
                        raise SafetyAbort(result.reason or result.status)
                    truncated = result.status == "timeout" or self.gateway.waypoints_executed >= self.cfg.limits.max_waypoints
                    if truncated and result.status != "timeout":
                        payload["status"] = "waypoints_exhausted"
        elif not terminated and operation not in {"budget_exhausted", "timeout"}:
            raise EmbodimentFault(f"unknown operation {operation!r}")
        elif operation in {"budget_exhausted", "timeout"}:
            truncated = True
        self.feedback = payload
        self.ended = terminated or truncated
        return StepResult(
            observation=self._observe(), terminated=terminated, truncated=truncated,
            termination_reason=str(payload["status"]) if self.ended else None,
            info={**info, "gateway": payload},
        )

    def close(self):
        """Release cameras and transport; never home or release a held object."""
        try:
            if self.cameras is not None:
                self.cameras.close()
        finally:
            if self.robot is not None:
                self.robot.close()
            self.robot = self.cameras = self.world = self.gateway = None


class AstraInspectPolicy(PolicyBase):
    """Use PromptRobots' strict Responses tools, prompts and image history unchanged."""

    def __init__(self, cfg: PipelineConfig, *, client_factory=make_astra_client):
        if cfg.planning.enabled:
            raise ValueError("planning mode uses the utils run loop; the Inspect adapter does not support it")
        if not cfg.astra.actions_only or cfg.reactive.enabled:
            raise ValueError("eval currently requires static-scene, actions-only mode")
        self.cfg = cfg
        self.client_factory = client_factory
        self.client = None
        self.items, self.audit = [], []
        self.pending, self.calls, self.usage, self.latency = None, 0, {}, 0.0
        self.info = PolicyInfo(name=f"astra/{cfg.astra.model}" if cfg.astra.is_remote
                               else "scripted-astra", action_space=action_space(cfg),
                               observation_space=observation_space(cfg))
        self.config = PolicyConfig(action_horizon=1)
        # Pinned to the single-target schema whatever motion.max_waypoints_per_call says: an Inspect
        # ActionChunk is a list of flat target vectors, and how the surrounding harness executes several of
        # them (one continuous motion, or one gateway pass each) is not this adapter's to assume. Raise the
        # waypoint limit for the live loop; eval runs keep the reference one-target contract.
        self.tools = build_tools(cfg.bounds, cfg.prompts_path, actions_only=True, max_waypoints=1)

    def reset(self, scene):
        """Clear conversation, budgets and scripted replay at every epoch."""
        self.close()
        self.items = [{"role": "system", "content": build_policy_prompt(self.cfg)}]
        self.audit = redact_images(self.items)
        self.pending = None
        self.calls = 0
        self.usage = {}
        self.latency = 0.0
        self.client = self.client_factory(self.cfg.astra, self.tools)

    def _feedback(self, payload):
        if self.pending is not None:
            item = {"type": "function_call_output", "call_id": self.pending,
                    "output": json.dumps(payload)}
            self.items.append(item)
            self.audit.append(copy.deepcopy(item))
            self.pending = None

    def act(self, observation):
        """Infer one tool call; encode null holds in metadata, never as NaNs."""
        self._feedback(observation.extra.get("gateway_feedback"))
        data = observation.state["eef_targets"].copy()
        if time.perf_counter() >= observation.extra["deadline"]:
            return ActionChunk([Action(data, meta={"operation": "timeout"})])
        if self.calls >= self.cfg.limits.max_llm_calls:
            return ActionChunk([Action(data, meta={"operation": "budget_exhausted"})])
        frames = {name: encode_jpeg(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), self.cfg.cameras.jpeg_quality)
                  for name, rgb in observation.images.items()}
        item = build_observation_item(
            observation.instruction or "", observation.state["joint_pos"], observation.extra["measured_eef"],
            observation.extra["waypoints_remaining"], self.calls, frames,
            self.cfg.cameras.detail, self.cfg.prompts_path,
        )
        self.items.append(item)
        self.audit.append(redact_images(item))
        self.items = prune_image_history(self.items, self.cfg.astra.image_history, self.cfg.prompts_path)
        started = time.perf_counter()
        try:
            response = self.client.create(self.items)
        except Exception as exc:
            raise PolicyError(f"Astra request failed: {exc}") from exc
        elapsed = time.perf_counter() - started
        self.calls += 1
        self.latency += elapsed
        for key, value in response.usage.items():
            self.usage[key] = self.usage.get(key, 0) + value
        self.items.extend(response.output_items)
        self.audit.extend(redact_images(response.output_items))
        error = action_output_error(response, reactive=False)
        if error:
            raise PolicyError(error)
        call = response.function_calls[0]
        self.pending = call.call_id
        active = []
        if call.name == "move_to":
            for name, value in call.arguments["targets"].items():
                if value is not None:
                    data[DIM_NAMES.index(name)] = value
                    active.append(name)
        return ActionChunk(
            [Action(data, meta={"operation": call.name, "active_dims": active, "call_id": call.call_id})],
            inference_latency_s=elapsed,
        )

    def on_trial_end(self, record, log_dir, run_id):
        """Record model usage and execution feedback, including terminal calls."""
        if record.steps and self.pending == record.steps[-1].action.meta.get("call_id"):
            self._feedback(record.steps[-1].result.info.get("gateway"))
            # Core collects transcript() before this hook. Append the terminal
            # tool result to that already-normalized snapshot when retained.
            if isinstance(record.policy_transcript, dict) and "messages" in record.policy_transcript:
                record.policy_transcript["messages"].append(copy.deepcopy(self.audit[-1]))
        record.metadata.update(astra_calls=self.calls, astra_usage=dict(self.usage),
                               astra_latency_s=self.latency, promptrobots_config=to_dict(self.cfg))
        self.close()

    def transcript(self):
        """Return a detached audit trail with image payloads redacted."""
        return {"messages": copy.deepcopy(self.audit), "model": self.info.name,
                "usage": dict(self.usage)}

    def close(self):
        """Close the provider client when it supports explicit cleanup."""
        if self.client is not None:
            close = getattr(self.client, "close", None)
            if close is not None:
                close()
            self.client = None


class Rot6dYamEmbodiment(PromptYamEmbodiment):
    """Expose the same gateway using the agent plugin's supported rotation format.

Each arm has xyz, the first two rotation-matrix columns, and gripper closure.
Rotations are relative to trial start, exactly like PromptRobots' Euler angles.
    """

    def __init__(self, cfg, **kwargs):
        super().__init__(cfg, **kwargs)
        dimensions = ("x", "y", "z", "r00", "r10", "r20", "r01", "r11", "r21", "gripper")
        labels = tuple(f"{arm}_{dim}" for arm in ARMS for dim in dimensions)
        bounds = [cfg.bounds.for_dim(dim) if dim in {"x", "y", "z", "gripper"} else (-1.0, 1.0)
                  for _ in ARMS for dim in dimensions]
        space = Box(shape=(20,), low=np.array([b[0] for b in bounds]),
                    high=np.array([b[1] for b in bounds]), semantics=ActionSemantics(
                        "eef_abs_pose", rotation_repr="rot6d", gripper="continuous",
                        frame="base", dim_labels=labels))
        observations = ObservationSpace(cameras=self.info.observation_space.cameras, state=StateSpec(fields=(
            StateField("joint_pos", (14,), "rad+normalized"),
            StateField("eef_targets", (20,), "m+rotation_columns+normalized"),
        )))
        self.info = replace(self.info, action_space=space, observation_space=observations, docs=(
            "Each arm: x,y,z,r00,r10,r20,r01,r11,r21,gripper. Positions are metres in "
            "that arm's base frame (+x forward, +y left, +z up). Right base is at "
            "left-frame y=-0.61 m. Rotation is relative to trial start: the first two "
            "columns of a rotation matrix, normalized by Gram-Schmidt before execution. "
            "Identity/start rotation is [1,0,0,0,1,0]. Gripper 0=closed, 1=open. "
            "eef_targets contains measured poses and commanded gripper closure so "
            "holding an object does not relax the grip; joint_pos includes measured "
            "grippers. Each step executes a gateway Cartesian segment with IK, "
            "clearance, floor and tracking checks, and can take variable wall time. "
            "Keep rotation components unchanged when only translating or gripping."
        ))

    def _observe(self):
        observation = super()._observe()
        source = observation.state["eef_targets"]
        values = []
        for offset in (0, 7):
            x, y, z, yaw, pitch, roll, grip = source[offset:offset + 7]
            matrix = Rotation.from_euler("xyz", [roll, pitch, yaw]).as_matrix()
            values.extend([x, y, z, *matrix[:, :2].T.reshape(-1), grip])
        return replace(observation, state={"joint_pos": observation.state["joint_pos"],
                                           "eef_targets": np.asarray(values)})

    def step(self, action):
        """Convert rotation columns to the gateway's Euler targets without bypassing checks."""
        if action.meta.get("request_stop") or action.meta.get("operation") in {
                "done", "give_up", "budget_exhausted", "timeout"}:
            return super().step(action)
        data = np.asarray(action.data)
        if data.shape != (20,) or not np.all(np.isfinite(data)):
            raise SafetyAbort("expected a finite 20-dimensional rot6d action")
        values = []
        for offset in (0, 10):
            x, y, z = data[offset:offset + 3]
            a, b = data[offset + 3:offset + 6], data[offset + 6:offset + 9]
            if np.max(np.abs(np.r_[a, b])) > 1.0 + 1e-8 or np.linalg.norm(a) < 1e-8:
                raise SafetyAbort("invalid rot6d rotation columns")
            first = a / np.linalg.norm(a)
            second = b - np.dot(first, b) * first
            if np.linalg.norm(second) < 1e-8:
                raise SafetyAbort("rot6d rotation columns are collinear")
            second /= np.linalg.norm(second)
            matrix = np.column_stack((first, second, np.cross(first, second)))
            roll, pitch, yaw = Rotation.from_matrix(matrix).as_euler("xyz")
            values.extend([x, y, z, yaw, pitch, roll, data[offset + 9]])
        return super().step(Action(np.asarray(values), meta={**action.meta, "active_dims": DIM_NAMES}))


class GatewayTraceSink(NullSink):
    """Save gateway execution diagnostics for every policy through the sink API."""

    def __init__(self, log_dir):
        self.log_dir = Path(log_dir)

    def on_eval_start(self, spec):
        """Allocate a unique namespace for this evaluation's motion traces."""
        self.trace_dir = self.log_dir / "gateway" / uuid.uuid4().hex
        self.trace_dir.mkdir(parents=True, exist_ok=True)

    def on_trial_end(self, record):
        """Persist completed steps and attach their path to framework metadata."""
        scene_key = hashlib.sha256(record.scene_id.encode()).hexdigest()[:12]
        path = self.trace_dir / f"{scene_key}-e{record.epoch}.json"
        trace = [{"t": step.t, "action": step.action.data.tolist(),
                  "meta": dict(step.action.meta), "result": dict(step.result.info),
                  "joint_pos": step.result.observation.state["joint_pos"].tolist()}
                 for step in record.steps]
        path.write_text(json.dumps(trace, indent=2, allow_nan=False) + "\n")
        record.metadata["gateway_trace"] = str(path.relative_to(self.log_dir))
