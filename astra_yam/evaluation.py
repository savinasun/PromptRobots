"""Environment-owned resets and independent scoring for repeatable YAM trials.

Follows ENPIRE's reset/execute/verify contract. Simulator state stays here;
the policy receives only the runner's normal camera/proprioception observations.
"""
from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import yaml

from astra_yam.config import PipelineConfig


@dataclass(frozen=True)
class Case:
    id: str
    goal: str
    scene: str
    verifier: str
    seed: int
    split: str
    jitter_m: float = 0.0
    # (one-based model response number OR "before_release", bowl dx, bowl dy). Applied while the
    # model's action is pending, never exposed to its prompt or observations as coordinates.
    disturbances: tuple = ()


def load_suite(path: str) -> list[Case]:
    data = yaml.safe_load(Path(path).read_text())
    if not isinstance(data, dict) or set(data) != {"cases"} or not isinstance(data["cases"], list):
        raise ValueError("suite must contain a cases list")
    cases = [Case(**{**item, "disturbances": tuple(tuple(d) for d in item.get("disturbances", []))}) for item in data["cases"]]
    if not cases or len({c.id for c in cases}) != len(cases):
        raise ValueError("suite needs nonempty, unique case IDs")
    for c in cases:
        if not c.id or not c.goal or c.split not in ("train", "validation"):
            raise ValueError(f"invalid case: {c.id}")
        if (c.scene, c.verifier) not in (("airpods", "airpods_open"), ("blocks", "blue_on_green"),
                                       ("airpod_bowl", "case_in_green_bowl")):
            raise ValueError(f"unsupported scene/verifier in {c.id}")
        if type(c.seed) is not int or c.seed < 0 or not np.isfinite(c.jitter_m) or not 0 <= c.jitter_m <= 0.02:
            raise ValueError(f"invalid seed or jitter in {c.id}")
        for disturbance in c.disturbances:
            if (c.scene != "airpod_bowl" or len(disturbance) != 3
                    or not ((type(disturbance[0]) is int and disturbance[0] >= 1) or disturbance[0] == "before_release")
                    or any(not np.isfinite(v) or abs(v) > 0.15 for v in disturbance[1:])):
                raise ValueError(f"invalid bowl disturbance in {c.id}")
    if {c.split for c in cases} != {"train", "validation"}:
        raise ValueError("suite requires train and validation cases")
    if {c.seed for c in cases if c.split == "train"} & {c.seed for c in cases if c.split == "validation"}:
        raise ValueError("train and validation seeds must be disjoint")
    return cases


def reset_scene(world, case: Case) -> None:
    world.reset_objects()
    rng = np.random.default_rng(case.seed)
    # Translate the complete scene together, preserving support relationships.
    offset = np.r_[rng.uniform(-case.jitter_m, case.jitter_m, 2), 0.0]
    for obj in world.objects.values():
        obj.pos += offset


def verify(world, case: Case) -> dict:
    if case.verifier == "case_in_green_bowl":
        obj, bowl = world.objects["airpods case"], world.objects["green bowl"]
        inside = bowl.contains_xy(obj)
        supported = abs(float(obj.pos[2] - obj.height / 2) - bowl.floor_z) < 0.005
        below_rim = obj.pos[2] + obj.height / 2 <= bowl.pos[2] + bowl.height / 2
        success = bool(inside and supported and below_rim and obj.held_by is None)
        return {"success": success, "score": 1.0 if success else 0.2 if obj.held_by is not None else 0.0,
                "reason": "released inside the current bowl interior" if success else "case not verified inside current bowl",
                "metrics": {"inside_rim": inside, "supported_inside": supported, "released": obj.held_by is None}}
    if case.verifier == "airpods_open":
        obj = world.objects["airpods case"]
        released = obj.lid_holder is None
        success = bool(obj.is_open and released and obj.held_by is not None)
        # Dense progress helps diagnose a failed attempt; only released, latched
        # openings count as success. The body must remain supported by an arm.
        score = (0.2 * (obj.held_by is not None) + 0.2 * obj.hanging
                 + 0.4 * min(float(obj.lid_angle) / obj.OPEN_LATCH_RAD, 1.0)
                 + 0.2 * success)
        return {"success": success, "score": float(score),
                "reason": "lid latched open and released; body held" if success else "opening not independently verified",
                "metrics": {"lid_angle_deg": float(np.degrees(obj.lid_angle)), "lid_released": released,
                            "body_held": obj.held_by is not None}}
    if case.verifier == "blue_on_green":
        blue, green = world.objects["blue block"], world.objects["green block"]
        xy = float(np.linalg.norm(blue.pos[:2] - green.pos[:2]))
        dz = abs(float(blue.pos[2] - green.pos[2]) - (blue.height + green.height) / 2)
        success = bool(blue.held_by is None and green.held_by is None and xy < 0.015 and dz < 0.005)
        return {"success": success, "score": float(success), "reason": "measured stacking geometry",
                "metrics": {"xy_error_m": xy, "height_error_m": dz}}
    raise ValueError(f"unknown verifier: {case.verifier}")


def fingerprint(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


class BowlDisturbances:
    """Environment-owned perturbations, including a matched release-stage trigger."""
    def __init__(self, world, case):
        self.world, self.case = world, case
        self.calls = 0
        self.applied_ids = set()
        self.events = []

    @property
    def applied(self):
        return len(self.applied_ids)

    def on_astra_response(self, response):
        self.calls += 1
        for index, (trigger, dx, dy) in enumerate(self.case.disturbances):
            if index in self.applied_ids:
                continue
            matches = trigger == self.calls
            if trigger == "before_release":
                obj = self.world.objects["airpods case"]
                arm = obj.held_by
                for call in response.function_calls:
                    targets = (call.arguments or {}).get("targets", {})
                    if call.name == "move_to" and arm and isinstance(targets, dict):
                        value = targets.get(f"{arm}_gripper")
                        matches = matches or (type(value) in (int, float) and np.isfinite(value)
                                               and self.world.grippers[arm] < value <= 1)
            if matches:
                bowl = self.world.objects["green bowl"]
                before = bowl.pos.copy()
                self.world.set_object_position("green bowl", before + np.array([dx, dy, 0.0]))
                self.applied_ids.add(index)
                self.events.append({"call": self.calls, "trigger": trigger, "before": before.tolist(),
                                    "after": bowl.pos.tolist()})
                break  # repeated before_release entries fire on separate release attempts


def run_case(cfg: PipelineConfig, case: Case, root: Path, notes: Path | None = None,
             feedback_file: str | None = None, client_factory=None) -> dict:
    """Fresh robot, scene, client, and transcript for every repetition. No hardware fallback."""
    if cfg.robot.backend != "sim" or cfg.cameras.backend != "sim":
        raise ValueError("research evaluation currently requires sim robot and cameras")
    from astra_yam.astra_client import make_astra_client
    from astra_yam.cli import _make_robot_and_cameras
    from astra_yam.embodiment import build_tools
    from astra_yam.kinematics import ArmKinematics
    from astra_yam.session import OperatorInput, TrialRunner

    cfg = copy.deepcopy(cfg)
    cfg.sim.scene, cfg.viz.enabled, cfg.home_on_end = case.scene, False, False
    cfg.log_dir = str(root)
    cfg.policy_notes_path = str(notes) if notes else None
    kin = ArmKinematics(cfg.robot.yam_xml_path, joint_lower=cfg.robot.joint_lower,
                        joint_upper=cfg.robot.joint_upper, limit_margin=cfg.motion.joint_limit_margin_rad)
    robot = cameras = operator = client = None
    try:
        robot, cameras, world = _make_robot_and_cameras(cfg, kin)
        reset_scene(world, case)
        client = (client_factory or make_astra_client)(cfg.astra, build_tools(cfg.bounds, cfg.prompts_path, reactive=cfg.reactive.enabled))
        operator = OperatorInput(use_stdin=False, file_path=feedback_file) if feedback_file else None
        disturbances = BowlDisturbances(world, case)
        runner = TrialRunner(cfg, robot, cameras, kin, client, operator=operator,
                             realtime=False, sim_world=world, verbose=False, hooks=disturbances)
        outcome = runner.run(case.goal)
        verification = verify(world, case)
        verification["metrics"]["disturbances_applied"] = disturbances.applied
        if disturbances.applied != len(case.disturbances):
            verification.update(success=False, score=0.0, reason="trial ended before the disturbance schedule completed")
        if outcome.status in ("error", "aborted", "estop", "operator_stop"):
            verification.update(success=False, score=0.0, reason=f"invalid trial: {outcome.status}")
        assisted = False
        trace = []
        for line in (Path(outcome.log_dir) / "transcript.jsonl").read_text().splitlines():
            event = json.loads(line)
            if event["kind"] == "operator_feedback":
                assisted = True
            if event["kind"] == "tool_call":
                trace.append({k: event[k] for k in ("name", "arguments", "result") if k in event})
        result = {"case": asdict(case), "outcome": outcome.as_dict(), "verification": verification,
                  "assisted": assisted, "trace": trace, "disturbances": disturbances.events}
        write_json(Path(outcome.log_dir) / "evaluation.json", result)
        return result
    finally:
        for resource in (operator, client, cameras, robot):
            close = getattr(resource, "close", None)
            if close:
                close()


def metrics(records: list[dict]) -> dict:
    if not records:
        raise ValueError("cannot score an empty evaluation")
    return {"trials": len(records),
            "success_rate": sum(r["verification"]["success"] for r in records) / len(records),
            "score": sum(r["verification"]["score"] for r in records) / len(records),
            "mean_calls": sum(r["outcome"]["llm_calls"] for r in records) / len(records),
            "rejections": sum(r["outcome"]["rejections"] for r in records),
            "tokens": sum(r["outcome"]["usage"].get("total_tokens", 0) for r in records),
            "assisted_trials": sum(r["assisted"] for r in records)}


def promotion_gate(baseline: list[dict], candidate: list[dict]) -> tuple[bool, str]:
    """Paired per-case regression gate; validation is a gate, not a held-out test set."""
    if [r["case"] for r in baseline] != [r["case"] for r in candidate] or not baseline:
        return False, "evaluation cases do not match"
    for r in baseline + candidate:
        if r["assisted"] or r["outcome"]["status"] in ("error", "aborted", "estop", "operator_stop"):
            return False, "assisted or invalid trial; comparison excluded"
    for old, new in zip(baseline, candidate):
        a, b = old["verification"], new["verification"]
        if (a["success"] and not b["success"]) or b["score"] + 1e-9 < a["score"]:
            return False, f"regression on {old['case']['id']}"
        if new["outcome"]["rejections"] > old["outcome"]["rejections"]:
            return False, f"more gateway rejections on {old['case']['id']}"
    a, b = metrics(baseline), metrics(candidate)
    improved = (b["success_rate"] > a["success_rate"] or b["score"] > a["score"] + 1e-6
                or (b["success_rate"] == 1 and b["mean_calls"] < a["mean_calls"]))
    return improved, "measured improvement without paired regressions" if improved else "no measured improvement"
