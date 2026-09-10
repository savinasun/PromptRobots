"""Bounded Astra prompt evolution, with frozen evaluation and durable evidence."""
from __future__ import annotations

import copy
import base64
import difflib
import json
import subprocess
from dataclasses import asdict
from pathlib import Path

from astra_yam.astra_client import OpenAIAstraClient
from astra_yam.config import to_dict
from astra_yam.evaluation import fingerprint, load_suite, metrics, promotion_gate, run_case, write_json

ROOT = Path(__file__).resolve().parent.parent
ENPIRE_COMMIT = "99ee90acf65b5b18957c8382ad580db999528be3"
STRATEGY_PROMPT = ROOT / "configs" / "RESEARCH_PROMPT.md"
OFFLINE_STRATEGY = ROOT / "configs" / "RESEARCH_SMOKE_STRATEGY.md"


class ResearchStopped(Exception):
    """Operator requested a stop at a boundary, without invalidating past trials."""

PROPOSAL_TOOL = {
    "type": "function", "name": "propose_strategy",
    "description": "Propose one small task-strategy revision to test in the next paired evaluation.",
    "strict": True,
    "parameters": {"type": "object", "additionalProperties": False,
                   "properties": {key: {"type": "string"} for key in
                                  ("hypothesis", "evidence", "policy_notes", "expected_improvement")},
                   "required": ["hypothesis", "evidence", "policy_notes", "expected_improvement"]},
}


def validate_proposal(proposal: dict) -> dict:
    required = set(PROPOSAL_TOOL["parameters"]["required"])
    if not isinstance(proposal, dict) or set(proposal) != required:
        raise ValueError("proposal must contain exactly the four strategy fields")
    for key, value in proposal.items():
        limit = 6000 if key == "policy_notes" else 2000
        if not isinstance(value, str) or not value.strip() or len(value) > limit:
            raise ValueError(f"invalid or oversized proposal field: {key}")
    return proposal


class AstraOptimizer:
    def __init__(self, cfg, root: Path):
        cfg = copy.deepcopy(cfg)
        cfg.tool_choice = {"type": "function", "name": "propose_strategy"}
        cfg.parallel_tool_calls = False
        cfg.max_output_tokens = cfg.max_output_tokens or 4096
        self.client = OpenAIAstraClient(cfg, [PROPOSAL_TOOL])
        self.root = root

    def propose(self, context: dict, iteration: int) -> dict:
        context = copy.deepcopy(context)
        frame_paths = context.pop("training_frame_paths", [])
        items = [{"role": "system", "content": STRATEGY_PROMPT.read_text()},
                 {"role": "user", "content": json.dumps(context)}]
        # Only training evidence is provided. Images are normal observation frames,
        # never simulator object coordinates or simulator implementation details.
        for path in frame_paths:
            image = base64.b64encode(Path(path).read_bytes()).decode("ascii")
            items.append({"role": "user", "content": [
                {"type": "input_text", "text": f"Final training observation: {Path(path).name}"},
                {"type": "input_image", "image_url": f"data:image/jpeg;base64,{image}", "detail": "high"}]})
        from astra_yam.observation import redact_images
        write_json(self.root / f"optimizer_{iteration:03d}_request.json",
                   redact_images(self.client.build_request_dict(items)))
        response = self.client.create(items)
        write_json(self.root / f"optimizer_{iteration:03d}_response.json", asdict(response))
        if len(response.function_calls) != 1:
            raise ValueError("optimizer must return exactly one proposal")
        call = response.function_calls[0]
        if call.name != "propose_strategy" or call.parse_error:
            raise ValueError("optimizer returned an invalid proposal tool call")
        return validate_proposal(call.arguments)

    def close(self):
        self.client.close()


class ScriptedOptimizer:
    """Offline plumbing check, not evidence of model or prompt improvement."""
    def propose(self, context: dict, iteration: int) -> dict:
        return {"hypothesis": "Verify observable goal conditions before declaring success.",
                "evidence": "Offline fixture; does not infer a strategy from model calls.",
                "policy_notes": OFFLINE_STRATEGY.read_text(),
                "expected_improvement": "Fewer premature completion claims."}

    def close(self):
        pass


def _source_state() -> dict:
    def git(*args):
        return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()
    try:
        return {"commit": git("rev-parse", "HEAD"), "dirty": bool(git("status", "--porcelain"))}
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None}


def _contract(cfg, cases):
    config = to_dict(cfg)
    for name in ("log_dir", "policy_notes_path", "operator_feedback_file"):
        config.pop(name, None)
    paths = [Path(cfg.system_prompt_path), Path(cfg.tilt_note_path), Path(cfg.prompts_path),
             Path(cfg.robot.yam_xml_path), *sorted((ROOT / "astra_yam").glob("*.py")), STRATEGY_PROMPT,
             OFFLINE_STRATEGY]
    if cfg.astra.backend == "scripted" and cfg.astra.script_path:
        paths.append(Path(cfg.astra.script_path))
    return {"config": config, "cases": [asdict(c) for c in cases],
            "files": {str(p): fingerprint(p.read_text()) for p in paths}}


def training_context(records, notes: str, history: list, feedback: str) -> dict:
    frames = []
    for record in records:
        if record["case"]["split"] != "train" or not record["outcome"].get("log_dir"):
            continue
        paths = sorted((Path(record["outcome"]["log_dir"]) / "frames").glob("*.jpg"))
        if paths:
            latest = max(p.name.split("_")[1] for p in paths)
            frames.extend(str(p) for p in paths if p.name.split("_")[1] == latest)
    return {"current_policy_notes": notes, "operator_research_feedback": feedback[:4000],
            "training_frame_paths": frames[-6:],
            "training_trials": [{"case_id": r["case"]["id"], "goal": r["case"]["goal"],
                                 "verification": r["verification"],
                                 "status": r["outcome"]["status"], "hindsight": r["outcome"]["hindsight"],
                                 "trace": r["trace"][-24:]}
                                for r in records if r["case"]["split"] == "train"],
            "previous_experiments": history[-5:]}


def run_research(cfg, suite_path: str, output: str, *, iterations: int = 2, repeats: int = 1,
                 mock_optimizer: bool = False, feedback_file: str | None = None,
                 optimizer_factory=None, evaluate=run_case) -> dict:
    if cfg.robot.backend != "sim" or cfg.cameras.backend != "sim":
        raise ValueError("research runs require simulation; real tasks need a commissioned reset/verifier")
    if not 0 <= iterations <= 20 or not 1 <= repeats <= 20:
        raise ValueError("iterations must be 0..20 and repeats 1..20")
    if cfg.limits.max_llm_calls <= 0 or cfg.limits.max_trial_seconds <= 0 or cfg.limits.max_waypoints <= 0:
        raise ValueError("positive per-trial call, time and waypoint budgets are required")
    if mock_optimizer and cfg.astra.backend != "scripted":
        raise ValueError("mock optimizer requires scripted rollout backend")
    if iterations and cfg.astra.backend == "scripted" and not mock_optimizer:
        raise ValueError("scripted rollouts require --mock-optimizer")
    cases = load_suite(suite_path)
    if feedback_file and not Path(feedback_file).is_file():
        raise ValueError("research feedback file must exist before starting")
    root = Path(output).resolve()
    root.mkdir(parents=True, exist_ok=False)  # never overwrite previous evidence
    contract = _contract(cfg, cases)
    contract_hash = fingerprint(contract)
    manifest = {"schema_version": 1, "source": _source_state(), "contract": contract,
                "contract_hash": contract_hash, "enpire_commit": ENPIRE_COMMIT,
                "iterations": iterations, "repeats": repeats,
                "max_trials": (iterations + 1) * repeats * len(cases),
                "max_model_requests_excluding_retries": (iterations + 1) * repeats * len(cases)
                    * cfg.limits.max_llm_calls + iterations,
                "mode": "scripted" if cfg.astra.backend == "scripted" else "live_astra_simulation"}
    write_json(root / "manifest.json", manifest)
    write_json(root / "source_snapshot.json", {name: Path(name).read_text() for name in contract["files"]})
    initial = Path(cfg.policy_notes_path).read_text() if cfg.policy_notes_path else ""
    champion = root / "baseline.md"
    champion.write_text(initial)
    (root / "best_policy.md").write_text(initial)
    history, incumbent, optimizer = [], [], None
    seen = {fingerprint(initial)}
    report = {"status": "running", "mode": manifest["mode"], "champion": str(champion),
              "iterations": history, "baseline": {}, "best": {}}

    def check_contract():
        if fingerprint(_contract(cfg, cases)) != contract_hash or load_suite(suite_path) != cases:
            raise RuntimeError("frozen experiment contract changed; stop and start a new experiment")

    def evaluate_variant(notes, number):
        records = []
        policy_hash = fingerprint(notes.read_text())
        for i, case in enumerate(cases):
            for repeat in range(repeats):
                check_contract()
                if feedback_file and any(line.strip().lower() in ("/stop", "stop")
                                         for line in Path(feedback_file).read_text().splitlines()):
                    raise ResearchStopped("operator stopped research at the trial boundary")
                if fingerprint(notes.read_text()) != policy_hash:
                    raise RuntimeError("policy changed during paired evaluation")
                print(f"[research] variant {number}, {case.id}, repeat {repeat + 1}/{repeats}", flush=True)
                trial_root = root / f"variant_{number:03d}" / f"case_{i:03d}_repeat_{repeat:03d}"
                record = evaluate(cfg, case, trial_root, notes)
                records.append(record)
                write_json(root / f"variant_{number:03d}" / "results.json", records)
                if fingerprint(notes.read_text()) != policy_hash:
                    raise RuntimeError("policy changed during paired evaluation")
                if record["outcome"]["status"] in ("error", "estop", "operator_stop", "aborted"):
                    raise RuntimeError(f"trial stopped: {record['outcome']['status']}; inspect {trial_root}")
        check_contract()
        return records

    try:
        incumbent = evaluate_variant(champion, 0)
        report["baseline"] = report["best"] = metrics(incumbent)
        write_json(root / "report.json", report)
        if iterations:
            optimizer = (optimizer_factory(cfg.astra, root) if optimizer_factory else
                         ScriptedOptimizer() if mock_optimizer else AstraOptimizer(cfg.astra, root))
        for iteration in range(1, iterations + 1):
            check_contract()
            # Interactive feedback is read between experiments, and never fed to the
            # physical controller or scorer. STOP is checked before any next trial.
            feedback = Path(feedback_file).read_text() if feedback_file else ""
            if any(line.strip().lower() in ("/stop", "stop") for line in feedback.splitlines()):
                report["status"] = "operator_stop"
                break
            context = training_context(incumbent, champion.read_text(), history, feedback)
            proposal = validate_proposal(optimizer.propose(context, iteration))
            candidate = root / f"candidate_{iteration:03d}.md"
            candidate.write_text(proposal["policy_notes"])
            diff = "".join(difflib.unified_diff(champion.read_text().splitlines(True),
                                               candidate.read_text().splitlines(True),
                                               fromfile=champion.name, tofile=candidate.name))
            (root / f"candidate_{iteration:03d}.diff").write_text(diff)
            write_json(root / f"candidate_{iteration:03d}.json", proposal)
            candidate_hash = fingerprint(proposal["policy_notes"])
            if candidate_hash in seen:
                history.append({"iteration": iteration, "hypothesis": proposal["hypothesis"],
                                "accepted": False, "reason": "unchanged or previously evaluated candidate", "training": []})
            else:
                seen.add(candidate_hash)
                records = evaluate_variant(candidate, iteration)
                accepted, reason = promotion_gate(incumbent, records)
                # Future optimizer context contains training evidence only. Validation
                # measurements stay in the report, outside optimizer-visible history.
                history.append({"iteration": iteration, "hypothesis": proposal["hypothesis"],
                                "accepted": accepted, "reason": "accepted" if accepted else "not promoted",
                                "training": training_context(records, "", [], "")["training_trials"]})
                write_json(root / f"decision_{iteration:03d}.json",
                           {"accepted": accepted, "reason": reason, "metrics": metrics(records)})
                if accepted:
                    champion, incumbent = candidate, records
                    (root / "best_policy.md").write_text(candidate.read_text())
                print(f"[research] candidate {iteration}: {reason}; accepted={accepted}", flush=True)
            report.update(champion=str(champion), best=metrics(incumbent))
            write_json(root / "report.json", report)
        else:
            report["status"] = "complete"
    except ResearchStopped as error:
        report.update(status="operator_stop", reason=str(error))
    except BaseException as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        all_records = []
        for path in root.glob("variant_*/results.json"):
            all_records.extend(json.loads(path.read_text()))
        report["total_evaluation"] = metrics(all_records) if all_records else {}
        report["optimizer_tokens"] = sum(json.loads(path.read_text())["usage"].get("total_tokens", 0)
                                         for path in root.glob("optimizer_*_response.json"))
        write_json(root / "report.json", report)
        _write_report(root, report)
        if optimizer:
            optimizer.close()
    return report


def _write_report(root: Path, report: dict):
    lines = ["# YAM prompt research", "", f"Status: {report['status']}; mode: {report['mode']}", "",
             "| Policy | Success rate | Progress score | Mean calls | Rejections |",
             "| --- | ---: | ---: | ---: | ---: |"]
    for name in ("baseline", "best"):
        m = report.get(name)
        if m:
            lines.append(f"| {name} | {m['success_rate']:.1%} | {m['score']:.3f} | {m['mean_calls']:.1f} | {m['rejections']} |")
    lines += ["", "These are simulator results, not physical robot performance. Validation cases are reused",
              "for promotion and are not an untouched test set. Small samples do not establish reliability.", "",
              f"Selected prompt: {report['champion']}", ""]
    (root / "report.md").write_text("\n".join(lines))
