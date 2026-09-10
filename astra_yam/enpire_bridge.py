"""Optional thin adapter to ENPIRE's actual artifact and result implementations.

ENPIRE is deliberately imported only when exporting; the YAM gello environment
does not need its planning, camera, training, or Python 3.11 dependencies.
"""
from __future__ import annotations

import json
from pathlib import Path


def export_experiment(experiment: str, output: str) -> int:
    from enpire.env.forge.artifacts import ArtifactStore
    from enpire.env.forge.interface import VerificationResult
    from enpire.env.forge.loop import TrialResult

    source, target = Path(experiment).resolve(), Path(output).resolve()
    if not (source / "manifest.json").is_file():
        raise ValueError("experiment manifest missing")
    target.mkdir(parents=True, exist_ok=False)
    count = 0
    for path in sorted(source.glob("variant_*/case_*/*/evaluation.json")):
        record = json.loads(path.read_text())
        outcome, verification = record["outcome"], VerificationResult(**record["verification"])
        store = ArtifactStore(target / f"trial_{count:04d}")
        result = TrialResult(success=verification.success, steps=outcome["llm_calls"],
                             total_reward=float(verification.score or 0), termination=outcome["status"],
                             verification=verification, elapsed_s=outcome["elapsed_s"],
                             info={"source_evaluation": str(path), "case": record["case"],
                                   "assisted": record["assisted"], "reward_semantics": "terminal verification score"})
        store.write_json("result.json", result)
        store.append_event({"event": "verify", "verification": verification})
        count += 1
    ArtifactStore(target).write_json("source_manifest.json", json.loads((source / "manifest.json").read_text()))
    return count
