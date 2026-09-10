# AirPods validation, 2026-09-09

Two live `gpt-6-astra` prompt revisions were evaluated across six simulated
trials: baseline, candidate 1, and candidate 2, each on nominal and shifted
scenes. Each trial had a 24-call, 240-second budget; reasoning effort was low.
The selected advice is saved in [AIRPODS_LEARNED.md](../configs/AIRPODS_LEARNED.md).

| Measurement, two trials per policy | Baseline | Selected candidate 1 |
| --- | ---: | ---: |
| Independently verified lid openings | 0/2 | 0/2 |
| Mean progress score | 0.20 | 0.40 |
| Mean model calls | 20 | 10 |
| Total gateway rejections | 8 | 4 |
| Rollout tokens | 485,841 | 187,270 |

Candidate 1 preserved body support in both scenes and reduced ineffective
attempts. It did **not** complete the lid-opening task. Candidate 2 was rejected
because it regressed on the shifted scene. Across all six trials there were
zero independently verified openings. The full live run consumed 1,061,120
rollout tokens and 9,698 optimizer tokens (including cached input in reported
token totals, not a dollar cost calculation).

The main observed limitation was visual: the schematic feeds did not provide
enough hinge/seam detail for Astra to reliably choose a lid grasp. Rendering
the articulated body/lid fixes identical open/closed images, but does not
replace physically accurate wrist-camera views or a commissioned real task.
These small samples show partial progress and fewer calls/rejections; they do
not establish improved physical opening reliability.

The deterministic offline fixture completed all four integration trials through
the same gateway and independent verifier. That checks harness wiring and
reset/replay behavior, not model generalization. The four successful records
were also exported with the actual upstream ENPIRE artifact/result classes.

Local experiment evidence (ignored by Git):

- `runs/airpods_astra_research/report.md` and `report.json`
- `runs/airpods_astra_research/decision_001.json` and `decision_002.json`
- `runs/airpods_astra_research/source_snapshot.json` records exact source used
  before subsequent artifact-integrity hardening.
- `runs/airpods_harness_verified/report.json`
- `runs/airpods_verified_enpire_export/trial_*/result.json`

To inspect the selected strategy interactively:

```bash
scripts/run_astra_yam.sh run --sim --scene airpods --viser \
  --policy-notes configs/AIRPODS_LEARNED.md \
  --goal "Open the AirPods case lid and release the lid while supporting the body."
```

Use the [research guide](RESEARCH.md) to run a new experiment with more
repetitions or commission independent physical verification. No hardware was
moved during these experiments.
