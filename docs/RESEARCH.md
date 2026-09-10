# YAM research harness

The harness evaluates a shared policy on goals and independent verifiers selected
by an explicit suite. `configs/multitask_research.yaml` covers stacking, lid
opening, and placement into a moving bowl. See [the prompt guide](PROMPTS.md)
for task-independent setup and the exact context sent to Astra.

The original suite, retained as the CLI default, targets opening the AirPods case,
maintaining body support, and releasing the lid so it stays open. Examples and
progress weights below describe that suite. The harness extends `TrialRunner`,
`OpenAIAstraClient`, `SafetyGateway`, `SimWorld`, and the existing operator UI.
It does not install or replace the station's gello environment.

## Experiment loop

1. Load `configs/airpods_research.yaml`. The nominal scene is training data; a
   deterministically shifted scene supplies a separate validation gate.
2. Evaluate the starting policy on every case. Each repetition gets fresh
   objects, robot state, model conversation, client, and logs.
3. Ask Astra for one falsifiable hypothesis and a complete revised task advice
   file. Provide training tool traces, hindsight, independent measurements,
   recent training camera frames, and previous experiments. Validation traces
   and images are excluded from optimizer input.
4. Append the candidate advice to the unchanged system prompt. Evaluate it on
   the same cases, seeds, and number of repetitions.
5. Reject any candidate that loses an existing success, reduces progress on any
   paired case, or increases gateway rejections. Select a candidate only for a
   measured increase in success/progress, or fewer calls when all cases succeed.
   Equal results keep the incumbent. Repeated identical proposals skip trials.

This is bounded prompt search, not unrestricted self-modifying Python. Astra
cannot edit files, gateway settings, or the reward implementation through its
proposal tool. The proposal schema admits only hypothesis, evidence, policy
notes, and expected improvement; notes are limited to 6,000 characters. Python,
robot XML, base prompts, resolved configuration, and evaluation cases are hashed
and checked between trials. A changed contract stops the experiment.

Progress is an environment-owned diagnostic: body held (0.2), upright (0.2),
lid opening up to its latch threshold (0.4), and verified completion (0.2).
Success requires an open, released lid and an arm holding the body. Invalid or
operator-stopped trials cannot be promoted. Human-assisted rollouts cannot be
compared as autonomous results. Scoring happens after the runner returns and
does not trust the model's `done` text. Final homing is disabled during research
so it cannot disturb the evaluated scene.

## Commands and budgets

```bash
# Baseline only, using a manually written strategy
scripts/run_astra_yam.sh research --sim --iterations 0 \
  --policy-notes configs/AIRPODS_STRATEGY.md --output runs/airpods-baseline-001

# More repetitions to assess variability, with explicit request limits
scripts/run_astra_yam.sh research --sim --iterations 2 --repeats 3 \
  --max-calls 24 --max-seconds 240 --max-waypoints 10000 \
  --effort low --set astra.max_output_tokens=4096 \
  --set astra.max_retries=1 --set astra.request_timeout_s=90 \
  --output runs/airpods-repeat-001
```

The maximum trial count is `(iterations + 1) * repeats * case_count`. Each
iteration adds at most one optimizer request. SDK retries can add API attempts;
they are separately configured by `astra.max_retries`. Per-trial time limits
are checked between model calls and during motion; a model request can extend
past that limit by its request timeout. Budgets are not dollar spending caps.

The model remains `gpt-6-astra` unless explicitly overridden. Credentials use
the existing key lookup; no key belongs in an experiment configuration or
feedback file. Requests use the Responses API with `store=False`.

## Interactive use

```bash
touch /tmp/airpods-research-feedback.txt
scripts/run_astra_yam.sh research --sim --iterations 2 \
  --research-feedback /tmp/airpods-research-feedback.txt \
  --output runs/airpods-feedback-001
```

Edit that file with observations or constraints while the experiment runs.
The optimizer reads it between iterations. A line containing `/stop` stops at
the next trial/iteration boundary; this file is not an emergency stop. During
interactive single trials, `run --viser` retains the existing live feedback,
Start, and E-stop controls, and `run --feedback-file PATH` retains its in-trial
feedback behavior. Research evaluation currently uses the schematic cameras;
the `--viser` options inherited by the command do not enable browser rendering.

## Artifacts

- `manifest.json`: source commit/dirty state, source and prompt hashes, ENPIRE
  reference commit, frozen configuration/cases, budgets, and scripted/live mode.
- `source_snapshot.json`: exact source/prompt/robot XML content matching those
  hashes, including a scripted policy fixture when configured. Strategy content
  is also checked before and after every paired trial.
- `baseline.md`, `candidate_NNN.md`, `.json`, `.diff`: policy versions,
  hypotheses, supporting evidence, and reviewable changes.
- `variant_NNN/results.json`: all completed evaluations, including failures.
- Each trial: requests, responses, images, transcripts, ordinary summary, and
  independent `evaluation.json` containing measured verification.
- `optimizer_NNN_request.json` / `_response.json`: reproducible optimizer
  context (images redacted in request JSON), responses, and token usage.
- `decision_NNN.json`: promotion decision and reason. `report.json` / `.md`:
  baseline versus selected policy. `total_evaluation` includes all rollout
  tokens; `optimizer_tokens` is separate.
- `best_policy.md`: selected advice for `run --policy-notes`.

An interrupted run retains completed trials and a failed report. Automatic
resume is intentionally not implemented: use a new experiment directory and
start from a previous strategy with `--policy-notes`.

## ENPIRE reuse and provenance

Inspected upstream: [NVlabs/ENPIRE](https://github.com/NVlabs/ENPIRE), `main` at
`99ee90acf65b5b18957c8382ad580db999528be3`.

The loop and immutable policy/environment boundary follow upstream
[`autoresearch_instruction.md`](https://github.com/NVlabs/ENPIRE/blob/99ee90acf65b5b18957c8382ad580db999528be3/enpire/policy/autoresearch_instruction.md)
and the
[`00_hello_environment` example](https://github.com/NVlabs/ENPIRE/blob/99ee90acf65b5b18957c8382ad580db999528be3/enpire/env/examples/00_hello_environment/example.py).
`astra_yam.enpire_bridge` directly uses upstream `ArtifactStore`,
`VerificationResult`, and `TrialResult` through lazy imports. No upstream
implementation is vendored. The adapter exports evaluated trials; it is not
an ENPIRE robot driver, RL learner, or replacement control stack.

In an environment where ENPIRE is importable:

```bash
python -m astra_yam export-enpire \
  --experiment runs/airpods-search-001 --output runs/airpods-enpire-001
```

This exports actual ENPIRE `result.json` and verification events per trial,
with references to the original recordings and a copy of the source manifest.
The exported total reward is the terminal verification score. The YAM runtime
stays hardware-free at import and does not acquire ENPIRE's heavy dependencies.

OpenAI references: [Astra model](https://developers.openai.com/api/docs/models/gpt-6-astra)
and [structured output](https://developers.openai.com/api/docs/guides/structured-outputs).

## What the measurements establish

The simulator is kinematic with heuristic grasp/lid mechanics and schematic
cameras. Its wrist images look downward and do not model full camera optics or
contact forces; labels and height annotations also simplify perception. The
updated case rendering exposes lid articulation but does not make the simulator
photorealistic. The offline fixture uses nominal simulator coordinates and is
only an integration check. A scripted optimizer is not an Astra improvement.

Small evaluation samples and repeatedly consulted validation gates do not
establish real-world reliability. Validation is not an untouched test set;
run a separate final suite with new seeds and repeated trials for that purpose.
Promoting partial progress is explicitly different from achieving the task.

Autonomous research currently accepts only simulated robots/cameras. For a
physical station, first commission a task-specific reset/readiness check and
an independent vision/operator verifier using ENPIRE's station and calibration
workflow. Then compare strategies under a fixed physical trial budget. The
existing `run --config ... --policy-notes ... --viser` entry point supports
reviewed strategies on hardware with its existing motion confirmation and
gateway. No physical performance gain is claimed from simulator experiments.
