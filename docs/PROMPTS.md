# Shared prompts and task configuration

The live runner accepts arbitrary natural-language goals. Its system prompt and
tools depend on the configured YAM embodiment and execution mode, not the goal
or simulator scene. No task strategy is loaded unless explicitly requested.

## What Astra receives

Each request contains:

1. **System message:** `configs/SYSTEM_PROMPT.md`, with the configured embodiment
   name and call budget; `configs/TILT_NOTE.md` when pitch/roll are actuated.
2. **Optional advice:** the file explicitly selected by `--policy-notes`, if any.
   Nothing automatically imports `AIRPODS_LEARNED.md`, `AIRPODS_STRATEGY.md`, a
   previous conversation, or an experiment's `best_policy.md`.
3. **Reactive instructions:** with `--dynamic-scene`, `session.reactive_rules`
   from `configs/PROMPTS.yaml`. These cover deriving observable subgoals, checking
   action results, reacquiring moved objects, short motions, and tentative lessons.
4. **User goal:** `Goal: <instruction>`, repeated in each observation. Clarify
   object identity here, for example “charging case” rather than “airpod.”
5. **Observations:** labeled camera images, joint positions, end-effector state,
   and remaining waypoints. Reactive mode adds observation sequence, elapsed time,
   and up to six tentative lessons from this episode.
6. **Conversation:** earlier tool calls/results, available model reasoning items,
   and operator feedback. Older images follow the configured history limit;
   lessons and conversation reset at the next trial.

Tool schemas come from `configs/PROMPTS.yaml` and configured workspace bounds:
`move_to`, `done`, `give_up`, plus `observe` in dynamic mode. `move_to` and
`observe` accept optional episode lessons in that mode. The model sees neither
the evaluation verifier nor simulator coordinates used to schedule disturbances.

The shared reactive guidance includes:

> Derive the next subgoal and observable completion conditions from the user's goal and current scene.
> Before each action, check its preconditions; after it, compare the observed result with the intended effect.

## Inspect the actual prompt

```bash
scripts/run_astra_yam.sh show-prompt --config configs/skild_yam_8.yaml \
  --dynamic-scene --max-calls 60

# Machine-readable system message and tools; no connection or API key needed
scripts/run_astra_yam.sh show-prompt --config configs/skild_yam_8.yaml \
  --dynamic-scene --max-calls 60 --bundle-json
```

Use the same config, `--set` overrides, budget, and optional notes as the run.
`show-prompt` and `TrialRunner` share one assembly function. `--json` retains its
older tools-only output. The bundle does not fabricate camera observations or
conversation history. For a specific live call, inspect that trial's
`requests/request_NNNN.json`: it records the assembled system message, tools,
goal, observations, and history. Image payloads are replaced by blob markers;
the corresponding images are saved under `frames/`. `transcript.txt` also starts
with the assembled prompt. Research snapshots preserve the source used at the
time; inspecting today's prompt cannot reconstruct an older experiment.

## Change tasks without changing the shared policy

```bash
scripts/run_astra_yam.sh run --sim --scene airpod_bowl --dynamic-scene --viser \
  --max-calls 60 --goal "Pick up the charging case and place it inside the green bowl."

scripts/run_astra_yam.sh run --sim --scene blocks --dynamic-scene --viser \
  --max-calls 60 --goal "Pick up the blue block and stack it on the green block."
```

These use identical system messages and tools. Only the goal and observations
change. Station geometry, camera selection, and motion limits remain embodiment
configuration. The generic runner performs no task-name matching.

`--task-profile airpod-bowl` remains an explicit legacy preset for reproducing
task-focused setups: it enables dynamic mode, loads `configs/AIRPOD_BOWL.md`, and
sets the default image-change threshold to 0.005 instead of the generic 0.01.
Use `--dynamic-scene` for the shared policy. The existing files in your tabs
remain optional task advice; they are not all concatenated into the prompt.

Dynamic mode uses the same controller behavior across tasks: bounded trajectory
duration, stale-decision rejection, reobservation events, grip-preserving holds,
and separate gripper opening from pose changes. The last rule is conservative
for manipulation tasks involving release; it is not suitable for actions that
require simultaneous motion and opening. Image change is a fixed-camera pixel
heuristic, with no continuous semantic object tracking. See the
[disturbance guide](AIRPOD_BOWL.md) for timing and controls.

## Evaluate transferable improvements

```bash
# Shared zero-shot baseline: six fresh episodes, no optimizer or task advice
scripts/run_astra_yam.sh research --sim --dynamic-scene \
  --suite configs/multitask_research.yaml --iterations 0 \
  --max-calls 40 --max-seconds 240 --output runs/shared-baseline-001

# One shared strategy revision: at most twelve rollouts and one optimizer call
scripts/run_astra_yam.sh research --sim --dynamic-scene \
  --suite configs/multitask_research.yaml --iterations 1 \
  --max-calls 40 --max-seconds 240 --output runs/shared-search-001
```

The suite covers stacking, lid opening, and placement into a disturbed container,
with a training and validation reset for each. The optimizer sees training
evidence from all tasks and proposes one advice file used on every case.
`configs/RESEARCH_PROMPT.md` requests evidence-supported, transferable revisions
without assuming an object or task. The promotion gate checks every paired case,
so a higher average cannot hide regression on another task. Once advice has been
optimized using task trials, report it as prompt search rather than an untouched
zero-shot baseline. Current-episode lessons remain separate from this search.

`run --goal` needs no task registry. Automated research currently supports three
scene/verifier pairs; extending it requires environment-owned reset and success
checks in `evaluation.py` and a suite entry. The scheduled bowl disturbances
are also environment-specific. These are not general-purpose perception or
physical reset implementations. Always specify `--suite`; omitting it retains
the earlier AirPods lid suite for compatibility.

The generalized prompt has integration coverage, including exact request/prompt
equality and shared candidate evaluation across tasks. Its manipulation success
has not yet been measured with live Astra. Earlier
[bowl results](AIRPOD_BOWL_RESULTS.md) used the task-specific preset.
