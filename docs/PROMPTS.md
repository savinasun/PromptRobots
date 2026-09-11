# Shared prompts and task configuration

On `astra-actions-only`, the default output contract is now strict actions with
no language fields or written episode lessons. See [action-only mode](ACTIONS_ONLY.md).
Descriptions of notes and hindsight below apply when explicitly
using `--language-output`.

The live runner accepts arbitrary natural-language goals. Its system prompt and
tools depend on the configured YAM embodiment and execution mode, not the goal
or simulator scene. No task strategy is loaded unless explicitly requested.

## What Astra receives

Each request contains:

1. **System message:** `configs/SYSTEM_PROMPT.md`, with the configured embodiment
   name and call budget; `configs/TILT_NOTE.md` when pitch/roll are actuated.
2. **Optional advice:** the file explicitly selected by `--policy-notes`, if any.
   Advice and previous conversations are never automatically imported.
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
`observe` accept optional episode lessons when language output is enabled.

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
with the assembled prompt used for that trial.

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

Dynamic mode uses the same controller behavior across tasks: bounded trajectory
duration, stale-decision rejection, reobservation events, grip-preserving holds,
and separate gripper opening from pose changes. The last rule is conservative
for manipulation tasks involving release; it is not suitable for actions that
require simultaneous motion and opening. Image change is a fixed-camera pixel
heuristic, with no continuous semantic object tracking. Use the UI reobserve
control to interrupt movement and request fresh observations.
