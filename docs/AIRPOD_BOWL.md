# Charging case → moving green bowl

Use the shared `--dynamic-scene` mode with an explicit charging-case goal.
System prompts and controller behavior then match other tasks; see
[the prompt guide](PROMPTS.md) for the complete request contents. The optional
legacy `--task-profile airpod-bowl` additionally loads charging-case advice and
a lower image-change threshold. Earlier [measured experiments](AIRPOD_BOWL_RESULTS.md)
used that task-specific preset.

Dynamic mode supplies episode-local lessons and reactive execution.
It does not load the previous lid-opening strategy, past trial transcripts,
demonstration trajectories, known bowl coordinates, or simulator object state
into Astra's context. This is zero-shot task execution with learning from the
current episode, rather than a trained policy or prompt search between trials.

## Use on the configured station

```bash
scripts/run_astra_yam.sh run --config configs/skild_yam_8.yaml \
  --dynamic-scene --viser --max-calls 60 \
  --goals-file tasks/affordance/airpod_bowl_goal.txt
```

The station's motion limits and existing confirmation remain in effect. The
dynamic mode enables `reactive.enabled` and adds shared instructions; it does
not load task advice or calibrate hardware during setup. Before running, use the existing
station checks and ensure the fixed overhead view includes the full area in
which the bowl will move. The default comparison camera is `top_cam`.

For an interactive simulation:

```bash
scripts/run_astra_yam.sh run --sim --scene airpod_bowl --dynamic-scene \
  --viser --max-calls 60 --goals-file tasks/affordance/airpod_bowl_goal.txt
```

The simulator now has a green bowl with an interior support surface and a rim,
plus the charging case. Enable **Edit objects** to move the bowl using its
gizmo. Moving a simulator object requests reobservation automatically. These
are heuristic contact mechanics, not a physical bowl/robot dynamics model.

## How disturbances are handled

1. **During model inference:** the robot is stationary. Immediately before
   executing the returned command, the harness compares a new fixed-camera
   image with the observation on which Astra decided. A material image change
   discards the proposed command, including a release or `done`, and supplies
   a fresh observation. New operator feedback during inference also invalidates
   the pending action. Discarded actions remain in the transcript with a
   `stale_observation` result and do not count as gateway rejections.
2. **During movement:** each `move_to` executes at most three seconds of its
   planned trajectory before returning `observation_required`. Only that portion
   ran. Astra gets measured pose and new images and must replan the remainder.
   Full-plan IK, joint, and clearance checks still apply before execution.
3. **At release:** dynamic mode rejects commands combining gripper opening with
   pose targets. Alignment/descent and opening must be separate decisions. The
   shared prompt asks for current action preconditions and observed results;
   optional bowl advice specifies checking the interior before opening and
   checking containment after release.
4. **Within the episode:** optional `lesson` text on `move_to` and `observe`
   retains up to six short, tentative action/result lessons. These are model
   notes, not measured facts. Latest observations supersede old bowl positions.
   Lessons reset for every new trial; raw conversation/tool results also remain
   available, with existing image-history pruning.

The new `observe` tool obtains a stationary observation without motion
waypoints, while preserving the commanded grasp. Observation sequence numbers
and capture times distinguish fresh frames from historical ones. Repeated
observations at the same pose now receive unique frame filenames.

## Live controls

**Scene moved — reobserve** pauses an in-flight command at the next control
check while preserving its last commanded gripper setting, or invalidates a
decision being computed. The trial then continues from a fresh observation.
It is separate from the existing E-stop, which terminates the trial.

Through the existing feedback file or terminal interface, `/reobserve` and
`/scene_changed` request the same refresh at the next polling boundary.
Unlike the UI callback, file/terminal notifications are not polled inside the
motion stream. Ordinary feedback can describe uncertainty or a failed grasp;
explicit human advice is recorded as assistance. A scene-change notification
provides no bowl coordinates, but notified trials should still be reported
separately from fully unannounced disturbances.

Useful overrides:

```bash
--set reactive.max_motion_seconds=2
--set reactive.camera_name=top_cam
--set reactive.change_fraction=0.005
--set reactive.pixel_difference=25
```

`change_fraction` is the fraction of downsampled, blurred pixels that differ
by more than `pixel_difference` on any color channel. The profile uses 0.005;
generic `--dynamic-scene` defaults to 0.01. Tune against recorded stationary
and disturbed camera pairs for the actual camera scale and lighting. This is
a conservative image-change heuristic, **not a semantic bowl tracker or a
continuous collision sensor**. Small movements may be missed; hands, shadows,
auto-exposure, or camera movement can cause false positives. Invalid or missing
comparison frames stop the trial instead of permitting an unobserved action.

Three seconds bounds commanded trajectory duration between observations,
not end-to-end response latency: camera capture and Astra calls add latency.
An unannounced change during motion is seen at the next observation; use the
UI notification or E-stop when an immediate interruption is needed.

## Reproducible zero-shot evaluation

```bash
scripts/run_astra_yam.sh research --sim --dynamic-scene \
  --suite configs/airpod_bowl_disturbances.yaml --iterations 0 \
  --max-calls 40 --max-seconds 240 --output runs/bowl-zero-shot-001
```

`--iterations 0` evaluates a fresh episode per case without a prompt optimizer.
The suite has a single bowl displacement and a repeated-displacement case.
Scheduled moves occur while the model response is pending, at fixed call
numbers, and are not announced to the policy. An episode ending before all
scheduled disturbances occur cannot score success. The verifier uses current
bowl containment, interior support, release, and rim height—not a model's
completion claim or the bowl's reset location.

Compare with the same suite, model, effort, budgets, and repetitions without
`--dynamic-scene`. Add the optional `--task-profile airpod-bowl` to investigate
task advice separately. Call-indexed disturbances can occur at different task phases
under different policies, so this is a reproducible stress test, not a matched
physical-phase benchmark. Use more repetitions and distinct scenes for a
reliability estimate. Controller horizon and prompt change together in this
comparison; an ablation is needed to attribute their individual contributions.

For a disturbance at a matched task stage, use
`--suite configs/airpod_bowl_release_eval.yaml`. Each case moves the bowl 8 cm
once, while the first gripper-opening command for a held case is pending.
The two cases move in different directions. The environment uses held-object
state only to schedule the perturbation; it never sends that state or the
new coordinates to Astra. This prevents fast policies from finishing before
a late call-number disturbance. Applied events and their response indices
are retained in `evaluation.json` under `disturbances`.

Inspect `evaluation.json`, `summary.json` (`stale_actions`, `observation_pauses`),
the unique camera frames, and tool responses. A frozen source snapshot accompanies
each research experiment. See [research details](RESEARCH.md) for artifact layout.

Prompt design reference: [official Astra guidance](https://developers.openai.com/api/docs/guides/latest-model).
No external task demonstrations or benchmark transcripts were used.
