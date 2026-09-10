# Charging-case placement with bowl disturbances

These September 9, 2026 experiments used live `gpt-6-astra` API calls with the
schematic YAM simulator. No physical robot was tested. The reactive policy used
the task-specific `configs/AIRPOD_BOWL.md` advice plus the reactive instructions
as captured in each experiment's source snapshot. They predate the shared-prompt
generalization described in [PROMPTS.md](PROMPTS.md).

## Release-stage comparison

`configs/airpod_bowl_release_eval.yaml` moves the bowl 8 cm while the first
release command for a held case is pending. Two cases move it in different
directions, with one repetition per case. Each variant used 40 calls, 240 seconds,
low reasoning effort, 4,096 maximum output tokens, one SDK retry, and a 90-second
request timeout. No prompt optimizer or prior-episode lessons were loaded.

| Measurement | Original prompt/controller | Task advice + reactive controller |
| --- | ---: | ---: |
| Final verified successes | 2/2 | 2/2 |
| Releases outside the current bowl | 2 | 0 |
| Mean model calls | 14.5 | 30.5 |
| Mean wall time | 51.5 s | 134.2 s |
| Gateway rejections | 0 | 0 |

Both original trials released at the stale position, then regrasped the case and
recovered. The reactive trials each discarded one stale action and released
inside the bowl. This shows fewer misplaced releases in these two trials, at
considerably higher latency and call count. It does not demonstrate better final
success rates, isolate the contribution of the prompt from the controller, or
establish reliability on hardware or across tasks.

The outside-release diagnostic compares consecutive simulator observations:
the case transitions from held to released, and its full horizontal footprint
does not fit inside the current rim. Final success independently requires
release, containment, interior support, and placement below the rim.

Artifacts in the local workspace:

- `runs/airpod_bowl_release_baseline/`
- `runs/airpod_bowl_release_reactive/`
- `runs/airpod_bowl_release_comparison.json`

Each experiment retains requests, images, tool outputs, measured verification,
the configuration manifest, and frozen source/prompt content. These local run
artifacts are ignored by Git.

## Earlier call-index stress test

`configs/airpod_bowl_disturbances.yaml` applies perturbations at fixed response
numbers. The task-specific reactive variant completed both cases with one and
two stale actions discarded respectively. The original variant completed the
first case but ended the second before its late scheduled disturbance. That
trial fails schedule coverage and cannot support a matched success-rate
comparison. This motivated the release-stage benchmark above.

Those runs are in `runs/airpod_bowl_zeroshot_baseline/` and
`runs/airpod_bowl_zeroshot_reactive/`.
