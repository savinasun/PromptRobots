# Action-only branch

The prior shared reactive harness is committed on `main` at `6d2b302`.
This branch defaults to `astra.actions_only: true`, including the station config.

```bash
# Inspect the action-only system prompt and strict schemas without connecting
scripts/run_astra_yam.sh show-prompt --dynamic-scene --bundle-json

# Run the generic policy in simulation
scripts/run_astra_yam.sh run --sim --scene airpod_bowl --dynamic-scene \
  --goal "Pick up the charging case and place it inside the green bowl."
```

`move_to` accepts only `targets`. All 14 dimension names must be present, each
with a numeric absolute target or `null` to hold its current value. At least one
target must be numeric. The harness removes nulls before the existing gateway
checks and motion execution. `done`, `give_up`, and `observe` take empty objects;
`observe` is available only in dynamic mode. There are no model-generated note,
lesson, summary, reason, or hindsight fields.

Requests enforce `tool_choice: required`, `parallel_tool_calls: false`, and
strict schemas with no extra fields. Local validation stops the trial with
`invalid_action_output` before dispatch if a response contains language,
assistant prose, multiple actions, or invalid arguments. API reasoning summaries
are permitted as separate diagnostic metadata. Raw rejected
responses remain in the audit logs. Human goals, images, state, feedback, and
gateway results remain available as input. Written episode lessons are disabled;
adaptation uses observation and action-result history.

Action-only requests force `reasoning.effort: low`. They request reasoning
summaries with `astra.reasoning_summary: auto` by default and print any returned
summaries after each response, also saving them in `transcript.txt`. To request
summaries explicitly with any config, add `--set astra.reasoning_summary=auto`.
A response may have no summary; unsupported summary requests use the existing
fallback without changing the effort or action schema. **This does not
disable internal reasoning:** GPT-6 Astra does not support effort `none`.
Encrypted reasoning metadata is retained for stateless tool-call continuity.
See [official Astra guidance](https://developers.openai.com/api/docs/guides/latest-model)
and [function calling](https://developers.openai.com/api/docs/guides/function-calling).

`--language-output` explicitly restores the earlier language contract and
configurable reasoning controls.

Changes were verified with offline tests; no hardware or live Astra performance
validation was performed for this branch.
