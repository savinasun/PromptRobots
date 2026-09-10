# PromptRobots (Astra + YAM)

Closed-loop runner for bimanual YAM arms.

- Goal + observations go to Astra
- Astra returns one tool call (`move_to`, `done`, `give_up`)
- Gateway validates, runs IK/safety checks, and executes motion

## Setup

Use the `gello` conda env via:

```bash
scripts/run_astra_yam.sh --help
scripts/run_astra_yam.sh run --help
```

API key lookup order:
1. `OPENAI_API_KEY` env var
2. `.env`
3. `.secrets/OPENAI_API_KEY`
4. `~/.secrets/OPENAI_API_KEY`

Example:

```bash
mkdir -p .secrets
chmod 700 .secrets
printf '%s' 'sk-...' > .secrets/OPENAI_API_KEY
chmod 600 .secrets/OPENAI_API_KEY
```

## Common commands

```bash
# connectivity check (robot/cameras/key)
scripts/run_astra_yam.sh check --save-frames

# simulation (no hardware, scripted Astra)
scripts/run_astra_yam.sh run --sim --mock-astra --fast-sim --yes --goal "Pick up blue and place on top of green block."

# simulation with real Astra model
scripts/run_astra_yam.sh run --sim --goal "Pick up blue and place on top of green block."

# real robot run
scripts/run_astra_yam.sh run --config configs/skild_yam_8.yaml --goal "Pick up blue and place on top of green block."

# run multiple goals
scripts/run_astra_yam.sh run --config configs/skild_yam_8.yaml --goals-file tasks/spatial/goals_spatial.txt
```

## Your command (argument guide)

```bash
scripts/run_astra_yam.sh run --config configs/skild_yam_8.yaml --viser --yes \
  --goal "Pick only one pencil and place it in the green plate." \
  --effort low \
  --image-history 2 \
  --fast --fast-sim \
  --max-calls 100 \
  --prompt-budget 100 \
  --max-seconds 2100
```

- `run`: start one closed-loop trial.
- `--config ...yaml`: load base config.
- `--viser`: browser 3D/operator UI.
- `--yes`: skip confirmation prompts.
- `--goal "..."`: task instruction.
- `--effort low`: model reasoning effort (`low|medium|high|xhigh|max`).
- `--image-history 2`: keep fewer past images (lower token growth).
- `--fast`: faster motion profile.
- `--fast-sim`: sim-only skip real-time waits (ignored on real robot backend).
- `--max-calls 100`: hard cap on model calls.
- `--prompt-budget 100`: call budget announced to model (usually same as `--max-calls`).
- `--max-seconds 2100`: wall-clock timeout (35 min).

## Notes on latest structure

- Legacy reference transcript files such as `configs/0000_example_input.json` and `configs/0002_example_input.json` are no longer part of this repository.
- Use `scripts/run_astra_yam.sh show-prompt` to inspect current prompt/tool schemas.
- Use `scripts/run_astra_yam.sh workspace` to derive current workspace bounds from URDF.

## Safety

- Keep E-stop available on real hardware.
- Validate with `check` before runs.
- Test new goals in simulation first.
