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

## Advanced Usage Guide

### Core arguments

- `run`: start one closed-loop trial.
- `--config PATH`: load base YAML config (recommended for real robot sessions).
- `--goal "TEXT"`: single natural-language task instruction.
- `--goals-file PATH`: run multiple goals (one per line, `#` comments allowed).
- `--sim`: shorthand for `--robot sim --cameras sim`.
- `--viser`: enable browser 3D/operator UI.
- `--yes` / `-y`: skip confirmation prompts.

### Argument reference

- `--model NAME`: override model from config.
- `--effort {low|medium|high|xhigh|max}`: reasoning effort.
- `--image-history N`: keep images for latest N observations (lower N saves tokens).
- `--max-calls N`: hard cap on LLM calls.
- `--prompt-budget N`: budget announced to the model (usually same as `--max-calls`).
- `--max-seconds S`, `--max-waypoints N`: hard time/waypoint limits.
- `--fast`: use faster motion defaults (overridden by explicit `--speed` or `--set`).
- `--speed MPS`: set linear Cartesian speed directly.
- `--release-tilt DEG`: constrain pitch/roll bounds symmetrically to `±DEG`.
- `--no-home`: do not move to home pose at trial start.
- `--home-on-end`: return to home pose at trial end.
- `--strict-gateway`: first rejected command packet ends the session.
- `--robot {zmq|sim}`, `--cameras {realsense|sim|none}`: backend selection.
- `--host HOST`, `--port PORT`: robot server endpoint override.
- `--scene {blocks|kitchen|airpods|chili|empty}`: simulator scene preset.
- `--fast-sim`: sim-only skip real-time sleeps.
- `--set KEY=VALUE`: override config keys (dotted paths supported, JSON-parsed values).

## Notes on latest structure

- Legacy reference transcript files such as `configs/0000_example_input.json` and `configs/0002_example_input.json` are no longer part of this repository.
- Use `scripts/run_astra_yam.sh show-prompt` to inspect current prompt/tool schemas.
- Use `scripts/run_astra_yam.sh workspace` to derive current workspace bounds from URDF.

## Safety

- Keep E-stop available on real hardware.
- Validate with `check` before runs.
- Test new goals in simulation first.
