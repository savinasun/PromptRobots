# PromptRobots (Astra + YAM / UR5e)

Closed-loop runner for a bimanual rig: two i2rt YAM arms (default) or two UR5e arms with Robotiq 2F-85
grippers. The rig is chosen per run; see [Embodiments](#embodiments-yam-or-ur5e).

For scored, repeatable evaluations using inspect-robots and interchangeable
agent policies, see [the evaluation pipeline](docs/INSPECT_EVAL.md).
Start with `scripts/setup_inspect_yam.sh` and
`scripts/run_inspect_yam.sh smoke --epochs 2`.

On `astra-actions-only`, robot policy output defaults to strict action calls:
numeric `move_to` targets and empty `done`, `give_up`, or (in dynamic mode)
`observe` arguments. Notes, lessons, and action explanations are disabled. Available API reasoning
summaries are printed in the terminal and saved in the transcript. Astra still performs internal reasoning at `low`; it does not support
turning reasoning off. See [action-only mode](docs/ACTIONS_ONLY.md).
Use `--language-output` to enable action notes and justifications.

- Goal + observations go to Astra
- Astra returns one tool call (`move_to`, `done`, `give_up`)
- Gateway validates, runs IK/safety checks, and executes motion

Use `--planning` for a two-model loop: a System 2 model reads the cameras and writes a text plan;
Astra (System 1) turns that plan into motion targets for the existing IK/gateway.
Both conversations persist throughout each trial, and all returned outputs/reasoning are logged.
In `scripts/run_example.py`, use `--planning` to enable this mode or `--no-planning` to disable it.
See [planning mode](docs/PLANNING.md) for configuration, histories, and log files.

## Embodiments: YAM or UR5e

Every run drives one of the rigs described in `utils/arm_models.py`:

| `embodiment_name` | arms | gripper (open span) | bases | station YAML |
|---|---|---|---|---|
| `yam_arms` (default) | i2rt YAM v2 | compact parallel jaw (9.5 cm) | upright, 0.61 m apart | `configs/skild_yam_8.yaml` |
| `ur5e_arms` | Universal Robots UR5e | Robotiq 2F-85 (8.5 cm) | 0.50 m apart, pitched 45 deg outward (`configs/skild_ur5e.urdf`) | `configs/skild_ur5e.yaml` |

Pick the rig with the station YAML or the `--embodiment` flag (any subcommand):

```bash
# UR5e, real robot (gello: python experiments/launch_nodes.py --robot=bimanual_ur, port 6001)
scripts/run_astra.sh run --config configs/skild_ur5e.yaml --goal "Pick up the blue block."
# UR5e in simulation, no YAML needed
scripts/run_astra.sh run --embodiment ur5e_arms --sim --viser --goal "Pick up the blue block."
# back to the YAM for the next run
scripts/run_astra.sh run --config configs/skild_yam_8.yaml --goal "Pick up the blue block."
scripts/run_astra.sh workspace --embodiment ur5e_arms --arm right     # UR5e reach envelope / joint limits
scripts/run_astra.sh show-prompt --embodiment ur5e_arms                # UR5e system prompt + tools
```

What the model sees is the same on both rigs: 14-DoF joint state, one gravity-aligned base frame per arm
(+x forward, +y left, +z up, origin at that arm's base mount), yaw/pitch/roll relative to the trial start, grippers
0 = closed .. 1 = open. Everything that differs is data on the `ArmModel`: the single-arm MuJoCo model used for
FK/IK (`utils/assets/yam.xml`, `utils/assets/ur5e.xml`, the latter generated from the URDF by
`scripts/build_ur5e_mjcf.py`), the joint box, the home pose, how each base is mounted (the UR5e result is rotated
from the tilted base into the level arm frame), the gripper geometry used by the tilt guard and the arm-to-arm
clearance capsules, the bimanual URDF for the visualizer, and the prose prompts (`docs/SYSTEM_PROMPT_UR5E.md`,
`docs/TILT_NOTE_UR5E.md`). A YAML only has to state what differs from its rig's defaults; `python -m utils run ...`
prints the resolved config.

UR5e specifics worth knowing before a hardware run:
- The table height is taken from the stand model (URDF root on the table, base mounts 0.113 m above it), so the
  tilt-guard floor is `motion.tool_floor_z_m: -0.113` in each arm's frame. Re-measure it on the rig.
- The UR5e home pose (`robot.home_joints_*`) puts both grasp points 0.35 m ahead of the base line, 0.20 m above
  the table, fingers straight down. Homing refuses to sweep any joint more than
  `robot.max_homing_joint_delta_rad`; park the arms near that pose first.
- gello's `bimanual_ur` server hardcodes the arm IPs; the arm it calls "left" must be the one mounted at +y of the
  stand (`left_base` in the URDF), or the frames the model reasons in are mirrored.
- `cameras.station_config_path` must point at the UR5e station's camera serials.

## Shared live prompts

Use `--dynamic-scene` for reobservation and short actions when objects move.
Change tasks through `--goal` or `--goals-file`; `--policy-notes` optionally adds
advice from a file you supply. See [the prompt guide](docs/PROMPTS.md) for request
contents and `show-prompt --bundle-json` for the assembled prompt and tools.

## Setup

Use the `gello` conda env via:

```bash
scripts/run_astra.sh --help
scripts/run_astra.sh run --help
```

Model backends (`astra.backend` in the YAML, or `--model` for the model ID):
- `openrouter` (default): OpenAI-compatible Chat Completions through OpenRouter. Default model
  `qwen/qwen3.8-max-0902` (Qwen3.8 Max; tool calling + image input). Key: `OPENROUTER_API_KEY`.
  Any other tool-capable, vision-capable OpenRouter model ID works, e.g. `--model qwen/qwen3.8-flash`.
- `openai`: the OpenAI Responses API with `gpt-6-astra`. Key: `OPENAI_API_KEY`.
- `scripted`: offline replay of a fixed tool-call list (tests / plumbing checks).

Shadow models (`shadow:` in the YAML, or `--shadow-model MODEL` repeated / `--no-shadow`): one or more
extra models receive the identical request every turn, in parallel. Each proposed action and note is printed
after the controlling model's (`[compare call N]`), written to `transcript.txt` and `notes_compare.txt`, and
saved as `responses/shadow_NNNN.json` (`shadow_1_NNNN.json`, ... for the extra models). Nothing a shadow
returns is executed or shown to the controlling model. Each shadow keeps its own conversation branch: it
reads the shared system prompt, goal and observations, but in place of the controlling model's turns it sees
its own earlier proposals, each answered with a "not executed" result naming the targets the robot actually
followed. Shadows never read the controlling model's notes or each other's (`shadow.independent_history: false`
restores the old behaviour of sending every shadow the controlling model's exact request). The request each
shadow received is saved as `requests/shadow_NNNN.json`. An `extra_models` entry is a model ID on the shadow backend,
or a mapping with its own settings, e.g. `{backend: openai, model: gpt-6-astra}` to shadow with Astra while another
model controls. Shadow models must accept tool calling and image input.

API key lookup order (`NAME` = `OPENROUTER_API_KEY` or `OPENAI_API_KEY`, per backend):
1. `NAME` env var
2. `.env`
3. `.secrets/NAME`
4. `~/.secrets/NAME`

Example:

```bash
mkdir -p .secrets
chmod 700 .secrets
printf '%s' 'sk-or-...' > .secrets/OPENROUTER_API_KEY
chmod 600 .secrets/OPENROUTER_API_KEY
```

## Common commands

```bash
# connectivity check (robot/cameras/key)
scripts/run_astra.sh check --save-frames

# simulation with real Astra model
scripts/run_astra.sh run --sim --goal "Pick up blue and place on top of green block."

# real robot run
scripts/run_astra.sh run --config configs/skild_yam_8.yaml --goal "Pick up blue and place on top of green block."

# run multiple goals
scripts/run_astra.sh run --config configs/skild_yam_8.yaml --goals-file tasks/spatial/goals_spatial.txt
```

## Advanced Usage Guide

### Core arguments

- `run`: start one closed-loop trial.
- `--config PATH`: load base YAML config (recommended for real robot sessions).
- `--goal "TEXT"`: single natural-language task instruction.
- `--goals-file PATH`: run multiple goals (one per line, `#` comments allowed).
- `--sim`: shorthand for `--robot sim --cameras sim`.
- `--viser`: enable browser 3D/operator UI.
- Viser includes live notes, recent activity, camera presets, and render-timeout recovery; see [the Viser guide](docs/VISER.md).
- `--yes` / `-y`: skip confirmation prompts.

### Argument reference

- `--model NAME`: override model from config.
- `--planning` / `--no-planning`: enable/disable System 2 text planning before each motion-model call.
- `--planner-model NAME`, `--planner-backend BACKEND`, `--planner-effort LEVEL`: configure System 2.
- `--effort {low|medium|high|xhigh|max}`: reasoning effort.
- `--image-history N`: keep images for latest N observations (lower N saves tokens). N counts observations, not images: each observation carries every camera, so N=4 on a 3-camera rig holds 12-24 images (the boundary is floored to a multiple of N, so it floats between N and 2N to keep the prompt-cache prefix stable).
- `--max-calls N`: hard cap on LLM calls and the budget announced to the model; counts both planner and motion calls.
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

## Repository layout

- `utils/` package code (`arm_models.py` describes the rigs), `scripts/` executables, `configs/` YAML/URDF, `docs/` documentation and the
  model-facing prose prompts (`SYSTEM_PROMPT.md`, `TILT_NOTE.md`, their `*_UR5E.md` variants, `PLANNER_PROMPT.md`, `MOTION_PROMPT.md`),
  `tasks/` goal files, `runs/` trial logs. Prompt paths are configurable; these are only the defaults.
- A trial directory is named `<yyyymmdd>_<HHmmss>_<success|fail>`. It is created as `_fail` and promoted to
  `_success` only when the run ends with `done`, so a run that is killed mid-flight keeps `_fail` without any
  of our code having to run. `give_up`, a budget exhaustion and a crash all stay `_fail`.
- A finished run is filed by outcome: `runs/succeeded/` if it ended with `done`, `runs/recycled/`
  otherwise. Only a run killed before it could be filed is left loose in `runs/`, still named `_fail`.
- `runs/.succeeded_map.json` and `runs/.recycled_map.json` index each folder in filing order and are
  appended to as runs are filed. `logging_utils.rebuild_map(log_root, folder)` regenerates one from disk
  if it is deleted or corrupted.

## Notes on latest structure

- Legacy reference transcript files such as `configs/0000_example_input.json` and `configs/0002_example_input.json` are no longer part of this repository.
- Use `scripts/run_astra.sh show-prompt` to inspect current prompt/tool schemas.
- Use `scripts/run_astra.sh workspace [--embodiment ur5e_arms] [--arm right]` to derive workspace bounds from the rig's URDF.

## Safety

- Keep E-stop available on real hardware.
- Validate with `check` before runs.
- Test new goals in simulation first.
