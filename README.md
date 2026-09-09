# PromptRobots — GPT-6 Astra driving the bimanual YAM arms

Closed-loop pipeline: a sequence of prompts (system prompt → goal → observation) goes to OpenAI's
`gpt-6-astra` through the **Responses API**; Astra answers with one tool call (`move_to` with Cartesian
targets, `done`, or `give_up`); a **safety gateway** turns the targets into IK joint waypoints and streams
them to the robot; fresh proprioception + camera images become the next observation; and so on until the
task is declared finished or a budget (LLM calls, waypoints, wall-clock) runs out.

Every piece of text the model sees lives in `configs/` - `SYSTEM_PROMPT.md`, `TILT_NOTE.md` and
`PROMPTS.yaml` (tool descriptions, observation message, session lines) - and is loaded through
`astra_yam/prompts.py`; editing a prompt never means touching code. Tool results are the exception: the
gateway's rejection reasons are generated next to the values that caused them.

The prompt/tool format is byte-for-byte the one used by the reference trials (`0000_example_input.json` is a
first request, `0002_example_input.json` a second one, `000{1,3}_example_astra.json` are Astra's tool calls).
Those four transcripts are **not currently in the repo**, so `tests/test_observation.py` and
`tests/test_kinematics.py` skip themselves; drop them into `configs/` to re-arm the comparison. It asserts
equality of the tool schemas and of the system prompt, which has since grown a camera-use section at the end,
so the reference text is asserted to be its prefix.

```
                     ┌──────────────────────────────────────────────────────────────┐
                     │ input items (grow every turn)                                │
 configs/SYSTEM_PROMPT.md ─► system │ "Goal: …" │ observation(step 0) │ fc │ fc_out │ observation(step 52) │ …
                     └──────────────────────────────────────────────────────────────┘
                                          │ responses.create(model="gpt-6-astra", tools, store=False,
                                          │                  include=["reasoning.encrypted_content"])
                                          ▼
                     Astra → reasoning item + function_call {"targets": {...}, "note": "..."}
                                          │
                                          ▼
      astra_yam/gateway.py   validate names/bounds/pinned axes → straight Cartesian path @10 Hz
                             → per-waypoint IK (MuJoCo yam.xml, grasp_site) → joint-limit / config-flip
                             checks → joint pacing → waypoint budget → stream @30 Hz + tracking monitor
                                          │ command_joint_state(14-d)            ▲ get_observations()
                                          ▼                                      │
      gello robot server (launch_nodes.py --robot=bimanual_yam, ZMQ tcp://127.0.0.1:6001)  ──  RealSense D405 ×3
                                          │
                                          ▼
      function_call_output {"ok": true, "steps": 52, "status": "completed", "cadence_hz": 10}
      + new observation: state[joint_pos], state[eef_state], waypoints remaining, top_cam/left_cam/right_cam
```

## Layout

| Path | What |
|---|---|
| `astra_yam/config.py` | All knobs as dataclasses (`Bounds`, `MotionConfig`, `RobotConfig`, `CameraConfig`, `AstraConfig`, `LimitsConfig`), YAML + dotted overrides |
| `astra_yam/kinematics.py` | MuJoCo FK/IK for one YAM arm at `grasp_site` (between the fingertips); relative yaw/pitch/roll conventions |
| `astra_yam/prompts.py` | Loader for `configs/PROMPTS.yaml` (`prompt("tools.note")`); no model-facing text is hard-coded |
| `astra_yam/embodiment.py` | Tool schemas (`move_to`/`done`/`give_up`), system prompt + tilt-note loaders, `state[eef_state]` computation |
| `astra_yam/observation.py` | Observation message builder (text + JPEG data URIs), `$blob:` redaction for logs, image-history pruning |
| `astra_yam/gateway.py` | The safety gateway (validation → plan → execute) |
| `astra_yam/robot_interface.py` | `ZmqYamRobot`: speaks the gello robot-server protocol (REQ/REP pickle, 14-d joints, gripper 0 = closed … 1 = open) with LINGER=0 and hard timeouts so a dead server fails fast |
| `astra_yam/cameras.py` | `RealSenseSource` (gello `RealSenseCameraFast`, serials from `metadata/station_config.json`) |
| `astra_yam/astra_client.py` | `OpenAIAstraClient` (Responses API) and `ScriptedAstraClient` (offline stand-in) |
| `astra_yam/session.py` | `TrialRunner`: the loop, budgets, operator feedback, logging |
| `astra_yam/sim.py` | Kinematic simulator (box/cylinder objects, `blocks`/`kitchen` scenes, grasp emulation) + schematic cameras + a ZMQ server speaking the gello protocol |
| `astra_yam/viser_ui.py` | viser 3D view + operator UI: URDF/meshes, bounds, plan path, browser-rendered cameras, digital-twin mirror, manual bench |
| `astra_yam/cli.py` | `run`, `check`, `viz`, `sim-server`, `show-prompt` |
| `configs/skild_yam_8.yaml` | Station defaults (bounds, speeds, ports, camera names) - same values as the dataclass defaults |
| `configs/SYSTEM_PROMPT.md` | The system prompt (budget and embodiment name are injected at load) |
| `configs/TILT_NOTE.md` | Tilt convention + clearance paragraphs, appended when pitch/roll are actuated |
| `configs/PROMPTS.yaml` | Tool descriptions, the observation message, and the lines the session injects |
| `scripts/run_astra_yam.sh` | Activates conda env `gello` and runs the CLI |
| `tests/` | `pytest` suite (kinematics, prompt format, gateway, simulated sessions, ZMQ protocol) |

## Setup

The pipeline runs in the existing `gello` conda env (zmq, pyrealsense2, mujoco, opencv). `openai>=3`,
`python-dotenv` and `pytest` were added to it. API key:

The key is looked up in this order: `OPENAI_API_KEY` in the environment, `PromptRobots/.env`, then the file
`PromptRobots/.secrets/OPENAI_API_KEY` or `~/.secrets/OPENAI_API_KEY` (file content = the key). The station
uses `.secrets/OPENAI_API_KEY`, which is git-ignored:

```bash
mkdir -p .secrets && chmod 700 .secrets && printf '%s' 'sk-...' > .secrets/OPENAI_API_KEY && chmod 600 .secrets/OPENAI_API_KEY
```

## Running on the real robot

Three terminals, in this order:

```bash
# 1) robot server (resets CAN, powers both arms, ZMQ on 6001). Leave running.
bash ~/bimanual_manipulation/launch_yam_node.sh

# 2) sanity check: API key, robot server, cameras (nothing moves; saves one frame per camera)
cd ~/savina/PromptRobots && scripts/run_astra_yam.sh check --save-frames

# 3) a trial
scripts/run_astra_yam.sh run --config configs/skild_yam_8.yaml \
    --goal "pick up blue and place ontop of green block"
```

What happens in `run`:

1. The config is printed. The robot is moved in joint space (5 s) to the **home pose** – the same pose the
   reference trials start from (grasp point ≈ (0.30, 0.00, 0.19) m in each arm's base frame, tool tilted
   58° down, grippers open). You are asked to confirm before it moves (skip with `--yes`; `--no-home` keeps
   the current pose). The home orientation defines yaw = pitch = roll = 0 for the trial.
2. Observation 0 is captured and the first request is sent. Every turn prints Astra's `note`, the gateway
   result and token usage.
3. The loop ends on `done` / `give_up`, after `limits.max_llm_calls` (100), `limits.max_waypoints` (3000),
   `limits.max_trial_seconds` (20 min), `max_consecutive_rejections` (5) rejected packets in a row, a
   tracking abort, an emergency stop (the viser button or `TrialRunner.request_estop()`), or when you type
   `/stop`. The arms **hold position** at the end (`--home-on-end` to return home).
4. **Operator feedback**: any line you type in the terminal is sent to Astra as
   `Operator feedback: …` before the next request (the system prompt tells it to trust these). With
   `--feedback-file path`, lines appended to that file are used instead (handy under `nohup`).

Several goals back-to-back (you are prompted to reset the scene in between):

```bash
scripts/run_astra_yam.sh run --config configs/skild_yam_8.yaml --goals-file goals.example.txt
```

### Where a turn's wall clock goes

Local compute is not the bottleneck: planning a 13 cm move (132 IK waypoints) takes ~90 ms, streaming it
costs 9 ms of CPU, and capturing + encoding three frames ~3 ms. A turn is **model latency + motion time**,
so those are the two things worth tuning, and every trial's `transcript.txt` ends with a **TIME** table
splitting the run into Astra / motion / observation / everything else so you can see which one you are
paying for.

* **Motion time** is `waypoints / motion.cadence_hz`, and waypoints come from `motion.linear_speed_mps`.
  At the reference 1 cm/s a 13 cm move is 132 waypoints = **13.2 s**; `--fast` (5 cm/s, 0.6 rad/s yaw,
  gripper 1.5/s, 0.1 s settle) makes it 27 waypoints = **2.7 s**. Joint pacing does not eat the gain -
  even at 10 cm/s the per-waypoint joint step stays under `max_joint_step_rad` for ordinary moves - and the
  profile changes no safety number: the same IK/limit/clearance checks and the same 0.25 rad tracking abort
  apply, and an explicit `--speed` or `--set` still wins over the profile. Faster motion does mean a larger
  tracking error on the real arms, so raise it in steps and watch for aborts.
* **Model latency** grows with the request, and the request used to carry every image of the run: by turn 60
  that is 180 images, 4.9 MB re-uploaded per call and ~138k image tokens re-prefilled per call. Images are
  now kept for the newest `astra.image_history` (4) to 2N observations, which holds a turn at ~12 images and
  0.4 MB no matter how long the trial runs (turn 20: 60 images/1.6 MB before, 12/0.35 MB now). The window
  moves in blocks of N precisely so the request *prefix* stays byte-identical for N-1 turns out of N -
  a sliding window would rewrite the prefix every turn and defeat the provider's prompt cache. The
  `[reasoning]`/usage line of each call reports how many images went out and what share of the input the
  provider served from cache, and `astra.prompt_cache_key` (default `astra-yam`) keeps the system-prompt +
  tools prefix on one cache across trials. `--image-history 0` restores the old keep-everything behaviour,
  and `--set cameras.detail=low` drops an image from ~765 to ~85 tokens at a real cost in perception.

Useful flags: `--fast` (motion profile above), `--effort high` (reasoning effort), `--max-calls 40`, `--prompt-budget 100` (budget announced to Astra; keep it realistic or it gives up), `--max-seconds 600`, `--speed 0.02`
(m/s), `--image-history 0` (keep every camera image in context; default keeps the newest 4-8 observations),
`--strict-gateway` (first rejected packet ends the session, like the reference gateway),
`--set motion.settle_seconds=0.5` (any dotted config key).

## Dry runs without hardware or an API key

```bash
# simulator + scripted "Astra" (a canned pick-and-place that completes in the sim world)
scripts/run_astra_yam.sh run --sim --mock-astra --fast-sim --yes --goal "pick up blue and place ontop of green block"

# simulator + the real model (needs OPENAI_API_KEY) – Astra sees schematic camera renderings
scripts/run_astra_yam.sh run --sim --goal "pick up blue and place ontop of green block"

# exercise the real ZMQ client path against a simulated robot server (tests also serve the simulator with gello's own ZMQServerRobot)
scripts/run_astra_yam.sh sim-server &            # SimYamRobot on tcp://127.0.0.1:6001
scripts/run_astra_yam.sh run --robot zmq --cameras none --mock-astra --yes --goal "test"

# tests
~/miniconda3/envs/gello/bin/python -m pytest tests -q
```

## 3D visualization and operator UI (viser)

`--viser` starts a [viser](https://viser.studio) server (default `http://<this-host>:8080`) next to the pipeline.
The scene is built from the bimanual URDF `skild_yam_v2.urdf` (left arm at the origin, right arm at y = −0.61 m,
FK verified against the pipeline's MuJoCo model) with the i2rt link meshes from
`third_party/robot_models/yam`, and shows: both arms with the grasp point and jaws (opening = gripper × 9.5 cm),
the per-arm Cartesian bounds as wireframe boxes, the table with a 10 cm grid, the sim objects, the three camera
poses, and the planned grasp-point path of every accepted `move_to` before it executes.

The side panel carries the trial status (goal, phase, Astra calls, waypoints), Astra's latest note, the last
camera frames, a **Start trial** button, the two stop buttons below, a feedback box
(lines reach Astra as `Operator feedback: …`), and gizmos to drag the sim objects around before or during a run.

* **EMERGENCY STOP** (red) halts a motion that is already streaming: `TrialRunner.request_estop()` sets an
  event the gateway checks before every joint command, so the arms stop within one control tick (33 ms at
  30 Hz), hold their measured pose — with each gripper kept at its last command, so an object being squeezed
  is not dropped — and the trial ends with status `estop` (a multi-goal run stops too). The tool result tells
  Astra the same thing. On the `viz` manual bench the button stops the current bench move and re-arms.
* **Stop after this motion** (orange) is the old graceful stop: it queues `/stop`, so the current motion
  finishes and the session ends before the next Astra call.

```bash
# policy test in simulation, Astra sees renders of the 3D scene from the three camera poses
scripts/run_astra_yam.sh run --sim --viser --scene kitchen --goal "Pick up the teal cup, lift it about 10 cm, and set it back down"

# same, but keep the schematic camera images
scripts/run_astra_yam.sh run --sim --viser --no-viser-render --goal "..."

# manual IK/gateway bench without Astra: drag a target gizmo, press "Move <arm> arm to target"
scripts/run_astra_yam.sh viz --scene kitchen

# digital twin next to the real robot (commanded/measured joints mirrored into the 3D view)
scripts/run_astra_yam.sh run --config configs/skild_yam_8.yaml --viser --goal "..."
```

In simulation the agent's camera images are rendered by the connected browser (`ClientHandle.get_render`) from
the top and wrist camera poses, so Astra sees the meshes, table and objects. Viewer-only helpers (bounds,
frustums, axes, plan path, gizmos) are hidden for those captures. 3D text labels are off by default
(`viz.show_labels`) because the viewer applies their removal asynchronously and they leaked into renders as white
streaks; object names and positions are listed in the *Scene objects* panel instead. If no browser is connected,
or a render times out, the schematic images are used and the observation still goes out. The wrist cameras are
mounted rigidly in the grasp-site frame 7 cm above the gripper housing and look 10 cm past the grasp point, so the
jaws appear at the bottom of the image like on the real D405 frames. The run waits for the
**Start trial** button (or Enter in the terminal) before each goal unless `--yes` is given. Scene presets:
`blocks` (default, two 3 cm cubes), `kitchen` (cup, mug, bowl, plates as cylinders sized like items.jpg),
`airpods` (the 2026-09-08 layout: an articulated AirPods Pro case on the round wooden dish, cup and plate
nearby), `chili` (open-top chili powder can, orange bowl, distractors), `empty`. Held boxes and cylinders follow
the tool's rotation, so released pitch/roll tilts them; a `SimCan` pours once tilted past 50° (fully at 95°): what
leaves it lands in the target bowl when the opening is above the bowl's footprint, otherwise it counts as spilled,
and the poured amount is drawn as a growing disc in the bowl. The case (`SimCase`) is reconstructed from Apple's dimensions (60.6 × 45.2 × 21.7 mm, 16 mm
lid, hinge on the top rear edge): grasped across its width it pivots to hang lid-up, the other gripper can pinch
the lid across its depth (jaws along x, i.e. yaw ≈ ±1.57) and swing it about the hinge; released above 57° the
lid stays open, below it snaps shut. A downloaded print of a toy-sized case (Printables model 11292, 44 mm wide,
split into half shells) is kept in `astra_yam/assets/reference/` but was not used because it is not to scale. Object interaction is kinematic but contact-aware: a grasp happens only when jaws that straddled an
object (open wider than it, axis within 1.5 cm of the grasp point, at its height) close below its width; the held
object follows the grasp point and is released onto the table or a supporting object once the jaws open 4 mm past
its width. Any other tool contact pushes the object away (closed fingertips act as a 2.4 cm pusher), so
closed-gripper pushes work and a closed gripper can never "catch" an object. The sim mug is a plain cylinder
without a handle, so use "Pick up the white mug and place it on the white plate" in simulation.

## Trial logs (`runs/<timestamp>_<goal>/`)

* `requests/request_NNNN.json` – the exact Responses API request of call NNNN with images replaced by
  `data:image/jpeg;base64,$blob:<sha1>` (same convention as the reference examples).
* `responses/response_NNNN.json` – Astra's output items (reasoning + function_call) and token usage.
* `frames/step_SSSSS_<cam>.jpg` – the camera frames of the observation taken after waypoint SSSSS.
* `transcript.txt` – **the run as plain text, readable start to finish**: the system prompt that produced it,
  the goal, then per turn the observation message (state lines plus the path of each saved frame - never
  pixels), Astra's `[reasoning]` trace, any `[message]`, the `[move_to]` targets, its `[note]`, the
  `[gateway]` result, a `[chunk]`/`[counters]` pair, operator feedback, and the final `[done]`/`[give_up]`
  with `[hindsight]` and outcome. `[chunk]` is the action-chunk size of that call - waypoints predicted
  (Cartesian path at `motion.cadence_hz`, and after joint pacing when it applies) versus executed and how
  many seconds of motion that is; `[counters]` is the running total (calls, waypoints executed/predicted,
  budget left, consecutive rejections, mean/min/max chunk). An **ACTION CHUNKS** table before the outcome
  lists every call's sizes with the totals, and the same numbers land in `summary.json` as
  `waypoints_predicted` and `chunk_sizes` and in `transcript.jsonl` as `chunk` events.
  The reasoning text is the API's reasoning **summary** (`astra.reasoning_summary`, default `auto`); the raw
  chain of thought is never returned in the clear, only as the encrypted blob the next request needs. When a
  model returns no summary the line reads `[reasoning] (not returned as text; N reasoning tokens)`, and if the
  endpoint rejects the parameter the client says so once, drops it, and the trial continues
  (`--set astra.reasoning_summary=null` to not ask at all).
* `transcript_notes.txt` – **note-only transcript**: Astra's prose and nothing else, one numbered entry per
  `note` (labelled with the call and the observation step it was written from) and the closing
  `done`/`give_up` with its `hindsight` - the run as narration, without state lines, reasoning, gateway
  results or counters.
* `notes.md` – Astra's notes per move, rejections, done/give_up summary and hindsight, final outcome.
* `transcript.jsonl` – every event (observations with eef state, tool calls with gateway plan/result,
  operator lines, errors).
* `summary.json` – status, counts, elapsed time, accumulated usage.

## Conventions and safety details

* **Joint layout** `[left_arm(6) | left_gripper | right_arm(6) | right_gripper]`, gripper normalized
  0 = closed, 1 = open – identical to gello's `command_joint_state` / `joint_positions`.
* **Frames**: each arm's base frame is the MuJoCo world frame of `yam.xml` (+x forward, +y left, +z up).
  The right arm base sits 0.61 m to the right of the left one (hw v2.x); Astra is told bases differ per rig.
  `z` is the grasp-point height above the arm's base plane, and the default lower bound is −0.516 m (the arm
  really does reach below its base plane), so set `motion.tool_floor_z_m` to your rig's table height – or
  narrow `bounds.z` – before the first real trial.
* **Grasp point** = `grasp_site` (13.47 cm past the flange). Our FK agrees with the reference trials'
  `eef_state` to within ~8 mm (the reference gateway apparently uses a slightly different link model).
* **Orientation**: yaw/pitch/roll are reported relative to the home orientation as extrinsic base-frame
  rotations (yaw about +z, CCW from above positive). All three are actuated in the default bounds (pitch ±π/2,
  yaw/roll ±π); `--release-tilt 0` pins pitch and roll at 0 instead, so the IK holds the start orientation as
  the reference rig did.
* **Gateway** (`astra_yam/gateway.py`): unknown names, non-numbers, out-of-bounds values and non-zero
  pinned axes are rejected (never clamped). The endpoint and every 1 mm waypoint must be IK-reachable
  within 2 mm / 0.02 rad, inside the joint limits (the exact `skild_yam_v2.urdf` limits, which is what the
  driver clamps to; gello's rounded `robot_constants.JOINTS_*_LIMIT_YAM` sit inside them) and without
  configuration flips (> 0.35 rad joint jump). Joint steps above 0.05 rad per waypoint are subdivided
  ("joint pacing"). The plan must fit the remaining waypoint budget. While streaming, measured joints are
  compared to commands every 5 waypoints; > 0.25 rad stops the motion, holds, and ends the session.
* **Speed**: 1 cm/s linear (≈1 mm per 10 Hz waypoint – the reference trace shows 52 waypoints for a
  5.5 cm move), 0.15 rad/s yaw, gripper 0.5/s. Commands are interpolated at 30 Hz.
* **Arm-to-arm clearance** (`astra_yam/collision.py`, added after the two arms collided on 2026-09-08 with
  `bounds.y=[-0.40,0.40]`): each arm is a chain of capsules (base, upper arm, forearm, wrist, 5 cm gripper
  housing, 3 cm neck, 0.8 cm fingers). Every planned waypoint must keep 2 cm between arm links/housings of the
  two arms and 0.5 cm between their necks/fingers (`motion.arm_clearance_m`, `motion.arm_tip_clearance_m`;
  0 disables). Motions that increase an existing violation are rejected, motions away from the other arm are
  always allowed. Geometric consequence: two parallel tools cannot work within ~5 cm of each other; the second
  tool must be yawed (about +1.57 for the right arm) so its body stays on its own side.
* **Detours** (`motion.detour_enabled`): when only the straight path violates the clearance rule, the planner
  tries over-the-top, to-the-side, yaw-first, and combined piecewise paths (gripper and yaw held until the
  last segment) and reports the deviation in the tool result (`"detour": ...`) so Astra knows what happened.
* **Bounds come from the URDF, not from judgement.** `python -m astra_yam workspace` reads the exact revolute
  limits out of `configs/skild_yam_v2.urdf`, sweeps that joint box with FK (200k samples + all 64 corners, fixed seed) and
  prints a ready-to-paste YAML block. Those numbers are the dataclass defaults in `astra_yam/config.py`, and
  `configs/skild_yam_8.yaml` spells the same values out explicitly: positions are the reachable
  grasp-point envelope rounded outward to the millimetre (x, y ±0.777 m, z −0.516…0.864 m, max reach 0.866 m) and
  rotations are the full range the reported Euler convention can represent (yaw/roll ±π, pitch ±π/2). It is a
  bounding box, not the reachable set; unreachable targets are still rejected per target by IK.
  `tests/test_workspace.py` fails if a bound cuts off reachable space or pads beyond the robot's reach.
  With `z` opened to the full envelope the tilt floor guard is inactive: set `motion.tool_floor_z_m` (e.g. `0.0`)
  to keep the jaw tips above the table. `--release-tilt DEG` still overrides the tilt bounds either way; the system prompt then gains `configs/TILT_NOTE.md`, which defines the convention
  (pitch about base +y, roll about base +x, applied to the start orientation before yaw), and the gateway
  rejects tilts that would dip a jaw tip or the housing underside below the workspace floor.
* **Responses API details**: `store=False` + `include=["reasoning.encrypted_content"]`, all output items
  (including encrypted reasoning) are appended back into `input`, followed by the `function_call_output`
  and the new observation. `tool_choice="required"` and `parallel_tool_calls=False` enforce exactly one
  tool call per turn (`AstraConfig`). Retries/backoff come from the SDK (`max_retries=4`).
