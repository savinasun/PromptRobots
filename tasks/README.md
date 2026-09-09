# Astra evaluation tasks — bimanual YAM (station `skild-yam-8`)

40 short tasks, 10 per category, for evaluating `gpt-6-astra` in closed loop on the real arms.

| Category | Folder | What it probes |
|---|---|---|
| Spatial | [spatial/](spatial/) | grounding spatial language into each arm's own base frame: near/far, left/right, middle, midpoint, ordering, arm selection |
| Affordance | [affordance/](affordance/) | grasping an object at the part that makes it work — handle, rim, edge, stem — and choosing the right object for a stated purpose |
| Collision | [collision/](collision/) | keeping the whole arm (not just the grasp point) clear of obstacles, the table, and the other arm |
| Dexterous | [dexterous/](dexterous/) | fine placement, tilt/pour, thin and small objects, orientation control while holding, insertion, handover |

Each folder holds a runnable goals file plus a `README.md` with the per-task scene setup and pass criteria.

```bash
scripts/run_astra_yam.sh run --config configs/skild_yam_8.yaml \
  --goals-file tasks/spatial/goals_spatial.txt --fast
```

The runner prompts for a scene reset between goals, so a whole category can be run as one batch.

## Props

Everything here uses the kitchen props in [items.jpg](items.jpg): blue / white / mint plates, orange bowl,
white mug (with handle), teal cup (no handle), dark watering can (handle + spout), wooden spoon, fork,
spatula, bamboo cutting board, metal tray, white round lid.

## Scene frame and the placement grid

Targets are in **each arm's own base frame** (+x forward out of the base, +y left, +z up). The right base sits
0.61 m to the right of the left one, so a point's coordinates in the two frames differ only in y:

```
y_right = y_left + 0.61
```

Setups below are written in the **left arm's frame**. The lanes used by the tasks:

| Lane | y (left frame) | y (right frame) | Reachable by |
|---|---|---|---|
| left lane | +0.20 | +0.81 | left arm |
| left-center | −0.10 | +0.51 | left arm (right arm only at full stretch) |
| **midline** | **−0.305** | **+0.305** | both |
| right-center | −0.50 | +0.11 | right arm |
| right lane | −0.81 | −0.20 | right arm |

Rows: **near** x ≈ 0.22, **mid** x ≈ 0.32, **far** x ≈ 0.42 — all inside the 15–48 cm band in front of a base
that earlier trials worked in.

**Shared band.** Kinematics is not the limit here (max reach is 0.866 m; the midline is only ~0.43 m from
either base). Taking ±0.40 m as each arm's comfortable working half-width — well inside the URDF-derived
bound of ±0.777 — the left arm covers y_left ∈ [−0.40, +0.40] and the right arm covers y_left ∈ [−1.01, −0.21].
The **overlap is y_left ∈ [−0.40, −0.21]**, a 19 cm band centred on the midline. Bimanual and handover tasks
put their target in that band. (The "11 cm dead zone, no shared workspace" note in `goals_kitchen.txt`
came from the older hand-set `bounds.y = ±0.25`, not from the arms.)

## Before the first run

* **Set the table height.** `bounds.z` now spans the full FK envelope (−0.516 … 0.864 m), so nothing stops a
  command into the table. Set `motion.tool_floor_z_m` to the measured table surface height in the base frame
  before running any of these — measure it, do not guess (rest the jaws on the table and read `eef_state[2]`,
  or use `python -m astra_yam viz`). Every height in a task is written relative to the table ("5 cm above"),
  so no task text depends on that number.
* **Tilt.** The pour/tilt tasks need pitch and roll actuated: pass `--release-tilt 80` (the value the chili
  pour traces used). Tasks that must not tilt say so and use `--release-tilt 0`.
* **Budget.** At the default 1 cm/s a 30 cm move costs 30 s and 300 of the 3000 waypoints, and 100 LLM calls
  at ~8 s each already fill 13 of the 20 allowed minutes. Run these with `--fast` (5 cm/s) unless a task
  says otherwise; the gateway checks and the 0.25 rad tracking abort are unchanged. Tasks marked **long**
  also want `--max-seconds 1800`.
* **Clearance.** Two parallel tools cannot work within about 5 cm of each other — the second must be yawed
  (about +1.57 for the right arm) so its housing stays on its own side. Several collision and dexterous tasks
  exist precisely to see whether Astra works this out.

## Scoring

Record one of these per trial, plus the LLM-call count and any gateway rejections:

| Result | Meaning |
|---|---|
| **PASS** | goal achieved, Astra called `done` |
| **SOFT** | goal achieved, but only after operator feedback, or the budget ran out before `done` |
| **FAIL** | wrong object or place, something knocked over or dragged, or `done` called on an unachieved goal |
| **REFUSED** | `give_up` with a correct reason (the thing really is not in the scene or not reachable) — a correct refusal, scored apart from FAIL |
| **BLOCKED** | ended by a rejection loop, tracking abort, or e-stop |

The `done` / `give_up` answer to "what do you wish you had known from the start" is worth logging verbatim —
it is the clearest signal of which part of the scene Astra misread.
