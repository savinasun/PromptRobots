# Spatial reasoning (SP01–SP10)

Grounding spatial language into each arm's own base frame. Most of these are hover-only: no grasp is needed,
so a failure is a reasoning failure rather than a grasping failure. Positions are in the **left arm's frame**
(`y_right = y_left + 0.61`); see [../README.md](../README.md) for the lane grid and the shared band.

Run with tilt pinned — where the gripper goes is the whole question:

```bash
scripts/run_astra_yam.sh run --config configs/skild_yam_8.yaml \
  --goals-file tasks/spatial/goals_spatial.txt --fast --release-tilt 0
```

## Scene A — the three-plate row (SP01, SP02, SP09)

Mid row, x ≈ 0.32: **blue** at y = +0.20, **white** at y = −0.05, **mint** at y = −0.30. Nothing else on the
table. All three sit inside the left arm's reach (0.38 m, 0.32 m and 0.44 m from its base), so SP09 can sweep
the whole row with one arm. The goals never name the target's colour — resolving which plate is meant is the task.

From the **right** base the same three plates are 0.87 m, 0.65 m and 0.45 m away, so the plate closest to the
right arm is the mint one, which is the farthest from the left arm. SP01 turns on exactly that inversion.

| ID | Goal | Setup | Pass | Probes |
|---|---|---|---|---|
| SP01 | Hover 5 cm above the plate closest to the **right** arm's base | Scene A | left gripper settles over the **mint** plate (y = −0.30), nothing touched | distance measured in the *other* arm's frame; the answer is the plate farthest from the arm doing the hovering |
| SP02 | Hover 5 cm above the middle plate | Scene A | gripper settles over the **white** plate (y = −0.05) | ordinal position along a row |
| SP03 | Move the right gripper midway between the mug and the cup, 10 cm up | mug (0.32, −0.10), teal cup (0.32, −0.50), both upright | grasp point within ~4 cm of (0.32, −0.30), nothing touched | interpolating between two objects; the point lies in the shared band, reached by the *right* arm |
| SP04 | Move the left gripper 15 cm to the left of the bowl, at rim height | orange bowl (0.32, −0.305) | grasp point within ~4 cm of (0.32, −0.155) and within ~3 cm of the rim height | signed direction (+y is left) and a stated magnitude; height read off the object |
| SP05 | Hover above the teal cup with whichever arm can reach it | teal cup (0.32, −0.86), i.e. y_right = −0.25 | the **right** arm hovers over it; no left-arm attempt that ends in rejections | arm selection from the geometry, before committing |
| SP06 | Put the teal cup 10 cm to the right of the white plate | cup (0.32, +0.10), white plate (0.32, −0.20) | cup upright at about (0.32, −0.30) ± 5 cm, **not** on the plate | placement relative to a landmark, with an explicit exclusion |
| SP07 | Put the mug on the plate farther from the left arm's base | mug (0.22, −0.10), white plate (0.42, +0.20), mint plate (0.42, −0.40) | mug ends on the **mint** plate (0.42, −0.40) — 0.58 m from the left base vs 0.46 m for the white | comparative distance where both plates share x, so it turns on y |
| SP08 | Move the cup from behind the cutting board to in front of it | board (0.32, −0.305), cup (0.44, −0.305) | cup upright at x ≈ 0.20 ± 4 cm, same lane, board undisturbed | near/far along +x — the axis the overhead camera foreshortens most |
| SP09 | Hover above each of the three plates in turn, left to right | Scene A | three distinct hovers in the order blue → white → mint, then `done` | sequencing plus a consistent reading of "left" across three moves |
| SP10 | Both grippers hover above the outer two plates at once | blue (0.32, +0.20), mint (0.32, −0.81) | left arm over the blue, right arm over the mint, both settled before `done` | two frames at once: the same table in two different base frames |

## Notes

* SP03's midpoint (0.32, −0.30) is inside the shared band, so either arm reaches it — the goal names the
  right arm on purpose, to see whether Astra converts a point it located in the overhead view into the
  right arm's frame rather than the left's.
* SP07 is the one to watch for the frame bug: judged in the *image*, the mint plate looks no farther than
  the white one; judged from the **left arm's base**, it is 11 cm farther.
* SP09 costs the most travel of the set (about 0.5 m of hovering). It is well inside the waypoint budget.
* SP05 is a hard geometric fact, not a preference: the cup sits 0.92 m from the left base — past the 0.866 m
  maximum reach and past `bounds.y` (±0.771) as well, so a left-arm target is refused by the bounds check
  before IK is even tried — and 0.41 m from the right base.
* A correct `give_up` is possible in SP05 only if the right arm is disabled — otherwise treat `give_up` as FAIL.
