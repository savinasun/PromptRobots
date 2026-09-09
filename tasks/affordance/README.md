# Affordance (AF01–AF10)

Grasping an object at the part that makes it work — handle, rim, edge, stem — and, in the last two, picking
the right object for a stated purpose. Every one of these is achievable with a body grasp *except* where the
goal names the part; the test is whether Astra grasps the functional feature when it is asked to, and whether
its wrist camera is used to confirm the jaws are actually straddling that feature before closing.

All positions are in the **left arm's frame** and inside its comfortable band (|y| ≤ 0.40, x 0.20–0.45), so
arm choice is never the difficulty here. `y_right = y_left + 0.61`.

```bash
scripts/run_astra_yam.sh run --config configs/skild_yam_8.yaml \
  --goals-file tasks/affordance/goals_affordance.txt --fast
```

Tilt is left at its default range. None of these need a large tilt, but AF04 and AF08 (flat plate, flat board)
go better if the tool can pitch slightly, so do not pin it with `--release-tilt 0`.

| ID | Goal | Setup | Pass | Probes |
|---|---|---|---|---|
| AF01 | Mug by its **handle**, lift 10 cm, set back down | mug (0.30, −0.10) upright, handle pointing +y | jaws close on the handle (not the body), mug lifts and returns upright within ~3 cm of its start | the canonical handle grasp; the handle is a thin loop, so the jaws must be aligned with it |
| AF02 | Watering can by its **handle**, 15 cm to the left, upright | can (0.32, −0.305), handle at the back, spout +y | can ends upright at about (0.32, −0.155), spout still clear of the table | a heavy, top-loaded object whose only good grasp is the handle; body grasps tip it |
| AF03 | Wooden spoon by its **handle**, into the orange bowl | spoon (0.30, −0.10) handle toward the arms, bowl (0.32, −0.36) | spoon is picked up by the shaft, ends resting in the bowl | the goal explicitly excludes the easier grasp (the spoon's bowl) — does Astra obey the named part |
| AF04 | Blue plate by its **rim**, 10 cm forward, flat | plate (0.30, −0.10) | plate ends at about (0.40, −0.10), still flat, not flipped or dragged on its face | a flat object has exactly one graspable feature; needs a near-vertical approach onto the rim |
| AF05 | Fork by its **handle**, onto the white plate | fork (0.28, −0.10) handle toward the arms, plate (0.34, −0.36) | fork ends on the plate, having been held by the handle | thin, low-profile object; wrist camera is the only view that resolves the handle |
| AF06 | Teal cup lying on its side → **standing upright** | cup (0.30, −0.10) on its side, mouth +y | cup ends standing upright on the table | the functional state of a cup; needs a yaw or a re-grasp, not just a lift |
| AF07 | Spatula by its **handle**, onto the cutting board | spatula (0.28, −0.10) handle toward the arms, board (0.34, −0.36) | spatula lies on the board, having been held by the handle | same shape class as AF05 but wider and floppier at the working end |
| AF08 | Cutting board by its **edge**, 10 cm right, flat | board (0.32, −0.20), long edge facing the arms | board ends at about (0.32, −0.30), still flat | a large flat object: the edge grasp needs the jaws roughly vertical and the board is wider than the 9.5 cm jaw opening in every other direction |
| AF09 | Pick the utensil you would **stir soup** with → into the bowl | spoon (0.30, −0.05), fork (0.30, −0.20), bowl (0.34, −0.36) | the **wooden spoon** ends in the bowl | function named instead of the object; the fork is the nearer distractor by nothing, the spoon by purpose |
| AF10 | Pick the thing you can **drink from** → onto the mint plate | mug (0.30, −0.10), fork (0.30, −0.20), blue plate (0.30, +0.15), mint plate (0.36, −0.36) | the **white mug** ends on the mint plate | function naming again, with a plate in the scene as the shape-similar distractor to the target plate |

## Notes

* AF01 is the task the kitchen sim could not run — the simulated mug is a plain cylinder with no handle, and
  Astra correctly gave up on it there. On the real arms the handle exists, so `give_up` here is a FAIL.
* AF02, AF04 and AF08 are the three where a wrong grasp does visible damage (tipping the can, flipping the
  plate, dragging the board). Watch the first close of the jaws; abort if the approach is not vertical.
* AF09 and AF10 are the only two where the observation cannot be turned into a target by geometry alone —
  they need the object's purpose. If Astra picks correctly but then fumbles the transport, score SOFT and
  note that the selection was right: the selection is what the task is for.
* Grasp width: the jaws open about 9.5 cm. The mug handle, spoon shaft, fork handle, spatula handle, plate
  rim and board edge all fit; the watering can body, the bowl and the board face do not. That asymmetry is
  what makes the named part the only workable grasp on several of these.
