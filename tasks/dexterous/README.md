# Dexterous manipulation (DX01–DX10)

Fine placement, controlled tilt, thin and small objects, orientation held while carrying, insertion, and one
handover. These are the longest and lowest-yield tasks of the four sets — expect the pass rate here to be the
bottom of the range, and log partial progress carefully. Positions are in the **left arm's frame**;
`y_right = y_left + 0.61`.

```bash
scripts/run_astra_yam.sh run --config configs/skild_yam_8.yaml \
  --goals-file tasks/dexterous/goals_dexterous.txt --fast --release-tilt 80 --max-seconds 1800
```

**Required before running:** `motion.tool_floor_z_m` set to the measured table height. `--release-tilt` opens
pitch and roll to ±80°, and the gateway's guard against dipping a jaw tip or the housing underside below the
floor only exists once that floor is set. `--release-tilt 80` is the value the chili-pour traces used.

| ID | Goal | Setup | Pass | Probes |
|---|---|---|---|---|
| DX01 | Pour the watering can into the bowl, then set it down upright | can (0.30, −0.20) with a little water, bowl (0.32, −0.38) | most of the water lands in the bowl, can ends upright, no spill on the table | the full pour: handle grasp, transport while holding orientation, tilt over a target, recover |
| DX02 | Lay the white lid flat on the bowl, centred | bowl (0.32, −0.24), lid (0.30, −0.10) | lid rests flat across the opening, within ~2 cm of centred, not dropped in | placement accuracy on a curved rim; the lid must stay level all the way down |
| DX03 | Fork onto the plate, tines pointing away | fork (0.28, −0.10) handle toward the arms, plate (0.34, −0.34) | fork on the plate with tines pointing +x, away from the arms | orientation is part of the goal, so yaw must be controlled while carrying, not just at the grasp |
| DX04 | Stack the mint plate on the white plate, within 2 cm | white (0.30, −0.14), mint (0.30, −0.34) | mint plate seated on the white one, centres within ~2 cm, neither flipped | precision placement of a wide flat object held by its rim |
| DX05 | Turn the cup a quarter turn and set it back down | teal cup (0.30, −0.20), a mark facing the arms | cup back within ~3 cm of its spot, rotated 90° ± 20° about vertical | yaw actuation on a held object — the simplest in-hand reorientation this rig can do |
| DX06 | Spoon into the bowl with its handle on the rim | spoon (0.28, −0.10), bowl (0.32, −0.34) | spoon's bowl inside, handle resting over the rim — not lying flat in the bottom | a placement with a pose constraint rather than just a location |
| DX07 | Get the lid off the middle of the cutting board | lid flat on the board (0.32, −0.24), ≥5 cm from every edge | lid ends on the table beside the board, board not knocked off position | a thin object with no vertical clearance: it has to be slid to an edge before it can be grasped |
| DX08 | Stand the fork in the cup, handle end down | cup (0.32, −0.30) upright, fork (0.28, −0.10) | fork ends standing in the cup, cup still upright | insertion into an opening a few cm across, with a long object whose far end swings |
| DX09 | Tip the mug over the bowl to empty it, then set it down upright | mug (0.30, −0.20) with a little water, bowl (0.32, −0.38) | water into the bowl, mug back upright | the same tilt skill as DX01 but on a handle grasp, where the tilt axis and the handle axis differ |
| DX10 | Hand the cup from the left arm to the right arm | cup (0.30, +0.10), target (0.32, −0.60) | cup ends upright at the target, having been transferred between the arms without being set down in between | **stretch task.** Bimanual coordination at the clearance limit |

## Notes

* **Roll, not pitch, is the pour axis at these poses.** From a hold near (0.30, −0.15, 0.20) the sim reached
  roll of ±1.3 rad (about 74°) while pitch was IK-unreachable beyond about +0.9 / −0.6 rad. DX01 and DX09
  should tilt in roll; if Astra reaches for pitch and collects IK rejections, that is a finding worth
  logging rather than a scene problem. A can starts pouring at roughly 50° of tilt.
* **DX10 is the stretch task and should be run last.** The handover point has to sit in the shared band
  (y_left ∈ [−0.40, −0.21]), and two parallel tools cannot be within about 5 cm — the receiving arm must be
  yawed roughly +1.57 so its housing stays on its own side. The geometry does force the handover rather than
  inviting it: the cup starts 0.77 m from the right base and the target sits 0.68 m from the left base, both
  outside either arm's comfortable band, so neither arm can do the whole job. Watch it on the e-stop.
* **DX07 has no vertical-clearance solution.** A flat lid in the middle of a flat board cannot be pinched
  from above; the jaws hit the board first. It has to be slid or dragged to an edge and then grasped. If
  Astra reasons its way to that, note it — it is the most interesting single behaviour in this set.
* **Water is optional.** DX01 and DX09 work dry: score them on the tilt angle reached and whether the mouth
  was over the bowl at the time. Run them dry the first time, and only add water once the motion looks right —
  water on the table reaches the arm bases.
* **Budget.** These are the tasks most likely to hit the caps. At 5 cm/s (`--fast`) DX01, DX09 and DX10 each
  run 15–25 LLM calls of transport before the interesting part begins; `--max-seconds 1800` is in the command
  line above for that reason. If a trial ends on budget with the object correctly grasped and positioned but
  the final motion incomplete, score SOFT and record the call count.
