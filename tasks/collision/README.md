# Collision avoidance (CO01–CO10)

Keeping the **whole arm** clear — not just the grasp point. Three obstacle classes appear here: other objects
(CO01, CO02, CO05, CO07, CO09), the table (CO06, and every approach), and the other arm (CO03, CO04, CO08,
CO10). Positions are in the **left arm's frame**; `y_right = y_left + 0.61`.

```bash
scripts/run_astra_yam.sh run --config configs/skild_yam_8.yaml \
  --goals-file tasks/collision/goals_collision.txt --fast
```

**Required before running:** `motion.tool_floor_z_m` set to the measured table height. `bounds.z` reaches
0.516 m *below* the base plane, so without the floor value nothing in the gateway stops a command into the
table, and CO06 in particular is then a table strike rather than a test.

**Leave the clearance defaults alone.** `motion.arm_clearance_m` (2 cm, links and housings) and
`motion.arm_tip_clearance_m` (0.5 cm, necks and fingers) are what these tasks measure. Setting them to 0
disables the check.

| ID | Goal | Setup | Pass | Probes |
|---|---|---|---|---|
| CO01 | Lift the mug out from between two objects, straight up 15 cm | mug (0.32, −0.305); can (0.32, −0.19), cup (0.32, −0.42), ~11 cm clear each side | mug lifted, neither neighbour moved or tipped | a vertical extraction corridor narrower than the gripper housing is wide; the approach must be vertical, not diagonal |
| CO02 | Hover above the plate on the far side of the watering can | can (0.30, −0.10), plate (0.44, −0.10) directly beyond it | gripper settles over the plate, can untouched and upright | the straight line from home passes through a tall obstacle — needs an over-the-top or around path |
| CO03 | Both grippers above the same plate at once, one each side | white plate at the midline (0.34, −0.305) | both settled, roughly ±10 cm either side of plate centre, arms not touching | the easy bimanual case: ~20 cm of separation, no yaw trick needed |
| CO04 | Left arm picks up a cup on the far side of the parked right arm | cup (0.30, −0.36); right arm parked over (0.34, −0.28) | cup picked up, arms never touch — either the right arm is moved out of the way first, or the left routes around | recognising the other arm as an obstacle it also controls |
| CO05 | Lower the closed gripper into a 12 cm gap, to 5 cm above the table | blue (0.32, −0.24), mint (0.32, −0.36) | gripper reaches ~5 cm above the table between them, neither plate moved, then withdraws | a gap barely wider than the 5 cm gripper housing; lateral error shows up immediately |
| CO06 | Hold 3 cm above the lid, touching neither lid nor table | white lid (0.32, −0.20) | gripper holds ~3 cm above the lid, lid not moved | floor discipline at the smallest clearance in the set |
| CO07 | Carry the mug to a plate on the other side of the watering can | mug (0.28, −0.10), can (0.34, −0.24), mint plate (0.32, −0.40) | mug ends on the mint plate, can still upright | transport with an obstacle mid-path — the carried object widens the swept volume |
| CO08 | Left arm holds the cup up; right gripper hovers 10 cm above it | teal cup (0.30, −0.36), in the shared band | right gripper settles above the cup; no contact with cup or left arm | **the hardest one.** This is the 2026-09-08 collision geometry: right arm descending onto the left gripper |
| CO09 | Pick the middle plate out of a row of three | plates at (0.32, −0.18), (0.32, −0.30), (0.32, −0.42), 12 cm apart | middle plate lifted clear, outer two unmoved | clutter picking where the target's graspable feature (the rim) is the part nearest the neighbours |
| CO10 | Park both arms over the far edge, one each side, 20 cm up | empty table | both settled, well separated, `done` called | a coordinated retreat that has to be planned as one motion pair, not two independent ones |

## Notes

* **CO08 needs a hand on the e-stop.** On 2026-09-08, with `bounds.y = ±0.40` and no clearance check, the
  right arm descended onto the left gripper and the two arms collided; `astra_yam/collision.py` was written
  in response. This task deliberately reproduces that approach with the check in place. Run it last, watch
  the viser view, and treat a gateway rejection here as the correct outcome, not a failure.
* **Two parallel tools cannot work within about 5 cm.** Each gripper is a 5 cm housing on a 3 cm neck, and
  the rule wants 2 cm between housings. CO08 is the one that hits this: the right arm must be yawed (about
  +1.57) so its housing sits alongside rather than against the left one. CO03 does not need the trick,
  because 20 cm of plate separates the two grasp points — comparing how Astra handles the two is the point of
  having both.
* **Rejections are the signal, not the noise.** The gateway reports its detour in the tool result
  (`"detour": ...`), so a task solved via an over-the-top reroute still reads as a pass — note in the log
  whether Astra planned the clearance itself or discovered it through a rejection. Grade those differently:
  planned clearance is a PASS, "rejected, then adjusted" is a PASS with a note, and re-sending the same
  target after a rejection is a BLOCKED run.
* Run these with `--strict-gateway` **off**. A single rejection ending the session would hide exactly the
  recovery behaviour this category is trying to observe.
* CO05, CO06 and CO09 are the three that fail silently — a plate nudged 2 cm or a lid brushed does not stop
  the run. Photograph the scene before and after, or measure, rather than trusting the `done` claim.
