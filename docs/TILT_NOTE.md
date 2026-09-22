Tilt is actuated on this rig: pitch and roll may be commanded within their
bounds. Pitch is a rotation of the tool about the base +y axis and roll about
the base +x axis (right-hand rule, applied to the start orientation before
yaw). At yaw 0 and pitch 0 the fingers point forward and down at about 58
degrees below horizontal; positive pitch swings the fingertips further down and
back toward the base (negative pitch points them more forward and level), and
positive roll swings the fingertips toward the arm's left (+y). The jaw tips
and the gripper housing must stay above the workspace floor when tilted; the
gateway rejects tilts that would dip them below it. Use tilt sparingly and
re-check the observation after each change.

Tilting also changes what the arm sweeps: the gripper housing, the wrist and
the elbow swing wider than the grasp point, so a tilted tool can meet the
table, an object or the other arm while the grasp point itself still looks
clear. Both arms share one workspace. Work out the clearance yourself before
you command a motion: judge from the observation what the whole arm will sweep,
move the other arm out of the way first when it is in that volume, and leave
more room around tall or fragile things than around the target. The gateway
also checks arm-to-arm clearance and may reroute or reject a motion, but a
rejection is information about the geometry, not a substitute for planning:
after one, change the approach (back off, raise, turn, or clear the other arm)
instead of re-sending the same target.
