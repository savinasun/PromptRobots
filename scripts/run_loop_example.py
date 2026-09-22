#!/usr/bin/env python3

import subprocess
import sys
from pathlib import Path


def main():
    repo_root = Path(__file__).resolve().parent.parent

    goal = r"""
Your goal is to autonomously learn a fast, reliable, and safe manipulation skill for picking up the white pepper can from arbitrary reachable positions on the tabletop.

Treat this as a long-horizon self-improvement task rather than a single pickup attempt.

FINAL OBJECTIVE

Starting from the robot's default initial configuration, reliably locate, approach, grasp, and lift the white pepper can from any safely reachable position and orientation on the table.

Over repeated autonomous practice, improve toward smooth, efficient, human-like pickup speed while maintaining very high reliability and avoiding collisions, unstable grasps, unnecessary motion, and unsafe behavior.

Do not merely complete one pickup. Repeatedly practice, evaluate your performance, create useful new practice situations, and improve your behavior from experience.

==================================================
1. SAFETY IS THE HIGHEST PRIORITY
==================================================

At the beginning, behave conservatively.

Move slowly, maintain generous clearance from the table and surrounding objects, and prefer simple, predictable trajectories.

Never trade safety for speed.

Do not intentionally create dangerous situations, use excessive force, make abrupt high-speed motions near the table, or execute a motion when you are substantially uncertain that it is safe.

When uncertainty is high, slow down.

When approaching the object, table, robot body, other arm, or other obstacles, use extra caution.

If necessary, stop, observe again, or choose a safer trajectory rather than blindly continuing an uncertain motion.

Speed should increase only as competence and confidence are earned through repeated successful experience.

==================================================
2. USE THE ROBOT'S FULL BIMANUAL CAPABILITIES
==================================================

The robot has two grippers.

Each gripper has its own wrist-mounted camera.

The robot also has a separate head camera.

You may use EITHER the left gripper OR the right gripper to pick up the white pepper can.

There is no requirement to use a particular hand.

Choose the grasping arm dynamically based on the current situation.

Consider:
- pepper-can position,
- pepper-can orientation,
- left-arm reachability,
- right-arm reachability,
- collision risk,
- trajectory simplicity,
- visibility,
- expected grasp quality,
- and experience from similar previous attempts.

For some configurations, the left arm may be better.

For others, the right arm may be better.

For some configurations, either arm may work.

Do not unnecessarily force one arm to solve every configuration.

Learn from experience which arm tends to work best for different regions and configurations.

==================================================
3. USE ACTIVE PERCEPTION
==================================================

The cameras are not merely passive sensors.

You may deliberately move the robot's arms and wrists to obtain better visual observations.

In particular, if one arm is being used to grasp the pepper can, the OTHER arm may be moved so that its wrist camera provides a better viewpoint.

Think of the non-grasping arm as a movable eye.

You may move the non-grasping wrist camera to:
- observe the pepper can from another angle,
- resolve uncertainty about object position or orientation,
- inspect an area occluded from the head camera,
- inspect an area occluded from the grasping wrist camera,
- observe the gap between the gripper and the object,
- verify gripper alignment before grasping,
- verify grasp quality after contact,
- detect whether the can has slipped,
- inspect clearance from the table,
- or obtain another view when the current visual information is ambiguous.

Use information from:
- the head camera,
- the left wrist camera,
- the right wrist camera,
- previous observations,
- and the consequences of previous actions.

However, do not move the second arm unnecessarily.

Moving a camera has a time and motion cost.

Acquire an additional viewpoint when the expected reduction in uncertainty is worth the extra motion.

As you become more experienced, learn the difference between:

"I genuinely need another observation."

and:

"I have already seen enough and can safely execute the pickup."

The long-term objective is NOT to maximize observations.

The objective is to obtain enough information to act safely and reliably while progressively becoming faster and more efficient.

When moving the non-grasping arm for perception:
- avoid collisions with the table,
- avoid collisions with the other arm,
- avoid blocking the grasping trajectory,
- avoid unnecessarily entering the object's workspace,
- and return it to a safe configuration when appropriate.

You may learn useful wrist-camera observation poses and reuse them when similar situations occur.

==================================================
4. LEARN PROGRESSIVELY
==================================================

Initially prioritize:

1. Correctly perceiving and locating the pepper can.
2. Choosing an appropriate arm.
3. Planning a safe approach trajectory.
4. Reliable gripper alignment.
5. Stable grasping.
6. Clean lifting without collision or slipping.
7. Effective use of additional camera views when necessary.
8. Returning both arms to safe configurations.

Only after these behaviors become consistently reliable should you gradually increase speed and reduce unnecessary conservatism.

Do NOT increase speed because a single attempt succeeded.

Increase speed incrementally only when repeated experience demonstrates that a particular behavior or trajectory segment is predictable and safe.

Different parts of the motion should be allowed to use different levels of speed.

For example:

PERCEPTION / PLANNING:
Take enough time to understand the scene and select a safe strategy.

FREE-SPACE REACH:
This can eventually become fast and direct once confidence is high.

FINAL APPROACH:
Remain controlled and precise.

GRASP:
Prioritize alignment and grasp stability rather than raw speed.

INITIAL LIFT:
Remain controlled until the object is clearly secured.

RETREAT:
Once the object is stable and the surrounding space is clear, motion can become faster.

==================================================
5. LEARN FROM EVERY ATTEMPT
==================================================

Treat previous attempts as experience.

Do not behave as though every trial is completely unrelated to previous trials.

After every attempt, evaluate what happened.

Consider:

- Was the pepper can localized correctly?
- Was the chosen arm appropriate?
- Would the other arm have been safer, simpler, or faster?
- Was the head-camera view sufficient?
- Did a wrist camera provide useful additional information?
- Would moving the unused wrist camera have reduced uncertainty?
- Did you acquire unnecessary additional views?
- Was the chosen grasp appropriate?
- Was the approach unnecessarily long?
- Did either arm move unnecessarily close to the table?
- Was there excessive wrist motion?
- Was there excessive arm motion?
- Did the gripper require corrections before grasping?
- Did the object move before the grasp became secure?
- Did the object slip?
- Was the lift stable?
- Were there unnecessary pauses?
- Were there unnecessary replans?
- Which parts of the trajectory were clearly safe enough to perform faster next time?
- Which parts remain uncertain and should stay slow?
- Could a simpler trajectory have achieved the same result?
- What is the most important lesson to apply on the next attempt?

Use these observations to change future behavior.

==================================================
6. CREATE YOUR OWN CURRICULUM
==================================================

You are responsible for progressively teaching yourself this skill.

Begin with easy object configurations and conservative behavior.

As competence improves, progressively practice more difficult configurations.

After successfully picking up the pepper can, when it is safe and practical, place it at a new reachable location on the table to create the next training example.

You may use either arm for repositioning when appropriate.

Choose new positions intentionally rather than randomly repeating easy cases.

Gradually explore:

- positions near the center,
- positions farther away,
- positions on the left,
- positions on the right,
- positions that favor the left arm,
- positions that favor the right arm,
- positions where either arm is reasonable,
- different object orientations,
- different approach directions,
- locations with poorer head-camera visibility,
- configurations where wrist-camera active perception is useful,
- locations that previously caused failure,
- locations that previously required excessive correction,
- and locations where previous trajectories were inefficient.

Do not immediately create the hardest possible configuration.

Increase difficulty progressively.

Prefer training examples that are informative:

situations that are not yet completely mastered, but can still be attempted safely.

Do NOT intentionally create unsafe placements.

Do NOT place the object where retrieving it would require dangerous motion, collisions, excessive reach, or operation outside the robot's safe workspace.

==================================================
7. REGULARLY RESET TO THE TRUE STARTING CONDITION
==================================================

Repeatedly return to the robot's default initial configuration.

The goal is to learn:

DEFAULT INITIAL CONFIGURATION
    ->
OBSERVE
    ->
CHOOSE ARM / VIEWPOINT
    ->
REACH
    ->
GRASP
    ->
LIFT

Do not allow the skill to depend on the ending pose of the previous attempt.

The final behavior must work from the robot's default initial configuration.

==================================================
8. GENERALIZE, DO NOT MEMORIZE
==================================================

Do not merely memorize a small number of pickup positions.

Learn a general strategy that adapts to:

- pepper-can position,
- pepper-can orientation,
- visibility,
- occlusion,
- arm reachability,
- scene geometry,
- previous experience,
- and uncertainty.

The final skill should work across a broad distribution of safely reachable tabletop positions.

==================================================
9. MEASURE EVERY ATTEMPT
==================================================

Treat every pickup attempt as one data point.

For each attempt, keep track of at least:

- attempt number,
- approximate pepper-can position,
- approximate pepper-can orientation,
- which arm was used,
- whether active perception was used,
- which additional camera/viewpoint was used,
- success or failure,
- whether any unintended collision occurred,
- pickup time,
- number of major corrections or replans,
- number of additional observation motions,
- grasp quality,
- approximate confidence before execution,
- and the main lesson learned.

Define PICKUP TIME as:

elapsed time from beginning the pickup behavior from the default initial configuration until the pepper can has been securely grasped and lifted clearly off the table.

==================================================
10. LEARNING OBJECTIVE
==================================================

The desired learning curve is:

ATTEMPT NUMBER increases
        ->
PICKUP TIME decreases

while simultaneously maintaining:

SUCCESS RATE >= 95%

and:

UNINTENDED COLLISION RATE approximately 0%.

Do NOT optimize pickup time before reliability and safety have been established.

The optimization priorities are STRICTLY:

1. Safety and near-zero unintended collisions.
2. Reliability.
3. Success rate >= 95%.
4. Generalization across diverse object positions.
5. Intelligent arm selection.
6. Effective visual perception and active perception.
7. Stable grasps.
8. Smooth and simple motion.
9. Short pickup time.

A faster pickup is NOT an improvement if it:
- increases collision risk,
- causes unstable grasps,
- reduces success rate,
- produces unsafe near-misses,
- or ignores important uncertainty.

==================================================
11. USE A ROLLING PERFORMANCE WINDOW
==================================================

Once enough attempts have been collected, evaluate competence over approximately the most recent 20 diverse attempts.

A useful milestone is:

- at least 19 successful pickups out of the most recent 20 attempts,
- zero unintended collisions,
- no recurring unsafe near-misses,
- stable grasps across diverse locations,
- appropriate use of both arms,
- and sensible use of active perception.

Only after achieving stable performance should you significantly increase speed or increase curriculum difficulty.

If performance becomes worse after increasing speed:

1. reduce speed,
2. determine which part of the behavior became unreliable,
3. practice that component,
4. then cautiously increase speed again.

==================================================
12. BECOME FASTER BY BECOMING BETTER, NOT RISKIER
==================================================

As competence increases, progressively remove unnecessary:

- pauses,
- observations,
- camera movements,
- corrections,
- waypoints,
- replanning,
- conservative detours,
- excessive vertical clearance,
- and redundant motions.

For familiar configurations that strongly resemble repeatedly successful previous attempts, act more decisively.

For unfamiliar configurations, unusual orientations, occlusion, ambiguous perception, or trajectories near obstacles, become conservative again.

Confidence must be calibrated to experience.

Do not confuse aggressiveness with competence.

The desired mature behavior should resemble a confident human reaching for a familiar object:

observe enough to understand the situation,
choose an appropriate hand,
make a relatively direct free-space reach,
decelerate near the object,
perform a precise grasp,
and execute a stable lift.

==================================================
13. HANDLE FAILURE INTELLIGENTLY
==================================================

If an attempt fails, do not simply become more aggressive.

Diagnose the likely failure cause.

For example:

PERCEPTION FAILURE:
Get a better view or use another camera.

ARM-SELECTION FAILURE:
Try the other arm or learn that this region favors one arm.

APPROACH FAILURE:
Change trajectory or approach direction.

GRASP FAILURE:
Improve alignment, grasp location, or final approach.

OCCLUSION:
Move the unused wrist camera to obtain another view.

SPEED-RELATED FAILURE:
Slow down the relevant segment.

REPEATED FAILURE:
Practice an easier version before attempting the difficult case again.

Learn specific lessons from specific failures.

==================================================
14. PERIODICALLY SUMMARIZE THE LEARNING CURVE
==================================================

After several attempts, compare recent performance with earlier performance.

Ask:

- Is median pickup time decreasing?
- Is success rate >= 95%?
- Has collision rate remained approximately zero?
- Are difficult positions improving?
- Are trajectories becoming shorter?
- Are trajectories becoming smoother?
- Are fewer corrections required?
- Are fewer unnecessary observations required?
- Is arm selection improving?
- Are you learning when the second wrist camera is useful?
- Are you learning when an additional view is unnecessary?
- Are you becoming faster because of genuine competence rather than increased risk?

When possible, summarize progress in a compact form such as:

Attempts 1-10
Success rate: 80%
Collision rate: 0%
Median pickup time: 18.2 s

Attempts 11-20
Success rate: 95%
Collision rate: 0%
Median pickup time: 12.5 s

Attempts 21-30
Success rate: 100%
Collision rate: 0%
Median pickup time: 8.7 s

The desired trend is:

pickup time DOWN

while:

success rate stays >= 95%

and:

unintended collisions stay approximately ZERO.

==================================================
15. COMPLETION CRITERIA
==================================================

The task is complete only when you can repeatedly start from the robot's default initial configuration and safely pick up the white pepper can from a broad range of arbitrary, safely reachable tabletop positions with:

- >= 95% success rate over recent diverse trials,
- approximately zero unintended collisions,
- no recurring unsafe near-misses,
- stable grasps,
- appropriate autonomous selection of left or right gripper,
- effective use of the head camera and both wrist cameras,
- active perception when useful,
- minimal unnecessary observation motion,
- smooth and efficient trajectories,
- and relatively fast, human-like pickup times.

Until then, continue autonomously creating useful practice situations, learning from previous attempts, and improving the strategy used for the next attempt.

REMEMBER:

The second arm is not merely idle when it is not grasping.

It can also be a movable sensor.

Use your body to improve what you can see.

But move only when the expected information is worth the motion.

Confidence should be earned through repeated successful experience.

First become SAFE.
Then become RELIABLE.
Then become PERCEPTIVE.
Then become GENERAL.
Then become SMOOTH.
Finally become FAST.
""".strip()

    command = [
        str(repo_root / "scripts" / "run_astra_yam.sh"),
        "run",
        "--config",
        str(repo_root / "configs" / "skild_yam_8_loop.yaml"),

        # Start the browser-based 3D visualization and operator interface.
        "--viser",

        # Skip the confirmation prompt before moving the robot.
        "--yes",

        # ============================================================
        # TEST IN SIMULATION
        # ============================================================
        # Switch both hardware backends to simulation.
        # "--sim",
        # "--scene",
        # "airpods",
        # "--goal",
        # "Pick up the AirPods charging case and open the lid.",
        # "--set",
        # "motion.arm_clearance_m=0",

        # ============================================================
        # TEST ON REAL ROBOT
        # ============================================================

        "--goal",
        goal,

        # Allow action notes, justifications, progress summaries,
        # and other language output.
        "--language-output",

        # Keep current reasoning setting.
        "--effort",
        "low",

        # Retain recent visual observations in model context.
        "--image-history",
        "2",

        # NOTE:
        # Intentionally NOT using "--fast" at the beginning.
        # The goal asks the agent to become faster only after
        # demonstrating safe and reliable behavior.
        #
        "--fast",

        # Allow enough calls for repeated autonomous practice.
        "--max-calls",
        "300",

        # Maximum session duration: 3600 seconds = 60 minutes.
        "--max-seconds",
        "3600",

        # Request automatic readable reasoning summaries from Astra.
        "--set",
        "astra.reasoning_summary=auto",
    ]

    print("Running command:")
    print(" ".join(command))
    print()

    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as e:
        print(
            f"Command failed with exit code {e.returncode}",
            file=sys.stderr,
        )
        sys.exit(e.returncode)
    except KeyboardInterrupt:
        print("\nInterrupted.")
        sys.exit(130)


if __name__ == "__main__":
    main()