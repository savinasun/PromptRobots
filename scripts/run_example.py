#!/usr/bin/env python3

import subprocess
import sys
from pathlib import Path


def main():
    repo_root = Path(__file__).resolve().parent.parent
    command = [
        str(repo_root / "scripts" / "run_astra.sh"),
        "run",
        # Bimanual UR5e + Robotiq 2F-85 station (the YAM station is configs/skild_yam_8.yaml).
        # Equivalent shorthand without a YAML: "--embodiment", "ur5e_arms".
        "--config", str(repo_root / "configs" / "skild_ur5e.yaml"),
        # Qwen plans in text; Astra converts each plan into motion targets.
        "--planning",
        # Skip the confirmation prompt before moving the robot.
        "--yes",
        # Set goal for planner.
        "--goal",
        "Please keep your end effector closed and act as a pointer, please push to move the airpod case, the orange fork and the tennis ball in sequence.",

        # ************************************* TEST IN REAL ****************************************
        # Before a hardware run (see README "UR5e specifics worth knowing before a hardware run"):
        #   1. Start the gello node server for this rig:
        #        cd $GELLO_SOFTWARE_PATH && python experiments/launch_nodes.py --robot=bimanual_ur
        #      (URRobot 172.17.0.2 = left / 172.17.0.3 = right; the arm gello calls "left" must be the one
        #      mounted at +y of the stand, i.e. left_base in skild_ur5e.urdf, or every frame is mirrored.)
        #   2. Point cameras.station_config_path in configs/skild_ur5e.yaml at the UR5e station's serials.
        #   3. Park both arms near robot.home_joints_* — homing refuses joint sweeps wider than 2.5 rad.
        #   4. Re-measure the table: the tilt-guard floor is motion.tool_floor_z_m = -0.113 in each arm's frame.

        # System 2 reasoning effort. This is the single biggest cost in the loop: measured on this exact
        # planner request (qwen/qwen3.8-max-0902, the only OpenRouter provider is Alibaba at ~45 tok/s),
        #   high   130 s  (5468 output tokens, 4900 of them reasoning)
        #   medium  75 s  (3522 / 2973)
        #   low     37 s  (1564 / 1034)      <- plan text stayed ~2 k characters at every level
        # Omitting the flag is NOT cheap: Qwen then thinks as if effort were high (112 s), so set it.
        "--planner-effort", "low",
        # Faster alternative: run System 2 on Astra through OpenAI's own API (14 s at high effort, 8 s at
        # medium, same request) instead of Qwen via OpenRouter.
        # "--planner-backend", "openai",
        # "--planner-model", "gpt-6-astra",

        # NOT using "--language-output": with planning on, System 2 writes the scene assessment and
        # progress notes, and System 1 restating them in its own words was ~60% of its output tokens on the
        # robot's critical path. System 1 is on the action-only contract; the plan text is in notes.md.
        # System 1 (Astra) here is gpt-6-astra on OpenAI, with the UR5e system prompt
        # (docs/SYSTEM_PROMPT_UR5E.md) and its own prompt cache key (astra.prompt_cache_key: astra-ur5e).
        "--effort", "high",
        # Retain images from the three most recent observations in model context.
        "--image-history", "3",
        # Use the faster preset for arm, wrist, gripper, and settling motion: 5 cm/s linear, 0.6 rad/s yaw,
        # gripper 1.5/s, 0.1 s settle. The UR5e is heavier and faster than the YAM and the YAML already paces
        # joint motion finer for it (motion.max_joint_step_rad = 0.03); drop this flag for the 1 cm/s default
        # on a first shakedown run.
        "--fast",
        # Stop after at most 100 total calls: up to 50 planner + motion cycles.
        "--max-calls", "100",
        # Stop the trial after at most 2,100 seconds (35 minutes).
        "--max-seconds", "2100",
        # Request automatic readable reasoning summaries from the Astra API.
        "--set", "astra.reasoning_summary=auto",
        # The YAM run disabled the arm-to-arm clearance guard. Left at the 2 cm default here: the UR5e bases
        # are only 0.50 m apart and pitched 45 deg outward, so the two arms can actually meet in the middle.
        # Uncomment to match the YAM run exactly.
        # "--set", "motion.arm_clearance_m=0",

        # ************************************* TEST IN SIMULATION ****************************************

        # Start the browser-based 3D visualization and operator interface.
        # "--viser",
        # Switch both hardware backends to simulation (no YAML needed: "--embodiment", "ur5e_arms", "--sim").
        # "--sim",
        # "--scene",
        # "blocks",
        # "--goal",
        # "Pick up the blue block.",
        # "--set",
        # "motion.arm_clearance_m=0",
    ]

    print("Running command:")
    print(" ".join(command))
    print()

    try:
        # Raise CalledProcessError if the command exits unsuccessfully.
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Command failed with exit code {e.returncode}", file=sys.stderr)
        sys.exit(e.returncode)
    except KeyboardInterrupt:
        print("\nInterrupted.")
        sys.exit(130)


if __name__ == "__main__":
    main()
