#!/usr/bin/env python3

import subprocess
import sys
from pathlib import Path


def main():
    repo_root = Path(__file__).resolve().parent.parent
    command = [
        str(repo_root / "scripts" / "run_astra_yam.sh"),
        "run",
        "--config", str(repo_root / "configs" / "skild_yam_8.yaml"),
        # Qwen plans in text; Astra converts each plan into motion targets.
        "--planning",
        # Skip the confirmation prompt before moving the robot.
        "--yes",
        # Set goal for planner.
        "--goal",
        "Please keep your end effector closed and act as a pointer, please push to move the CeraVe lotion, the airpod case, the toothbrush and the hand sanitizer in sequence.",

        # ************************************* TEST IN REAL ****************************************
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
        # Set System 1 (Astra)'s reasoning effort.
        "--effort", "high",
        # Retain images from the three most recent observations in model context.
        "--image-history", "3",
        # Use the faster preset for arm, wrist, gripper, and settling motion.
        "--fast",
        # Stop after at most 100 total calls: up to 50 planner + motion cycles.
        "--max-calls", "100",
        # Stop the trial after at most 2,100 seconds (35 minutes).
        "--max-seconds", "2100",
        # Request automatic readable reasoning summaries from the Astra API.
        "--set", "astra.reasoning_summary=auto",
        # Reset arm to home position
        "--set", "motion.arm_clearance_m=0",

        # ************************************* TEST IN SIMULATION ****************************************

        # Start the browser-based 3D visualization and operator interface.
        # "--viser",
        # Switch both hardware backends to simulation.
        # "--sim",
        # "--scene",
        # "airpods",
        # "--goal",
        # "Pick up the AirPods charging case and open the lid.",
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
