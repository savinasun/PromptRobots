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
        # Start the browser-based 3D visualization and operator interface.
        "--viser",
        # Skip the confirmation prompt before moving the robot.
        "--yes",

        # TEST IN SIMULATION
        # Switch both hardware backends to simulation.
        "--sim",
        "--scene",
        "airpods",
        "--goal",
        "Pick up the AirPods charging case and open the lid.",
        "--set", 
        "motion.arm_clearance_m=0",
        # TEST IN REAL
        # "--goal",
        # "Pick up the AirPods charging case and put it in the teal bowl.",

        # Allow action notes, justifications, and other language output.
        "--language-output",
        # Use the model's lowest supported reasoning-effort level.
        "--effort", "low",
        # Retain images from the two most recent observations in model context.
        "--image-history", "2",
        # Use the faster preset for arm, wrist, gripper, and settling motion.
        "--fast",
        # Stop the trial after at most 100 model calls.
        "--max-calls", "100",
        # Stop the trial after at most 2,100 seconds (35 minutes).
        "--max-seconds", "2100",
        # Request automatic readable reasoning summaries from the Astra API.
        "--set", "astra.reasoning_summary=auto",
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
