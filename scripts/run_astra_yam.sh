#!/bin/bash
# Run the Astra <-> YAM pipeline inside the `gello` conda env (has zmq, pyrealsense2, mujoco, openai).
# Usage: scripts/run_astra_yam.sh run --goal "pick up blue and place on top of green block"
#        scripts/run_astra_yam.sh check
set -euo pipefail
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda deactivate 2>/dev/null || true
conda activate gello
export PYTHONUNBUFFERED=1
export GELLO_SOFTWARE_PATH="${GELLO_SOFTWARE_PATH:-$HOME/bimanual_manipulation/skild-gello/gello_software}"
cd "$(dirname "$0")/.."
exec python -m astra_yam "$@"
