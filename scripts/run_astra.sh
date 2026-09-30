#!/bin/bash
# Run the Astra <-> robot pipeline inside the `gello` conda env (has zmq, pyrealsense2, mujoco, openai).
# The rig comes from the arguments, not from this script: --config / --embodiment pick YAM or UR5e.
# Usage: scripts/run_astra.sh run --config configs/skild_yam_8.yaml --goal "pick up blue and place on top of green block"
#        scripts/run_astra.sh run --config configs/skild_ur5e.yaml --goal "pick up the blue block"
#        scripts/run_astra.sh run --embodiment ur5e_arms --sim --goal "pick up the blue block"
#        scripts/run_astra.sh check --config configs/skild_ur5e.yaml
set -euo pipefail
source "$HOME/miniconda3/etc/profile.d/conda.sh"
# Conda's activate.d/deactivate.d hooks are not `set -u` clean: this env's qt-main hook does
# `export CONDA_BACKUP_QT_XCB_GL_INTEGRATION=$QT_XCB_GL_INTEGRATION` on an unset variable, which
# under `set -u` aborts the script before python ever starts. Relax nounset across activation only
# (conda's own docs recommend this), then restore it for the rest of the script.
set +u
conda deactivate 2>/dev/null || true
conda activate gello
set -u
export PYTHONUNBUFFERED=1
export GELLO_SOFTWARE_PATH="${GELLO_SOFTWARE_PATH:-$HOME/bimanual_manipulation/skild-gello/gello_software}"
cd "$(dirname "$0")/.."
exec python -m utils "$@"
