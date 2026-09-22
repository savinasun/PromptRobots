#!/usr/bin/env bash
# Keep robot dependencies in gello; install framework metadata in a local overlay.
set -euo pipefail
task_root="$(cd "$(dirname "$0")/.." && pwd)"
inspect_root="${INSPECT_ROBOTS_ROOT:-$(dirname "$task_root")/inspect-robots}"
base_python="${YAM_BASE_PYTHON:-$HOME/miniconda3/envs/gello/bin/python}"
"$base_python" -m venv --system-site-packages "$task_root/.venv-inspect"
"$task_root/.venv-inspect/bin/python" -m pip --isolated install --disable-pip-version-check \
    --no-deps -e "$inspect_root" -e "$inspect_root/plugins/inspect-robots-agent"
"$task_root/.venv-inspect/bin/python" -c \
    'import inspect_robots, inspect_robots_agent, httpx, websockets, openai, mujoco, cv2, yaml; print("Inspect Robots and YAM imports ready")'
