#!/usr/bin/env bash
# Use the existing gello environment and sibling inspect-robots checkout.
set -euo pipefail
task_root="$(cd "$(dirname "$0")/.." && pwd)"
inspect_root="${INSPECT_ROBOTS_ROOT:-$(dirname "$task_root")/inspect-robots}"
task_python="${YAM_EVAL_PYTHON:-$task_root/.venv-inspect/bin/python}"
if [[ ! -x "$task_python" ]]; then
    echo "Run scripts/setup_inspect_yam.sh, or set YAM_EVAL_PYTHON to your evaluation Python." >&2
    exit 1
fi
if [[ ! -d "$inspect_root/src/inspect_robots" ]]; then
    echo "Set INSPECT_ROBOTS_ROOT to your inspect-robots checkout." >&2
    exit 1
fi
export PYTHONPATH="$task_root:$inspect_root/src:$inspect_root/plugins/inspect-robots-agent/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
cd "$task_root"
case "${1:-}" in
    inspect|view|list)
        exec "$task_python" -m inspect_robots.cli "$@"
        ;;
    *)
        exec "$task_python" -m utils.inspect_eval "$@"
        ;;
esac
