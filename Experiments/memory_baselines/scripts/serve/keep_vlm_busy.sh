#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON:-/root/anaconda3/envs/vllm_env/bin/python}"
if [[ ! -x "$PYTHON" ]]; then
  echo "Python not found: $PYTHON (set PYTHON=/path/to/python)" >&2
  exit 2
fi
exec "$PYTHON" "$SCRIPT_DIR/keep_vlm_busy.py" "$@"

# ```
# bash Experiments/memory_baselines/scripts/serve/keep_vlm_busy.sh --list
# bash Experiments/memory_baselines/scripts/serve/keep_vlm_busy.sh --port 8170 --target-gpu-util 60
# ```