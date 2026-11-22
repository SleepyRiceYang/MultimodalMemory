#!/usr/bin/env bash
# Starts/resumes three detached model services and their shared router.
set -euo pipefail
BASE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../../.." && pwd)"
export PYTHONPATH="${BASE}/Methods/MAGNET/src${PYTHONPATH:+:${PYTHONPATH}}"
exec /root/anaconda3/envs/vllm_env/bin/python -u -m gui_baseline.pool_manager "$@"
