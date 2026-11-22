#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
export PYTHONPATH="$ROOT/Benchmark/OSWorld:$ROOT/Methods/MMSkills_for_Visual_Agents_baseline/osworld_integration${PYTHONPATH:+:$PYTHONPATH}"
export NO_PROXY="${NO_PROXY:-},127.0.0.1,localhost"
export no_proxy="$NO_PROXY"
cd "$ROOT/Benchmark/OSWorld"
# Do not install/copy agent modules while the existing formal evaluation is running.
exec "${PYTHON:-/root/anaconda3/envs/osworld/bin/python}" -u \
  "$ROOT/Experiments/memory_baselines/mmskills_osworld/run_dense_smoke.py" "$@"
