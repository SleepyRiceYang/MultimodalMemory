#!/usr/bin/env bash
set -euo pipefail
BASE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../../.." && pwd)"
STAGE="${1:-all}"
if (( $# )); then shift; fi
export PYTHONPATH="${BASE}/Methods/MAGNET/src:${BASE}/Benchmark/OSWorld${PYTHONPATH:+:${PYTHONPATH}}"
export NO_PROXY="${NO_PROXY:-},127.0.0.1,localhost"
export no_proxy="${NO_PROXY}"
export OSWORLD_MAX_RETRY_TIMES=2
export OSWORLD_OPENAI_TIMEOUT=180
mkdir -p "${BASE}/Experiments/memory_baselines/artifacts/logs"
cd "${BASE}/Benchmark/OSWorld"
/root/anaconda3/envs/vision2code/bin/python -u -m gui_baseline.evolution \
  --benchmark androidworld --stage "${STAGE}" "$@" \
  2>&1 | tee -a "${BASE}/Experiments/memory_baselines/artifacts/logs/androidworld_evolution_${STAGE}.log"
