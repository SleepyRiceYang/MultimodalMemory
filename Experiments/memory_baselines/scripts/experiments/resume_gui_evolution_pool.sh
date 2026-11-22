#!/usr/bin/env bash
# Both benchmarks resume their original committed tasks and banks using one shared pool.
set -euo pipefail
BASE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../../.." && pwd)"
export PYTHONPATH="${BASE}/Methods/MAGNET/src:${BASE}/Benchmark/OSWorld${PYTHONPATH:+:${PYTHONPATH}}"
export NO_PROXY="${NO_PROXY:-},127.0.0.1,localhost"
export no_proxy="$NO_PROXY"
mkdir -p "$BASE/Experiments/memory_baselines/artifacts/logs"
cd "$BASE/Benchmark/OSWorld"
/root/anaconda3/envs/vision2code/bin/python -u -m gui_baseline.pooled_evolution "$@" \
  2>&1 | tee -a "$BASE/Experiments/memory_baselines/artifacts/logs/gui_evolution_pool.log"
