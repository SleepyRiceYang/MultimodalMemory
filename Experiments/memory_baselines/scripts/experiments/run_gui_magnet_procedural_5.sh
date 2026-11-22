#!/usr/bin/env bash
set -euo pipefail
BASE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../../.." && pwd)"
BENCHMARK="${1:?Usage: bash run_gui_magnet_procedural_5.sh androidworld|osworld [options]}"
shift
case "${BENCHMARK}" in
  androidworld) PYTHON=/root/anaconda3/envs/vision2code/bin/python ;;
  osworld) PYTHON=/root/anaconda3/envs/osworld/bin/python ;;
  *) echo 'Benchmark must be androidworld or osworld' >&2; exit 2 ;;
esac
export PYTHONPATH="${BASE}/Methods/MAGNET/src:${BASE}/Experiments/memory_baselines:${BASE}/Benchmark/OSWorld${PYTHONPATH:+:${PYTHONPATH}}"
export NO_PROXY="${NO_PROXY:-},127.0.0.1,localhost"
export no_proxy="${NO_PROXY}"
export OSWORLD_MAX_RETRY_TIMES=2
export OSWORLD_OPENAI_TIMEOUT=180
mkdir -p "${BASE}/Experiments/memory_baselines/artifacts/logs"
cd "${BASE}/Benchmark/OSWorld"
"${PYTHON}" -u -m gui_baseline.runner --workspace "${BASE}" --benchmark "${BENCHMARK}" "$@" 2>&1 | tee -a "${BASE}/Experiments/memory_baselines/artifacts/logs/gui_magnet_procedural_${BENCHMARK}.log"
