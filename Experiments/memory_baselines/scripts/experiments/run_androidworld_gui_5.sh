#!/usr/bin/env bash
set -euo pipefail
BASE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../../.." && pwd)"
CONDITION="${1:?Usage: bash run_androidworld_gui_5.sh no_memory|magnet_procedural_text|magnet_procedural_visual [extra options]}"
shift
case "${CONDITION}" in
  no_memory|magnet_procedural_text|magnet_procedural_visual) ;;
  *) echo 'Invalid condition' >&2; exit 2 ;;
esac
export PYTHONPATH="${BASE}/Methods/MAGNET/src:${BASE}/Benchmark/OSWorld${PYTHONPATH:+:${PYTHONPATH}}"
export NO_PROXY="${NO_PROXY:-},127.0.0.1,localhost"
export no_proxy="${NO_PROXY}"
RUN_ID="${RUN_ID:-comparison_5_v1}"
LOG_DIR="${BASE}/Experiments/memory_baselines/artifacts/logs"
mkdir -p "${LOG_DIR}"
cd "${BASE}/Benchmark/OSWorld"
/root/anaconda3/envs/vision2code/bin/python -u -m gui_baseline.comparison \
  --workspace "${BASE}" --benchmark androidworld --condition "${CONDITION}" \
  --run "${RUN_ID}" --model-url "${MODEL_URL:-http://127.0.0.1:8104/v1}" \
  --embedding-url "${EMBEDDING_URL:-http://127.0.0.1:8087}" \
  --seed "${SEED:-20261028}" --max-steps "${MAX_STEPS:-30}" \
  --max-tokens "${MAX_TOKENS:-2048}" "$@" \
  2>&1 | tee -a "${LOG_DIR}/${RUN_ID}_androidworld_${CONDITION}.log"
