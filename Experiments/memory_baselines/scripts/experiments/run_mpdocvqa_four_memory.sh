#!/usr/bin/env bash
# Build four memory baselines from a completed shared no-memory trajectory run.
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
BASE="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
SOURCE_RUN="${1:?Usage: bash ${BASH_SOURCE[0]} <completed-no-memory-source-run-id> [Python arguments]}"
shift
RUN_ID="${FORMAL_RUN:-${SOURCE_RUN}_four_memory_v2}"
export PYTHONPATH="${BASE}${PYTHONPATH:+:${PYTHONPATH}}"
mkdir -p "${BASE}/artifacts/logs"
LOG_PATH="${LOG_PATH:-${BASE}/artifacts/logs/${RUN_ID}.log}"
mkdir -p "$(dirname -- "${LOG_PATH}")"
/root/anaconda3/envs/vision2code/bin/python -u -m vlm_memory.pipeline.formal_100_100 \
  --config "${BASE}/config/qwen3_vl_8b_mpdocvqa_four_memory_baselines_v2.json" \
  --source-run "${SOURCE_RUN}" --run "${RUN_ID}" \
  --benchmarks "mpdocvqa" "$@" \
  2>&1 | tee -a "${LOG_PATH}"
