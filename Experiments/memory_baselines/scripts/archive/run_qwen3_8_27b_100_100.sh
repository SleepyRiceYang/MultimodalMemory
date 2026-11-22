#!/usr/bin/env bash
# Usage: bash scripts/run_qwen3_8_27b_100_100.sh prompt-pilot|no-memory|formal [runner arguments]
set -euo pipefail
TASK_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PHASE="${1:-prompt-pilot}"
if [[ $# -gt 0 ]]; then shift; fi
EXPERIMENT_PYTHON="${EXPERIMENT_PYTHON:-/root/anaconda3/envs/vision2code/bin/python}"
export PYTHONPATH="${TASK_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
mkdir -p "${TASK_ROOT}/artifacts/logs"
LOG_PATH="${LOG_PATH:-${TASK_ROOT}/artifacts/logs/qwen3_8_27b_${PHASE}_$(date -u +%Y%m%dT%H%M%S)_$$.log}"
case "${PHASE}" in
  prompt-retest)
    "${EXPERIMENT_PYTHON}" -u -m vlm_memory.pipeline.prompt_retest \
      "$@" 2>&1 | tee -a "${LOG_PATH}"
    ;;
  prompt-pilot|no-memory)
    "${EXPERIMENT_PYTHON}" -u -m vlm_memory.pipeline.run_100_100_nomemory \
      --mode "${PHASE}" "$@" 2>&1 | tee -a "${LOG_PATH}"
    ;;
  formal)
    "${EXPERIMENT_PYTHON}" -u -m vlm_memory.pipeline.formal_100_100 \
      "$@" 2>&1 | tee -a "${LOG_PATH}"
    ;;
  continue)
    "${EXPERIMENT_PYTHON}" -u -m vlm_memory.pipeline.continue_after_prompt \
      "$@" 2>&1 | tee -a "${LOG_PATH}"
    ;;
  *)
    echo "Usage: $0 prompt-pilot|no-memory|formal [runner arguments]" >&2
    exit 2
    ;;
esac
