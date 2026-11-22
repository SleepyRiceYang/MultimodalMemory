#!/usr/bin/env bash
# Analyze saved construction outputs and run the fixed eight-case prompt check.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TASK_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
EXPERIMENT_PYTHON="${EXPERIMENT_PYTHON:-/root/anaconda3/envs/vision2code/bin/python}"
SOURCE_RUN="${SOURCE_RUN:-qwen3_vl_8b_mathvision_8192_32768_v1_nomemory}"
RUN_ID="${RUN_ID:-qwen3_vl_8b_mathvision_truncation_prompt_20261024}"
PROFILE="${PROFILE:-mathvision_concise_v1}"
PHASE="${1:-analyze}"
case "${PHASE}" in analyze|test|all) ;; *) echo "Usage: $0 [analyze|test|all]" >&2; exit 2 ;; esac
export PYTHONPATH="${TASK_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
mkdir -p "${TASK_ROOT}/artifacts/logs"
LOG_PATH="${LOG_PATH:-${TASK_ROOT}/artifacts/logs/${RUN_ID}_${PHASE}_$(date -u +%Y%m%dT%H%M%S)_$$.log}"
run_diagnostic() {
  if [[ "${PHASE}" == test || "${PHASE}" == all ]]; then
    "${EXPERIMENT_PYTHON}" -u -m vlm_memory.pipeline.mathvision_prompt_diagnostic \
      --source-run "${SOURCE_RUN}" --run "${RUN_ID}" --profile "${PROFILE}" || return "$?"
  fi
  "${EXPERIMENT_PYTHON}" -m vlm_memory.analysis.mathvision_truncation \
    --source-run "${SOURCE_RUN}" --run "${RUN_ID}"
}
run_diagnostic 2>&1 | tee -a "${LOG_PATH}"
