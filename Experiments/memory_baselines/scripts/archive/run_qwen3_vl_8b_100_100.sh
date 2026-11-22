#!/usr/bin/env bash
# Usage: run_qwen3_vl_8b_100_100.sh mathvision|vision2code|mpdocvqa [all|no-memory|formal]
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TASK_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
BENCHMARK="${1:?Usage: $0 mathvision|vision2code|mpdocvqa [all|no-memory|formal]}"
PHASE="${2:-all}"
case "${BENCHMARK}" in mathvision|vision2code|mpdocvqa) ;; *) echo "Unknown benchmark: ${BENCHMARK}" >&2; exit 2 ;; esac
case "${PHASE}" in all|no-memory|formal) ;; *) echo "Unknown phase: ${PHASE}" >&2; exit 2 ;; esac
CONFIG="${FORMAL_CONFIG:-${TASK_ROOT}/config/qwen3_vl_8b_${BENCHMARK}_100_100.json}"
SOURCE_RUN="${SOURCE_RUN:-qwen3_vl_8b_${BENCHMARK}_8192_32768_v1_nomemory}"
FORMAL_RUN="${FORMAL_RUN:-qwen3_vl_8b_${BENCHMARK}_8192_32768_v1_baselines}"
EXPERIMENT_PYTHON="${EXPERIMENT_PYTHON:-/root/anaconda3/envs/vision2code/bin/python}"
export PYTHONPATH="${TASK_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
mkdir -p "${TASK_ROOT}/artifacts/logs"
LOG_PATH="${LOG_PATH:-${TASK_ROOT}/artifacts/logs/qwen3_vl_8b_${BENCHMARK}_${PHASE}_$(date -u +%Y%m%dT%H%M%S)_$$.log}"
RETRY_ARGS=()
if [[ "${RETRY_FAILED:-0}" == 1 ]]; then RETRY_ARGS+=(--retry-failed); fi
run_pipeline() {
  if [[ "${PHASE}" == all || "${PHASE}" == no-memory ]]; then
    "${EXPERIMENT_PYTHON}" -u -m vlm_memory.pipeline.run_100_100_nomemory \
      --config "${CONFIG}" --mode no-memory --run "${SOURCE_RUN}" \
      --benchmarks "${BENCHMARK}" "${RETRY_ARGS[@]}" || return "$?"
  fi
  if [[ "${PHASE}" == all || "${PHASE}" == formal ]]; then
    "${EXPERIMENT_PYTHON}" -u -m vlm_memory.pipeline.formal_100_100 \
      --config "${CONFIG}" --source-run "${SOURCE_RUN}" --run "${FORMAL_RUN}" \
      --stage all --benchmarks "${BENCHMARK}" "${RETRY_ARGS[@]}" || return "$?"
  fi
}
run_pipeline 2>&1 | tee -a "${LOG_PATH}"
