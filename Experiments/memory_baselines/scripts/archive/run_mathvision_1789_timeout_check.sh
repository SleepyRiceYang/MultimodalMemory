#!/usr/bin/env bash
# Single-item diagnosis for MathVision task 1789; requires the local vLLM service.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TASK_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
EXPERIMENT_PYTHON="${EXPERIMENT_PYTHON:-/root/anaconda3/envs/vision2code/bin/python}"
CONFIG="${FORMAL_CONFIG:-${TASK_ROOT}/config/qwen3_8_27b_formal_100_100.json}"
RUN_ID="${RUN_ID:-qwen38_mathvision_1789_timeout1800}"
export PYTHONPATH="${TASK_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
mkdir -p "${TASK_ROOT}/artifacts/logs"
LOG_PATH="${LOG_PATH:-${TASK_ROOT}/artifacts/logs/${RUN_ID}_$(date -u +%Y%m%dT%H%M%S).log}"
exec "${EXPERIMENT_PYTHON}" -u -m vlm_memory.pipeline.single_mathvision_task \
  --task-id 1789 --run "${RUN_ID}" --config "${CONFIG}" 2>&1 | tee -a "${LOG_PATH}"
