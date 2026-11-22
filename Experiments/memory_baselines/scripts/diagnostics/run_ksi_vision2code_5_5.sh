#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
BASE="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
export PYTHONPATH="${BASE}${PYTHONPATH:+:${PYTHONPATH}}"
LOG_DIR="${BASE}/artifacts/vision2code/qwen3_vl_8b_ksi_dense_5_5_20261024"
mkdir -p "${LOG_DIR}"
/root/anaconda3/envs/vision2code/bin/python -u -m vlm_memory.pipeline.retrieval_pilot --config "${BASE}/config/qwen3_vl_8b_ksi_vision2code_5_5.json" "$@" 2>&1 | tee -a "${LOG_DIR}/runner.log"
