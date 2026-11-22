#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
BASE="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
export PYTHONPATH="${BASE}${PYTHONPATH:+:${PYTHONPATH}}"
mkdir -p "${BASE}/artifacts/logs"
/root/anaconda3/envs/vision2code/bin/python -u -m vlm_memory.pipeline.representation_pilot \
  --config "${BASE}/config/qwen3_vl_8b_five_baselines_1_1.json" "$@" \
  2>&1 | tee -a "${BASE}/artifacts/logs/five_baselines_1_1.log"
