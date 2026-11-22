#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
BASE="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
export PYTHONPATH="${BASE}${PYTHONPATH:+:${PYTHONPATH}}"
RUN_ID="${RUN_ID:-qwen3_vl_8b_androidworld_nomemory_5_20261026_png_v1}"
mkdir -p "${BASE}/artifacts/logs" "${BASE}/artifacts/locks"
/root/anaconda3/envs/vision2code/bin/python -u -m vlm_memory.pipeline.androidworld_smoke \
  --run "${RUN_ID}" --base-url "${BASE_URL:-http://127.0.0.1:8104/v1}" \
  --max-steps "${MAX_STEPS:-30}" --max-tokens "${MAX_TOKENS:-2048}" "$@" \
  2>&1 | tee -a "${BASE}/artifacts/logs/${RUN_ID}.log"
