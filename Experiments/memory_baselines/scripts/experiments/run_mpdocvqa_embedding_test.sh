#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
BASE="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
export PYTHONPATH="${BASE}${PYTHONPATH:+:${PYTHONPATH}}"
SOURCE_RUN="${SOURCE_MEMORY_RUN:-qwen3_vl_8b_mpdocvqa_8192_32768_v1_nomemory_four_memory_v2}"
RUN_ID="${FORMAL_RUN:-${SOURCE_RUN}_embedding_v1}"
mkdir -p "${BASE}/artifacts/logs"
/root/anaconda3/envs/vision2code/bin/python -u -m vlm_memory.pipeline.embedding_test \
  --config "${BASE}/config/qwen3_vl_8b_mpdocvqa_four_memory_baselines_v2_embedding.json" \
  --benchmark mpdocvqa --source-run "${SOURCE_RUN}" --run "${RUN_ID}" "$@" \
  2>&1 | tee -a "${BASE}/artifacts/logs/${RUN_ID}.log"
