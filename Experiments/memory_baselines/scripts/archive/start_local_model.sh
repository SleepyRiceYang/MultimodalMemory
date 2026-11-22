#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"

export MODEL_PATH="${MODEL_PATH:-/data/pretrained_models/Qwen3-VL-8B-Instruct}"
export SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-qwen3-vl-8b-instruct}"
export GPU_INDEX="${GPU_INDEX:-7}"
export PORT="${PORT:-8000}"
export GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.7}"
export MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"

exec bash "${REPO_ROOT}/Benchmark/start_qwen3vl_vllm.sh" \
  --enforce-eager \
  --max-num-seqs 1 \
  --limit-mm-per-prompt '{"image":16}' \
  "$@"
