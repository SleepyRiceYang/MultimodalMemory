#!/usr/bin/env bash
# Standalone multimodal Qwen3.5-9B service.
set -euo pipefail
MODEL_PATH="${MODEL_PATH:-/data/pretrained_models/Qwen/Qwen3.5-9B}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-qwen3.5-9b}"


GPU_INDEX="${GPU_INDEX:-3}"
PORT="${PORT:-8166}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-65536}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.8}"
VLLM_BIN="${VLLM_BIN:-/root/anaconda3/envs/vllm_env/bin/vllm}"
[[ -f "${MODEL_PATH}/config.json" ]] || { echo "Missing model config: ${MODEL_PATH}" >&2; exit 1; }
[[ "${GPU_INDEX}" != *,* ]] || { echo 'Select one GPU' >&2; exit 2; }
export CUDA_VISIBLE_DEVICES="${GPU_INDEX}"
export PATH="$(dirname "${VLLM_BIN}"):${PATH}"
exec "${VLLM_BIN}" serve "${MODEL_PATH}" \
  --served-model-name "${SERVED_MODEL_NAME}" --host 127.0.0.1 --port "${PORT}" \
  --tensor-parallel-size 1 --max-model-len "${MAX_MODEL_LEN}" \
  --max-num-seqs "${MAX_NUM_SEQS:-4}" \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" "$@"
