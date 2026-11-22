#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH="${MODEL_PATH:-/data/pretrained_models/Qwen/Qwen3.8-27B}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-qwen3.8-27b}"
GPU_INDEX="${GPU_INDEX:-4,5}"
PORT="${PORT:-8077}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.6}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-262144}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-1}"

if [[ ! -f "${MODEL_PATH}/config.json" ]]; then
  echo "Model config not found: ${MODEL_PATH}/config.json" >&2
  exit 1
fi

export CUDA_VISIBLE_DEVICES="${GPU_INDEX}"
export PATH="/root/anaconda3/envs/vllm_env/bin:${PATH}"
exec /root/anaconda3/envs/vllm_env/bin/vllm serve "${MODEL_PATH}" \
  --served-model-name "${SERVED_MODEL_NAME}" \
  --host 127.0.0.1 \
  --port "${PORT}" \
  --tensor-parallel-size 2 \
  --max-num-seqs "${MAX_NUM_SEQS}" \
  --max-model-len "${MAX_MODEL_LEN}" \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
  "$@"
