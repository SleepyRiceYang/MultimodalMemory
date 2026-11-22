#!/usr/bin/env bash
# One GPU per service. Usage: start_qwen3_vl_8b.sh mathvision|vision2code|mpdocvqa [vLLM args]
set -euo pipefail
ROLE="${1:?Usage: $0 mathvision|vision2code|mpdocvqa [vLLM args]}"
shift
case "${ROLE}" in
  mathvision) DEFAULT_GPU=4; DEFAULT_PORT=8078 ;;
  vision2code) DEFAULT_GPU=5; DEFAULT_PORT=8079 ;;
  mpdocvqa) DEFAULT_GPU=6; DEFAULT_PORT=8080 ;;
  *) echo "Unknown benchmark: ${ROLE}" >&2; exit 2 ;;
esac
MODEL_PATH="${MODEL_PATH:-/data/pretrained_models/Qwen3-VL-8B-Instruct}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-qwen3-vl-8b-instruct}"
GPU_INDEX="${GPU_INDEX:-${DEFAULT_GPU}}"
PORT="${PORT:-${DEFAULT_PORT}}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-65536}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.8}"
VLLM_BIN="${VLLM_BIN:-/root/anaconda3/envs/vllm_env/bin/vllm}"
if [[ ! -f "${MODEL_PATH}/config.json" ]]; then
  echo "Model config not found: ${MODEL_PATH}/config.json" >&2
  exit 1
fi
if [[ "${GPU_INDEX}" == *,* ]]; then
  echo "This service requires one GPU_INDEX, received ${GPU_INDEX}" >&2
  exit 2
fi
export CUDA_VISIBLE_DEVICES="${GPU_INDEX}"
export PATH="$(dirname "${VLLM_BIN}"):${PATH}"
exec "${VLLM_BIN}" serve "${MODEL_PATH}" \
  --served-model-name "${SERVED_MODEL_NAME}" \
  --host 127.0.0.1 --port "${PORT}" \
  --tensor-parallel-size 1 --max-num-seqs 1 \
  --max-model-len "${MAX_MODEL_LEN}" \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" "$@"
