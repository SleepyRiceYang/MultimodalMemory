#!/usr/bin/env bash
# AndroidWorld model service; default GPU 4, port 8104.
set -euo pipefail
MODEL_PATH="${MODEL_PATH:-/data/pretrained_models/Qwen3-VL-8B-Instruct}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-qwen3-vl-8b-instruct}"
GPU_INDEX="${GPU_INDEX:-4}"
PORT="${PORT:-8104}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-65536}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.8}"
VLLM_BIN="${VLLM_BIN:-/root/anaconda3/envs/vllm_env/bin/vllm}"
if [[ ! -f "${MODEL_PATH}/config.json" ]]; then
  echo "Model config not found: ${MODEL_PATH}/config.json" >&2
  exit 1
fi
if [[ "${GPU_INDEX}" == *,* ]]; then
  echo "GPU_INDEX must identify one GPU" >&2
  exit 2
fi
export CUDA_VISIBLE_DEVICES="${GPU_INDEX}"
export PATH="$(dirname "${VLLM_BIN}"):${PATH}"
JSON_ARGS=()
if [[ "${STRICT_JSON_WHITESPACE:-1}" == 1 ]]; then
  JSON_ARGS=(--structured-outputs-config '{"backend":"xgrammar","disable_any_whitespace":true}')
fi
exec "${VLLM_BIN}" serve "${MODEL_PATH}" \
  --served-model-name "${SERVED_MODEL_NAME}" \
  --host 127.0.0.1 --port "${PORT}" \
  --tensor-parallel-size 1 --max-num-seqs 1 \
  --max-model-len "${MAX_MODEL_LEN}" \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" "${JSON_ARGS[@]}" "$@"
