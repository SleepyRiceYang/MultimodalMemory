#!/usr/bin/env bash
# Native vLLM data parallelism: one complete model per GPU, one HTTP endpoint.
set -euo pipefail

if [[ "${1:-}" == '--help' ]]; then
  cat <<'HELP'
Usage:
  GPU_INDEX=3 PORT=8170 bash start_qwen3_8B.sh [vLLM arguments...]
  GPU_INDEX=3,7 PORT=8170 bash start_qwen3_8B.sh [vLLM arguments...]
  GPU_INDEX=0,1,2,3 PORT=8170 bash Experiments/memory_baselines/scripts/serve/start_qwen3_8B.sh
CUDA_VISIBLE_DEVICES is set to GPU_INDEX. Data parallel size is the GPU count;
tensor parallel size is 1. All replicas serve the same MODEL_PATH through
http://127.0.0.1:PORT/v1, with native vLLM internal load balancing.
PORTS is no longer supported: use one PORT, not a port per replica.
MAX_NUM_SEQS applies to each replica. Logs stay in the foreground; Ctrl+C is
handled by vLLM. To save logs, use shell redirection or tee.
Other settings: MODEL_PATH, SERVED_MODEL_NAME, MAX_MODEL_LEN,
GPU_MEMORY_UTILIZATION, MAX_NUM_SEQS, VLLM_BIN.
HELP
  exit 0
fi

MODEL_PATH="${MODEL_PATH:-/data/pretrained_models/Qwen3-VL-8B-Instruct}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-qwen3-vl-8b-instruct}"
GPU_INDEX="${GPU_INDEX:-3}"
PORT="${PORT:-8166}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-65536}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.8}"
# The host driver (535.54.03 / CUDA 12.2) cannot run the migrated vllm_env
# build (PyTorch CUDA 13.0). Use the locally validated CUDA-12-compatible env.
VLLM_BIN="${VLLM_BIN:-/root/vllm_compat/bin/vllm}"
[[ -f "${MODEL_PATH}/config.json" ]] || { echo "Missing model config: ${MODEL_PATH}" >&2; exit 1; }
[[ -x "${VLLM_BIN}" ]] || { echo "vLLM executable not found: ${VLLM_BIN}" >&2; exit 1; }
[[ -z "${PORTS:-}" ]] || { echo 'PORTS is obsolete in native data parallel mode; unset PORTS and set one PORT.' >&2; exit 2; }
[[ "${GPU_INDEX}" != ,* && "${GPU_INDEX}" != *, && "${GPU_INDEX}" != *,,* ]] || { echo 'GPU_INDEX contains an empty item' >&2; exit 2; }
IFS=',' read -r -a gpus <<< "${GPU_INDEX}"
declare -A seen_gpus=()
for gpu in "${gpus[@]}"; do
  [[ "${gpu}" =~ ^[A-Za-z0-9_/-]+$ ]] || { echo "Invalid GPU identifier: ${gpu}" >&2; exit 2; }
  [[ -z "${seen_gpus[$gpu]:-}" ]] || { echo "Duplicate GPU: ${gpu}" >&2; exit 2; }
  seen_gpus[$gpu]=1
done
[[ "${PORT}" =~ ^[0-9]{1,5}$ ]] || { echo 'PORT must be an integer in 1..65535' >&2; exit 2; }
PORT=$((10#${PORT}))
(( PORT >= 1 && PORT <= 65535 )) || { echo "Port out of range: ${PORT}" >&2; exit 2; }
# Avoid overriding the single-node, single-endpoint replica topology.
for arg in "$@"; do
  case "${arg}" in
    --port|--port=*|--host|--host=*|--data-parallel*|--tensor-parallel*|--pipeline-parallel*|--headless|--nnodes|--nnodes=*|--node-rank|--node-rank=*|-dp|-dp=*|-tp|-tp=*|-pp|-pp=*)
      echo "${arg} conflicts with the script-managed topology; use GPU_INDEX and PORT." >&2; exit 2 ;;
  esac
done
python3 - "${PORT}" <<'PY'
import socket,sys
with socket.socket() as sock:
    try: sock.bind(('127.0.0.1',int(sys.argv[1])))
    except OSError as exc: raise SystemExit(f'Port {sys.argv[1]} unavailable: {exc}')
PY
export CUDA_VISIBLE_DEVICES="${GPU_INDEX}"
export PATH="$(dirname "${VLLM_BIN}"):${PATH}"
printf 'START native vLLM: GPUs=%s DP=%s TP=1 endpoint=http://127.0.0.1:%s/v1\n' "${GPU_INDEX}" "${#gpus[@]}" "${PORT}"
echo 'Model loading logs follow. The endpoint is ready only after vLLM finishes startup.'
exec "${VLLM_BIN}" serve "${MODEL_PATH}" \
  --served-model-name "${SERVED_MODEL_NAME}" --host 127.0.0.1 --port "${PORT}" \
  --data-parallel-size "${#gpus[@]}" --tensor-parallel-size 1 \
  --max-model-len "${MAX_MODEL_LEN}" --max-num-seqs "${MAX_NUM_SEQS:-4}" \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" "$@"
