#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="${SCRIPT_DIR}/../..${PYTHONPATH:+:${PYTHONPATH}}"
export CUDA_VISIBLE_DEVICES="${GPU_INDEX:-7}"
export EMBEDDING_MODEL="${EMBEDDING_MODEL:-/data/pretrained_models/BAAI-bge-large-en-v1.5}"
exec /root/anaconda3/envs/vllm_env/bin/python -m uvicorn vlm_memory.embedding_server:app --host 127.0.0.1 --port "${PORT:-8087}"
