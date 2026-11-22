#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
PYTHON="${EMBEDDING_PYTHON:-/root/.venvs/mmskills_embedding/bin/python}"
export CUDA_VISIBLE_DEVICES="${GPU_INDEX:-6}"
export EMBEDDING_MODEL="${EMBEDDING_MODEL:-/data/pretrained_models/MiniCPM-Embedding}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTHONPATH="$ROOT/Experiments/memory_baselines/mmskills_osworld${PYTHONPATH:+:$PYTHONPATH}"
[[ -f "$EMBEDDING_MODEL/download_manifest.json" ]] || { echo 'Download and verify MiniCPM-Embedding first.' >&2; exit 1; }
exec "$PYTHON" -m uvicorn embedding_server:app --host 127.0.0.1 --port "${PORT:-8187}"
