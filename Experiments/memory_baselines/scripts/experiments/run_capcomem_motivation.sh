#!/usr/bin/env bash
set -euo pipefail
BASE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../../.." && pwd)"
export PYTHONPATH="${BASE}/Experiments/memory_baselines:${BASE}/Methods/MAGNET/src${PYTHONPATH:+:${PYTHONPATH}}"
exec /root/anaconda3/envs/vllm_env/bin/python -u -m vlm_memory.pipeline.capcomem_motivation "$@"
