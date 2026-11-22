#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../../.."
export PYTHONPATH="Methods/MAGNET/src${PYTHONPATH:+:$PYTHONPATH}"
output="${1:-Experiments/memory_baselines/artifacts/capcomem_rq1_b2b3/factorial_easy61_r3}"
exec "${MAGNET_PYTHON:-/root/magnet_cold_env/bin/python}" -u -m gui_baseline.evidence_factorial \
  --all-easy --rollouts 3 --seed 42 --workers 4 --device cpu \
  --model-url http://127.0.0.1:8177/v1 \
  --evidence-memory Experiments/memory_baselines/artifacts/capcomem_rq1_b2b3/memory_v3_multistate/memory.json \
  --output "$output"
