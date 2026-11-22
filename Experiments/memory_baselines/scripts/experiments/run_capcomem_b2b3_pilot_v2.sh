#!/usr/bin/env bash
set -euo pipefail
cd /root/VLM-Memory
export PYTHONPATH=Methods/MAGNET/src
for condition in B2 B3; do
  /root/magnet_cold_env/bin/python -u -m gui_baseline.evidence_eval_v2 \
    --evidence-memory Experiments/memory_baselines/artifacts/capcomem_rq1_b2b3/memory_v1/memory.json \
    --output "Experiments/memory_baselines/artifacts/capcomem_rq1_b2b3/iteration_2/${condition}" \
    --condition "$condition" --model-url http://127.0.0.1:8177/v1 --device cuda:5 --workers 4
 done
