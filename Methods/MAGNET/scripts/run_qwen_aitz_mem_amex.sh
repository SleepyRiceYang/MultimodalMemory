#!/bin/bash
set -euo pipefail

# for i in {1..1}; do
#     echo "Run ${i}/10: amex val_id_app"
#     python scripts/workflows/run_all_model_pairs.py \
#         --base-config configs/presets/coat/qwenvl_coat.yaml \
#         --output-root artifacts/runs/exhaustive \
#         --dataset amex \
#         --subsets val_id_app \
#         --ratio 1.0 \
#         --num-workers 64 \
#         --demo-mode coat \
#         --predict-click-mode bbox \
#         --models qwen2.5-vl-32b qwen2.5-vl-32b \
#         --auto-eval \
#         --flow-procedural \
#         --predict-stationary \
#         --procedural-rag artifacts/storage/odyssey_procedural \
#         --stationary-rag artifacts/storage/odyssey_stationary_omni
#     # --limit 3500 \
#     # --log-prompts
# done
# sleep 300
for i in {1..1}; do
    echo "Run ${i}/10: odyssey val_sample"
    python scripts/workflows/run_all_model_pairs.py \
        --base-config configs/presets/coat/qwenvl_coat.yaml \
        --output-root artifacts/runs/exhaustive \
        --dataset odyssey \
        --subsets val_sample \
        --ratio 1.0 \
        --num-workers 64 \
        --demo-mode coat \
        --predict-click-mode bbox \
        --models qwen2.5-vl-32b qwen2.5-vl-32b \
        --auto-eval \
        --flow-procedural \
        --predict-stationary \
        --procedural-rag artifacts/storage/aitz_procedural \
        --stationary-rag artifacts/storage/aitz_stationary_omni
    # --limit 3500 \
    # --log-prompts
done
