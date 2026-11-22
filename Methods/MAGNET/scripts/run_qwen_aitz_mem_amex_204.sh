# python scripts/workflows/run_all_model_pairs.py \
#     --base-config configs/presets/coat/qwenvl_coat.yaml \
#     --output-root artifacts/runs/exhaustive \
#     --dataset amex \
#     --subsets val_id_app \
#     --ratio 1.0 \
#     --num-workers 64 \
#     --demo-mode coat \
#     --predict-click-mode bbox \
#     --models qwen2.5-vl-32b qwen2.5-vl-32b \
#     --flow-procedural \
#     --procedural-rag artifacts/storage/aitz_procedural\
#     --predict-stationary \
#     --stationary-rag artifacts/storage/aitz_stationary_omni \
#     --auto-eval \
#     # --limit 20 \
#     # --log-prompts


python scripts/workflows/run_all_model_pairs.py \
    --base-config configs/presets/coat/qwenvl_coat.yaml \
    --output-root artifacts/runs/204 \
    --dataset aitz \
    --subsets test \
    --ratio 1.0 \
    --num-workers 16 \
    --demo-mode coat \
    --predict-click-mode bbox \
    --models qwen2.5-vl-32b qwen2.5-vl-32b \
    --flow-procedural \
    --procedural-rag artifacts/storage/aitz_procedural\
    --predict-stationary \
    --stationary-rag /artifacts/storage/aitz_stationary_omni \
    --auto-eval \
    # --limit 3500 \
    # --log-prompts
