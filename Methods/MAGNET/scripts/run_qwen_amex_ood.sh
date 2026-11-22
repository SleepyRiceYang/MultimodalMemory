CUDA_VISIBLE_DEVICES=0,1,6,7

for i in {1..1}; do
    python scripts/workflows/run_all_model_pairs.py \
        --base-config configs/presets/coat/qwenvl_coat.yaml \
        --output-root artifacts/runs/exhaustive \
        --dataset amex \
        --subsets val_id_app \
        --ratio 1.0 \
        --num-workers 64 \
        --demo-mode coat \
        --predict-click-mode bbox \
        --models qwen2.5-vl-32b qwen2.5-vl-32b \
        --auto-eval \
        --flow-procedural \
        --predict-stationary \
        --limit 1000

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
    #     --auto-eval \
    #     --no-flow-procedural \
    #     --no-predict-stationary \
    #     --limit 1000

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
    #     --auto-eval \
    #     --flow-procedural \
    #     --no-predict-stationary \
    #     --limit 1000

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
    #     --auto-eval \
    #     --no-flow-procedural \
    #     --predict-stationary \
    #     --limit 1000
done
