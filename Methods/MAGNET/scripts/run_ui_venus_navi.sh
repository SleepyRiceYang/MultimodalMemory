# # python scripts/workflows/run_all_model_pairs.py \
# #     --base-config configs/presets/coat/ui_venus_navi.yaml \
# #     --output-root artifacts/runs/ui_venus_navi \
# #     --models "ui-venus-navi-7b" \
# #     --demo-mode ui_venus \
# #     --flow-procedural \
# #     --predict-stationary \
# #     --predict-click-mode point \
# #     --dataset odyssey \
# #     --subsets val_sample \
# #     --num-workers 64
# #     # --limit 20 \
    
# # python scripts/workflows/run_all_model_pairs.py \
# #     --base-config configs/presets/coat/ui_venus_navi.yaml \
# #     --output-root artifacts/runs/ui_venus_navi \
# #     --models "ui-venus-navi-7b" \
# #     --demo-mode ui_venus \
# #     --flow-procedural \
# #     --predict-stationary \
# #     --predict-click-mode point \
# #     --dataset amex \
# #     --subsets val_id_app \
# #     --num-workers 64
#     # --limit 20 \
    

# python scripts/workflows/run_all_model_pairs.py \
#     --base-config configs/presets/coat/ui_venus_navi.yaml \
#     --output-root artifacts/runs/ui_venus_navi \
#     --models "ui-venus-navi-7b" \
#     --demo-mode ui_venus \
#     --flow-procedural \
#     --predict-stationary \
#     --predict-click-mode point \
#     --dataset aitz \
#     --subsets test \
#     --num-workers 64
#     # --limit 20 \
    
python scripts/workflows/run_all_model_pairs.py \
    --base-config configs/presets/coat/ui_venus_navi.yaml \
    --output-root artifacts/runs/exhaustive \
    --dataset aitz \
    --subsets test \
    --ratio 1.0 \
    --num-workers 64 \
    --demo-mode ui_venus \
    --predict-click-mode point \
    --models ui-venus-navi-7b \
    --auto-eval