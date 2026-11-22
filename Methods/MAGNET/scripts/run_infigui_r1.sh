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
    --base-config configs/presets/coat/infigui_r1.yaml \
    --output-root artifacts/runs/exhaustive \
    --dataset amex \
    --data-dir /attached/remote-home1/lbsun/data/amex \
    --subsets val_id_app \
    --ratio 1.0 \
    --num-workers 32 \
    --demo-mode infigui_r1 \
    --predict-click-mode point \
    --models infigui-r1-3b \
    --auto-eval \
    # --limit 1000