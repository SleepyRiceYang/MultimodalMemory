for i in {1..1}; do
    env -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY \
CUDA_VISIBLE_DEVICES=0,1,6,7 \
python scripts/workflows/run_all_model_pairs.py \
  --base-config configs/presets/coat/config_gemini_android.yaml \
  --output-root artifacts/runs/exhaustive \
  --dataset amex \
  --subsets val_id_app \
  --ratio 1.0 \
  --num-workers 1 \
  --demo-mode coat \
  --predict-click-mode bbox \
  --models gemini-2.5-pro gemini-2.5-pro \
  --auto-eval \
  --flow-procedural \
  --predict-stationary \
  --limit 500

    env -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY \
CUDA_VISIBLE_DEVICES=0,1,6,7 \
python scripts/workflows/run_all_model_pairs.py \
  --base-config configs/presets/coat/config_gemini_android.yaml \
  --output-root artifacts/runs/exhaustive \
  --dataset amex \
  --subsets val_id_app \
  --ratio 1.0 \
  --num-workers 1 \
  --demo-mode coat \
  --predict-click-mode bbox \
  --models gemini-2.5-pro gemini-2.5-pro \
  --auto-eval \
  --no-flow-procedural \
  --no-predict-stationary \
  --limit 500

    env -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY \
CUDA_VISIBLE_DEVICES=0,1,6,7 \
python scripts/workflows/run_all_model_pairs.py \
  --base-config configs/presets/coat/config_gemini_android.yaml \
  --output-root artifacts/runs/exhaustive \
  --dataset amex \
  --subsets val_id_app \
  --ratio 1.0 \
  --num-workers 1 \
  --demo-mode coat \
  --predict-click-mode bbox \
  --models gemini-2.5-pro gemini-2.5-pro \
  --auto-eval \
  --no-flow-procedural \
  --predict-stationary \
  --limit 500

    env -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY \
CUDA_VISIBLE_DEVICES=0,1,6,7 \
python scripts/workflows/run_all_model_pairs.py \
  --base-config configs/presets/coat/config_gemini_android.yaml \
  --output-root artifacts/runs/exhaustive \
  --dataset amex \
  --subsets val_id_app \
  --ratio 1.0 \
  --num-workers 1 \
  --demo-mode coat \
  --predict-click-mode bbox \
  --models gemini-2.5-pro gemini-2.5-pro \
  --auto-eval \
  --flow-procedural \
  --no-predict-stationary \
  --limit 500
done
