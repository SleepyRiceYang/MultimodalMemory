#!/usr/bin/env bash
set -euo pipefail

python scripts/evaluate_predictions.py \
  --config configs/presets/coat/qwenvl_coat.yaml \
  --dataset-name amex \
  --dataset-dir /remote-home/lbsun/data/amex \
  --split val_id_app \
  --pred-dir artifacts/runs/qwenvl32_test/amex/val_id_app/pairs/qwen2-5-vl-32b__proc1__qwen2-5-vl-32b__stat0__bbox \
  --model-name qwen2.5-vl-32b \
  --demo-mode COAT \
  --screen-mode tag \
  --click-mode bbox \
  --html-report artifacts/metrics/amex_val_id_app.html \
  --metrics-json artifacts/metrics/amex_val_id_app.json \
  --serve --serve-host 0.0.0.0 --serve-port 8992  
  # "$@"

report_path=$(python - <<'PY'
from pathlib import Path
print(Path("artifacts/metrics/amex_val_id_app.html").resolve())
PY
)
printf '\nInteractive report: file://%s\n' "$report_path"
