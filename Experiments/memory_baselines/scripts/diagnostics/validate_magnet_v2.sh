#!/usr/bin/env bash
# Construction-only validation. Reuses saved calls; does not start formal evaluation.
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
BASE="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
BENCHMARK="${1:?Usage: validate_magnet_v2.sh <mathvision|vision2code|mpdocvqa> [describe|calibrate|workflows|visual|replay|integration|all]}"
STAGE="${2:-all}"
export PYTHONPATH="${BASE}${PYTHONPATH:+:${PYTHONPATH}}"
mkdir -p "${BASE}/artifacts/logs"
/root/anaconda3/envs/vision2code/bin/python -u -m vlm_memory.pipeline.validate_magnet_v2 \
  --benchmark "${BENCHMARK}" --stage "${STAGE}" \
  2>&1 | tee -a "${BASE}/artifacts/logs/magnet_v2_${BENCHMARK}_${STAGE}.log"
