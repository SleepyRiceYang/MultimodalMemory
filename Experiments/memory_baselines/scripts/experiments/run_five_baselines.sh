#!/usr/bin/env bash
# Use an already completed common 100/100 no-memory run; never starts model servers.
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
BASE="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
BENCHMARK="${1:?Usage: run_five_baselines.sh <mathvision|vision2code|mpdocvqa> <source-run> [extra Python arguments]}"
SOURCE_RUN="${2:?Provide the completed common no-memory source run ID}"
shift 2
case "${BENCHMARK}" in mathvision|vision2code|mpdocvqa) ;; *) exit 2 ;; esac
export PYTHONPATH="${BASE}${PYTHONPATH:+:${PYTHONPATH}}"
mkdir -p "${BASE}/artifacts/logs"
/root/anaconda3/envs/vision2code/bin/python -u -m vlm_memory.pipeline.formal_100_100 \
  --config "${BASE}/config/qwen3_vl_8b_${BENCHMARK}_five_baselines.json" \
  --source-run "${SOURCE_RUN}" --run "${FORMAL_RUN:-${SOURCE_RUN}_five_representations_v1}" \
  --benchmarks "${BENCHMARK}" "$@" \
  2>&1 | tee -a "${BASE}/artifacts/logs/${BENCHMARK}_five_baselines.log"
