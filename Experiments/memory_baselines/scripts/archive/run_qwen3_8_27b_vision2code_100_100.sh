#!/usr/bin/env bash
# Independent Vision2Code 100/100 pipeline; safe to run alongside MathVision.
# Usage: bash scripts/run_qwen3_8_27b_vision2code_100_100.sh [all|no-memory|formal]
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PHASE="${1:-all}"
CONFIG="${FORMAL_CONFIG:-${BASE_DIR}/config/qwen3_8_27b_formal_100_100.json}"
SOURCE_RUN="${SOURCE_RUN:-qwen38_vision2code_formal_timeout1800_20261023_nomemory}"
FORMAL_RUN="${FORMAL_RUN:-qwen38_vision2code_formal_timeout1800_20261023_baselines}"
case "${PHASE}" in all|no-memory|formal) ;; *) echo "Usage: $0 [all|no-memory|formal]" >&2; exit 2 ;; esac

if [[ "${PHASE}" == all || "${PHASE}" == no-memory ]]; then
  bash "${SCRIPT_DIR}/run_qwen3_8_27b_100_100.sh" no-memory \
    --config "${CONFIG}" --run "${SOURCE_RUN}" --benchmarks vision2code
fi
if [[ "${PHASE}" == all || "${PHASE}" == formal ]]; then
  bash "${SCRIPT_DIR}/run_qwen3_8_27b_100_100.sh" formal \
    --config "${CONFIG}" --source-run "${SOURCE_RUN}" --run "${FORMAL_RUN}" \
    --stage all --benchmarks vision2code
fi
