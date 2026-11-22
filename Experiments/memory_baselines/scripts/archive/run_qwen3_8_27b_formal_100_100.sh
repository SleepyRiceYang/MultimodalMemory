#!/usr/bin/env bash
# Run the fixed 100-memory/100-test Qwen3.8-27B experiment with the frozen
# formal prompt profile. The no-memory phase creates common trajectories;
# the formal phase builds all configured memories from memory split only.
#
# Usage:
#   bash scripts/run_qwen3_8_27b_formal_100_100.sh [all|no-memory|formal]
# Re-run the same phase/run IDs to resume from saved task results.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PHASE="${1:-all}"
CONFIG="${FORMAL_CONFIG:-${BASE_DIR}/config/qwen3_8_27b_formal_100_100.json}"
SOURCE_RUN="${SOURCE_RUN:-qwen38_formal_prompt_timeout1800_20261023_nomemory}"
FORMAL_RUN="${FORMAL_RUN:-qwen38_formal_prompt_timeout1800_20261023_baselines}"

case "${PHASE}" in
  all|no-memory|formal) ;;
  *) echo "Usage: $0 [all|no-memory|formal]" >&2; exit 2 ;;
esac

if [[ "${PHASE}" == "all" || "${PHASE}" == "no-memory" ]]; then
  bash "${SCRIPT_DIR}/run_qwen3_8_27b_100_100.sh" no-memory \
    --config "${CONFIG}" --run "${SOURCE_RUN}" --benchmarks mathvision vision2code
fi

if [[ "${PHASE}" == "all" || "${PHASE}" == "formal" ]]; then
  bash "${SCRIPT_DIR}/run_qwen3_8_27b_100_100.sh" formal \
    --config "${CONFIG}" --source-run "${SOURCE_RUN}" --run "${FORMAL_RUN}" \
    --stage all --benchmarks mathvision vision2code
fi
