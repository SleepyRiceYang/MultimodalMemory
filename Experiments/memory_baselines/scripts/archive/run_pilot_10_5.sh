#!/usr/bin/env bash
set -euo pipefail
TASK_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export PYTHONPATH="${TASK_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
PILOT_PYTHON="${PILOT_PYTHON:-/root/anaconda3/envs/vision2code/bin/python}"
PILOT_RUN="${PILOT_RUN:-pilot_10_5_new}"
for BENCHMARK_NAME in mathvision mpdocvqa vision2code; do
  "${PILOT_PYTHON}" -u -m vlm_memory.pipeline.pilot \
    --benchmark "${BENCHMARK_NAME}" --run "${PILOT_RUN}" --stage all "$@"
done
"${PILOT_PYTHON}" -m vlm_memory.analysis.pilot_report \
  --math-run "${PILOT_RUN}" --mp-run "${PILOT_RUN}" --vision-run "${PILOT_RUN}" \
  --output "${TASK_ROOT}/../../Docs/实验分析/${PILOT_RUN}-实验报告.md"
