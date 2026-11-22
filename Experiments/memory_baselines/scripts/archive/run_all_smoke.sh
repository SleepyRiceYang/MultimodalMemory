#!/usr/bin/env bash
set -euo pipefail
SUITE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ARTIFACT_DIR="${1:-${SUITE_ROOT}/artifacts/smoke_$(date -u +%Y%m%dT%H%M%SZ)}"
cd "${SUITE_ROOT}"
V2C_PYTHON=/root/anaconda3/envs/vision2code/bin/python
MATH_PYTHON=/root/anaconda3/envs/mathvision/bin/python
"${V2C_PYTHON}" -m vlm_memory.scripts.prepare_history --artifacts "${ARTIFACT_DIR}"
"${V2C_PYTHON}" -m vlm_memory.scripts.build_memories --artifacts "${ARTIFACT_DIR}"
"${MATH_PYTHON}" -m vlm_memory.scripts.run_smoke --benchmark mathvision --artifacts "${ARTIFACT_DIR}"
"${V2C_PYTHON}" -m vlm_memory.scripts.run_smoke --benchmark vision2code --artifacts "${ARTIFACT_DIR}"
"${V2C_PYTHON}" -m vlm_memory.scripts.summarize --artifacts "${ARTIFACT_DIR}" --require-complete
