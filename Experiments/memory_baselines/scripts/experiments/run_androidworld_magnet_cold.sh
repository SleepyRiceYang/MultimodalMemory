#!/usr/bin/env bash
# Dedicated dual-memory run: empty initialization, Easy61, existing background vLLM.
set -euo pipefail
BASE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../../.." && pwd)"
STAGE="${1:-plan}"
if (( $# )); then shift; fi
export PYTHONPATH="${BASE}/Methods/MAGNET/src${PYTHONPATH:+:${PYTHONPATH}}"
export NO_PROXY="${NO_PROXY:-},127.0.0.1,localhost"
export no_proxy="${NO_PROXY}"
export PYTHONUNBUFFERED=1
# Isolate auxiliary MiniCPM/OCR models; the existing vLLM process is untouched.
export CUDA_VISIBLE_DEVICES="${MAGNET_CUDA_VISIBLE_DEVICES:-4}"
cd "${BASE}"
exec "${MAGNET_PYTHON:-/root/magnet_cold_env/bin/python}" -m gui_baseline.cold_continual --stage "${STAGE}" "$@"
