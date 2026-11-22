#!/usr/bin/env bash
set -euo pipefail
BASE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../../.." && pwd)"
BENCHMARK="${1:?Specify androidworld or osworld}"; shift
case "$BENCHMARK" in
  androidworld) PYTHON=/root/anaconda3/envs/vision2code/bin/python ;;
  osworld) PYTHON=/root/anaconda3/envs/osworld/bin/python ;;
  *) echo "Unsupported benchmark: $BENCHMARK" >&2; exit 2 ;;
esac
export PYTHONPATH="${BASE}/Methods/MAGNET/src:${BASE}/Benchmark/OSWorld${PYTHONPATH:+:${PYTHONPATH}}"
export NO_PROXY="${NO_PROXY:-},127.0.0.1,localhost"
export no_proxy="$NO_PROXY"
export OSWORLD_MAX_RETRY_TIMES=2 OSWORLD_OPENAI_TIMEOUT=180
cd "$BASE/Benchmark/OSWorld"
exec "$PYTHON" -u -m gui_baseline.parallel_eval --benchmark "$BENCHMARK" "$@"
