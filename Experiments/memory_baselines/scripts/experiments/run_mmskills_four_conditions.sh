#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
PYTHON="${PYTHON:-/root/anaconda3/envs/osworld/bin/python}"
export PYTHONPATH="$ROOT/Benchmark/OSWorld:$ROOT/Methods/MMSkills_for_Visual_Agents_baseline/osworld_integration${PYTHONPATH:+:$PYTHONPATH}"
export NO_PROXY="${NO_PROXY:-},127.0.0.1,localhost"
export no_proxy="$NO_PROXY"
"$PYTHON" "$ROOT/Methods/MMSkills_for_Visual_Agents_baseline/scripts/setup_osworld_local.py" "$ROOT/Benchmark/OSWorld"
cd "$ROOT/Benchmark/OSWorld"
exec "$PYTHON" -u "$ROOT/Experiments/memory_baselines/mmskills_osworld/run.py" "$@"
