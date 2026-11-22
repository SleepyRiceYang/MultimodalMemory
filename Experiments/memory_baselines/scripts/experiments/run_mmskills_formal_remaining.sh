#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
export NO_PROXY="${NO_PROXY:-},127.0.0.1,localhost"
export no_proxy="$NO_PROXY"
exec "${PYTHON:-/root/anaconda3/envs/osworld/bin/python}" -u \
  "$ROOT/Experiments/memory_baselines/mmskills_osworld/formal_remaining.py" "$@"
