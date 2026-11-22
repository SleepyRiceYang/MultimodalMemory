#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)"
INTERVAL_SECONDS="${INTERVAL_SECONDS:-600}"
BENCHMARKS=(mathvision vision2code mpdocvqa)
ACTIVE=0
SOURCES=(
  qwen3_vl_8b_mathvision_concise_v1_8192_32768_nomemory
  qwen3_vl_8b_vision2code_8192_32768_v1_nomemory
  qwen3_vl_8b_mpdocvqa_8192_32768_v1_nomemory
)

snapshot() {
  local i benchmark source run log dir pids failed
  ACTIVE=0
  echo "===== $(date -u '+%Y-%m-%d %H:%M:%S UTC') ====="
  for i in "${!BENCHMARKS[@]}"; do
    benchmark="${BENCHMARKS[$i]}"
    source="${SOURCES[$i]}"
    run="${source}_four_memory_v2"
    log="${ROOT}/Experiments/memory_baselines/artifacts/logs/${run}.log"
    dir="${ROOT}/Experiments/memory_baselines/artifacts/${benchmark}/${run}"
    pids="$(ps -eo pid=,comm=,args= | awk -v r="${run}" '$2 ~ /^python/ && index($0,"vlm_memory.pipeline.formal_100_100") && index($0,r) {print $1}')"
    if [[ -n "${pids}" ]]; then ACTIVE=$((ACTIVE + 1)); fi
    echo "[$benchmark] run=$run process_pids=${pids:-none}"
    if [[ -f "${log}" ]]; then
      stat -c 'log_bytes=%s log_mtime=%y' "${log}"
      tail -n 2 "${log}" | sed 's/^/log_tail: /'
    else
      echo 'log=missing'
    fi
    if [[ -f "${dir}/bank.json" ]]; then
      python3 - "${dir}/bank.json" <<'PY'
import json,sys
x=json.load(open(sys.argv[1])); print("bank=frozen:%s entries:%s"%(x.get("frozen"),{k:len(v) for k,v in x.get("methods",{}).items()}))
PY
    else
      echo 'bank=not-yet-written'
    fi
    if [[ -f "${dir}/progress.json" ]]; then
      echo "progress=$(tr '\n' ' ' < "${dir}/progress.json")"
    else
      echo 'progress=not-yet-written'
    fi
    for method in raw_multimodal ksi_style ksi_visual magnet_adapted; do
      local n
      if [[ -d "${dir}/runs/${method}" ]]; then
        n="$(find "${dir}/runs/${method}" -type f -name result.json | wc -l)"
      else
        n=0
      fi
      echo "results_${method}=${n}"
    done
    failed="$(find "${dir}" -type f -name '*.failed.json' 2>/dev/null | wc -l)"
    echo "failed_json_count=${failed}"
  done
  echo "active_benchmarks=${ACTIVE}"
}

while true; do
  snapshot
  if [[ "${ACTIVE}" -eq 0 ]]; then
    echo "All three v2 experiment processes have exited; monitor stopped."
    exit 0
  fi
  sleep "${INTERVAL_SECONDS}"
done
