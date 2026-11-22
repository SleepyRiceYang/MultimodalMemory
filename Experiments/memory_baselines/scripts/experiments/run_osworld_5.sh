#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
BASE="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
REPO="$(cd -- "${BASE}/../../Benchmark/OSWorld" && pwd)"
RUN_ID="${RUN_ID:-qwen3_vl_8b_osworld_nomemory_5_20261026}"
RESULT="${BASE}/artifacts/osworld/${RUN_ID}"
mkdir -p "${RESULT}" "${BASE}/artifacts/logs"
cp -n "${BASE}/splits/gui_smoke_5_seed20261026/osworld.json" "${RESULT}/tasks.json"
cd "${REPO}"
export NO_PROXY="${NO_PROXY:-},127.0.0.1,localhost"
export no_proxy="${NO_PROXY}"
export OSWORLD_MAX_RETRY_TIMES=2
/root/anaconda3/envs/osworld/bin/python -u scripts/python/run_multienv_qwen.py \
  --provider_name docker \
  --path_to_vm /data/yangshengjie/workspace/VLM-Memory/Benchmark/OSWorld/docker_vm_data/Ubuntu.qcow2 \
  --headless --no-enable_proxy --model qwen3-vl-8b-instruct \
  --base_url "${BASE_URL:-http://127.0.0.1:8105/v1}" --api_key EMPTY \
  --test_all_meta_path "${RESULT}/tasks.json" --num_envs 1 --max_steps "${MAX_STEPS:-30}" \
  --max_tokens "${MAX_TOKENS:-2048}" --temperature 0 --result_dir "${RESULT}" \
  2>&1 | tee -a "${BASE}/artifacts/logs/${RUN_ID}.log"
