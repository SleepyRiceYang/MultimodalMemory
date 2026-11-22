#!/usr/bin/env bash
set -euo pipefail
BASE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../../.." && pwd)"
STAGE="${1:?Usage: bash run_osworld_50_50.sh construction|build-memory|test-no-memory|test-text|test-visual [--dry-run]}"
shift
SPLIT="${BASE}/Experiments/memory_baselines/splits/gui_50_50_seed20261026/osworld.json"
CONSTRUCTION_RUN="${CONSTRUCTION_RUN:-formal_50_construction_v1}"
TEST_RUN="${TEST_RUN:-formal_50_test_v1}"
BANK_RUN="${BANK_RUN:-formal_50_bank_v1}"
BANK_DIR="${BASE}/Experiments/memory_baselines/artifacts/gui_magnet_procedural/osworld/banks/${BANK_RUN}"
MODEL_URL="${MODEL_URL:-http://127.0.0.1:8105/v1}"
EMBEDDING_URL="${EMBEDDING_URL:-http://127.0.0.1:8087}"
export PYTHONPATH="${BASE}/Methods/MAGNET/src:${BASE}/Benchmark/OSWorld${PYTHONPATH:+:${PYTHONPATH}}"
export NO_PROXY="${NO_PROXY:-},127.0.0.1,localhost"
export no_proxy="${NO_PROXY}"
export OSWORLD_MAX_RETRY_TIMES=2
export OSWORLD_OPENAI_TIMEOUT=180
case "${STAGE}" in
  construction)
    MODULE=gui_baseline.comparison
    ARGS=(--workspace "${BASE}" --benchmark osworld --condition no_memory
      --split-file "${SPLIT}" --split-role memory_construction --run "${CONSTRUCTION_RUN}"
      --bank-dir "${BANK_DIR}" --model-url "${MODEL_URL}" --embedding-url "${EMBEDDING_URL}"
      --vm-path "${VM_PATH:-/data/yangshengjie/workspace/VLM-Memory/Benchmark/OSWorld/docker_vm_data/Ubuntu.qcow2}"
      --max-steps "${MAX_STEPS:-50}" --max-tokens "${MAX_TOKENS:-2048}") ;;
  build-memory)
    MODULE=gui_baseline.formal
    ARGS=(--workspace "${BASE}" --benchmark osworld --split-file "${SPLIT}"
      --source-run "${CONSTRUCTION_RUN}" --bank-run "${BANK_RUN}"
      --model-url "${MODEL_URL}" --embedding-url "${EMBEDDING_URL}") ;;
  test-no-memory|test-text|test-visual)
    MODULE=gui_baseline.comparison
    case "${STAGE}" in
      test-no-memory) CONDITION=no_memory ;;
      test-text) CONDITION=magnet_procedural_text ;;
      test-visual) CONDITION=magnet_procedural_visual ;;
    esac
    ARGS=(--workspace "${BASE}" --benchmark osworld --condition "${CONDITION}"
      --split-file "${SPLIT}" --split-role test --run "${TEST_RUN}" --source-run "${CONSTRUCTION_RUN}"
      --bank-dir "${BANK_DIR}" --model-url "${MODEL_URL}" --embedding-url "${EMBEDDING_URL}"
      --vm-path "${VM_PATH:-/data/yangshengjie/workspace/VLM-Memory/Benchmark/OSWorld/docker_vm_data/Ubuntu.qcow2}"
      --max-steps "${MAX_STEPS:-50}" --max-tokens "${MAX_TOKENS:-2048}") ;;
  *) echo "Unknown stage: ${STAGE}" >&2; exit 2 ;;
esac
LOG_DIR="${BASE}/Experiments/memory_baselines/artifacts/logs"
mkdir -p "${LOG_DIR}"
cd "${BASE}/Benchmark/OSWorld"
/root/anaconda3/envs/osworld/bin/python -u -m "${MODULE}" "${ARGS[@]}" "$@" \
  2>&1 | tee -a "${LOG_DIR}/osworld_50_50_${STAGE}.log"
