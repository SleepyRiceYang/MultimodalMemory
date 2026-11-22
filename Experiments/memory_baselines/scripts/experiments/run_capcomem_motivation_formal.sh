#!/usr/bin/env bash
# Full fixed lists: MP (20+60)*4 + Android (20+50)*5 = 670 executions.
# --check validates services and configuration without starting task execution.
set -euo pipefail
BASE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../../.." && pwd)"
cd "$BASE"
export PYTHONPATH="${BASE}/Experiments/memory_baselines:${BASE}/Methods/MAGNET/src${PYTHONPATH:+:${PYTHONPATH}}"
PYTHON=/root/anaconda3/envs/vllm_env/bin/python
OUTPUT="${BASE}/Experiments/memory_baselines/artifacts/capcomem_motivation/formal_20261104_v1"
STAGE=all
CHECK=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --check) CHECK=1; shift ;;
    --stage) STAGE="${2:?--stage requires a value}"; shift 2 ;;
    --output) OUTPUT="${2:?--output requires a path}"; shift 2 ;;
    -h|--help)
      echo 'Usage: bash run_capcomem_motivation_formal.sh [--check] [--stage prepare|assets|select|evaluate|report|all] [--output PATH]'
      exit 0 ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done
case "$STAGE" in prepare|assets|select|evaluate|report|all) ;; *) echo "Invalid stage: $STAGE" >&2; exit 2 ;; esac
if [[ "$CHECK" == 1 ]]; then
  "$PYTHON" - <<'PY'
import json, urllib.request
from vlm_memory.pipeline.capcomem_motivation import MP, GB, model_config, LocalModel
from gui_baseline.runner import Android
for port in (8113,8114):
    url=f'http://127.0.0.1:{port}/v1/models'
    with urllib.request.urlopen(url,timeout=15) as r: data=json.load(r)
    assert any(m['id']=='qwen3-vl-8b-instruct' for m in data['data']), url
    print('PASS model',port)
with urllib.request.urlopen('http://127.0.0.1:8087/health',timeout=15) as r: encoder=json.load(r)
assert encoder['model']=='/data/pretrained_models/BAAI-bge-large-en-v1.5'
print('PASS embedding',encoder['gpu'])
for port in (15003,15004):
    with urllib.request.urlopen(f'http://127.0.0.1:{port}/health',timeout=15) as r:
        assert json.load(r)['status']=='success'
    print('PASS Android HTTP',port)
for path in (MP/'bank.json',MP/'dataset_manifest.json',GB/'bank.json',GB/'source_episodes.json'):
    assert path.is_file(),path
print('PASS source artifacts and Python imports')
print('READY: MP 320 + Android 350 = 670 task-condition executions; no inference or environment reset performed.')
PY
  exit 0
fi
exec "$PYTHON" -u -m vlm_memory.pipeline.capcomem_motivation \
  --output "$OUTPUT" \
  --benchmarks mpdocvqa androidworld \
  --limit-per-role 0 \
  --workers 2 \
  --max-steps 30 \
  --model-urls http://127.0.0.1:8113/v1 http://127.0.0.1:8114/v1 \
  --embedding-url http://127.0.0.1:8087 \
  --stage "$STAGE"
