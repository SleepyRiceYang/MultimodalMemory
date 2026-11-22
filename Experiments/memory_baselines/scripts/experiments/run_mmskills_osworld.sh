#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
OSWORLD="$ROOT/Benchmark/OSWorld"
MMSKILLS="$ROOT/Methods/MMSkills_for_Visual_Agents_baseline"
PYTHON="${PYTHON:-/root/anaconda3/envs/osworld/bin/python}"

if [[ ! -x "$PYTHON" ]]; then
  echo "Python not found: $PYTHON (set PYTHON=/path/to/python)" >&2
  exit 2
fi
if [[ -z "${OPENAI_API_KEY:-}" ]]; then
  echo "Set OPENAI_API_KEY (or an OpenAI-compatible provider key) first." >&2
  exit 2
fi
if [[ -z "${MMSKILLS_META:-}" || ! -f "$MMSKILLS_META" ]]; then
  echo "Set MMSKILLS_META to an OSWorld test metadata JSON (domain -> task IDs)." >&2
  exit 2
fi

"$PYTHON" "$MMSKILLS/scripts/setup_osworld_local.py" "$OSWORLD"
mkdir -p "$OSWORLD/logs"

export PYTHONPATH="$OSWORLD${PYTHONPATH:+:$PYTHONPATH}"
cd "$OSWORLD"
exec "$PYTHON" "$MMSKILLS/osworld_integration/run.py" \
  --agent_type "${MMSKILLS_AGENT_TYPE:-mm_skill}" \
  --provider_name "${OSWORLD_PROVIDER:-docker}" \
  --observation_type "${OSWORLD_OBSERVATION_TYPE:-screenshot}" \
  --action_space "${OSWORLD_ACTION_SPACE:-pyautogui}" \
  --test_all_meta_path "$MMSKILLS_META" \
  --test_config_base_dir "$OSWORLD" \
  --skills_library_dir "${MMSKILLS_SKILLS_DIR:-$ROOT/Datasets/MMSkills/ubuntu}" \
  --task_skill_mapping_root "$MMSKILLS/task_skill_mappings/task_skill_mapping.json" \
  --base_url "${OPENAI_BASE_URL:-}" \
  --api_key "$OPENAI_API_KEY" \
  --model "${MMSKILLS_MODEL:-gpt-4o}" \
  --api_model "${MMSKILLS_API_MODEL:-${MMSKILLS_MODEL:-gpt-4o}}" \
  --max_steps "${MMSKILLS_MAX_STEPS:-20}" \
  --max_tokens "${MMSKILLS_MAX_TOKENS:-2048}" \
  --result_dir "${MMSKILLS_RESULT_DIR:-$ROOT/Experiments/memory_baselines/artifacts/mmskills/osworld}" \
  --enable_skills \
  --skill_mode "${MMSKILLS_SKILL_MODE:-multimodal}" \
  --save_conversation_json
