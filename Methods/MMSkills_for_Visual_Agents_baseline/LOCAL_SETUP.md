# MMSkills baseline: local setup and repository map

## What this repository contains

This ZIP is the public MMSkills runtime release, not the OSWorld benchmark itself and not a model-training pipeline. Its main pieces are:

- `mm_agents/`: model-agnostic agents (`general`, text-skill, and multimodal `mm_skill`), skill loading and task-to-skill resolution. `_mmskills_internal/` contains the branch-planner implementation used by the public adapter.
- `skills_library/`: the compact, self-contained skill subset shipped with the code release. A skill package contains `SKILL.md`, state-card JSON, and optional `Images/` references.
- `task_skill_mappings/task_skill_mapping.json`: one global mapping from OSWorld domain/task IDs to skill names.
- `osworld_integration/`: MMSkills-aware OSWorld runner and single-task/result helpers.
- `scripts/install_into_osworld.py`: upstream installer. It replaces runner files and deletes/re-copies skill directories, so it is unsafe for this workspace's locally modified OSWorld checkout.
- `agent_integrations/`, `macosworld_integration/`, `vab_minecraft_integration/`, `gaming_agent_integration/`: integrations for other agent products and benchmarks.

At inference, `mm_skill` reads the screenshot, decides whether a relevant skill and visual evidence are useful, consults the selected skill, then emits GUI actions through the OpenAI-compatible chat-completions client. It does not train or fine-tune the model.

## Workspace configuration

The ZIP was extracted intact to `Methods/MMSkills_for_Visual_Agents_baseline/`. The original ZIP and the pre-existing partial clone were left untouched. OSWorld is at `Benchmark/OSWorld` and uses the existing `osworld` conda environment.

The additive installer below copies only MMSkills-specific agent modules into OSWorld. It preserves OSWorld's `mm_agents/agent.py` and its `mm_agents/utils/qwen_vl_utils.py`; MMSkills image preprocessing is installed as `mmskills_qwen_vl_utils.py`.

```bash
cd /root/VLM-Memory
/root/anaconda3/envs/osworld/bin/python \
  Methods/MMSkills_for_Visual_Agents_baseline/scripts/setup_osworld_local.py \
  Benchmark/OSWorld
```

The runner expects task metadata in OSWorld's format (`{"domain": ["task-id", ...]}`), plus per-task configs at `Benchmark/OSWorld/evaluation_examples/examples/<domain>/<task-id>.json`. It defaults to the downloaded full Ubuntu skill dataset at `Datasets/MMSkills/ubuntu`, which contains all 170 unique skills referenced by the release's 360-task mapping. To use only the much smaller compact code-release subset, set `MMSKILLS_SKILLS_DIR=/root/VLM-Memory/Methods/MMSkills_for_Visual_Agents_baseline/skills_library`; some mapped skills will then be unavailable.

## Run

Set a task metadata JSON and API credentials, then run the experiment wrapper. An OpenAI-compatible gateway can be selected with `OPENAI_BASE_URL` and `MMSKILLS_API_MODEL`.

```bash
export MMSKILLS_META=/path/to/osworld_test_meta.json
export OPENAI_API_KEY='...'
export OPENAI_BASE_URL='http://127.0.0.1:8000/v1' # optional
export MMSKILLS_MODEL='gpt-4o'                    # runner/model label
export MMSKILLS_API_MODEL='your-served-model'     # optional backend model name
bash Experiments/memory_baselines/scripts/experiments/run_mmskills_osworld.sh
```

Useful overrides: `MMSKILLS_AGENT_TYPE=general` for the no-skill control, `MMSKILLS_AGENT_TYPE=general_skill` for text skills, `MMSKILLS_AGENT_TYPE=mm_skill` for MMSkills; `OSWORLD_PROVIDER`, `OSWORLD_OBSERVATION_TYPE`, `MMSKILLS_MAX_STEPS`, `MMSKILLS_RESULT_DIR`, `MMSKILLS_SKILLS_DIR`.

This config does not start OSWorld's VM or model server. Start the environment and compatible model endpoint using their normal project procedures before running a benchmark. The first validation should use a one-task metadata file.
