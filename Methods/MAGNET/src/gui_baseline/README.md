# MAGNET GUI procedural adaptation

This is an **adaptation**, not the original MAGNET offline evaluation. The original
`agent/`, LightRAG, stationary patch memory and OmniParser paths remain separate.

Two conditions share a frozen procedural bank, embedding query, selected workflow
IDs, procedural text, actor and environment protocol. `text` receives that text;
`visual` additionally receives each selected workflow's BEFORE/AFTER screenshots.
Screenshots are historical workflow evidence, **not detected target patches**.
OmniParser and its weights are not required or invoked. Current-screen grounding
is performed by the VLM actor, using native benchmark actions.

## Modules

- `build.py`: export recorded no-memory trajectories, bounded extraction, source
  action citation checks, independent VLM transition verification, dense index.
- `memory.py`: actor-field whitelist, task-literal parameterization, benchmark and
  source-task exclusion, frozen text/image paired retrieval.
- `prompts.py`: extraction, verification, Android actor and common memory notice.
- `runner.py`: real AndroidWorld HTTP/ADB and OSWorld DesktopEnv/QwenAgent adapters;
  reset each condition, official scoring, resume, logs, pairing/freeze checks.

Memory fields delivered to the actor: `id`, `title`, `app`, `precondition`, `goal`,
`parameters` (slot names only), `steps` (`instruction`, `expected_state`),
`postcondition`. Raw trajectories, source coordinates, verification output and
source literal values stay in provenance. Visual view adds two evidence images.
Historical images can still display source literals; instructions prohibit copying
those values. Text/visual comparisons measure the effect of that entire image input.

Index: Topic=`title`, Scope=`app; precondition`, Question Anchor=`goal`.
Query: public current task goal, fixed once per paired task in this pilot.
BGE-large-en-v1.5 cosine similarity, Top20 candidates, shared final budget max3 units,
2 historical images and16000 memory text characters. Since each workflow has2
images, current budget admits1 unit. No LLM reranking or retrieval-count updates.
There is currently no relevance threshold: an unrelated workflow may be selected;
the actor must ignore it. This is logged and must be addressed before a formal
large-bank study. No memory crosses benchmarks or incorporates validation outcomes.

The pilot reuses the previous five recorded source tasks in each benchmark and
validates on five different task IDs/types. Sparse successful source transitions
mean only one unit per bank survived verification. These are functional smoke
tests, not evidence of a memory benefit or a full MAGNET reproduction. The saved
`visual_review.json` records inspection corrections to automatically accepted
units before evaluation; automated verification alone did not catch all semantic
errors. Future fully automated studies need stronger verification, not silent
manual fixes based on test results.

## Run

From workspace root (services on8104/8105 and embedding8087 must be running):

```bash
bash Experiments/memory_baselines/scripts/experiments/run_gui_magnet_procedural_5.sh androidworld
bash Experiments/memory_baselines/scripts/experiments/run_gui_magnet_procedural_5.sh osworld
```

`--stage build|evaluate|all` (default all), Android 默认 `--run paired_5_v2`、OS 默认 `paired_5_v1`, `--max-steps30`,
`--max-tokens2048`. Identical commands resume completed condition runs; partial/error
folders are archived before retry. A changed configuration requires a new run name.
One exclusive environment lock per benchmark prevents simultaneous conflicting runs.
Each benchmark executes text/visual serially; the two benchmark scripts can run in
parallel using GPUs4/5. Native OSWorld current-episode screenshot history is capped
at3, separate from the2 historical memory images. Android uses the current screen
and last10 actions; no evaluator feedback is exposed to either actor.

Android: terminal state score is recorded separately; reported agent score requires
explicit agent termination. OSWorld: official `env.evaluate()` numeric score.
Unfinished tasks remain score0 where the native protocol requires termination;
infrastructure exceptions are errors and are not recast as task failures/successes.

## Matched No Memory / Text / Visual test scripts

`comparison.py` drives three independent conditions on the same five held-out
GUI tasks in each benchmark. It uses the same online actor and task budget;
No Memory omits retrieval and historical memory messages entirely. Each condition
has its own shell entry point selection through:

```bash
bash Experiments/memory_baselines/scripts/experiments/run_androidworld_gui_5.sh no_memory
bash Experiments/memory_baselines/scripts/experiments/run_androidworld_gui_5.sh magnet_procedural_text
bash Experiments/memory_baselines/scripts/experiments/run_androidworld_gui_5.sh magnet_procedural_visual
bash Experiments/memory_baselines/scripts/experiments/run_osworld_gui_5.sh no_memory
bash Experiments/memory_baselines/scripts/experiments/run_osworld_gui_5.sh magnet_procedural_text
bash Experiments/memory_baselines/scripts/experiments/run_osworld_gui_5.sh magnet_procedural_visual
```

Default run `comparison_5_v1` is separate from the completed earlier two-arm
pilot. Completed task-condition folders are resumed, interrupted/error folders
archived and rerun. Conditions for one benchmark run serially due to the same
environment lock; benchmarks may run in parallel. See
`Docs/实验方案/20261026-05-GUI三组同题测试脚本.md` for settings and outputs.

## Frozen 50/50 study entry points

`run_androidworld_50_50.sh` and `run_osworld_50_50.sh` in
`Experiments/memory_baselines/scripts/experiments/` each accept:
`construction`, `build-memory`, `test-no-memory`, `test-text`, `test-visual`.
Run in that order per benchmark. Each stage loads the frozen manifest in
`Experiments/memory_baselines/splits/gui_50_50_seed20261026/` and stores its own
run configuration. The builder reads exactly 50 completed construction No Memory
trajectories. Test memory conditions reject a bank whose source IDs differ from
the manifest, and share one retrieval record per held-out task. Android uses the
native `10×complexity` step policy (capped at 120); OS uses 50 steps. Both use
2048 output tokens per action by default. Existing 5-task artifacts are separate
and are never imported into the formal bank.

See `Docs/实验方案/20261026-07-GUI五十对五十Baseline执行脚本.md` for the ten commands
and the environment/data prerequisites. The scripts have passed dry-run and
synthetic trajectory integrity tests; the 50/50 experiments have not run yet.
