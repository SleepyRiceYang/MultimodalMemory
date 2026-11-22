# Baseline setup

当前纳入 B0、B1、B1-T（Raw Text）、B1-V（Raw Visual）、B3（KSI-style）、B4（MAGNET）、B6（Darwinian）与 B8（EvoSkill-GUI）。UI-Mem、EpiCon、SkillLens 本轮不运行。可运行的推理领域适配层和单样本结果见 [Experiments/memory_baselines](../Experiments/memory_baselines/README.md)；以下表格保留原生源码的获取状态。

This directory holds upstream methods separately from the benchmark repositories. The [experiment plan](../Docs/ExperimentLogs/20261017-实验方案.md) distinguishes original reproductions, adapted methods, and reference controls; do not report an adaptation as an original-paper result.

| ID | Method | Implementation status |
| --- | --- | --- |
| B0 | No Memory | Reference control: current-task visual input is retained; no historical experience is injected. The existing MATH-Vision/Vision2Code smoke tests use this setting. |
| B1 | Raw Episodic | Reference control implemented with multimodal, text-only and visual-only variants. Smoke histories are independent procedural tasks, disjoint from the benchmark inputs. |
| B2 | UI-Mem-style Text | Adaptation required; [official project page](https://ui-mem.github.io/) still says code/model/data are coming soon. Full UI-Mem includes online RL and is out of scope for the first frozen-memory comparison. |
| B3 | KSI-style Knowledge | [Official KSI source](https://github.com/recursive-knowledge/KSI) cloned to `KSI/`; base package installed in Conda env `ksi` (Python 3.12). Its native coding/reasoning harness is not a MATH-Vision runner; visual-math use requires an adapter. |
| B4 | MAGNET | [Paper source](https://github.com/sunlibo2390/MAGNET) cloned to `MAGNET/`. Its included example uses Android-in-the-Zoo, not AndroidWorld. |
| B5 | SkillLens | The [Visual Skill Cards paper](https://arxiv.org/abs/2608.10775) has no identified official release. Do not substitute Microsoft's unrelated `SkillLens` repository. |
| B6 | Darwinian Memory | The [published paper](https://proceedings.mlr.press/v306/mi26b.html) links [DMS-C48C](https://anonymous.4open.science/r/DMS-C48C/). Its repository API returns HTTP 410 `repository_expired`; see `DarwinianMemory/SOURCE.md`. A runnable domain adaptation is available in the shared runner. |
| B7 | EpiCon | The [official project page](https://zzzmyyzeng.github.io/EpiCon/) says GitHub is coming soon. Its trained memory controller/tree organizer are not available here; a hand-built text-plus-image store is **not** original EpiCon. |
| B8 | EvoSkill-GUI | [Official source](https://github.com/ZJU-REAL/EvoSkill-GUI) cloned to `EvoSkill-GUI/`. Optional GUI extension; requires its own Python 3.12 environment, Android runtime, Docker/KVM, and benchmark-specific integration. |

## Shared experimental contract

Before building formal memory banks, define disjoint `D_memory` and `D_eval` task IDs. Keep the backbone, prompt, visual input, action budget, and evaluation protocol fixed within each benchmark. Record memory provenance, retrievals, injected text/images, latency, tokens, and outcome. Disable online memory updates in the first comparison. A single smoke-test score is not a baseline comparison.

The local model endpoint is `http://127.0.0.1:8000/v1` with served model `qwen3-vl-8b-instruct`; start it with `bash ../Benchmark/start_qwen3vl_vllm.sh` if it is not already running. Upstream methods may need API adapters and must be validated independently before their scores are compared.

## Source revisions

The initial shallow clones are pinned for reproducibility: KSI `ac709263bc3d`, MAGNET `04274741d32e`, and EvoSkill-GUI `b3b976f391eb`. EvoSkill-GUI's three application submodules have not been initialized. Do not update these sources during an experiment without recording the new revisions.

KSI's `ksi --help` works, but `ksi-doctor` still reports three native-run blockers: missing `ksi-agent:bench` image, Node.js 22.16.x, and a provider profile. Its quickstart would build a Docker image and run API-backed ARC tasks; it has not been run. The optional vector-memory extra is also absent. The separate KSI-style adapter has completed one MathVision and one Vision2Code smoke sample using the benchmark environments.
