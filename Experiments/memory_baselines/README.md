# Multimodal memory baseline runner

2026-10-25：当前四记忆脚本已切换 MAGNET v2，默认新目录后缀 `_four_memory_v2`。采用问题/执行双描述、非重叠聚类、来源与裁图证据检查、局部去重；详见 [构建集验证报告](../../Docs/实验分析/20261025-01-MAGNET-v2-构建集验证报告.md)。旧 v1 产物保留用于审计，不能混用缓存。

当前新增五组表征实验：`no_memory`、`raw_multimodal`、`ksi_style`、`ksi_visual`、`magnet_adapted`。
KSI 两组共享知识文本、索引、排序和预算选择，只有历史视觉证据的注入不同。
MAGNET 采用成功轨迹筛选、多任务聚类/工作流抽象，以及先观察原图区域、再关联执行操作的视觉记忆流程。
该入口使用已有的 BGE embedding + VLM rerank；下方旧入口配置仍可用于查阅和恢复历史运行。

单构建样本 / 单测试样本的 Vision2Code 验证：

```bash
bash Experiments/memory_baselines/scripts/diagnostics/run_five_baselines_1_1.sh
```

从已完成的共同 100/100 no-memory 轨迹运行新五组 baseline：

```bash
bash Experiments/memory_baselines/scripts/experiments/run_five_baselines.sh vision2code qwen3_vl_8b_vision2code_8192_32768_v1_nomemory
```

新正式配置为 `config/qwen3_vl_8b_<benchmark>_five_baselines.json`；单样本配置为 `config/qwen3_vl_8b_five_baselines_1_1.json`。
单样本 runner 支持 `--stage prepare|build|audit|index|evaluate|all`，重复运行复用已有结果。
审查采用同一本地模型，不能代替人工核对；单来源不会强制生成 MAGNET 跨任务工作流。

当前模型为本地 Qwen3-VL-8B-Instruct。任务输出上限 8,192 tokens，记忆构建输出上限 32,768；上下文为 65,536。客户端流式生成，超时 1,800 秒，关闭自动重试。

## 当前入口

脚本按用途分类，服务脚本直接调用 vLLM，实验脚本直接调用 Python。完整参数和恢复方式见 [scripts/README.md](scripts/README.md)。

| Benchmark | 启动服务 | 运行实验 | 默认 GPU / 端口 |
|---|---|---|---|
| MathVision | scripts/serve/start_mathvision.sh | scripts/experiments/run_mathvision.sh | 4 / 8078 |
| Vision2Code | scripts/serve/start_vision2code.sh | scripts/experiments/run_vision2code.sh | 5 / 8079 |
| MP-DocVQA | scripts/serve/start_mpdocvqa.sh | scripts/experiments/run_mpdocvqa.sh | 6 / 8080 |

在仓库根目录运行，例如：

```bash
bash Experiments/memory_baselines/scripts/serve/start_mathvision.sh
# 在另一个终端执行：
bash Experiments/memory_baselines/scripts/experiments/run_mathvision.sh no-memory
```

实验脚本接受 `no-memory`、`formal`、`all`（默认）。`no-memory` 收集每个 benchmark 内 100 个记忆构建任务和 100 个独立测试任务的共同轨迹；`formal` 从同一批构建轨迹形成记忆并冻结，再评测 100 个测试任务，直接复用 no-memory 测试结果；`all` 顺序执行两个阶段。测试轨迹不作为记忆来源。

每个 benchmark 的配置在 `config/qwen3_vl_8b_<benchmark>_100_100.json`。基础提示词集中在 `vlm_memory/prompts/base_tasks.py`，MathVision 正式配置使用 `mathvision_concise_v1`，其余 benchmark 保留 `formal_100_100`；经验构建提示词按 baseline 分文件存储于同一 `prompts/` 目录，索引见 [提示词说明](vlm_memory/prompts/README.md)。不同 benchmark 使用独立服务、run ID 和产物目录。

## 当前 MathVision 诊断

批量 MathVision 已由用户停止。对构建集的 51 个截断样本进行类型分析，并固定 8 个典型构建集任务测试 `mathvision_concise_v1`。用户已选择该第一版提示词，复测停止于 6/8，完整结果保留；第二版未运行。MathVision 配置和默认 run ID 已切换，批量实验尚未启动。

```bash
bash Experiments/memory_baselines/scripts/diagnostics/mathvision_truncation.sh analyze
bash Experiments/memory_baselines/scripts/diagnostics/mathvision_truncation.sh all
```

分析和配对报告在 [Docs/实验分析](../../Docs/实验分析/20261024-01-MathVision-截断原因与提示词诊断.md)。这些样本按旧截断结果有意选择，不能将其正确率当作整个 benchmark 性能。测试集不参与提示词选择。

## 日志与恢复

日志在 `artifacts/logs/`；结果在 `artifacts/<benchmark>/<run>/`。逐任务保存输入、实际注入记忆、模型请求与响应、评分以及 token 用量。每 30 秒显示生成进度，任务完成后显示进度和 ETA。流式 chunk 数不等于 token 数，准确 token 数以最终 `usage` 为准。

重复原命令会复用完成结果；使用 `RETRY_FAILED=1` 显式重试失败任务。中断生成不能从半段文本继续，未完成输出保存在 `calls/*.partial.json`，重新发起前归档到 `partial_history/`。同一 run 的实验配置和提示词保持冻结，只有客户端超时可按现有恢复协议单独调整。

每次实验保存 config、实际提示词、划分清单和模型路径，不要求复制代码或保存代码版本。改变正式提示词、解码参数或 token 预算时使用新的 run ID。

## 方法与评价边界

新五组配置使用统一 embedding 检索与重排，经验预算默认 Top-k=3、最多 2 张历史图像、16,000 字符，评测期间冻结记忆库。KSI-style 与 MAGNET-adapted 均为推理任务适配，不能作为原论文完整复现分数。旧配置中的 Darwinian、BM25 与其他扩展分支仅保留历史兼容。

MathVision 使用本地官方评价器。Vision2Code 使用上游渲染和 rubric 评分，当前生成器与评分器为同一本地模型；评分不独立。MP-DocVQA 按页面读取与聚合协议评价。模型服务启动与实验执行独立。

## 历史入口

旧 27B、旧 8B 包装入口、10/5 pilot、smoke、早期单题诊断位于 `scripts/archive/`。迁移前完整说明保存在 [README_legacy.md](scripts/archive/README_legacy.md)。当前脚本不再调用这些历史 Shell 包装层。
