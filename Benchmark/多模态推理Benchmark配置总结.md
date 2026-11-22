# 多模态推理 Benchmark 配置总结

本文汇总 MATH-Vision、Vision2Code 与 MP-DocVQA Task 4 的数据、运行环境和本地模型验证。记录中的单样本结果用于确认评测链路可运行，不代表 benchmark 总体性能。不同 benchmark 的指标和协议不同，分数不可直接横向比较。

## 配置概览

| Benchmark | 任务与数据 | 环境和本地评测 | 当前验证 |
| --- | --- | --- | --- |
| MATH-Vision | 多学科图文数学推理；官方数据集 3,040 题，`testmini` 304 题 | Conda `mathvision`；复用本地 Qwen3-VL-8B-Instruct 接口 `http://127.0.0.1:8000/v1`；官方 evaluator 计算准确率 | 记忆基线 smoke 使用 `testmini` ID 4，8 种配置均为 0/1 |
| Vision2Code | 多领域视觉到代码；官方 `test_mini` 539 题、`test` 2,169 题；Kaggle 数据已下载并校验 | Conda `vision2code`；同一 Qwen 本地接口；上游生成/渲染/rubric 流程 | 记忆基线 smoke 在 `test_mini` 的一个 ChartQA 样本上完成 8 种配置；均成功渲染，评分见下表 |
| MP-DocVQA Task 4 | 多页文档问答；train 36,230、val 5,187、test 5,019，共 46,436 题 | Conda `mpdocvqa`；同一 Qwen 本地接口；原仓库 Exact Match、ANLS、APPA 指标 | val 索引 2，12 页；EM=1、ANLS=1.0、APPA=0 |

三个环境独立管理 benchmark 依赖，模型服务共享本地接口。记忆基线的配置、检索与提示代码集中在 [`Experiments/memory_baselines`](../../Experiments/memory_baselines/README.md)，目前不要求每个 baseline 单独建环境。

## 数据与结果详情

### MATH-Vision

单样本基线输入为 `testmini` ID `4`，采用官方 evaluator 判定答案。无记忆以及 7 种记忆配置都答错（0/1）。这仅说明生成、记忆注入和官方评分能完整运行；一题结果不能推断模型或记忆方法准确率。

### Vision2Code

记忆基线实验使用 `test_mini` 样本 `ChartQA_test_000509_2ac3907c97d2`，各配置使用同一生成模型、上游渲染器和数据集 rubric，并由同一本地 Qwen 模型评分。结果如下：

| 配置 | 代码渲染 | 本地 rubric 评分（0–5） |
| --- | --- | ---: |
| B0 No Memory | 成功 | 2.0 |
| B1 Raw Multimodal | 成功 | 3.0 |
| B1-T Raw Text | 成功 | 0.0 |
| B1-V Raw Visual | 成功 | 0.0 |
| B3 KSI-style | 成功 | 0.0 |
| B4 MAGNET-adapted | 成功 | 0.0 |
| B6 Darwinian-adapted | 成功 | 2.1 |
| B8 EvoSkill-adapted | 成功 | 0.0 |

另有一条使用 benchmark 原生入口的独立本地 smoke 记录：同为一个 `test_mini` 样本，但生成代码未能渲染，评分 0/5。它与上表的统一 baseline runner 是不同运行，不能混为同一结果。两次记录均只有一个样本，且记忆实验使用合成历史、同模型评分，不适合给方法排序或作性能结论。所有论文方法在该 runner 中是面向推理任务的适配实现，并非论文原生系统的完整复现。

### MP-DocVQA Task 4

原始归档来自 `/root/data/Dateset`，在 benchmark 下解析至 `Benchmark/MP-DocVQA-Framework/data/MP-DocVQA/`。校验确认 46,436 道题、5,929 个文档和 47,952 个引用页面；对应的图像和 OCR 文件均存在。抽样解码图像和解析 OCR 通过。train 有 5,253 道题引用超过 20 页（最长 40 页），验证时保留原始页面列表。原始 split 按文档 ID 有交叠，后续构建记忆需按文档隔离。

本地 Qwen 按 `page_ids` 顺序读取该 val 样本的全部 12 页，每次两页，再汇总读页结果；不使用 OCR、答案或答案页预筛选。预测答案 `ITC Limited` 与参考答案一致（EM=1、ANLS=1.0），但预测页索引为 0、标注答案页索引为 10，因此 APPA=0。共 7 次调用、15,554 tokens、约 17 秒。此本地分批协议不是原始 Hi-VT5 模型评测的复现。

## 记忆基线和实验边界

MATH-Vision 与 Vision2Code 当前各验证 B0、B1、B1-T、B1-V、B3、B4、B6、B8 八种配置。历史记忆来自 4 个独立合成任务，测试样本答案未用于构建或检索；记忆在评测期间冻结。KSI-style、MAGNET、Darwinian、EvoSkill 均为领域适配版。单样本验证用于检查配置、模型调用、检索/注入、渲染或指标流程，不是正式对比实验。

MP-DocVQA 当前完成数据解析、独立本地运行入口和单样本验证，尚未接入上述八种记忆基线。因此三项 benchmark 的“已配置”不表示实验矩阵进度相同。

## 复现入口与详细记录

- 基线配置与单样本结果：[Baseline-配置与单样本验证.md](Baseline-配置与单样本验证.md)
- 模型、预算和运行说明：[memory_baselines README](../../Experiments/memory_baselines/README.md)、[smoke 配置](../../Experiments/memory_baselines/config/smoke.json)
- MP-DocVQA 数据、运行和结果：[Task 4 配置与测试记录](20261020-MP-DocVQA配置与测试.md)、[本地使用说明](../../Benchmark/MP-DocVQA-Framework/README_LOCAL.md)
- 实验总体设计：[20261017 实验方案](20261017-实验方案.md)

