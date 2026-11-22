# 多模态推理记忆构建集 / 测试集（100/100）

## 设计

固定随机种子 `20261023`。每个 benchmark 选择 100 个记忆构建任务和 100 个测试任务；不建立 Development 集。相同的 100 条记忆构建任务轨迹供所有 baseline 使用，各方法分别把这些共同轨迹转换为自己的记忆。测试阶段读取已冻结记忆库，禁止写入测试反馈。

| Benchmark | Memory Construction | Test | 分层方式 |
| --- | ---: | ---: | --- |
| MATH-Vision | 100 题 | 100 题 | 按 5 个官方难度级别按可用池比例分层；每个集合覆盖全部 16 个主题 |
| MP-DocVQA | 100 个问题 / 100 篇文档 | 100 个问题 / 100 篇文档 | 按题目输入页数分为 1–2、3–5、6–10、11–20 页四档；两集合使用相同配额并按 `doc_id` 隔离 |
| Vision2Code | 100 张图 | 100 张图 | 按本地可用的来源数据集比例分层；两集合来源配额相同，覆盖本地可用的六个领域 |

逐任务 ID、输入字段、图像路径、哈希和评测目标保存在各 benchmark 子目录的 `dataset_manifest.json` 与 `evaluation_targets/`。评测目标单独存放，不属于模型输入或检索索引。

## 划分依据与限制

- **MATH-Vision：** 使用完整 `test.jsonl` 作为自定义任务池；从未用于先前 pilot 的样本中按难度分层，两个集合每个难度的配额均为：Level 1 = 8、Level 2 = 24、Level 3 = 18、Level 4 = 27、Level 5 = 23。每个集合均覆盖全部 16 个主题。排除 10/5 pilot 已使用的 task ID 和图像哈希。MATH-Vision 没有官方训练集，因此这是 benchmark 内部的自定义记忆泛化协议，不能作为官方 3,040 题完整测试分数。
- **MP-DocVQA：** 从官方 train 构建记忆，从有本地答案可供评分的 validation 选测试任务；每个集合的页数档配额均为：1–2 页 = 51、3–5 页 = 19、6–10 页 = 13、11–20 页 = 17。每个文档最多选一个问题；记忆集和测试集的 `doc_id` 不重合，并排除 pilot 中用过的文档。筛选上限为 20 页，覆盖原 benchmark 的多页范围。官方 test 文件在当前本地副本中没有答案，因此本轮选择 validation 以便本地评分；该集是 pilot held-out set，不是未公开的官方封存测试集。
- **Vision2Code：** 本地 release 有 2,169 个官方 test 条目，其中 1,994 张图像已发布；排除此前 pilot 使用的条目/图像哈希后，本轮候选为 1,977 张。两个集合各 100 张，按来源数据集同配额分层，共覆盖 14 个本地有测试图像的来源数据集和六个领域。当前 release 未包含 source-train 图像，所以记忆构建样本也来自官方 test 池；这是 custom within-benchmark split，不能报告为官方 Vision2Code leaderboard 分数。测试/记忆样本按 ID 和图像哈希互斥。
- 三个 benchmark 均排除此前 10/5 pilot 中用过的样本；MP-DocVQA 另外排除其所属文档。完成了 ID、图像哈希和文档级隔离，但没有全面进行语义/模板近重复人工审查。

## 构建和重现

```bash
python3 Experiments/memory_baselines/vlm_memory/pipeline/build_100_100_splits.py
```

生成器：[build_100_100_splits.py](../../vlm_memory/pipeline/build_100_100_splits.py)。汇总统计：[selection_summary.json](selection_summary.json)。运行配置：[memory_generalization_100_100.json](../../config/memory_generalization_100_100.json)会让 pipeline 直接载入这些固定清单和独立评测目标，不重新抽样。

例如，按该配置启动 MathVision 100/100 实验：

```bash
python -m vlm_memory.pipeline.pilot --benchmark mathvision --run memory_100_100_v1 --config Experiments/memory_baselines/config/memory_generalization_100_100.json --stage all
```

MP-DocVQA 和 Vision2Code 使用相同命令结构，只需替换 `--benchmark`。
