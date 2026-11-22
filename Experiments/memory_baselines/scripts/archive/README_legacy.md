> 历史说明，路径以迁移前项目目录为基准；当前入口见 `scripts/README.md`。

# Multimodal memory baseline runner

当前新增 Qwen3-VL-8B 双单卡服务配置：MathVision 默认 GPU 4 / 8078，Vision2Code 默认 GPU 5 / 8079；任务输出 8,192 tokens，记忆构建输出 32,768 tokens，上下文 65,536。新脚本、独立配置、运行和恢复方式见 [Qwen3-VL-8B 双服务 100/100 协议](../../Docs/实验方案/20261023-Qwen3VL8B-双服务100-100.md)。

MP-DocVQA 现也可使用单卡 8B 流程，默认 GPU 6 / 8080，100/100 固定划分与运行命令见 [MP-DocVQA 100/100 协议](../../Docs/实验方案/20261023-Qwen3VL8B-MPDocVQA-100-100.md)。

## 长输出超时与恢复

本地生成使用流式响应，正式配置 `model.timeout` 设为 1,800 秒，控制等待首个响应或相邻读取之间的超时，
客户端关闭自动重试，避免超时后重复提交长任务。
每 30 秒打印已接收的 chunk 数和字符数；chunk 数不等于 token 数，准确 token 数以最终 `usage` 为准。
`calls/*.partial.json` 保存最近的生成进度及异常，只有完整响应才写入可复用的 `calls/*.json`。
中断的生成会重新请求，不能从半段文本接着解码；重试前的输出归档到 `calls/partial_history/`。

MathVision 专用入口默认携带 `--retry-failed`：重新执行
`bash Experiments/memory_baselines/scripts/run_qwen3_8_27b_mathvision_100_100.sh all`
会复用完成任务并重新执行失败任务，旧错误记录保留在任务的 `errors/` 下。
设置 `RETRY_FAILED=0` 可关闭这一行为。
服务启动脚本支持 `MAX_NUM_SEQS`，默认 1；此选项仅在下次启动服务时生效。
当前 262144 上下文、60% 显存预算下，并发 2 的启动实测因 KV cache 显存不足而失败；
提高并发需要重新规划显存或上下文预算。
这些修改保留已冻结的提示词、温度和输出 token 上限。
同一 run 可调整 `model.timeout`，原快照保持不变，实际超时写入 `runtime_transport.json`
及生成记录；其他实验配置变化仍要求使用新的 run ID。

## Qwen3.8-27B 固定 100/100 实验

当前选中配置为 `base_v2`：MathVision 使用新版并去掉 `in at most five short steps`；
Vision2Code 恢复原始 `old`。输出预算按用户配置为 **32678**。
当前只执行原五题的 v1/v2 同预算复测，100/100 保持停止。复测命令：

```bash
bash Experiments/memory_baselines/scripts/run_qwen3_8_27b_100_100.sh prompt-retest \
  --run qwen38_base_v2_32678_5sample
```

每个 benchmark 同样五题、两种提示词各一次，共 20 次新生成。上轮分数只列为历史参考，
报告在全部结果完成后写入 `Docs/实验分析/20261023-qwen38_base_v2_32678_5sample-复测对比报告.md`。

配置为 `config/qwen3_8_27b_100_100.json`，服务为 `127.0.0.1:8077`。
统一基础提示词在 `vlm_memory/prompts/base_prompts.py`，保留 old/base_v1；
经验构建提示词仍在 `templates.py`。执行协议见
[`20261023-Qwen38-100-100执行协议.md`](../../Docs/实验方案/20261023-Qwen38-100-100执行协议.md)。

```bash
# 每个 benchmark 从构建集抽 5 题，旧/新提示词共 20 次生成；测试集不参与选择。
bash Experiments/memory_baselines/scripts/run_qwen3_8_27b_100_100.sh prompt-pilot \
  --run qwen3_8_27b_fullbudget_prompt_20261023
# 每个 benchmark 的构建集与测试集各 100 题，共 400 个 no-memory 任务。
bash Experiments/memory_baselines/scripts/run_qwen3_8_27b_100_100.sh no-memory \
  --run qwen3_8_27b_base_v1_100_100
# 已有共同轨迹完成后，准备/构建记忆/评测；no-memory 结果复用。
bash Experiments/memory_baselines/scripts/run_qwen3_8_27b_100_100.sh formal \
  --source-run qwen3_8_27b_base_v1_100_100 --run qwen3_8_27b_memory_100_100 --stage all
```

同一命令重复执行会复用完成结果；失败任务用 `--retry-failed` 显式重试。
配置/提示词变化使用新 run ID。日志保存到 `artifacts/logs/`，每 30 秒打印等待状态，
任务结束后显示进度和 ETA。逐任务 JSON 原子写入；已保存的生成响应可在评分恢复时复用。
构建集轨迹和测试集轨迹分目录保存，记忆方法只导入构建集轨迹。报告写入 `Docs/实验分析`。
输出预算达到服务上下文上限时，先通过本地 `/tokenize` 计算输入长度，为输入预留上下文。
请求超时仍由配置 `timeout` 决定；超时不会当成任务答错。

## 真实 benchmark 内 10/5 pilot

新增批量入口按 benchmark 独立选择 10 个历史任务和 5 个 held-out 测试任务，运行
B0、B1、B3、B4-adapted、B6-adapted。配置见 [pilot_10_5.json](config/pilot_10_5.json)。
数据选择只读取公共任务字段，参考答案在生成完成后由评价端读取。
MathVision 使用官方 test 池内部划分；Vision2Code 从本地有已发布图像的 test-mini 池划分；
MP-DocVQA 使用文档 ID 不重叠的 train/val 子集且文档最多 6 页，所有页面均读取。
这是小规模流程与案例发现实验，不是官方完整 split 的性能评测。

先使用 `Benchmark/start_qwen3vl_vllm.sh` 启动本地服务，建议上下文 32768、
`--limit-mm-per-prompt '{"image":16}'`；选择 GPU 7 前确认显存可用。
批量驱动使用 `vision2code` 环境，MathVision 与 MP-DocVQA 的官方评价器分别在其
已有独立 Conda 环境中执行。以仓库根目录为工作目录：

```bash
PILOT_RUN=my_new_pilot bash Experiments/memory_baselines/scripts/run_pilot_10_5.sh
# 已完成的任务/建库单元可以跳过；失败的测试 attempt 不自动覆盖。
PILOT_RUN=my_new_pilot bash Experiments/memory_baselines/scripts/run_pilot_10_5.sh --resume
```

单阶段运行：

```bash
export PYTHONPATH=Experiments/memory_baselines
/root/anaconda3/envs/vision2code/bin/python -m vlm_memory.pipeline.pilot \
  --benchmark mathvision --run my_new_pilot --stage prepare
# 同一入口可选 collect / build / evaluate / all。
```

评测阶段支持 `--task-ids <ID...>` 运行不重叠的任务分片，并可通过
`--claim-pending` 原子预留结果文件。未完成的分片只写 `checkpoint.json`；
完整矩阵才写通过状态的 `validation.json`。`seed_0` 表示本轮的单次重复编号，
调用温度为 0，API 未显式设置服务端随机种子。

产物位于 `artifacts/<benchmark>/<run>/`：`dataset_manifest.json` 保存任务与划分，
`evaluation_targets/` 独立存放文档问答金标，`histories/` 保存 10 个来源任务及逐源构建调用，
`bank.json` 为合并并冻结的记忆快照，`runs/<method>/<task>/seed_0/attempt_0/`
保存输入、实际注入、模型请求/响应和官方评分。B3 在逐源蒸馏后进一步跨这 10 个任务
蒸馏知识；B4/B6 保留各自的 crop 或前提—目标—轨迹结构。允许空检索并保留失败输出。

离线分析命令为 `python -m vlm_memory.analysis.pilot_report`，可分别指定
`--math-run`、`--mp-run`、`--vision-run` 和 `--output`。分析导出逐样本 CSV、配对差异、
记忆/成本统计、案例路径及 Markdown 报告。Vision2Code 的 rubric 由同一本地模型评分，
报告中显式披露此限制，不作为独立评价结果。

下文保留原先八配置单样本 smoke 的运行说明与实现边界。

本项目在同一冻结 Qwen3-VL-8B、同一当前任务输入和同一评价器下运行 8 个配置。入口配置为 [config/smoke.json](config/smoke.json)。共享模型服务位于 `http://127.0.0.1:8000/v1`；MathVision 使用 `mathvision` Conda 环境，Vision2Code 使用 `vision2code` 环境。轻量适配层不需要为每个 baseline 再建立环境，也不启动 Docker。

| ID / 配置名 | 注入的历史经验 | 实现范围 |
| --- | --- | --- |
| B0 `no_memory` | 无 | 自建参照；始终保留当前任务图像 |
| B1 `raw_multimodal` | 原始任务、响应、动作/执行步骤、反馈及有序图像 | 自建原始多模态轨迹对照 |
| B1-T `raw_text` | 同一批轨迹的完整文本部分 | 不总结，不注入历史图像 |
| B1-V `raw_visual` | 同一批轨迹的有序图像 | 不注入历史任务文本、答案和动作文本；源任务描述仅用于检索索引 |
| B3 `ksi_style` | 经验主张、适用/排除条件、证据引用、失败假设与检查项 | 来源反馈审阅 → 结构化知识蒸馏；纯文本 |
| B4 `magnet_adapted` | Workflow 和 Object–Function 对应的视觉 Crop | 模型提取语义与归一化框，实际裁图；分别保存流程条目与补丁条目 |
| B6 `darwinian_adapted` | Precondition、Goal、子任务步骤及源图像 | 根据实际成功/失败统计平滑可靠性，按相关性、可靠性和步数排序，并过滤高风险单元 |
| B8 `evoskill_adapted` | `plan.md`、`backup.md`、`recover.md`、失败案例 | 初始技能 → 只看任务/图像/反馈的独立 critic → 离线修订；评测阶段冻结 |

四个论文方法均为推理领域适配版，身份写入结果。原生 KSI 的容器论坛编排、MAGNET 的图检索/视觉 embedding、Darwinian 的完整 Planner–Actor 与突变/淘汰机制、EvoSkill 的 GUI 文件工具及测试时多轮执行未在此复现。当前检索为可审计 BM25；Darwinian 另有可靠性/效率排序。方法差异还体现在记忆构建、条目粒度、图像裁剪、技能文件与反馈修订中。这些适配结果不能作为原论文复现分数。

## 工程结构

```text
config/                  # 模型、检索/视觉预算、冻结开关及方法矩阵
vlm_memory/
  models/                # OpenAI-compatible 本地 VLM；图像编码、用量和耗时
  prompts/               # 执行、构建、蒸馏与修订提示词
  memory/                # 源轨迹 → 方法记忆；BM25、风险排序、预算与冻结检查
  benchmarks/            # 官方数据加载、MathVision 评价及 Vision2Code 渲染/评分
  scripts/               # 历史收集、记忆构建、评测与报告
scripts/                 # Shell 一键入口
tests/                   # 模态隔离、泄漏拒绝、预算、冻结与风险排序测试
artifacts/<run>/          # 配置、原始调用、记忆库、Crop、技能和每方法输出
```

## 安装和运行

在仓库根目录安装适配层（复用已配置的两个 benchmark 环境）：

```bash
/root/anaconda3/envs/mathvision/bin/python -m pip install --no-deps -e Experiments/memory_baselines
/root/anaconda3/envs/vision2code/bin/python -m pip install --no-deps -e Experiments/memory_baselines
cd Experiments/memory_baselines
bash scripts/run_all_smoke.sh
```

服务未启动时先运行 `bash Benchmark/start_qwen3vl_vllm.sh`（在仓库根目录）。该脚本默认 GPU 6；服务已在运行时直接复用。

各步骤也可单独运行（`RUN` 为新输出目录）：

```bash
RUN=artifacts/my_run
conda run -n vision2code python -m vlm_memory.scripts.prepare_history --artifacts "$RUN"
conda run -n vision2code python -m vlm_memory.scripts.build_memories --artifacts "$RUN"
conda run -n mathvision python -m vlm_memory.scripts.run_smoke --benchmark mathvision --artifacts "$RUN"
conda run -n vision2code python -m vlm_memory.scripts.run_smoke --benchmark vision2code --artifacts "$RUN"
conda run -n vision2code python -m vlm_memory.scripts.summarize --artifacts "$RUN" --require-complete
conda run -n vision2code python -m unittest discover -s tests -v
```

`--method` 可选择单个配置，`--index` 选择样本。`--resume` 跳过已有结果；失败记录不自动覆盖，应保留现场后使用新目录复测。

## 数据与评价协议

Smoke history 由程序独立生成：形状计数、矩形面积、两张条形图；模型真实作答/生成代码并记录实际反馈。它不使用 benchmark 题目、答案、历史 GPT caption 或测试轨迹。图表历史的成功仅表示可渲染，尚未验证语义保真度。所有方法使用相同源历史；构建调用、critic、revision 和证据 ID 均保留。

评测使用 MATH-Vision `testmini` 第一个样本 ID `4` 与 Vision2Code `test_mini` 第一个 ChartQA 样本。参考答案仅在 MathVision 官方 evaluator 子进程内读取。库中源任务 ID 或图像 SHA256 与评测输入重合时拒绝运行；近重复检查仍需正式数据集治理。记忆预算默认 Top-k=3、历史图像最多 2 张、文本最多 6000 字符；超预算按整条跳过，图像按轨迹顺序截取，日志标明截断。

MathVision 调用原仓库评价器。Vision2Code 调用原仓库代码规范化器、渲染器和按数据集的 rubric 评分器；本地 Qwen 同时充当生成器和评分器，仅用于接口验证。所有配置使用统一的明确保存提示词 `plt.savefig(OUTPUT_PATH)`，这与上游原提示词存在文字差异，已在 `prompts/` 版本化。生成代码不做自动补写。像素相似度仅是诊断指标。

正式比较前需用许可允许、与评测集隔离的真实历史经验替换 smoke history，并确定独立评分器和方法适配协议。本次单样本结果只能证明管线和实际注入行为，不支持方法性能结论。UI-Mem、EpiCon、SkillLens 本轮不运行。

## 上游来源

- KSI：`Methods/KSI`，`ac709263bc3d`，参考 `src/ksi/distillation/prompts.py` 的证据驱动知识结构。
- MAGNET：`Methods/MAGNET`，`04274741d32e`，参考 `src/memory/stores/{procedural_mem,stationary_mem}.py` 的流程与静态视觉记忆。
- EvoSkill-GUI：`Methods/EvoSkill-GUI`，`b3b976f391eb`，参考 `src/mobile_world/skills/` 的技能文件、隔离 critic 与修订流程。
- Darwinian：[正式论文](https://proceedings.mlr.press/v306/mi26b.html)提供 [DMS-C48C](https://anonymous.4open.science/r/DMS-C48C/)；其 `/api/repo/DMS-C48C/files/` 当前返回 HTTP 410 `repository_expired`。本项目依据论文进行适配。
