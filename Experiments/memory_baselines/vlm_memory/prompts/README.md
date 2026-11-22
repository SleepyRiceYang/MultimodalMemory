# 提示词索引

本目录只保存三个 benchmark 当前采用的基础提示词，以及各 baseline 的记忆构建和通用经验利用文本。提示词以 Python 常量和占位模板定义；检索、建库、更新、裁图、执行与评价流程仍由各自模块负责。

## 基础任务提示词

所有 baseline 在同一 benchmark 内使用同一基础提示词，经验文本和图像作为额外输入。

| Benchmark | 文件与定义 | 当前配置标识 | 输入 |
|---|---|---|---|
| MathVision | `base_tasks.py` → `MATHVISION_SYSTEM` | `mathvision_concise_v1` | `{task}`，当前题目、选项、图像 |
| Vision2Code | `base_tasks.py` → `VISION2CODE_SYSTEM` | `formal_100_100` | `{task}`，当前参考图像；`VISION2CODE_TASK` 保存固定任务说明 |
| MP-DocVQA 页面读取 | `base_tasks.py` → `MPDOCVQA_READ_SYSTEM` / `MPDOCVQA_READ_USER` | `formal_100_100` | `{question}`、`{page_indices}`、页面图像 |
| MP-DocVQA 答案聚合 | `base_tasks.py` → `MPDOCVQA_AGGREGATE_SYSTEM` / `MPDOCVQA_AGGREGATE_USER` | `formal_100_100` | `{question_json}`、`{readings_json}` |

`BASE_TASKS` 是唯一当前清单。读取入口为 `get_task_prompt(benchmark, version=None, stage=None)`；MP-DocVQA 显式选择 `read_pages` 或 `aggregate`。版本标识用于验证配置，不代表源码中保留多份历史文本。已删除版本会明确报错，不会自动换成最新提示词继续旧实验。

## 记忆构建与经验利用

| 文件 | 定义 | 用途 / 输入 |
|---|---|---|
| `ksi_style.py` | `REVIEW_SYSTEM` | 审阅历史轨迹和实际反馈，提出带来源的主张 |
| `ksi_style.py` | `DISTILL_SYSTEM` | 将历史轨迹、反馈和讨论提炼为知识条目 |
| `ksi_style.py` | `CROSS_TASK_SUFFIX` | 跨构建任务蒸馏补充要求，变量 `{task_count}`；与 `DISTILL_SYSTEM` 组合 |
| `ksi_style.py` | `PAIRED_REVIEW_SYSTEM` / `PAIRED_DISTILL_SYSTEM` | 当前五组协议的证据审阅与参数化提炼；允许无可靠经验时为空 |
| `ksi_visual.py` | `ATTACH_SYSTEM` | 为固定 KSI 节点关联源图视觉证据，不改写知识文本 |
| `magnet_construction.py` | `WORKFLOW_SYSTEM` | 从同簇至少两条成功来源抽象共同工作流 |
| `magnet_construction.py` | `REGION_SYSTEM` / `ASSOCIATION_SYSTEM` | 先仅观察原图区域，再绑定真实执行中的操作，减少代码描述污染视觉观察 |
| `magnet_construction.py` | `GROUP_SYSTEM` | 对候选作语义归组；保留实际候选和维护记录 |
| `memory_audit.py` | `AUDIT_SYSTEM` | 诊断性记忆核对，非独立正确性证明 |
| `magnet_adapted.py` | `EXTRACT_SYSTEM` | 从历史轨迹提取 workflow、视觉区域与 object/function |
| `magnet_adapted.py` | `WORKFLOW_PREFIX` / `PATCH_TEMPLATE` | 注入模型时的工作流与视觉证据文字格式 |
| `darwinian_adapted.py` | `EXTRACT_SYSTEM` | 提取前提、目标、轨迹步骤索引、验证条件 |
| `evoskill_adapted.py` | `BUILD_SYSTEM` / `CRITIC_SYSTEM` / `REVISION_SUFFIX` | 现有扩展 baseline 的技能构建、反馈诊断和修订指令 |
| `memory_context.py` | `MEMORY_NOTICE` | 所有记忆方法共享的经验利用说明 |
| `memory_context.py` | `ENTRY_TEMPLATE` 等 | 经验条目、图像来源和技能段落的展示模板 |

记忆构建调用的动态用户内容是序列化的 `episodes`、`extra` 或跨任务知识列表；这些实际内容不是固定提示词。文本中的 JSON 示例随相应构建提示词存储；结构化输出校验 schema 仍由构建模块维护。

No Memory 不构建经验；Raw Multimodal 直接保留来源轨迹，没有额外 LLM 提炼指令。旧 `magnet_adapted.py` 的单条提取模板及 Darwinian/EvoSkill 模板为历史兼容；当前五组协议使用上表新增模板。

## 评价提示词

`evaluation.py` 提供 `vision2code_rater_prompts()` 统一入口，复用本地官方 `Benchmark/Vision2Code/vision2code/evaluation/dataset_rubrics.py` 的评分 system 文本、按数据集生成的用户内容和 JSON 修复提示词。评分文本与解题文本分开，官方 rubric 表保留单一来源，避免复制后出现不同评分标准。该入口采用延迟导入，普通任务提示词无需加载评价依赖。

MathVision 的规则评分和 MP-DocVQA 的 ANLS / exact match 没有额外 LLM 评分提示词。

## 修改与保存

- 固定指令在本目录定义，流程代码只填入变量和实际经验。
- 本次仅迁移当前采用的文本，未重新优化提示词或发起实验。
- 新运行保存所选提示词的快照，不再保存未使用的历史 profile。已有实验快照保留不变，可继续查阅旧文本和历史结果。
- 历史提示词对照入口只提供报告读取；不会从源码恢复已删除的旧版本进行新生成。
- 修改当前文本时使用新 run ID，避免与旧结果混用；不要求每次保存代码版本或复制代码。

## MAGNET v2（2026-10-25）

固定提示词集中在 `magnet_v2.py`：PROBLEM（仅公开题面和原图）、ANCHORED_PROCEDURE（选择原始片段编号）、ALIGNED_WORKFLOWS（对齐每个来源的实际操作）、VERIFY_BATCH（逐步骤核查）、CANDIDATE_DESCRIPTIONS / RELATIONS（短描述及局部去重）、INSPECT_CROP / MATCH_CROP（不看候选说明的裁图盲读及匹配）。区域建议及功能关联继续复用 `magnet_construction.py` 的 REGION_SYSTEM / ASSOCIATION_SYSTEM，实际支持必须绑定来源操作。旧全量分组模板只保留用于历史兼容。
