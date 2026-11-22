# 脚本入口

当前五组表征验证：`diagnostics/run_five_baselines_1_1.sh`，支持 `--stage build|audit|index|evaluate|all`；服务仍使用 `serve/start_vision2code.sh` 和 `serve/start_embedding.sh`。

当前五组正式入口：`experiments/run_five_baselines.sh <benchmark> <已完成的共同 no-memory run ID>`。
它直接调用 Python，使用 `config/qwen3_vl_8b_<benchmark>_five_baselines.json`，覆盖 No Memory、Raw Multimodal、KSI 文本、KSI 文本＋视觉、MAGNET-adapted。
默认写入新的 `<source-run>_five_representations_v1` 目录，可用 `FORMAL_RUN` 指定新目录；不会启动 100/100 no-memory 收集。

三个模型启动脚本默认启用 xgrammar 的 `disable_any_whitespace=true`，防止结构化 JSON 在字段之间重复生成换行直至截断。可用 `STRICT_JSON_WHITESPACE=0` 恢复旧格式；已运行服务需要重新启动才生效。正常任务的自由文本/代码生成不受这个 JSON 格式选项约束。

从仓库根目录运行。当前脚本按用途分目录；`serve/` 直接执行 vLLM，`experiments/` 和 `diagnostics/` 直接调用 Python 模块。当前入口之间不调用其他 Shell 脚本。

| 用途 | MathVision | Vision2Code | MP-DocVQA |
|---|---|---|---|
| 模型服务 | serve/start_mathvision.sh | serve/start_vision2code.sh | serve/start_mpdocvqa.sh |
| 100/100 实验 | experiments/run_mathvision.sh | experiments/run_vision2code.sh | experiments/run_mpdocvqa.sh |
| GPU / 端口默认值 | 4 / 8078 | 5 / 8079 | 6 / 8080 |

例如分别在两个终端启动服务和实验：

```bash
bash Experiments/memory_baselines/scripts/serve/start_mathvision.sh
bash Experiments/memory_baselines/scripts/experiments/run_mathvision.sh no-memory
```

实验阶段：`no-memory` 收集 100 个构建任务与 100 个测试任务的共同轨迹；`formal` 在完整共同轨迹上建库、冻结、评测；`all` 依次执行二者（默认）。重复同一命令恢复，`RETRY_FAILED=1` 显式重试失败任务。`FORMAL_CONFIG`、`SOURCE_RUN`、`FORMAL_RUN`、`EXPERIMENT_PYTHON`、`LOG_PATH` 可覆盖默认设置。仅调整提示词诊断不会自动恢复批量实验。

服务可通过 `MODEL_PATH`、`GPU_INDEX`、`PORT`、`MAX_MODEL_LEN`、`GPU_MEMORY_UTILIZATION` 覆盖默认参数；端口和模型名须与实验配置对应。服务默认显存比例 0.8、单卡 TP=1、并发 1、上下文 65,536。

MathVision 截断分析与固定 8 个构建集样本复测：

```bash
bash Experiments/memory_baselines/scripts/diagnostics/mathvision_truncation.sh analyze
bash Experiments/memory_baselines/scripts/diagnostics/mathvision_truncation.sh all
```

`analyze` 只读取已有结果并生成报告；`test` / `all` 运行候选提示词，复用原提示词的已保存结果进行配对比较。每个新提示词使用独立 `PROFILE` 和 `RUN_ID`；采样和输出预算保持固定。MathVision 已按用户决定采用 `mathvision_concise_v1`；本次复测已停止，第二版未运行。诊断命令仅在显式执行时才会发起生成。

旧 27B、旧 8B 包装入口、10/5 pilot、smoke 和 1789 单题诊断脚本都移到 `archive/`，内部项目根目录已适配新位置。历史文档中的旧路径是迁移前路径；后续实验使用上表入口。

实验保存 config、实际提示词、划分清单、模型路径和运行结果；不要求每次复制代码或保存代码版本。

MathVision 当前默认 run ID 包含 `mathvision_concise_v1`，与旧提示词结果分开；执行 `no-memory` 将按新提示词收集共同轨迹。Vision2Code 和 MP-DocVQA 保留现有配置与 run ID。


统一向量检索与 KSI 5/5 小测（分别在终端中运行服务和实验）：

```bash
# GPU 7，BGE-large-en-v1.5，端口 8087
bash Experiments/memory_baselines/scripts/serve/start_embedding.sh
# GPU 5，Qwen3-VL-8B，端口 8079；已有服务时不重复启动
bash Experiments/memory_baselines/scripts/serve/start_vision2code.sh
bash Experiments/memory_baselines/scripts/diagnostics/run_ksi_vision2code_5_5.sh
```

新检索由配置 `retrieval.enabled=true` 启用；索引与调用轨迹保存在 run 的 `retrieval/`。小测复用已完成的 no-memory 轨迹，重新构建 KSI 记忆并评测；重复实验命令会验证固定配置与划分，跳过已完成调用。五个最大类别各抽一个样本，未包含 Spatial，不代表全量 benchmark 分布。旧 100/100 入口不自动切换检索协议；正式使用新协议时须新建配置与 run ID。

Embedding 服务独立运行在 `vllm_env`，使用本地 BGE 的 CLS pooling 和 L2 归一化；实验进程通过 HTTP 调用，无需在 Vision2Code 环境安装 torch。query 使用 BGE 的检索指令前缀，索引不加该前缀；服务拒绝超过 512 tokens 的输入。

## 四组 Memory Baseline 正式实验

四组实验配置只运行 `raw_multimodal`、`ksi_style`、`ksi_visual`、`magnet_adapted`；它们使用已经完成的共同 no-memory 构建/测试轨迹，不会额外运行 No Memory。当前资源映射如下：

| 服务 | GPU | 端口 | 启动脚本 |
|---|---:|---:|---|
| MathVision Qwen3-VL-8B | 4 | 8104 | `serve/start_mathvision_four_memory.sh` |
| Vision2Code Qwen3-VL-8B | 5 | 8105 | `serve/start_vision2code_four_memory.sh` |
| MP-DocVQA Qwen3-VL-8B | 7 | 8107 | `serve/start_mpdocvqa_four_memory.sh` |
| BGE-large-en-v1.5 embedding | 7 | 8087 | `serve/start_embedding.sh` |

分别在四个终端启动服务，然后在另外的终端启动对应 benchmark 实验：

```bash
bash Experiments/memory_baselines/scripts/serve/start_mathvision_four_memory.sh
bash Experiments/memory_baselines/scripts/serve/start_vision2code_four_memory.sh
bash Experiments/memory_baselines/scripts/serve/start_mpdocvqa_four_memory.sh
GPU_INDEX=7 bash Experiments/memory_baselines/scripts/serve/start_embedding.sh

bash Experiments/memory_baselines/scripts/experiments/run_mathvision_four_memory.sh qwen3_vl_8b_mathvision_concise_v1_8192_32768_nomemory
bash Experiments/memory_baselines/scripts/experiments/run_vision2code_four_memory.sh qwen3_vl_8b_vision2code_8192_32768_v1_nomemory
bash Experiments/memory_baselines/scripts/experiments/run_mpdocvqa_four_memory.sh qwen3_vl_8b_mpdocvqa_8192_32768_v1_nomemory
```

以上三个源 run 均已完成 100 个记忆构建任务和 100 个测试任务，汇总文件的 `complete` 为 `true`、`errors` 为空。MathVision 对应 `mathvision_concise_v1`，Vision2Code 与 MP-DocVQA 对应 `formal_100_100`。每个 benchmark 使用与其 fixed manifest 和求解模型设置一致的源 run；三组可分别在终端运行，断点恢复沿用相同命令。使用 `FORMAL_RUN` 可以指定新的目标 run ID；默认目标 ID 为 `<source-run-id>_four_memory_v2`。专用配置位于 `config/qwen3_vl_8b_<benchmark>_four_memory_baselines_v2.json`。模型服务地址属于运行时连接设置，因此可与源 run 使用不同端口；模型路径、提示词、输出预算和划分仍需匹配。

embedding 脚本默认使用 GPU 7；可用 `GPU_INDEX`、`PORT`、`EMBEDDING_MODEL` 覆盖。三张 vLLM 服务使用独立端口，避免与既有 8078/8079/8080 服务冲突。

### 结构化响应解析失败时的恢复

四组配置的 `memory_construction.parse_retries` 默认是 3，即首次请求后最多再试 3 次。每次响应都检查 JSON 可解析性和 schema 必需字段（例如 KSI review 必须包含 `claims`）。响应缺失字段、JSON 无效或输出被截断时会重试；仍失败则写入对应 construction call 的 `.failed.json`，跳过该条记忆，继续构建其他样本。原始失败响应和每次错误保存在 `.attempt_N.json` / `.attempt_N.error.json` 中。

可直接用原命令恢复，不需要换 run ID：

```bash
bash Experiments/memory_baselines/scripts/experiments/run_mathvision_four_memory.sh qwen3_vl_8b_mathvision_concise_v1_8192_32768_nomemory
```

新的解析重试参数属于运行恢复设置，不改变源轨迹、提示词或评测划分；已有运行快照可以继续使用。若达到重试上限仍失败，该来源会被跳过并保留诊断记录，最终请检查 `artifacts/mathvision/<formal-run-id>/construction_calls/*.failed.json`，确认 KSI 记忆覆盖情况。

### MAGNET v2 构建集验证（2026-10-25）

当前四记忆脚本使用新版问题转述、带来源编号的执行摘要、非重叠 complete-link 聚类、逐步骤证据校验和局部去重。门槛来自构建集内部诊断，记录在各新版 config。改用 v2 必须使用新目录，不能将旧 v1 聚类/工作流缓存当成同一实验恢复；原始 no-memory 来源轨迹继续复用。

诊断入口：`bash Experiments/memory_baselines/scripts/diagnostics/validate_magnet_v2.sh <benchmark> <stage>`。阶段包含 describe、calibrate、workflows、visual、replay、integration、all；replay 仅对 Vision2Code 回放旧 118 个候选。所有诊断任务均取自原记忆构建集。报告见 `Docs/实验分析/20261025-01-MAGNET-v2-构建集验证报告.md`。

### 当前测试入口：embedding cosine（2026-10-25）

`experiments/run_mathvision_embedding_test.sh`、`run_vision2code_embedding_test.sh`、`run_mpdocvqa_embedding_test.sh` 分别复用对应 `_four_memory_v2` 冻结库，在 `_embedding_v1` 新目录测试四组 memory；不重建记忆、不复用旧 memory 成绩、不调用大模型 rerank。三个终端可并行。追加 `--prepare-only` 只准备，正常中断后原命令恢复，有错误结果时追加 `--retry-failed`。过程日志和 ETA 沿用正式评测器。

### GUI Agent 五任务测试（2026-10-26）

- `bash Experiments/memory_baselines/scripts/experiments/run_androidworld_5.sh`：GPU 4 / 8104，AndroidWorld HTTP 15000，官方 JSONAction 与 TaskEval 验证器；截图单步 JSON 代理，无历史记忆；默认通过本项目 Android 容器的只读 adb screencap 获取 PNG，动作/评分仍走官方 HTTP，动作后等待 2 秒。默认结果目录带 `_png_v1`；此前逐像素 HTTP 截图慢路径保留为诊断，不混入成绩。环境启动用 `bash Benchmark/AndroidWorld/docker_setup/start_android_world.sh`，不仅检查 health，还须确认 adb 设备与 boot_completed=1。失败初始化可用 `--retry-failed` 恢复并保留错误记录。
- `bash Experiments/memory_baselines/scripts/experiments/run_osworld_5.sh`：GPU 5 / 8105，复用本地 Ubuntu Docker VM，使用 OSWorld 原生 QwenAgent、官方任务和评分器。沿用官方结果恢复机制；不会启动/停止现有 vLLM。
- 两组各 5 个预先固定的离线任务；每任务最多 30 轮、每轮输出 2,048 tokens。用于端到端 smoke 验证，不代表完整 benchmark 或五组 memory baseline 对比。划分清单在 `splits/gui_smoke_5_seed20261026/`，结果分别在 `artifacts/androidworld/` 和 `artifacts/osworld/`。
- embedding 启动脚本默认 GPU 7，仍为 8087，可用 GPU_INDEX 覆盖。
