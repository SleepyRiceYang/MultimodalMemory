# AndroidWorld / OSWorld：各 50 构建任务 + 50 测试任务

生成日期 2026-10-26；固定划分 seed `20261026`。本划分是项目自定义的 **Memory Construction / Held-out Test 两分法**，不是 benchmark 官方 train/test，也不设 Development。记忆只在构建集形成和更新；测试集冻结读取，不写回。每个 benchmark 各自建库，不跨 benchmark 混合。脚本 `build_split.py` 只读取本地 benchmark 元数据，不读取模型分数、轨迹成功率或测试结果。

| Benchmark | 单位 | 构建 | 测试 | 难度平衡依据 |
|---|---|---:|---:|---|
| AndroidWorld | 不同任务类型，各 1 个任务实例 | 50 | 50 | 任务原生 complexity，项目三档 |
| OSWorld | 不同任务 ID | 50 | 50 | 项目定义的结构复杂度代理，按 domain 内三分位 |

## AndroidWorld

数据源：`Docs/实验方案/assets/20261026-gui-protocol/androidworld_registry_stats.json`，对应本地注册表 116 种任务类型。先排除之前 GUI 五样本来源、五样本测试中见过的10种类型及其存在的4个 `Verify` 变体，共14种；余102种。另排除两个编号重复变体 `ExpenseDeleteMultiple2` 与 `RecipeAddMultipleRecipesFromMarkor2`，以保留原始任务并使 complexity 三档总数为偶数。100种类型全部分配到两侧，固定 Android suite seed `20261026`、`task_idx=0`。**这是类型留出**：相同类型的不同 seed 实例不得跨两侧补充，否则泛化单位会改变。

项目难度档：Easy `complexity≤2`、Medium `2<complexity≤4`、Hard `>4`。`complexity` 是任务类中的原生数值；三档是本项目定义，不称作官方 Easy/Medium/Hard。对近似任务族及 `Verify` 变体实施同侧约束，再精确平衡难度。两侧各 **36 Easy、9 Medium、5 Hard**；每个应用标签（多应用任务计入每个标签）的计数差最多1。此前见过的 `ClockStopWatchRunning` 的保留近变体 `ClockStopWatchPausedVerify` 被约束在构建侧。

清单：[androidworld.json](androidworld.json)：`memory_construction` 与 `test` 均为任务类型列表；`suite_seed` 与 `task_index` 指定实例生成。`audit.json` 保留每个入选类型的 complexity、应用标签、任务族、所属集合与分布统计。

## OSWorld

数据源：`Benchmark/OSWorld/evaluation_examples/test_all.json` 和对应 `examples/<domain>/<id>.json`。369项中先排除官方 `test_infeasible.json` 标记26项，再要求 `proxy=false`（当前本地 OSWorld 环境未启用代理），再排除此前 GUI 轨迹/五样本验证中见过的10项，留下277项。对同一领域任务指令用 `SequenceMatcher` 相似度 `>0.90` 检出的一对近似题整体排除，候选剩275项。该近似检测只是保守字符串筛查，不保证排除所有语义近重复。对相同下载 URL 与本地路径的任务专用文件进一步检查跨集合重复：若出现冲突，在同一 domain×代理档内用未选任务替换；Thunderbird 公共 `thunderbird-profile.tar.gz` 视作环境基础资产，不作为任务专用文件隔离。

**OSWorld 没有统一官方难度字段。** 这里的结构复杂度代理分数在抽样前固定为：

`英文指令中的字母/数字词数 + 8×(related_apps 数−1) + 2×min(config 步数, 6)`。

每个 domain 内按此分数排序，分成三个近似等量档；它是**分层代理，不是已验证的真实任务难度**。先按可用候选数给10个 domain 分配配额，再在 domain × 代理档内成对选题，一题入构建、一题入测试。两侧在每一个 domain × 代理档单元上数量完全相同；总代理档均为 **16 Easy、18 Medium、16 Hard**。domain 数量每侧：Chrome3、GIMP3、Calc8、Impress8、Writer4、Multi-apps13、OS2、Thunderbird3、VLC3、VS Code3。

每侧43题包含 `download` 初始化步骤，也做了配对平衡；两侧任务专用下载文件无相同 URL+路径。**这不是运行环境预检成功证明**：网络、资产、账号以及下载资源能否执行，尚需逐题预检；不能因模型答错临时替换测试题。如果预检发现基础设施问题，应记录受影响 ID、修复环境或公布新的预注册划分版本，不改写本清单。`proxy=false` 仅说明 benchmark 配置没有请求代理，并不保证任务完全离线。

清单：[osworld.json](osworld.json) 存协议元数据；[osworld_memory.json](osworld_memory.json)、[osworld_test.json](osworld_test.json) 是 OSWorld 原生 `domain → task ID[]` 格式，可分别用于构建和测试。`audit.json` 包含每题指令、domain、代理分数/档位、应用数、初始化步骤类型及所属集合。

## 可复现与使用边界

```bash
/root/anaconda3/envs/vision2code/bin/python Experiments/memory_baselines/splits/gui_50_50_seed20261026/build_split.py
```

用相同本地数据运行应生成逐字相同的 JSON。AndroidWorld、OSWorld 均各有50个构建、50个测试任务，无 ID/类型交叉，也不含此前已查看结果的五样本任务。构建集上的运行轨迹可供所有记忆方法共同提取不同表征；测试集配置和顺序在方法确定后冻结。现有 GUI 五样本测试入口仍使用旧的5任务清单，**尚未改成运行此50/50划分**；已有 MAGNET 小型记忆库来源于此前五样本，不能直接当作本划分的正式构建库。
