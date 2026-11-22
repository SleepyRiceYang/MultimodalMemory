# CapCoMem 动机预实验

协议：`Docs/CapCoMem/Docs/20261103-01-动机预实验.md`。新模块为 `vlm_memory.pipeline.capcomem_motivation`，仅读取原实验产物，结果默认保存到 `artifacts/capcomem_motivation/pilot_v2`。

```bash
bash Experiments/memory_baselines/scripts/experiments/run_capcomem_motivation.sh --stage all
```

默认每个 benchmark 为构建重放 1 道、测试 1 道；MP-DocVQA 4 个条件、AndroidWorld 5 个条件，共 18 次任务执行。`--limit-per-role 0` 执行完整固定清单；改变任务规模、步数等配置必须使用新的 `--output`，不得将 pilot 当完整实验。`--max-steps` 默认为 30，所有 GUI 条件相同。默认双线程复用 GPU3/4 模型，GUI 每个并发任务独占一个既有 Android 环境与文件锁。不会使用带停用后端的旧路由。可用 `--workers` 限制并发。

可分阶段执行 `prepare → assets → select → evaluate → report`；阶段以外配置必须一致。完成条件可恢复；GUI 中断条件归档后重新初始化，MP 按原执行器校验请求后恢复调用。GUI 所有条件使用相同严格动作 schema，拒绝多余字段及嵌套动作。各任务保留输入、记忆追踪、模型原始响应、官方评分与耗时。`errors.json` 保存本次异常，成功恢复后也保留为历史调试记录，最终状态看 `summary.json`。不自动判定步骤级能力成功。

数据与控制：

- 固定种子 20261104；保存两组各 20 道构建清单，以及 MP 60 道 / Android 50 道测试清单。MP 的变化样本全部保留，短缺随机补齐、不重标类别。
- 只使用构建来源库。允许检索自身条目，但不强制命中，保存 `self_hit`。MP 源文档跨划分重叠即报错。
- Pilot 从固定清单选择任务，不按本轮成绩挑选；MP 优先页数少的样本，GUI 构建优先有经验节点覆盖的任务，再按 ID 排序。完整执行仍覆盖全部清单。
- 同一文本索引 cosine 排名固定 top-1，全部形式使用相同条目。该设置是在历史“最多三条、最多两图”预算内的保守实现；M4 预留两图，其他形式不用额外条目填补。
- MP 复用 KSI 核心、已有 ROI 和对应原页；GUI 复用 procedural 核心，取第一条引用动作前后两帧，而非多步工作流两端。
- 单帧说明只看单帧，ROI 说明只看实际裁图，变化说明才看前后两帧，防止将 AFTER 内容错误写进 BEFORE 描述。说明限制两句，原始调用保留。
- 实验二先提交逐题选择，再执行任何条件；任务描述与能力描述选择使用同一模型、说明及输出预算。十个随机种子离线计算，Oracle 仅为事后参照。执行顺序为先完成构建重放、冻结 `fixed_forms/*.json`、再执行测试；最佳固定形式仅由配对完整的构建重放计算，评分平分时优先选择实际 Token 成本低的形式。已有测试结果时计算的固定参照明确标记为 `retrospective_pilot_diagnostic`，不冒称测试前冻结；少于 20 题时明确标记为 pilot 参照。

依赖既有 vllm_env、mpdocvqa 评分环境及 Android HTTP 环境。若 8087 未启动，可在检查 GPU6 显存后运行：

```bash
GPU_INDEX=6 PORT=8087 bash Experiments/memory_baselines/scripts/serve/start_embedding.sh
```

验证：

```bash
PYTHONPATH=Experiments/memory_baselines:Methods/MAGNET/src /root/anaconda3/envs/vllm_env/bin/python -m unittest discover -s Experiments/memory_baselines/tests -p test_capcomem_motivation.py -v
```

当前范围：可执行任务效果和形式选择闭环。能力需求由冻结提示识别并保留；步骤级人工能力判据、人工标注、GUI 共同检查点和重复运行置信区间仍需后续分析，不能把预测能力标签当作已验证的能力提升。固定库重放也不是多轮演化实验。


## 正式运行入口（2026-11-04 交接）

```bash
bash Experiments/memory_baselines/scripts/experiments/run_capcomem_motivation_formal.sh --check
bash Experiments/memory_baselines/scripts/experiments/run_capcomem_motivation_formal.sh
```

正式入口固定 `--limit-per-role 0 --workers 2 --max-steps 30`，使用 GPU3/4 服务及 GPU6 embedding。MP 为 (20+60)×4=320，Android 为 (20+50)×5=350，共 670 次任务条件执行；实验二复用这些成绩，额外产生选择调用。默认输出为 `artifacts/capcomem_motivation/formal_20261104_v1/`，与 pilot 隔离。

本次仅通过 `bash -n`、`--check` 和 `--stage prepare` 验证，正式配置、20/20 构建与 60/50 测试清单及来源快照已经落盘。验证命令已经退出，没有启动正式 assets/select/evaluate，没有留下正式实验进程。既有模型、embedding、Android 服务保留，供后续运行 agent 使用。

分阶段顺序是 `prepare → assets → select → evaluate → report`；`evaluate` 内部先完成全部构建重放、冻结各基准最佳固定形式，再开始测试。任务/能力选择均先于任务执行。中断后重跑相同正式入口即可恢复；不要更换参数或手工修改已冻结清单。准备阶段重跑也会校验一致性。无自动外层重启循环；出现错误先保留日志、定位原因，再决定恢复。

建议运行 agent 重定向 stdout/stderr 到正式目录内独立日志，保存 PID，并监控进程、日志、逐条件 `result.json`、`progress.json`、`errors.json` 及 `fixed_forms/*.json`。`progress.json` 的 completed/expected 是当前 phase 的调度 job 数，不是 670 条结果数；实际结果数应数 `runs/*/*/*/M*/result.json` 中 status=completed 的记录。单个 GUI job 包含一个环境分片上的多题多条件，长时间不更新 job 计数时应检查逐步日志和 heartbeat。不要仅凭 summary.json 存在判断完成：最终应为 complete=true、completed_results=expected_results=670、无 missing/unexpected，且所有配对及资产冻结检查通过。正式 fixed_forms 的 retrospective_pilot_diagnostic 必须为 false。

最终 summary 的步骤级能力评价仍需人工标注与共同检查点分析；正式入口覆盖任务级对照及形式选择，不自动将预测能力标签转为能力完成率。模型服务和 GUI 环境由接手 agent 在启动前重新检查健康与占用。
