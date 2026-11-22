# Darwinian Memory implementation source

正式发表页面：https://proceedings.mlr.press/v306/mi26b.html

论文声明的实现地址：https://anonymous.4open.science/r/DMS-C48C/

已核查该来源，匿名站点的仓库接口
`https://anonymous.4open.science/api/repo/DMS-C48C/files/`
返回 HTTP **410**，正文为 `{"error":"repository_expired"}`。因此当前目录仅记录来源与访问状态，不包含原版源码。

可运行的推理领域适配位于 `Experiments/memory_baselines/`，方法名为 `darwinian_adapted`。它使用子任务单元、Precondition/Goal、源步骤与视觉证据、观测反馈统计和冻结的风险/效率排序。完整原生 Planner–Actor、在线突变、动态淘汰和原论文效用公式尚未复现。
