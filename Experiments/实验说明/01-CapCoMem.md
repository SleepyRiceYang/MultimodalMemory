# CapCoMem 动机预实验

- **实验设置**：AndroidWorld 测试 50 题，MP-DocVQA 测试 60 题。比较无记忆、纯文本、历史整图和局部 ROI，AndroidWorld 另外比较动作前后帧。任务选择、能力选择和 Oracle 复用固定形式的结果做离线比较。
- **脚本**：[运行入口](../memory_baselines/scripts/experiments/run_capcomem_motivation_formal.sh)。
- **主要结果**：含构建重放共 670 条记录。纯文本在两个测试集均值最高，AndroidWorld 为 13/50，MP-DocVQA ANLS 为 0.7545。Oracle 是事后上界。
- **本地结果**：`Experiments/memory_baselines/artifacts/capcomem_motivation/formal_20261104_v1/summary.json`。
