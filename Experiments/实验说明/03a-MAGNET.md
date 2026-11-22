# MAGNET 空记忆持续学习

- **实验设置**：AndroidWorld Easy61，从空库开始跑第 0–3 轮，每轮逐题使用相同轮初快照。每轮末从成功任务提炼流程，从有效点击提取控件功能与裁剪图，供下一轮使用。
- **脚本**：[并发持续学习](../memory_baselines/scripts/experiments/run_androidworld_magnet_cold_parallel.sh)、[回归任务重复实验](../memory_baselines/scripts/experiments/repeat_magnet_regressions.py)。
- **主要结果**：四轮成功数依次为 15、16、14、14／61。第 2 轮末有 28 条流程、64 个控件功能、247 张控件图。
- **本地结果**：`Experiments/memory_baselines/artifacts/magnet_cold_continual/easy61_empty_qwen3vl8b_parallel_v2/`。回归任务结果在同层 `magnet_regression_repeats_v1/`。
