# MMSkills OSWorld 复现

- **实验设置**：40 题，五组为无技能、文本技能、视觉技能映射、视觉技能检索、视觉技能结构保留但无图片。
- **脚本**：[正式实验](../memory_baselines/scripts/experiments/run_mmskills_formal_dense.sh)、[无图消融](../memory_baselines/scripts/experiments/run_mmskills_formal_without_images.sh)。
- **主要结果**：200 条记录，195 条可评分。视觉映射、视觉检索、无图消融全 40 题均分依次为 34.44%、29.48%、36.84%。无技能及文本组分别有 3、2 条环境异常。
- **本地结果**：`Experiments/memory_baselines/artifacts/mmskills_osworld/analysis_20261111_v3/`。能力约束开发案例另存于 `artifacts/capability_constraints/`。
