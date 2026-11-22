"""Retest base_v1 against selected base_v2 on the exact previous five tasks per benchmark."""
from __future__ import annotations

import argparse
import csv
import fcntl
import json
from datetime import datetime, timezone
from pathlib import Path
import time

from ..config import ROOT, WORKSPACE, load_config
from ..models.local import image_hash
from .pilot import empty_context, execute, read
from .run_100_100_nomemory import atomic_json, eta_text, heartbeat, snapshot_run


def make_report(root: Path, previous: Path, config: dict) -> dict:
    records = []
    stats = {}
    lines = ["# 基础提示词 v2：同一批 5 样本复测对比", "",
             "## 实验设置", "",
             f"运行 ID：`{root.name}`。每个 benchmark 仍使用上一轮的 5 个构建集样本，共 20 次新生成。",
             "本轮对照为 base_v1 与 base_v2，均重新生成和评分；没有复用上一轮响应。",
             f"模型：`{config['model']['name']}`；temperature={config['model']['temperature']}；生成输出上限 **{config['model']['max_tokens']:,} tokens**；图像最长边 {config['model']['image_max_side']}。",
             "MathVision 的 v2 只删除 v1 中的 `in at most five short steps`；Vision2Code 的 v2 与原 old 提示词完全相同。",
             "两组均无经验记忆注入。API 未显式设置服务端 seed；每题每条件只运行一次。",
             "100/100 实验保持停止，本入口仅进行这批抽样复测，不触发后续实验。", "",
             "## 实际提示词", ""]
    prompts = read(root / "base_prompts.snapshot.json")
    for benchmark in ("mathvision", "vision2code"):
        lines.extend([f"### {benchmark}", ""])
        for version in ("base_v1", "base_v2"):
            lines.extend([f"**{version}**", "", "```text", prompts[version][benchmark]["system"], "```", ""])
        lines.append("两组 user 文本相同：MathVision 为原始题目及选项；Vision2Code 为 `Generate Python code to recreate the current reference image.`。")
        lines.append("")
    lines.extend(["## 本轮同预算配对结果", "",
                  "| Benchmark | v1 均分 | v2 均分 | 差值 v2−v1 | v1/v2 渲染成功数 | v1/v2 截断数 |",
                  "|---|---:|---:|---:|---|---|"])
    details = []
    for benchmark in ("mathvision", "vision2code"):
        ids = read(root / benchmark / "sample_ids.json")
        tasks = {str(t["id"]): t for t in read(WORKSPACE / config["fixed_split_dir"] / benchmark / "dataset_manifest.json")["memory"]}
        groups = {}
        for version in ("base_v1", "base_v2"):
            group = []
            for task_id in ids:
                folder = root / benchmark / version / str(task_id)
                result = read(folder / "result.json")
                if result.get("status") != "completed" or result.get("score") is None:
                    raise RuntimeError(f"Incomplete result: {folder}")
                call = read(folder / "calls/generation.json")
                row = {"benchmark": benchmark, "task_id": str(task_id), "profile": version,
                       "score": result["score"], "answer": result["evaluation"].get("answer", ""),
                       "output_tokens": result["usage"]["completion_tokens"],
                       "input_tokens": result["usage"]["prompt_tokens"], "seconds": result["seconds"],
                       "render_success": result["evaluation"].get("render_success"),
                       "truncated": result["generation_truncated"], "finish_reason": call["finish_reason"],
                       "artifact": str(folder)}
                group.append(row)
                records.append(row)
            groups[version] = group
        values = {}
        for version, group in groups.items():
            values[version] = {"n": len(group), "mean_score": sum(r["score"] for r in group)/len(group),
                "mean_output_tokens": sum(r["output_tokens"] for r in group)/len(group),
                "mean_seconds": sum(r["seconds"] for r in group)/len(group),
                "truncated": sum(r["truncated"] for r in group),
                "renders": sum(bool(r["render_success"]) for r in group) if benchmark == "vision2code" else None}
        values["delta"] = values["base_v2"]["mean_score"] - values["base_v1"]["mean_score"]
        values["paired_outcomes"] = {
            "improved": sum(b["score"] > a["score"] for a, b in zip(groups["base_v1"], groups["base_v2"])),
            "worse": sum(b["score"] < a["score"] for a, b in zip(groups["base_v1"], groups["base_v2"])),
            "tied": sum(b["score"] == a["score"] for a, b in zip(groups["base_v1"], groups["base_v2"]))}
        stats[benchmark] = values
        a, b = values["base_v1"], values["base_v2"]
        renders = f"{a['renders']}/5、{b['renders']}/5" if benchmark == "vision2code" else "不适用"
        lines.append(f"| {benchmark} | {a['mean_score']:.3f} | {b['mean_score']:.3f} | {values['delta']:+.3f} | {renders} | {a['truncated']}/{b['truncated']} |")
        details.extend([f"### {benchmark} 逐样本结果", "",
                        "| 任务 | 元数据 | v1 分数 | v2 分数 | v1/v2 答案或渲染 | v1/v2 输出 token |",
                        "|---|---|---:|---:|---|---:|"])
        for a, b in zip(groups["base_v1"], groups["base_v2"]):
            meta = tasks[a["task_id"]]["metadata"]
            label = f"Level {meta['level']}; {meta['subject']}" if benchmark == "mathvision" else meta["category"]
            answer = f"{str(a['answer']).upper()}/{str(b['answer']).upper()}" if benchmark == "mathvision" else f"{'成功' if a['render_success'] else '失败'}/{'成功' if b['render_success'] else '失败'}"
            details.append(f"| {a['task_id']} | {label} | {a['score']} | {b['score']} | {answer} | {a['output_tokens']:,}/{b['output_tokens']:,} |")
        details.append("")
        outcomes = values["paired_outcomes"]
        details.extend([f"逐题配对：v2 提高 {outcomes['improved']} 题、下降 {outcomes['worse']} 题、持平 {outcomes['tied']} 题。", ""])
        for a, b in zip(groups["base_v1"], groups["base_v2"]):
            if a["score"] != b["score"]:
                details.append(f"**差异样本 `{a['task_id']}`：{a['score']} → {b['score']}。**")
                for row in (a, b):
                    folder = Path(row["artifact"])
                    if benchmark == "mathvision":
                        response = read(folder / "result.json")["prediction"].split("</think>")[-1].strip()
                        details.extend(["", f"{row['profile']} 的最终输出（原文）：", "", "```text", response, "```", ""])
                    else:
                        rating = folder / "official_rubric_rating.json"
                        if rating.exists():
                            judged = read(rating)
                            details.append(f"{row['profile']} 评分器解释（未经独立人工验证）：{judged.get('overall_summary', '')}")
                            details.append(f"[渲染图]({folder / 'rendered_image.png'}) · [完整 rubric]({rating})")
                        else:
                            details.append(f"{row['profile']} 执行错误：`{read(folder / 'evaluation.json').get('status', '')[-900:]}`")
                details.append("")
        for row in groups["base_v2"]:
            if benchmark == "vision2code" and not row["render_success"]:
                evaluation = read(Path(row["artifact"]) / "evaluation.json")
                details.extend([f"v2 未渲染样本 `{row['task_id']}`：", "", "```text", evaluation["status"], "```", ""])
    lines.extend(["", "MathVision 分数为官方正确率；Vision2Code 为上游数据集 rubric 的 0–5 分。", ""])
    lines.extend(details)
    lines.extend(["## 输出与耗时", "",
                  "| Benchmark | v1 平均输出 token | v2 平均输出 token | v1 平均任务秒数 | v2 平均任务秒数 |",
                  "|---|---:|---:|---:|---:|"])
    for benchmark, values in stats.items():
        a, b = values["base_v1"], values["base_v2"]
        lines.append(f"| {benchmark} | {a['mean_output_tokens']:.1f} | {b['mean_output_tokens']:.1f} | {a['mean_seconds']:.1f} | {b['mean_seconds']:.1f} |")
    lines.extend(["", "输出 token 包含推理与最终答案/代码。Vision2Code 任务耗时包含生成、渲染和评分；表中输出 token 不含评分器输出。", "",
                  "## 上一轮结果：描述性参考", "",
                  "| Benchmark | 上轮 old | 上轮 v1 | 本轮 v1 | 本轮 v2 |", "|---|---:|---:|---:|---:|"])
    historical = read(previous / "analysis.json")["benchmarks"]
    for benchmark in stats:
        old, prior = historical[benchmark]["old"]["mean"], historical[benchmark]["base_v1"]["mean"]
        current = stats[benchmark]
        lines.append(f"| {benchmark} | {old:.3f} | {prior:.3f} | {current['base_v1']['mean_score']:.3f} | {current['base_v2']['mean_score']:.3f} |")
    lines.extend(["", "上一轮上限为 262,144，本轮为 32,678。虽然上一轮所有实际输出均低于本轮上限，跨轮比较仍不如本轮同预算配对严格；不得把所有跨轮差异都解释为提示词变化。", "",
                  "## 解释与限制", ""])
    for benchmark, values in stats.items():
        delta = values["delta"]
        lines.append(f"- {benchmark}：本轮 v2 相比 v1 {'提高' if delta > 0 else '下降' if delta < 0 else '持平'} {abs(delta):.3f}；只有 5 个样本、每条件一次生成，不能据此证明稳定收益。")
    lines.extend(["- 本轮 MathVision 可检验移除五步限制这一文本改动，但不保证模型内部推理步骤受该表述直接控制。",
                  "- Vision2Code v2 是原始 old 提示词；与上一轮 old 评分不同也可能来自单次生成/评分波动，不能把恢复旧提示词等同于保证复现旧分数。",
                  "- 样本来自构建集；MathVision 未覆盖 Level 1/3，Vision2Code 为文档×3、空间×1、科学图×1，不能代表全部领域。",
                  "- Vision2Code 的生成器与评分器是同一模型；rubric 解释属于模型判断。渲染失败按 0 分处理，没有进行代码修复后重算。",
                  "- 不根据这 5 题推断记忆方法本身的效果；所有任务均为 no-memory。", "",
                  "## 原始证据", "",
                  f"- [配置快照]({root / 'config.snapshot.json'})",
                  f"- [提示词快照]({root / 'base_prompts.snapshot.json'})",
                  f"- [逐样本 CSV]({root / 'paired_results.csv'})", ""])
    for row in records:
        folder = Path(row["artifact"])
        lines.append(f"- `{row['benchmark']}/{row['task_id']}/{row['profile']}`：[响应]({folder / 'response.txt'}) · [评价]({folder / 'evaluation.json'}) · [调用]({folder / 'calls/generation.json'})")
    report = WORKSPACE / "Docs/实验分析" / f"20261023-{root.name}-复测对比报告.md"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text("\n".join(lines) + "\n")
    with (root / "paired_results.csv").open("w") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    summary = {"run": root.name, "max_tokens": config["model"]["max_tokens"],
               "conditions": ["base_v1", "base_v2"], "benchmarks": stats, "report": str(report)}
    atomic_json(root / "analysis.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description="Read historical prompt-comparison results from saved snapshots")
    parser.add_argument("--run", default="qwen38_base_v2_32678_5sample")
    parser.add_argument("--previous-run", default="qwen3_8_27b_fullbudget_prompt_20261023")
    parser.add_argument("--config", type=Path, help="Historical argument; the saved run config is authoritative")
    parser.add_argument("--report-only", action="store_true")
    args = parser.parse_args()
    if not args.report_only:
        parser.error("Historical prompt text was removed from source; only --report-only is supported")
    root = ROOT / "artifacts/prompt_comparison" / args.run
    previous = ROOT / "artifacts/prompt_comparison" / args.previous_run
    config = read(root / "config.snapshot.json")
    print(json.dumps(make_report(root, previous, config), ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
