"""Analyze a completed paired prompt pilot without any model calls."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from ..config import ROOT, WORKSPACE
from ..pipeline.pilot import read
from ..pipeline.run_100_100_nomemory import atomic_json


def report(run: str) -> dict:
    root = ROOT / "artifacts/prompt_comparison" / run
    config = read(root / "config.snapshot.json")
    rows = []
    stats = {}
    lines = [f"# Qwen3.8-27B 基础提示词配对分析：{run}", "",
             "## 实验记录", "",
             f"模型：`{config['model']['name']}`；请求输出上限：{config['model']['max_tokens']}。",
             "每个 benchmark 从记忆构建集按固定种子抽 5 题，分别执行 old/base_v1。测试集不参与提示词调整。",
             "两组使用同一模型、图像处理、输出预算与评价器；4,096-token 预算的旧部分运行不纳入统计。", "",
             "## 配对结果", "",
             "| Benchmark | n | 旧均分 | 新均分 | 差值 | 旧/新截断 | 旧/新平均输出 token |",
             "|---|---:|---:|---:|---:|---|---|"]
    for benchmark in ("mathvision", "vision2code"):
        ids = read(root / benchmark / "sample_ids.json")
        paired = []
        for task_id in ids:
            pair = {}
            for version in ("old", "base_v1"):
                folder = root / benchmark / version / str(task_id)
                result = read(folder / "result.json")
                if result.get("status") != "completed" or result.get("score") is None:
                    raise ValueError(f"Unfinished or unscored prompt pilot: {folder}")
                call = read(folder / "calls/generation.json")
                row = {"benchmark": benchmark, "task_id": str(task_id), "version": version,
                    "score": result["score"], "truncated": result["generation_truncated"],
                    "output_tokens": result["usage"]["completion_tokens"], "seconds": result["seconds"],
                    "render_success": result["evaluation"].get("render_success"),
                    "finish_reason": call["finish_reason"], "effective_output_budget": call["effective_max_tokens"],
                    "artifact": str(folder)}
                rows.append(row)
                pair[version] = row
            paired.append(pair)
        values = {}
        for version in ("old", "base_v1"):
            group = [p[version] for p in paired]
            values[version] = {"n": len(group), "mean": sum(r["score"] for r in group)/len(group),
                "mean_output_tokens": sum(r["output_tokens"] for r in group)/len(group),
                "truncated": sum(r["truncated"] for r in group),
                "render_success_count": sum(bool(r["render_success"]) for r in group) if benchmark == "vision2code" else None}
        old, new = values["old"], values["base_v1"]
        values["delta"] = new["mean"] - old["mean"]
        stats[benchmark] = values
        lines.append(f"| {benchmark} | 5 | {old['mean']:.3f} | {new['mean']:.3f} | {values['delta']:+.3f} | {old['truncated']}/{new['truncated']} | {old['mean_output_tokens']:.0f}/{new['mean_output_tokens']:.0f} |")
    lines.extend(["", "MathVision 分数为官方正确率；Vision2Code 分数为上游数据集 rubric 的 0–5 分。", "",
                  "## 案例与解释", ""])
    for benchmark in stats:
        lines.extend([f"### {benchmark}", "", "| 任务 | 旧分数 | 新分数 | 旧/新输出 token |", "|---|---:|---:|---|"])
        for row in [r for r in rows if r["benchmark"] == benchmark and r["version"] == "old"]:
            new = next(r for r in rows if r["benchmark"] == benchmark and r["task_id"] == row["task_id"] and r["version"] == "base_v1")
            lines.append(f"| {row['task_id']} | {row['score']} | {new['score']} | {row['output_tokens']}/{new['output_tokens']} |")
        lines.append("")
    lines.extend(["## 使用决定与限制", "",
        "base_v1 明确要求逐步推理，并保留官方适配评价所需的 boxed 答案或可执行 Python/OUTPUT_PATH 输出约定。",
        "5 题是提示词与管线的探索性检查，不能据此证明整体提升；本次简单随机抽样未覆盖所有难度和视觉类别。",
        "Vision2Code 生成器与评分器均为 Qwen3.8-27B，评分并不独立；渲染失败按 0 分记录，评分服务错误不应当成任务 0 分。",
        "此前 8B pilot 使用不同模型和不同任务，不能把其均分差异解释为提示词的因果效果。",
        "后续 100/100 所有方法固定同一基础提示词，并共同使用 no-memory 构建集轨迹；测试轨迹不进入记忆构建。", "",
        f"原始产物：`{root}`。"])
    output = WORKSPACE / "Docs/实验分析" / f"20261023-{run}-完整对比分析.md"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n")
    with (root / "paired_results.csv").open("w") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {"run": run, "benchmarks": stats, "report": str(output),
               "all_20_scored": len(rows) == 20, "any_truncated": any(r["truncated"] for r in rows)}
    atomic_json(root / "analysis.json", summary)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True)
    print(json.dumps(report(parser.parse_args().run), ensure_ascii=False, indent=2))
