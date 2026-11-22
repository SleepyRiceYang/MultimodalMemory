"""Construction-only truncation audit and a paired diagnostic prompt report."""
from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path

from ..config import ROOT, WORKSPACE
from ..pipeline.pilot import read
from ..pipeline.run_100_100_nomemory import atomic_json


# Reviewed from saved outputs before candidate generations were evaluated.
# These are descriptive primary labels, not proof that the remaining reasoning
# is mathematically valid. Secondary repetition/answer flags are kept separately.
GROUPS = {
    "逐字或近逐字重复": "71 569 605 1291 1319 1649 1835 2052 2115 2318 2433 2534 2605",
    "已提出答案后继续重算": "202 607 1084 1179 1932 2050 2492 2599 2797",
    "反复解释图像或题意": "209 505 717 785 807 834 968 1150 1178 1187 1728 1789 1902 2020 2116 2293 2509",
    "未结束的推导或搜索": "177 389 692 700 1096 1586 1655 1955 1987 2088 2592 2883",
}


def audit(source: Path, output: Path):
    rows, all_results = [], []
    labels = {task: group for group, ids in GROUPS.items() for task in ids.split()}
    for path in sorted((source / "runs/no_memory/memory").glob("*/attempt_0/result.json")):
        result = read(path)
        all_results.append(result)
        if not result.get("generation_truncated"):
            continue
        task = str(result["task_id"])
        text = result.get("prediction", "")
        inp = read(path.parent / "input.json")
        words = text[-6000:].split()
        grams = [tuple(words[i:i+12]) for i in range(max(len(words)-11, 0))]
        duplicate_ratio = 1 - len(set(grams)) / len(grams) if grams else 0
        row = {"task_id": task, "primary_type": labels.get(task, "待复核"),
               "level": inp["metadata"]["level"], "subject": inp["metadata"]["subject"],
               "completion_tokens": result["usage"]["completion_tokens"], "correct": result["success"],
               "boxed_count": len(re.findall(r"\\boxed\s*\{", text)),
               "tail_duplicate_12gram_ratio": round(duplicate_ratio, 4),
               "text_chars": len(text), "tail_excerpt": text[-800:], "result_path": str(path)}
        rows.append(row)
    if set(labels) != {r["task_id"] for r in rows}:
        raise ValueError("Reviewed task labels differ from current truncated construction set")
    atomic_json(output / "truncation_cases.json", rows)
    with (output / "truncation_cases.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {"scope": "memory construction only; no held-out text used for tuning",
               "completed": len(all_results), "correct": sum(bool(r["success"]) for r in all_results),
               "truncated": len(rows), "truncated_correct": sum(bool(r["correct"]) for r in rows),
               "nontruncated": sum(not r.get("generation_truncated") for r in all_results),
               "nontruncated_correct": sum(bool(r["success"]) for r in all_results if not r.get("generation_truncated")),
               "types": dict(Counter(r["primary_type"] for r in rows)),
               "truncated_with_boxed": sum(r["boxed_count"] > 0 for r in rows)}
    atomic_json(output / "truncation_summary.json", summary)
    return summary


def report(root: Path, source: Path, summary: dict):
    paired = read(root / "comparison.json") if (root / "comparison.json").exists() else []
    lines = ["# MathVision 截断原因与提示词诊断", "",
             "分析日期：2026-10-24。来源：固定 100/100 划分的 Qwen3-VL-8B no-memory 运行。", "",
             "## 分析范围", "",
             "仅使用记忆构建集输出分析和选取复测样本；未使用 held-out 测试集文本或分数选择提示词。"
             "各类别为对已保存输出的人工描述性主标签，可存在重叠；‘仍在推导’不表示推导正确或只需增加预算即可得到答案。", "",
             f"构建集已完成 {summary['completed']}/100，正确 {summary['correct']}/100；"
             f"截断 {summary['truncated']}/100，截断组正确 {summary['truncated_correct']}/{summary['truncated']}，"
             f"未截断组正确 {summary['nontruncated_correct']}/{summary['nontruncated']}。两组题目难度不同，不能据此估计截断的因果影响。", "",
             "## 截断类型", "", "| 主类型 | 数量 | 占 51 个截断样本 |", "|---|---:|---:|"]
    for label, count in summary["types"].items():
        lines.append(f"| {label} | {count} | {count / summary['truncated']:.1%} |")
    lines += ["", f"仅 {summary['truncated_with_boxed']} 个截断输出包含 `\\boxed`。"
              "例如 1291 连续重复同一短算式；1932 已多次输出 boxed 答案仍重新分析；"
              "1789/807 反复改写对图像的解释；177/2883 在末段仍展开坐标和代数推导。"
              "8,192 是硬输出上限，达到上限不代表超出上限。", "",
              "逐样本标签、重复率、末段原文及路径见同目录 `truncation_cases.csv` / `truncation_cases.json`。"
              "重复率为末 6,000 字符内重复 12 词片段比例，只作辅助指标，不能代替语义判断。", "",
              "## 候选提示词与控制变量", "",
              "候选只改变基础 system 提示词；模型、temperature=0、图像缩放、8,192 输出预算、32,768 建库预算和评价器保持一致。"
              "不使用额外重试、重复惩罚、答案补写或提前停止。旧组复用原运行已保存的结果，新组重新生成；耗时受服务负载影响。", ""]
    if (root / "actual_prompt.json").exists():
        lines += ["```text", read(root / "actual_prompt.json")["system"], "```", ""]
    lines += ["## 典型样本配对结果", "",
              "8 个样本在新结果产生前，按截断类型选定，难度覆盖 1–5。全部来自旧运行已截断的构建集，属于有意选择的困难诊断样本；"
              "正确率不能代表整个 benchmark，不能直接宣称整体性能提升。", "",
              "| 任务 | 级别 | 选择原因 | 旧/新正确 | 旧/新 tokens | 新截断 |", "|---|---:|---|---|---:|---|"]
    for row in paired:
        lines.append(f"| {row['task_id']} | {row['level']} | {row['selection_reason']} | "
                     f"{int(row['baseline_success'])}/{int(row['candidate_success'])} | "
                     f"{row['baseline_tokens']}/{row['candidate_tokens']} | {row['candidate_truncated']} |")
    if paired:
        old_tokens = sum(r["baseline_tokens"] for r in paired)
        new_tokens = sum(r["candidate_tokens"] for r in paired)
        lines += ["", f"已完成 {len(paired)}/8：正确数 {sum(r['baseline_success'] for r in paired)} → "
                  f"{sum(r['candidate_success'] for r in paired)}；截断数 "
                  f"{sum(r['baseline_truncated'] for r in paired)} → {sum(r['candidate_truncated'] for r in paired)}；"
                  f"总输出 {old_tokens:,} → {new_tokens:,} tokens（减少 {1-new_tokens/old_tokens:.1%}）。", ""]
    decision = read(root / "decision.json") if (root / "decision.json").exists() else {}
    if decision:
        lines += ["## 用户决定", "", "用户选择采用 `mathvision_concise_v1`，停止后续改进和复测。"
                  f"本轮已保存 {len(paired)}/8 个完整配对结果；未完成输出不计入上述比较。"
                  "第二版未执行。MathVision 正式配置已切换到第一版，使用独立 run ID；批量实验未自动启动。", ""]
    lines += ["## 保存与下一步", "",
              "每个新实验保存 config、实际提示词、划分/任务清单和模型路径；不要求复制代码或保存代码版本。"
              "旧运行和旧提示词不覆盖。采用新提示词时使用新配置和 run ID，"
              "各 baseline 使用同一基础提示词、同一输出预算和共同构建轨迹。", "",
              "本次结果仅支持评估输出收敛与截断改善；准确性需在更多、未按旧失败结果选择的构建任务上检查。", ""]
    target = WORKSPACE / "Docs/实验分析/20261024-01-MathVision-截断原因与提示词诊断.md"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("\n".join(lines) + "\n")
    (root / "report.md").write_text("\n".join(lines) + "\n")
    return target


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", default="qwen3_vl_8b_mathvision_8192_32768_v1_nomemory")
    parser.add_argument("--run", default="qwen3_vl_8b_mathvision_truncation_prompt_20261024")
    args = parser.parse_args()
    source = ROOT / "artifacts/mathvision" / args.source_run
    output = ROOT / "artifacts/prompt_comparison" / args.run
    output.mkdir(parents=True, exist_ok=True)
    summary = audit(source, output)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(report(output, source, summary))


if __name__ == "__main__":
    main()
