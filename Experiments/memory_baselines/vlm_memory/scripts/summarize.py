from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..config import load_config, write_json
from ..memory import MemoryBank


def summarize(root: Path, require_complete=False, config_path: Path | None = None):
    snapshot = root / "config.snapshot.json"
    config = load_config(config_path or (snapshot if snapshot.exists() else None))
    results = [json.loads(p.read_text()) for p in sorted((root / "results").glob("*/*/result.json"))]
    by_key = {(r["benchmark"], r["method"]): r for r in results}
    issues = []
    for benchmark in config["benchmarks"]:
        for method in config["baselines"]:
            result = by_key.get((benchmark, method))
            if not result or result["status"] != "completed":
                issues.append(f"Missing/failed run: {benchmark}/{method}")
                continue
            if not result.get("bank_frozen"):
                issues.append(f"Unfrozen memory: {benchmark}/{method}")
            if method != "no_memory" and result["memory_entries"] == 0:
                issues.append(f"Empty memory: {benchmark}/{method}")
            if method in ("no_memory", "raw_text", "ksi_style", "evoskill_adapted") and result["memory_images"]:
                issues.append(f"Unexpected historical images: {benchmark}/{method}")
            if method in ("raw_multimodal", "raw_visual", "magnet_adapted", "darwinian_adapted") and not result["memory_images"]:
                issues.append(f"Missing historical visual evidence: {benchmark}/{method}")
            if benchmark == "vision2code":
                evaluation = result["evaluation"]
                if not evaluation.get("render_success") or evaluation.get("rating_status") != "evaluated":
                    issues.append(f"Incomplete render/rating: {method}")
    bank = MemoryBank(root / "bank.json")
    bank.verify_frozen()
    calls = [json.loads(p.read_text()) for directory in ("history_calls", "construction_calls")
             for p in (root / directory).glob("*.json")]
    cost = {"history_and_construction_calls": len(calls),
            "history_and_construction_tokens": sum(c.get("usage", {}).get("total_tokens", 0) for c in calls),
            "solver_tokens": sum(r.get("usage", {}).get("total_tokens", 0) for r in results),
            "rater_tokens": sum(c.get("usage", {}).get("total_tokens", 0) for r in results
                                for c in r.get("evaluation", {}).get("rater_calls", []))}
    write_json(root / "validation.json", {"issues": issues, "completed": len(results), "cost": cost,
                                          "bank_sha256": bank.sha256,
                                          "asset_hashes": bank.asset_hashes})
    lines = ["# Baseline 单样本验证", "", "同一 Qwen3-VL-8B；独立合成历史；冻结记忆。论文方法均为推理领域适配版。",
             "MathVision 使用官方评价器；Vision2Code 使用上游渲染器与 rubric、本地同模型评分。", "",
             "| Benchmark | Baseline | 单题结果 | 历史条目/图像 | 生成调用总 Token |",
             "| --- | --- | --- | --- | --- |"]
    for benchmark in config["benchmarks"]:
        for method in config["baselines"]:
            result = by_key.get((benchmark, method), {})
            evaluation = result.get("evaluation", {})
            if benchmark == "mathvision":
                outcome = "正确" if evaluation.get("correct") else "错误"
            else:
                outcome = f"render={evaluation.get('render_success')}, rating={evaluation.get('rating_0_to_5', 'NA')}/5"
            if result.get("status") != "completed":
                outcome = result.get("error", "未完成")
            lines.append(f"| {benchmark} | {method} | {outcome} | "
                         f"{result.get('memory_entries', 0)}/{result.get('memory_images', 0)} | "
                         f"{result.get('usage', {}).get('total_tokens', 0)} |")
    lines += ["", f"验证问题：{'; '.join(issues) if issues else '无'}。", "",
              "这些结果用于验证构建、检索、模态注入、执行和评分接口；单题及合成记忆不支持性能排序。",
              "构建/反馈修订只使用合成历史，评测过程不回写记忆；技能修订尚无独立的修订收益验证。", "",
              "用量：`" + json.dumps(cost, ensure_ascii=False) + "`。", "",
              "每次运行的 model_call.json 保存实际提示词、图像角色/尺寸/哈希、Token 与耗时；",
              "memory_trace.json 保存检索条目、来源与实际注入文本；result.json 保存评价和冻结检查。"]
    (root / "report.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({"results": len(results), "issues": issues, "cost": cost}, ensure_ascii=False))
    if require_complete and issues:
        raise SystemExit(1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--require-complete", action="store_true")
    parser.add_argument("--config", type=Path)
    args = parser.parse_args()
    summarize(args.artifacts.resolve(), args.require_complete, args.config)


if __name__ == "__main__":
    main()
