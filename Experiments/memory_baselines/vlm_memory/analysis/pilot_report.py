"""Export reproducible per-task tables and Chinese pilot report without new inference."""
from __future__ import annotations

import argparse
import csv
import json
import statistics
import math
from collections import Counter
from pathlib import Path

from PIL import Image

from ..config import ROOT, WORKSPACE, write_json
from ..models.local import image_hash
from ..memory import MemoryBank

METHODS = {"no_memory": "B0 No Memory", "raw_multimodal": "B1 Raw Episodic",
           "ksi_style": "B3 KSI-style", "magnet_adapted": "B4 MAGNET-adapted",
           "darwinian_adapted": "B6 Darwinian-adapted"}


def read(path):
    return json.loads(path.read_text())


def mean(values):
    vals = [v for v in values if v is not None]
    return statistics.mean(vals) if vals else None


def fmt(value):
    return "—" if value is None else f"{value:.3f}"


def wilson(k, n):
    z = 1.959963984540054
    center = (k / n + z*z / (2*n)) / (1 + z*z/n)
    half = z * math.sqrt(k/n * (1-k/n)/n + z*z/(4*n*n)) / (1 + z*z/n)
    return f"[{center-half:.3f}, {center+half:.3f}]"


def usage(paths):
    calls = [read(p) for p in paths]
    return {"calls": len(calls), "tokens": sum(c.get("usage", {}).get("total_tokens", 0) for c in calls),
            "seconds": sum(c.get("seconds", 0) for c in calls)}


def perceptual_candidates(manifest):
    def dhash(path):
        with Image.open(path) as image:
            pixels = list(image.convert("L").resize((9, 8)).getdata())
        return sum((pixels[y*9+x] > pixels[y*9+x+1]) << (y*8+x) for y in range(8) for x in range(8))
    memory = [(t["id"], p, dhash(p)) for t in manifest["memory"] for p in t["images"]]
    test = [(t["id"], p, dhash(p)) for t in manifest["test"] for p in t["images"]]
    return [{"source_task": s, "test_task": t, "source_image": sp, "test_image": tp,
             "dhash_distance": (sh ^ th).bit_count(), "status": "manual_review_needed"}
            for s, sp, sh in memory for t, tp, th in test if (sh ^ th).bit_count() <= 8]


def recorded_spend():
    patterns = ("*/pilot_10_5_v*/histories/*/execution/calls/*.json",
                "*/pilot_10_5_v*/histories/*/construction_calls/*.json",
                "*/pilot_10_5_v*/construction_calls/*.json",
                "*/pilot_10_5_v*/runs/*/*/seed_0/attempt_0/calls/*.json")
    unique = {image_hash(p): p for pattern in patterns for p in (ROOT / "artifacts").glob(pattern)}
    calls = [read(p) for p in unique.values()]
    result_patterns = ("*/pilot_10_5_v*/histories/*/execution/result.json",
                       "*/pilot_10_5_v*/runs/*/*/seed_0/attempt_0/result.json")
    results = {image_hash(p): read(p) for pattern in result_patterns for p in (ROOT / "artifacts").glob(pattern)}
    judges = [c for r in results.values() for c in r.get("evaluation", {}).get("rater_calls", [])]
    return {"recorded_unique_generation_construction_calls": len(calls),
            "recorded_generation_construction_tokens": sum(c.get("usage", {}).get("total_tokens", 0) for c in calls),
            "recorded_judge_calls": len(judges),
            "recorded_judge_tokens": sum(c.get("usage", {}).get("total_tokens", 0) for c in judges),
            "policy": "deduplicate byte-identical copied call/result files; includes preserved preflight attempts; in-flight interrupted calls without saved responses excluded"}


def export(root):
    manifest = read(root / "dataset_manifest.json")
    validation = read(root / "validation.json")
    MemoryBank(root / "bank.json").verify_frozen()
    results = [read(p) for p in sorted((root / "runs").glob("*/*/seed_0/attempt_0/result.json"))]
    if len(results) != 25 or any(r["status"] != "completed" for r in results):
        raise ValueError(f"Incomplete matrix: {root}")
    baseline = {r["task_id"]: r for r in results if r["method"] == "no_memory"}
    bank = read(root / "bank.json")
    out = root / "analysis/analysis_v1"
    out.mkdir(parents=True, exist_ok=True)
    rows, aggregates = [], []
    for method in METHODS:
        group = [r for r in results if r["method"] == method]
        paired = [(r, baseline[r["task_id"]]) for r in group if r["score"] is not None and baseline[r["task_id"]]["score"] is not None]
        aggregate = {"method": method, "n": len(group), "scored": sum(r["score"] is not None for r in group),
            "score": mean([r["score"] for r in group]),
            "paired_gain": mean([r["score"] - b["score"] for r, b in paired]),
            "wins": sum(r["score"] > b["score"] for r, b in paired),
            "losses": sum(r["score"] < b["score"] for r, b in paired),
            "negative_transfer_count": sum(b["success"] and not r["success"] for r, b in paired),
            "render_successes": sum(bool(r["evaluation"].get("render_success")) for r in group),
            "mean_tokens": mean([r["usage"]["total_tokens"] for r in group]),
            "mean_seconds": mean([r["seconds"] for r in group]),
            "mean_images": mean([r["memory_images"] for r in group]),
            "mean_memory_chars": mean([r["memory_chars"] for r in group]),
            "retrieval_tasks": sum(r["memory_entries"] > 0 for r in group),
            "truncated_generations": sum(r["generation_truncated"] for r in group),
            "memory_truncations": sum(r["memory_truncated"] for r in group),
            "bank_entries": len(bank["methods"].get(method, []))}
        aggregates.append(aggregate)
        for result in group:
            rows.append({k: result[k] for k in ("benchmark", "method", "task_id", "score", "success", "seconds", "memory_entries", "memory_images", "generation_truncated")})
    with (out / "per_task.csv").open("w") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    write_json(out / "aggregates.json", aggregates)
    cases = [{"task_id": r["task_id"], "method": r["method"], "baseline_score": baseline[r["task_id"]]["score"],
              "score": r["score"], "trace": str((root / "runs" / r["method"] / r["task_id"] / "seed_0/attempt_0").relative_to(WORKSPACE))}
             for r in results if r["method"] != "no_memory" and r["score"] != baseline[r["task_id"]]["score"]]
    write_json(out / "cases.json", cases)
    cost = {"history_generation": usage((root / "histories").glob("*/execution/calls/*.json")),
            "memory_construction": usage(list((root / "histories").glob("*/construction_calls/*.json")) + list((root / "construction_calls").glob("*.json"))),
            "test_generation": usage((root / "runs").glob("*/*/seed_0/attempt_0/calls/*.json"))}
    for key, records in (("history_judging", [read(p) for p in (root / "histories").glob("*/execution/result.json")]), ("test_judging", results)):
        calls = [c for r in records for c in r["evaluation"].get("rater_calls", [])]
        cost[key] = {"calls": len(calls), "tokens": sum(c.get("usage", {}).get("total_tokens", 0) for c in calls)}
    write_json(out / "cost.json", cost)
    write_json(out / "near_duplicate_candidates.json", perceptual_candidates(manifest))
    provenance = {"benchmark": manifest["benchmark"], "run": root.name,
        "config_sha256": image_hash(root / "config.snapshot.json"), "dataset_sha256": image_hash(root / "dataset_manifest.json"),
        "bank_sha256": image_hash(root / "bank.json"), "code_hashes_at_report": {
            str(p.relative_to(ROOT)): image_hash(p) for p in (ROOT / "vlm_memory").rglob("*.py")},
        "source_episodes": {t["id"]: image_hash(root / "histories" / t["id"] / "episode.json") for t in manifest["memory"]}}
    evaluator_files = {
        "mathvision": ["Benchmark/MATH-Vision/evaluation/evaluate.py", "Benchmark/MATH-Vision/evaluation/utils.py"],
        "mpdocvqa": ["Benchmark/MP-DocVQA-Framework/metrics.py"],
        "vision2code": ["Benchmark/Vision2Code/vision2code/benchmark/run_benchmark.py",
                        "Benchmark/Vision2Code/vision2code/evaluation/dataset_rubrics.py",
                        "Benchmark/Vision2Code/vision2code/rendering/render_python.py"]}
    provenance["evaluator_hashes"] = {name: image_hash(WORKSPACE / name) for name in evaluator_files[manifest["benchmark"]]}
    write_json(root / "manifest.json", provenance)
    return manifest, validation, aggregates, cases, cost


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--math-run", default="pilot_10_5_v3")
    parser.add_argument("--mp-run", default="pilot_10_5_v4")
    parser.add_argument("--vision-run", default="pilot_10_5_v1")
    parser.add_argument("--output", type=Path, default=WORKSPACE / "Docs/实验分析/20261022-03-五个Baseline-三个推理Benchmark-10记忆5测试.md")
    args = parser.parse_args()
    parts = ["# 五个记忆 Baseline 在三个多模态推理 Benchmark 上的小规模实验\n",
        "实验日期：2026-10-22。模型：本地 `/data/pretrained_models/Qwen3-VL-8B-Instruct`，服务 `qwen3-vl-8b-instruct`，GPU 7。所有调用均为本地推理。\n",
        "## 实验设置\n",
        "每个 benchmark 独立收集 10 个历史任务的真实求解与反馈，构建 B1/B3/B4/B6 记忆；B0 使用相同测试输入但不注入历史记忆。随后在另外 5 个任务上运行全部五种方法，共 30 个历史任务、75 次测试求解。记忆评测期间冻结，未将测试反馈回写。\n",
        "数据选择采用固定随机种子 42，排除相同任务 ID 和图像 SHA256。MP-DocVQA 使用 train 建库、val 测试，并按 doc_id 隔离，只选择不超过 6 页的文档且读取全部页面。MATH-Vision 从官方 test 池内部划分 10/5；Vision2Code 从有本地发布图像的 test-mini 子池内部划分 10/5。这是研究者定义的 pilot split，不是官方完整测试集分数；未使用这些测试任务调参。语义/模板近重复尚未得到完整人工核验。\n",
        "模型温度 0，求解最多 4096 输出 Token，输入图像最长边 1024；检索 Top-k=3，历史图像最多 2 张，历史文本最多 16000 字符。MP-DocVQA 每次读取两页，JSON 输出上限 768 Token，再汇总读页结果。合法的空检索结果保留，预算截断显式记录。\n",
        "B3 为逐任务证据审阅/蒸馏后再跨十个任务进行一次知识蒸馏；B4 为 workflow 与功能视觉 crop；B6 为前提/目标/源轨迹与可靠性排序。B6 的来源成功定义是 MATH 答案正确、MP-DocVQA Exact Match 成功、Vision2Code 渲染成功且本地评分至少 3/5；后者是本轮操作性定义。三者均使用现有推理适配器和 BM25，而非原论文完整论坛、图检索或在线进化系统。此轮比较这些适配实现，不作为原论文复现排名。\n",
        "Vision2Code 使用上游规范化、Python 渲染和数据集 rubric，由同一本地 Qwen 评分，属于同模型质量诊断；同时报告渲染成功率。其分数不能替代独立评分器验证。\n",
        "## 主结果\n",
        "| Benchmark | 方法 | 有评分任务 | 主指标均值 | 相对 B0 配对差 | 胜/负 | 平均求解 Token | 平均执行与评分秒 |\n| --- | --- | --- | --- | --- | --- | --- | --- |"]
    details, all_cases = [], []
    for benchmark, run in (("mathvision", args.math_run), ("mpdocvqa", args.mp_run), ("vision2code", args.vision_run)):
        root = ROOT / "artifacts" / benchmark / run
        manifest, validation, aggregates, cases, cost = export(root)
        all_cases.extend((benchmark, c) for c in cases)
        for row in aggregates:
            parts.append(f"| {benchmark} | {METHODS[row['method']]} | {row['scored']}/5 | {fmt(row['score'])} | {fmt(row['paired_gain'])} | {row['wins']}/{row['losses']} | {fmt(row['mean_tokens'])} | {fmt(row['mean_seconds'])} |")
        details.extend([f"\n## {benchmark}：记忆与运行检查\n",
            f"产物：[运行目录](../../{root.relative_to(WORKSPACE)}/)。历史任务 {len(manifest['memory'])} 个，测试任务 {len(manifest['test'])} 个，完成 {validation['completed_results']}/25 个方法×任务结果。\n",
            "历史任务 ID：" + "、".join(t["id"] for t in manifest["memory"]) + "。\n",
            "测试任务 ID：" + "、".join(t["id"] for t in manifest["test"]) + "。\n",
            "| 方法 | 记忆单元 | 有检索任务/5 | 平均历史图像数 | 平均记忆字符 | 记忆截断任务数 | 输出截断任务数 |\n| --- | --- | --- | --- | --- | --- | --- |"])
        for row in aggregates:
            details.append(f"| {METHODS[row['method']]} | {row['bank_entries']} | {row['retrieval_tasks']} | {fmt(row['mean_images'])} | {fmt(row['mean_memory_chars'])} | {row['memory_truncations']} | {row['truncated_generations']} |")
        if benchmark == "vision2code":
            details.append("\n渲染成功数：" + "；".join(f"{METHODS[r['method']]} {r['render_successes']}/5" for r in aggregates) + "。")
        elif benchmark == "mpdocvqa":
            results = read(root / "summary.json")
            details.append("\n补充官方指标（Exact Match / Answer Page Accuracy）：" + "；".join(
                f"{METHODS[m]} {fmt(mean([r['evaluation']['exact_match'] for r in results if r['method']==m]))}/{fmt(mean([r['evaluation']['answer_page_accuracy'] for r in results if r['method']==m]))}"
                for m in METHODS) + "。")
            details.append("\n负迁移任务数（B0 Exact Match 成功而本方法失败）：" + "；".join(f"{METHODS[r['method']]} {r['negative_transfer_count']}/5" for r in aggregates) + "。")
        else:
            details.append("\n负迁移任务数（B0 成功而本方法失败，分母为全部 5 个任务）：" + "；".join(f"{METHODS[r['method']]} {r['negative_transfer_count']}/5" for r in aggregates) + "。")
        details.append("\n阶段调用与 Token：" + "；".join(f"{k}: {v['calls']} 次/{v['tokens']} Token" for k, v in cost.items()) + "。")
        histories = [read(p) for p in (root / "histories").glob("*/execution/result.json")]
        details.append(f"\n历史任务的操作性成功数：{sum(r['success'] for r in histories)}/10；历史求解输出截断数：{sum(r['generation_truncated'] for r in histories)}。失败来源同样保留并交给各方法处理。")
        bank = read(root / "bank.json")
        raw = bank["methods"]["raw_multimodal"]
        details.append(f"\nB1 文本索引的不同取值数：{len({e['index_text'] for e in raw})}/{len(raw)}。仅用文本 BM25，未复现原论文的视觉 embedding 或图检索；索引区分度不足时，分数并列会按条目 ID 排序。")
        details.append(f"\n逐样本表、成本、方法分歧案例索引分别位于该运行目录的 `analysis/analysis_v1/`；冻结快照与来源哈希见 `manifest.json`。")
        results = read(root / "summary.json")
        label = "Render SR" if benchmark == "vision2code" else ("Accuracy" if benchmark == "mathvision" else "Exact Match")
        details.append(f"\n{label} 的 Wilson 95% 区间（每方法 n=5）：" + "；".join(
            f"{METHODS[m]} {wilson(sum(bool(r['evaluation'].get('render_success')) if benchmark=='vision2code' else r['success'] for r in results if r['method']==m), 5)}"
            for m in METHODS) + "。区间仅描述这个抽样协议下二值指标的不确定性，不构成方法差异检验。")
        if benchmark == "mathvision":
            details.append("\n当前五题的难度等级分布：" + str(dict(Counter(t["metadata"]["level"] for t in manifest["test"]))) +
                "；题型分布：" + str(dict(Counter(t["metadata"]["subject"] for t in manifest["test"]))) +
                "。该实际子样本偏向高难几何题，不代表完整 benchmark 的学科与难度分布。")
        candidates = perceptual_candidates(manifest)
        details.append(f"\n历史/测试图像间 dHash 距离不超过 8/64 的候选对有 {len(candidates)} 组，详见 `near_duplicate_candidates.json`。此自动筛查只用于提示人工检查，不能证明语义重复或排除重复。")
        reviews_path = root / "analysis/analysis_v1/near_duplicate_reviews.json"
        if reviews_path.exists():
            details.append("\n该候选已查看两张原图：来源为稿件修改信，测试页为化学品监管讨论，页面文字内容不同，未发现同一页面复制；保留视觉核对记录于 `near_duplicate_reviews.json`。这不等于对整个样本池完成语义近重复认证。")
    parts += ["\nMATH-Vision 主指标为 Accuracy（0–1），MP-DocVQA 为官方 ANLS（0–1），Vision2Code 为本地 rubric（0–5）。胜/负以同一测试任务相对 B0 主分数升降计算；不对不同 benchmark 分数取混合均值。求解 Token 不包含独立列出的构建与评分开销。时间列从执行器开始到评价结束计时，未包含前置记忆检索；本轮未单独测量检索延迟。\n"]
    parts += details
    spend = recorded_spend()
    write_json(ROOT / "artifacts/pilot_10_5_recorded_spend.json", spend)
    parts += ["\n## 已记录调用总量\n",
        f"计入保留的预运行/失败尝试，并对复制复用的原始文件去重后，已有响应落盘的求解与构建调用为 {spend['recorded_unique_generation_construction_calls']} 次、{spend['recorded_generation_construction_tokens']} Token；评分调用为 {spend['recorded_judge_calls']} 次、{spend['recorded_judge_tokens']} Token。中断时尚未落盘的请求不能从这些文件计量，因此此表表示已记录调用量。各 benchmark 前面的阶段表表示最终实验所引用的调用，不应再与本表相加。\n"]
    parts += ["\n## 配对案例索引\n", "以下只记录观测到的分数差异；原因需要进一步核对原始轨迹和实际注入内容，不能由分数直接推断。\n",
        "| Benchmark | task ID | 方法 | B0 分数 | 本方法分数 |\n| --- | --- | --- | --- | --- |"]
    for benchmark, c in all_cases:
        parts.append(f"| {benchmark} | {c['task_id']} | {METHODS[c['method']]} | {fmt(c['baseline_score'])} | {fmt(c['score'])} |")
    parts += ["\n## 异常与解释边界\n",
        "MP-DocVQA 初始运行 `pilot_10_5_v1` 在历史收集阶段遇到 found=false 与非空答案不一致的输出；增加 JSON Schema 约束后，v2 完成历史收集，但建库输出把图片路径误写为来源任务 ID，触发校验失败。v3 增加合法来源和步骤引用约束；检查轨迹时发现来源 episode 尚未包含已保存的最终汇总响应，因此在测试前停止 v3，并在 v4 从原始模型记录补齐该步骤，重新建库和测试。v4 复用 v2 的十次完整历史任务求解，不重新生成历史答案。\n",
        "MATH-Vision v1 的一条来源审阅 JSON 达到输出上限，触发停止；v2 增加审阅结构约束后，蒸馏的自由文本仍达到上限。v3 复用十条历史求解及四个已完成来源记忆，对剩余构建进一步限定字段与列表长度，再完成建库和测试。所有修复发生于各自 benchmark 的测试之前，未根据测试分数挑选或修订记忆；旧运行和失败输出均保留。\n",
        "每个方法只有 5 个测试任务，每任务一次求解；API 温度为 0，未显式指定服务端随机种子，路径 seed_0 表示此轮单次重复编号。Accuracy 一道题对应 20 个百分点。因此本轮主要用于确认真实来源记忆的完整链路并筛选配对案例，不支持稳定排名、统计显著性或跨 benchmark 泛化结论。B6 源任务可靠性只由单次观测估计，不能视为长期效用学习。图像与文本预算也可能限制复杂历史的完整表达。\n",
        "Vision2Code 的一个已核对案例见 [B1 检索与坐标范围案例](20261022-02-Vision2Code-B1检索与坐标范围案例.md)：B1 的十条索引相同，柱状图任务检索到化学图经验，输出出现不适用的坐标范围与关闭坐标轴设置。该案例支持检查索引区分度与记忆适用性，尚未完成因果干预验证。\n",
        "MP-DocVQA 的 [数值格式扣分案例](20261022-01-MPDocVQA-数值格式扣分案例.md) 显示：B3/B6 在任务 44965 上省略千位逗号，数值和证据页一致，但官方 Exact Match 与 ANLS 下降。该样本解释了本轮两种方法相对 B0 的平均 ANLS 差值，不应直接视为数值推理能力下降。\n",
        "下一步先检查 cases.json 中的方法分歧、失败和负迁移任务，对照 memory_trace.json、calls/ 与评价输出形成原因假设，再在固定任务条件下做记忆信息干预；扩大样本前先完善独立评分与近重复检查。\n"]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(parts) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
