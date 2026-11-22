"""Paired base-prompt pilot and resumable no-memory runs on fixed 100/100 splits."""
from __future__ import annotations

import argparse
import json
import sys
import time
import threading
import contextlib
import fcntl
from datetime import datetime, timezone
from pathlib import Path

from ..config import ROOT, WORKSPACE, load_config, write_json
from ..models.local import image_hash
from ..prompts import DOCUMENT_PROMPTS, snapshot_prompts
from .data import select
from .pilot import empty_context, execute, read


@contextlib.contextmanager
def heartbeat(label: str):
    stopped = threading.Event()
    started = time.monotonic()
    def announce():
        while not stopped.wait(30):
            print(f"INFERENCE {label}: waiting for generation/evaluation, elapsed={time.monotonic()-started:.0f}s", flush=True)
    worker = threading.Thread(target=announce, daemon=True)
    worker.start()
    try:
        yield
    finally:
        stopped.set()
        worker.join()


def experiment_config(value: dict) -> dict:
    """Exclude HTTP endpoint and timeout when comparing experiment settings."""
    normalized = {**value, "model": {key: item for key, item in value.get("model", {}).items()
                                     if key not in {"timeout", "base_url"}}}
    if "memory_construction" in value:
        normalized["memory_construction"] = {key: item for key, item in value["memory_construction"].items()
                                              if key != "parse_retries"}
    return normalized


def _matches_saved_prompts(saved: dict, current: dict) -> bool:
    # Old snapshots may contain additional unused profiles. Validate only the
    # prompt actually selected by the current run, never replace saved text.
    for version, benchmarks in current.items():
        for benchmark, spec in benchmarks.items():
            previous = saved.get(version, {}).get(benchmark)
            if previous is None and benchmark == "mpdocvqa":
                if saved.get("__document_prompts__") == DOCUMENT_PROMPTS:
                    continue
            if previous != spec:
                return False
    return True


def snapshot_run(root: Path, config: dict) -> None:
    snapshot = root / "config.snapshot.json"
    prompts = root / "base_prompts.snapshot.json"
    values = snapshot_prompts(config)
    for path, data in ((snapshot, config), (prompts, values)):
        if path.exists():
            preserved = read(path)
            matches = (_matches_saved_prompts(preserved, data) if path == prompts else
                       experiment_config(preserved) == experiment_config(data))
            if not matches:
                raise ValueError(f"Preserved configuration/prompt changed: {path}; use a new run ID")
            if path == snapshot:
                runtime = root / "runtime_transport.json"
                old_model = preserved.get("model", {})
                new_model = data.get("model", {})
                transport = {"snapshot_timeout": old_model.get("timeout"), "timeout": new_model.get("timeout"),
                             "snapshot_base_url": old_model.get("base_url"), "base_url": new_model.get("base_url")}
                if old_model.get("timeout") != new_model.get("timeout") or old_model.get("base_url") != new_model.get("base_url") or runtime.exists():
                    atomic_json(runtime, {**transport, "updated_at": datetime.now(timezone.utc).isoformat()})
            continue
        atomic_json(path, data)
    actual_prompts = root / "actual_prompts.json"
    if not actual_prompts.exists():
        selected = {benchmark: spec for entries in values.values() for benchmark, spec in entries.items()}
        atomic_json(actual_prompts, selected)
    model_metadata = root / "model.json"
    if not model_metadata.exists():
        known_paths = {
            "qwen3-vl-8b-instruct": "/data/pretrained_models/Qwen3-VL-8B-Instruct",
            "qwen3.8-27b": "/data/pretrained_models/Qwen/Qwen3.8-27B",
        }
        name = config.get("model", {}).get("name")
        model_path = config.get("model", {}).get("path", known_paths.get(name))
        atomic_json(model_metadata, {"name": name, "path": model_path,
                    "path_source": "config" if config.get("model", {}).get("path") else "known_local_default"})


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def eta_text(done: int, total: int, started: float, avg_seconds: float | None = None) -> str:
    elapsed = time.monotonic() - started
    per_item = avg_seconds if avg_seconds is not None else elapsed / max(done, 1)
    remaining = max(total - done, 0) * per_item
    end = datetime.now(timezone.utc).timestamp() + remaining
    end_text = datetime.fromtimestamp(end, timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    return f"{done}/{total} ({100 * done / max(total, 1):.1f}%), elapsed={elapsed/3600:.2f}h, ETA={remaining/3600:.2f}h, finish~{end_text}"


def verify_manifest(benchmark: str, config: dict, path: Path) -> dict:
    manifest = read(path) if path.exists() else select(benchmark, config, path.parent)
    if len(manifest["memory"]) != 100 or len(manifest["test"]) != 100:
        raise ValueError(f"Expected fixed 100/100 split for {benchmark}")
    ids_mem = {str(t["id"]) for t in manifest["memory"]}
    ids_test = {str(t["id"]) for t in manifest["test"]}
    hashes_mem = {h for t in manifest["memory"] for h in t.get("image_hashes", [])}
    hashes_test = {h for t in manifest["test"] for h in t.get("image_hashes", [])}
    if ids_mem & ids_test or hashes_mem & hashes_test:
        raise ValueError(f"Fixed memory/test overlap for {benchmark}")
    for task in manifest["memory"] + manifest["test"]:
        for image in task["images"]:
            if not Path(image).is_file():
                raise FileNotFoundError(image)
    return manifest


def score_value(benchmark: str, result: dict) -> float | None:
    if benchmark == "mathvision":
        return float(bool(result.get("score")))
    score = result.get("score")
    return float(score) if score is not None else None


def prompt_pilot(config: dict, run: str, seed: int) -> None:
    raise ValueError("Historical prompt variants have been removed from source. "
                     "Read the saved experiment snapshots/reports; new experiments use current prompts.")


def make_episode(task: dict, result: dict, run_root: Path, split: str, manifest: dict, config: dict) -> dict:
    folder = run_root / "runs" / "no_memory" / split / str(task["id"]) / "attempt_0"
    images = [{"path": str(Path(p).resolve()), "role": f"source_observation_{i}", "sha256": image_hash(Path(p))}
              for i, p in enumerate(task["images"])]
    rendered = folder / "rendered_image.png"
    if task["benchmark"] == "vision2code" and rendered.exists():
        images.append({"path": str(rendered.resolve()), "role": "render_after_action", "sha256": image_hash(rendered)})
    return {"id": str(task["id"]), "benchmark": task["benchmark"], "split": split,
            "task": task["prompt"], "images": images, "steps": result.get("steps", []),
            "feedback": result.get("evaluation"), "success": result.get("success"),
            "score": result.get("score"), "model": config["model"]["name"],
            "prompt_version": config["prompt_version"],
            "provenance": {"split": split, "manifest_sha256": image_hash(run_root / "dataset_manifest.json"),
                           "task_id": str(task["id"]), "result_path": str((folder / "result.json").relative_to(run_root))}}


def no_memory(config: dict, run: str, benchmarks: list[str], retry_failed: bool) -> None:
    started = time.monotonic()
    all_counts = sum(200 for _ in benchmarks)
    done = 0
    durations = []
    for benchmark in benchmarks:
        run_root = ROOT / "artifacts" / benchmark / run
        run_root.mkdir(parents=True, exist_ok=True)
        snapshot_run(run_root, config)
        manifest_path = run_root / "dataset_manifest.json"
        manifest = verify_manifest(benchmark, config, manifest_path)
        atomic_json(manifest_path, manifest)
        for split in ("memory", "test"):
            tasks = manifest[split]
            for task in tasks:
                folder = run_root / "runs" / "no_memory" / split / str(task["id"]) / "attempt_0"
                result_path = folder / "result.json"
                error_path = folder / "error.json"
                if result_path.exists():
                    prev = read(result_path)
                    if prev.get("status") == "completed":
                        prev.update(method="no_memory", split=split, prompt_version=config["prompt_version"])
                        atomic_json(result_path, prev)
                        episode = make_episode(task, prev, run_root, split, manifest, config)
                        atomic_json(run_root / "trajectories" / split / f"{task['id']}.json", episode)
                        error_path.unlink(missing_ok=True)
                        atomic_json(folder / "status.json", {"status": "completed", "task_id": str(task["id"])})
                        durations.append(prev["seconds"])
                        done += 1
                        print(f"SKIP {benchmark}/{split}/{task['id']} {eta_text(done,all_counts,started, sum(durations)/len(durations) if durations else None)}", flush=True)
                        continue
                if error_path.exists() and not retry_failed:
                    done += 1
                    print(f"SKIP_FAILED {benchmark}/{split}/{task['id']} (use --retry-failed)", flush=True)
                    continue
                if error_path.exists():
                    archive = folder / "errors" / f"{time.time_ns()}.json"
                    archive.parent.mkdir(parents=True, exist_ok=True)
                    error_path.rename(archive)
                atomic_json(folder / "status.json", {"status": "running", "benchmark": benchmark,
                    "split": split, "task_id": str(task["id"]), "started_at": datetime.now(timezone.utc).isoformat()})
                task_started = time.monotonic()
                try:
                    print(f"START NOMEMORY {benchmark}/{split}/{task['id']}", flush=True)
                    with heartbeat(f"{benchmark}/{split}/{task['id']}"):
                        result = execute(task, empty_context(), folder, config, run_root,
                                         prompt_version=config.get("prompt_version", "current"))
                    result.update(method="no_memory", split=split, prompt_version=config.get("prompt_version", "current"))
                    atomic_json(result_path, result)
                    episode = make_episode(task, result, run_root, split, manifest, config)
                    episode_path = run_root / "trajectories" / split / f"{task['id']}.json"
                    atomic_json(episode_path, episode)
                    error_path.unlink(missing_ok=True)
                    atomic_json(folder / "status.json", {"status": "completed", "task_id": str(task["id"])})
                except KeyboardInterrupt:
                    atomic_json(run_root / "checkpoint.json", {"status": "interrupted", "done": done,
                        "total": all_counts, "updated_at": datetime.now(timezone.utc).isoformat()})
                    print("INTERRUPTED; completed tasks are saved. Rerun the same command to resume.", flush=True)
                    raise
                except Exception as exc:
                    atomic_json(error_path, {"status": "error", "error": f"{type(exc).__name__}: {exc}",
                        "benchmark": benchmark, "split": split, "task_id": str(task["id"])})
                    with (run_root / "runner.log").open("a") as log:
                        log.write(f"ERROR {benchmark}/{split}/{task['id']}: {type(exc).__name__}: {exc}\n")
                    print(f"ERROR {benchmark}/{split}/{task['id']}: {type(exc).__name__}: {exc}", flush=True)
                    atomic_json(folder / "status.json", {"status": "error", "task_id": str(task["id"]),
                        "error": f"{type(exc).__name__}: {exc}"})
                    raise
                duration = time.monotonic() - task_started
                durations.append(duration)
                done += 1
                line = f"NOMEMORY {benchmark}/{split}/{task['id']} {eta_text(done,all_counts,started,sum(durations)/len(durations))}"
                print(line, flush=True)
                with (run_root / "runner.log").open("a") as log:
                    log.write(line + "\n")
                atomic_json(run_root / "progress.json", {"attempted_in_invocation": done,
                    "expected_in_invocation": all_counts, "current_split": split,
                    "current_task": str(task["id"]), "last_task_seconds": duration,
                    "updated_at": datetime.now(timezone.utc).isoformat()})
    complete = True
    report = [f"# {run}：100/100 no-memory 实验", "",
              f"模型：`{config['model']['name']}`；提示词：`{config['prompt_version']}`。",
              "构建集轨迹用于后续经验记忆；测试集轨迹仅用于评测分析，不作为记忆来源。", "",
              "Vision2Code 采用上游渲染器和数据集 rubric；生成模型兼任评分模型，评分非独立。", "",
              "| Benchmark | 划分 | 已完成 / 100 | 成功数 / 成功率 | 平均分 | 截断数 |",
              "|---|---|---:|---:|---:|---:|"]
    for benchmark in benchmarks:
        run_root = ROOT / "artifacts" / benchmark / run
        results = []
        errors = []
        for split in ("memory", "test"):
            for task in read(run_root / "dataset_manifest.json")[split]:
                folder = run_root / "runs" / "no_memory" / split / str(task["id"]) / "attempt_0"
                if (folder / "result.json").exists(): results.append(read(folder / "result.json"))
                if (folder / "error.json").exists(): errors.append(read(folder / "error.json"))
            episodes = [read(run_root / "trajectories" / split / f"{task['id']}.json")
                        for task in read(run_root / "dataset_manifest.json")[split]
                        if (run_root / "trajectories" / split / f"{task['id']}.json").exists()]
            jsonl = run_root / "trajectories" / f"{split}.jsonl"
            jsonl.parent.mkdir(parents=True, exist_ok=True)
            jsonl.write_text("".join(json.dumps(e, ensure_ascii=False) + "\n" for e in episodes))
        metrics_by_split = {}
        for split in ("memory", "test"):
            group = [r for r in results if r.get("split") == split]
            scores = [float(r["score"]) for r in group if r.get("score") is not None]
            successful = sum(bool(r.get("success")) for r in group)
            metrics_by_split[split] = {"completed": len(group), "success_count": successful,
                "success_rate": successful / len(group) if group else None,
                "mean_score": sum(scores) / len(scores) if scores else None,
                "score_name": {"mathvision": "accuracy", "mpdocvqa": "ANLS",
                               "vision2code": "rubric_0_to_5"}[benchmark],
                "generation_truncated": sum(bool(r.get("generation_truncated")) for r in group)}
        atomic_json(run_root / "no_memory_summary.json", {"benchmark": benchmark, "model": config["model"],
            "prompt_version": config.get("prompt_version"), "memory_results": sum(r.get("split") == "memory" for r in results),
            "test_results": sum(r.get("split") == "test" for r in results), "errors": errors,
            "metrics_by_split": metrics_by_split,
            "total_results": len(results), "expected_results": 200, "complete": len(results) == 200 and not errors})
        complete = complete and len(results) == 200 and not errors
        for split in ("memory", "test"):
            group = [r for r in results if r.get("split") == split]
            scores = [r["score"] for r in group if r.get("score") is not None]
            mean = sum(scores)/len(scores) if scores else None
            successful = sum(bool(r.get("success")) for r in group)
            success_pct = f"{successful}/{len(group)} ({100*successful/len(group):.1f}%)" if group else "—"
            report.append(f"| {benchmark} | {split} | {len(group)} | {success_pct} | {mean} | {sum(r['generation_truncated'] for r in group)} |")
    report.extend(["", "MathVision 分数为官方正确率；MP-DocVQA 分数为 ANLS（同时统计 Exact Match）；Vision2Code 分数为 0–5 rubric 评分。", "",
                   f"完整状态：{'已完成' if complete else '存在未完成任务，详见各 benchmark 的 no_memory_summary.json'}。"])
    report_path = WORKSPACE / "Docs/实验分析" / f"20261023-{run}-no-memory.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(report) + "\n")
    if not complete:
        raise RuntimeError("No-memory matrix incomplete; failures preserved. Use --retry-failed to retry explicitly.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "config/qwen3_8_27b_100_100.json")
    parser.add_argument("--mode", choices=("prompt-pilot", "no-memory"), required=True)
    parser.add_argument("--run", default="qwen3_8_27b_base_v1")
    parser.add_argument("--benchmarks", nargs="+", choices=("mathvision", "vision2code", "mpdocvqa"),
                        default=("mathvision", "vision2code"))
    parser.add_argument("--seed", type=int, default=20261023)
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument("--resume", action="store_true", help="Completed tasks are always reused; accepted for explicit resume invocations")
    args = parser.parse_args()
    config = load_config(args.config)
    lock_dir = ROOT / "artifacts" / "locks"
    lock_dir.mkdir(parents=True, exist_ok=True)
    lock = (lock_dir / f"{args.mode}_{args.run}.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if args.mode == "prompt-pilot":
        prompt_pilot(config, args.run, args.seed)
    else:
        no_memory(config, args.run, args.benchmarks, args.retry_failed)


if __name__ == "__main__":
    main()
