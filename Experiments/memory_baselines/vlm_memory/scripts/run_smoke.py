from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from ..benchmarks.tasks import evaluate_math, evaluate_vision, load_task
from ..config import load_config, write_json, METHODS
from ..memory import MemoryBank
from ..models import LocalModel
from ..models.local import image_hash
from ..prompts import MATH_SYSTEM, VISION_SYSTEM


def run(args):
    root = args.artifacts.resolve()
    snapshot = root / "config.snapshot.json"
    config = load_config(args.config or (snapshot if snapshot.exists() else None))
    if snapshot.exists() and json.loads(snapshot.read_text()) != config:
        raise ValueError("Configuration differs from the frozen run snapshot; use a new artifact directory")
    if not snapshot.exists():
        write_json(snapshot, config)
    bank = MemoryBank(root / "bank.json")
    model = LocalModel(config["model"])
    task = load_task(args.benchmark, args.index)
    methods = [args.method] if args.method else config["baselines"]
    for method in methods:
        folder = root / "results" / args.benchmark / method
        if (folder / "result.json").exists():
            if args.resume:
                print(f"SKIP {args.benchmark}/{method}: existing result", flush=True)
                continue
            raise FileExistsError(f"{folder} exists; choose a fresh artifact directory or --resume")
        folder.mkdir(parents=True, exist_ok=True)
        print(f"RUN {args.benchmark}/{method} task={task['id']}", flush=True)
        started = time.monotonic()
        try:
            context = bank.context(method, args.benchmark, task["id"], task["query"], task["image"], config["memory"])
            trace = {**context, "images": [(label, str(path)) for label, path in context["images"]]}
            write_json(folder / "memory_trace.json", trace)
            if method != "no_memory" and not context["selected"]:
                raise RuntimeError("Smoke verification requires a non-empty memory retrieval")
            prompt = task["prompt"] + "\n\n" + context["text"]
            response = model.generate(MATH_SYSTEM if args.benchmark == "mathvision" else VISION_SYSTEM,
                prompt, [("CURRENT TASK reference (answer this task)", task["image"])] + context["images"])
            write_json(folder / "model_call.json", response)
            (folder / "response.txt").write_text(response["text"])
            if response["finish_reason"] == "length":
                raise RuntimeError("Model response exhausted token budget")
            if args.benchmark == "mathvision":
                evaluation = evaluate_math(folder, task, response["text"], sys.executable)
            else:
                evaluation = evaluate_vision(folder, task, response["text"], config["model"], config["vision2code"])
            frozen = bank.verify_frozen()
            result = {"status": "completed", "method": method, "benchmark": args.benchmark,
                      "task_id": task["id"], "split": task["split"], "evaluation": evaluation,
                      "identity": "reference_control" if method.startswith("raw_") or method == "no_memory" else "domain_adaptation",
                      "usage": response["usage"], "generation_seconds": response["seconds"],
                      "memory_entries": len(context["selected"]), "memory_images": len(context["images"]),
                      "memory_chars": len(context["text"]), "memory_truncated": context["truncated"],
                      "bank_sha256": bank.sha256, "bank_frozen": frozen,
                      "data_mode": config["data_mode"], "total_seconds": time.monotonic() - started}
        except Exception as exc:
            result = {"status": "error", "method": method, "benchmark": args.benchmark,
                      "task_id": task["id"], "error": f"{type(exc).__name__}: {exc}",
                      "total_seconds": time.monotonic() - started}
        write_json(folder / "result.json", result)
        print(json.dumps(result, ensure_ascii=False), flush=True)
    results = [json.loads(path.read_text()) for path in (root / "results").glob("*/*/result.json")]
    write_json(root / "summary.json", results)
    if any(result["status"] == "error" for result in results):
        raise SystemExit(1)


def main():
    parser = argparse.ArgumentParser(description="Run the same benchmark sample through each frozen-memory adapter")
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--benchmark", choices=("mathvision", "vision2code"), required=True)
    parser.add_argument("--method", choices=METHODS)
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--resume", action="store_true")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
