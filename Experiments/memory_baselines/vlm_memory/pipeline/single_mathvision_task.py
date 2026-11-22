"""Run one MathVision item against the fixed 100/100 split for diagnosis."""
from __future__ import annotations

import argparse
import fcntl
import json
from pathlib import Path

from ..config import ROOT, load_config
from ..prompts import get_task_prompt
from .pilot import empty_context, execute, read
from .run_100_100_nomemory import atomic_json, heartbeat, make_episode, snapshot_run


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-id", default="1789")
    parser.add_argument("--run", default="qwen3_vl_8b_mathvision_1789_current")
    parser.add_argument("--config", type=Path, default=ROOT / "config/qwen3_vl_8b_mathvision_100_100.json")
    args = parser.parse_args()

    config = load_config(args.config)
    get_task_prompt("mathvision", config.get("prompt_version"))
    manifest_source = ROOT.parent.parent / config["fixed_split_dir"] / "mathvision/dataset_manifest.json"
    manifest = read(manifest_source)
    task = next((dict(t) for t in manifest["memory"] if str(t["id"]) == str(args.task_id)), None)
    if task is None:
        raise ValueError(f"Task {args.task_id} is not in the fixed MathVision memory split")

    lock_dir = ROOT / "artifacts/locks"
    lock_dir.mkdir(parents=True, exist_ok=True)
    with (lock_dir / f"single_{args.run}.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        run_root = ROOT / "artifacts/mathvision" / args.run
        snapshot_run(run_root, config)
        atomic_json(run_root / "dataset_manifest.json", manifest)
        folder = run_root / "runs/no_memory/memory" / str(args.task_id) / "attempt_0"
        result_path = folder / "result.json"
        if result_path.exists() and read(result_path).get("status") == "completed":
            print(json.dumps(read(result_path), ensure_ascii=False, indent=2))
            return

        atomic_json(folder / "status.json", {"status": "running", "benchmark": "mathvision",
                    "split": "memory", "task_id": str(args.task_id), "client_timeout_seconds": config["model"]["timeout"]})
        try:
            with heartbeat(f"mathvision/memory/{args.task_id}"):
                result = execute(task, empty_context(), folder, config, run_root,
                                 prompt_version=config["prompt_version"])
            result.update(method="no_memory", split="memory", prompt_version=config["prompt_version"])
            atomic_json(result_path, result)
            episode = make_episode(task, result, run_root, "memory", manifest, config)
            atomic_json(run_root / "trajectories/memory" / f"{args.task_id}.json", episode)
            atomic_json(folder / "status.json", {"status": "completed", "task_id": str(args.task_id)})
            generation = read(folder / "calls/generation.json")
            summary = {"run": args.run, "task_id": str(args.task_id), "question": task["prompt"],
                       "client_timeout_seconds": config["model"]["timeout"],
                       "requested_max_tokens": generation["requested_max_tokens"],
                       "effective_max_tokens": generation["effective_max_tokens"],
                       "completion_tokens": generation.get("usage", {}).get("completion_tokens"),
                       "finish_reason": generation.get("finish_reason"), "seconds": result["seconds"],
                       "response_chars": len(result.get("prediction", "")),
                       "score": result.get("score"), "success": result.get("success")}
            atomic_json(run_root / "diagnostic_summary.json", summary)
            print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
        except KeyboardInterrupt:
            atomic_json(folder / "status.json", {"status": "interrupted", "task_id": str(args.task_id)})
            raise
        except Exception as exc:
            atomic_json(folder / "status.json", {"status": "error", "task_id": str(args.task_id),
                        "error": f"{type(exc).__name__}: {exc}"})
            raise


if __name__ == "__main__":
    main()
