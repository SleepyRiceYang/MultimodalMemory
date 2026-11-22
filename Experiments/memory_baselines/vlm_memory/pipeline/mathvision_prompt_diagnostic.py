"""Compare a candidate prompt on preselected, previously truncated construction items."""
from __future__ import annotations

import argparse
import fcntl
from datetime import datetime, timezone
from pathlib import Path

from ..config import ROOT
from ..prompts import get_task_prompt
from .pilot import empty_context, execute, read
from .run_100_100_nomemory import atomic_json, heartbeat, snapshot_run


CASES = {
    "1291": "短算式逐字重复",
    "1932": "多次给出 boxed 答案后继续重启",
    "2050": "已有正确答案，随后反复改写公式",
    "1789": "反复解释折叠图像，未收敛",
    "807": "反复改变图像边界解释，未收敛",
    "177": "末尾仍在展开数值和坐标推导",
    "2883": "末尾仍在展开坐标参数计算",
    "700": "反复枚举并核验候选图案",
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", default="qwen3_vl_8b_mathvision_8192_32768_v1_nomemory")
    parser.add_argument("--run", default="qwen3_vl_8b_mathvision_truncation_prompt_20261024")
    parser.add_argument("--profile", default="mathvision_concise_v1")
    args = parser.parse_args()
    source = ROOT / "artifacts/mathvision" / args.source_run
    config = read(source / "config.snapshot.json")
    config["prompt_version"] = args.profile
    manifest = read(source / "dataset_manifest.json")
    tasks = {str(t["id"]): t for t in manifest["memory"]}
    root = ROOT / "artifacts/prompt_comparison" / args.run
    lock_dir = ROOT / "artifacts/locks"
    lock_dir.mkdir(parents=True, exist_ok=True)
    with (lock_dir / f"diagnostic_{args.run}.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        snapshot_run(root, config)
        plan = []
        for task_id, reason in CASES.items():
            if task_id not in tasks:
                raise ValueError(f"Selection is not a construction task: {task_id}")
            folder = source / "runs/no_memory/memory" / task_id / "attempt_0"
            baseline = read(folder / "result.json")
            if not baseline.get("generation_truncated"):
                raise ValueError(f"Selected baseline is not truncated: {task_id}")
            plan.append({"task_id": task_id, "selection_reason": reason,
                         "metadata": tasks[task_id]["metadata"], "baseline_result": str(folder / "result.json")})
        atomic_json(root / "selected_cases.json", plan)
        atomic_json(root / "dataset_manifest.json", {"benchmark": "mathvision", "memory": [tasks[k] for k in CASES], "test": []})
        system, user = get_task_prompt("mathvision", args.profile)
        atomic_json(root / "actual_prompt.json", {"version": args.profile, "system": system, "user_template": user})
        atomic_json(root / "model.json", {"name": config["model"]["name"], "path": "/data/pretrained_models/Qwen3-VL-8B-Instruct"})
        results = []
        for index, case in enumerate(plan, 1):
            task_id = case["task_id"]
            folder = root / args.profile / task_id
            path = folder / "result.json"
            try:
                if path.exists() and read(path).get("status") == "completed":
                    result = read(path)
                else:
                    print(f"START DIAGNOSTIC {index}/{len(plan)} task={task_id}", flush=True)
                    with heartbeat(f"DIAGNOSTIC/{task_id}"):
                        result = execute(tasks[task_id], empty_context(), folder, config, root,
                                         prompt_version=args.profile)
                    result.update(method="no_memory", split="memory", prompt_version=args.profile)
                    atomic_json(path, result)
                baseline = read(Path(case["baseline_result"]))
                row = {"task_id": task_id, "selection_reason": case["selection_reason"],
                       "level": case["metadata"]["level"], "baseline_success": baseline["success"],
                       "candidate_success": result["success"],
                       "baseline_tokens": baseline["usage"]["completion_tokens"],
                       "candidate_tokens": result["usage"]["completion_tokens"],
                       "baseline_truncated": baseline["generation_truncated"],
                       "candidate_truncated": result["generation_truncated"],
                       "baseline_seconds": baseline["seconds"], "candidate_seconds": result["seconds"],
                       "candidate_result": str(path)}
                results.append(row)
                atomic_json(root / "comparison.json", results)
                atomic_json(root / "progress.json", {"stage": "running", "completed": index, "total": len(plan),
                            "updated_at": datetime.now(timezone.utc).isoformat()})
                print(f"DIAGNOSTIC {index}/{len(plan)} task={task_id} tokens={row['candidate_tokens']} "
                      f"correct={row['candidate_success']} truncated={row['candidate_truncated']}", flush=True)
            except BaseException as exc:
                atomic_json(root / "progress.json", {"stage": "interrupted" if isinstance(exc, KeyboardInterrupt) else "error",
                            "completed": len(results), "total": len(plan), "error": f"{type(exc).__name__}: {exc}"})
                raise
        atomic_json(root / "progress.json", {"stage": "completed", "completed": len(results), "total": len(plan)})


if __name__ == "__main__":
    main()
