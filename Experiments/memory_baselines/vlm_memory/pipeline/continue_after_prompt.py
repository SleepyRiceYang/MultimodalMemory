"""Wait for a running paired prompt pilot, analyze it, then collect common no-memory trajectories."""
from __future__ import annotations

import argparse
import fcntl
import time
from datetime import datetime, timezone
from pathlib import Path

from ..analysis.prompt_report import report
from ..config import ROOT, load_config
from ..prompts import get_task_prompt
from .pilot import read
from .run_100_100_nomemory import atomic_json, no_memory


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt-run", required=True)
    parser.add_argument("--run", required=True, help="Common no-memory trajectory run ID")
    parser.add_argument("--config", type=Path, default=ROOT / "config/qwen3_8_27b_100_100.json")
    args = parser.parse_args()
    lock_dir = ROOT / "artifacts/locks"
    lock_dir.mkdir(parents=True, exist_ok=True)
    owner = (lock_dir / f"pipeline_{args.run}.lock").open("w")
    fcntl.flock(owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
    status_path = ROOT / "artifacts/pipelines" / args.run / "status.json"
    def status(stage, **extra):
        value = {"stage": stage, "prompt_run": args.prompt_run, "no_memory_run": args.run,
                 "updated_at": datetime.now(timezone.utc).isoformat(), **extra}
        atomic_json(status_path, value)
        print(value, flush=True)
    try:
        status("waiting_for_prompt_pilot")
        pilot_lock = (lock_dir / f"prompt-pilot_{args.prompt_run}.lock").open("a")
        while True:
            try:
                fcntl.flock(pilot_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                time.sleep(30)
                print("WAIT: paired prompt pilot still running; no-memory collection will follow its analysis.", flush=True)
        status("analyzing_prompt_pilot")
        analysis = report(args.prompt_run)
        if analysis["any_truncated"]:
            raise RuntimeError("Pilot has truncated responses; inspect the saved comparison before collecting common trajectories")
        config = load_config(args.config)
        pilot_snapshot = ROOT / "artifacts/prompt_comparison" / args.prompt_run / "config.snapshot.json"
        if config != read(pilot_snapshot):
            raise ValueError("Configuration changed since the pilot; use the same frozen budget/settings")
        for benchmark in config["benchmarks"]:
            get_task_prompt(benchmark, config.get("prompt_version"),
                            "read_pages" if benchmark == "mpdocvqa" else None)
        collect_lock = (lock_dir / f"no-memory_{args.run}.lock").open("a")
        fcntl.flock(collect_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        status("collecting_no_memory", prompt_analysis=analysis,
               expected_tasks=400, formal_memory_methods_started=False)
        no_memory(config, args.run, ["mathvision", "vision2code"], retry_failed=False)
        status("complete", expected_tasks=400, completed_tasks=400,
               formal_memory_methods_started=False, prompt_report=analysis["report"])
    except BaseException as exc:
        status("error", error=f"{type(exc).__name__}: {exc}")
        raise


if __name__ == "__main__":
    main()
