"""Build memory baselines from completed common no-memory trajectories; evaluate frozen banks."""
from __future__ import annotations

import argparse
import fcntl
import json
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path

from ..config import ROOT, load_config
from ..memory import MemoryBank
from ..models.local import image_hash
from .pilot import construct, evaluate, read
from .run_100_100_nomemory import atomic_json, eta_text, experiment_config, heartbeat, snapshot_run, verify_manifest


def prepare_evaluation_targets(source: Path, destination: Path, manifest: dict) -> None:
    """Copy test references for the separate scorer; never add them to task inputs/history."""
    for task in manifest['test']:
        original = source / 'evaluation_targets' / f"{task['id']}.json"
        if not original.is_file():
            raise FileNotFoundError(f'Missing official scoring reference: {original}')
        target = destination / 'evaluation_targets' / original.name
        if target.exists():
            if target.read_bytes() != original.read_bytes():
                raise ValueError(f'Preserved scoring reference changed: {target}')
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(original, target)


def prepare(source: Path, destination: Path, benchmark: str, config: dict) -> dict:
    source_settings = experiment_config(read(source / "config.snapshot.json"))
    formal_settings = experiment_config(config)
    # Retrieval is downstream of the shared no-memory trajectories.
    source_settings.pop("retrieval", None)
    formal_settings.pop("retrieval", None)
    # Memory representation is downstream of the common no-memory solver.
    for key in ('baselines', 'memory_construction'):
        source_settings.pop(key, None)
        formal_settings.pop(key, None)
    if source_settings != formal_settings:
        raise ValueError("Shared trajectory configuration differs from formal experiment configuration")
    manifest = verify_manifest(benchmark, config, source / "dataset_manifest.json")
    # Validate both sets before importing anything. Test episodes are never memory sources.
    for split in ("memory", "test"):
        for task in manifest[split]:
            result_path = source / "runs/no_memory" / split / str(task["id"]) / "attempt_0/result.json"
            if not result_path.exists() or read(result_path).get("status") != "completed":
                raise ValueError(f"Missing completed common trajectory: {result_path}")
    snapshot_run(destination, config)
    atomic_json(destination / "dataset_manifest.json", manifest)
    if benchmark == 'mpdocvqa':
        prepare_evaluation_targets(source, destination, manifest)
    for task in manifest["memory"]:
        task_id = str(task["id"])
        source_episode = source / "trajectories/memory" / f"{task_id}.json"
        episode = read(source_episode)
        source_result = read(source / 'runs/no_memory/memory' / task_id / 'attempt_0/result.json')
        episode['generation_truncated'] = source_result.get('generation_truncated', False)
        if episode["split"] != "memory" or episode["prompt_version"] != config["prompt_version"]:
            raise ValueError(f"Untrusted memory source: {source_episode}")
        history = destination / "histories" / task_id
        imported = history / "episode.json"
        if imported.exists():
            if read(imported)["provenance"]["source_episode_sha256"] != image_hash(source_episode):
                raise ValueError("Common source trajectory changed after import")
            continue
        assets = []
        for index, asset in enumerate(episode["images"]):
            path = Path(asset["path"])
            if image_hash(path) != asset["sha256"]:
                raise ValueError(f"Source image changed: {path}")
            target = history / "images" / f"{index}{path.suffix}"
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
            assets.append({"path": str(target.relative_to(history)), "role": asset["role"]})
        episode["images"] = assets
        episode["provenance"].update(source_episode_sha256=image_hash(source_episode),
                                      source_episode=str(source_episode), model=config["model"]["name"])
        atomic_json(imported, episode)
        (history / "history.jsonl").write_text(json.dumps(episode, ensure_ascii=False) + "\n")
    for task in manifest["test"]:
        task_id = str(task["id"])
        target = destination / "runs/no_memory" / task_id / "seed_0/attempt_0"
        if not target.exists():
            shutil.copytree(source / "runs/no_memory/test" / task_id / "attempt_0", target)
    atomic_json(destination / "common_trajectory_provenance.json", {
        "source_run": str(source), "source_config_sha256": image_hash(source / "config.snapshot.json"),
        "source_manifest_sha256": image_hash(source / "dataset_manifest.json"),
        "memory_source_ids": [str(t["id"]) for t in manifest["memory"]],
        "test_source_ids": [], "common_sources_for_all_baselines": True})
    return manifest


def run_evaluation(root: Path, manifest: dict, config: dict, retry_failed: bool) -> None:
    bank = MemoryBank(root / "bank.json")
    for task in manifest["test"]:
        path = root / "runs/no_memory" / str(task["id"]) / "seed_0/attempt_0/result.json"
        result = read(path)
        result.update(bank_sha256=bank.sha256, bank_frozen=bank.verify_frozen(),
                      seed=0, attempt_id=0, identity="reference_control", reused_common_no_memory=True)
        atomic_json(path, result)
    started = time.monotonic()
    durations = []
    for index, task in enumerate(manifest["test"], 1):
        if retry_failed:
            for method in config["baselines"]:
                path = root / "runs" / method / str(task["id"]) / "seed_0/attempt_0/result.json"
                if path.exists() and read(path).get("status") != "completed":
                    archive = path.parent / "errors" / (str(time.time_ns()) + ".json")
                    archive.parent.mkdir(parents=True, exist_ok=True)
                    path.rename(archive)
        task_start = time.monotonic()
        with heartbeat(f"FORMAL {manifest['benchmark']}/{task['id']}"):
            evaluate(root, manifest, config, resume=True, task_ids=[task["id"]])
        durations.append(time.monotonic() - task_start)
        completed = sum(read(p).get("status") == "completed" for p in
                        (root / "runs").glob("*/*/seed_0/attempt_0/result.json")
                        if p.relative_to(root / 'runs').parts[0] in config['baselines'])
        atomic_json(root / "progress.json", {"completed_results": completed,
            "expected_results": len(manifest["test"]) * len(config["baselines"]),
            "updated_at": datetime.now(timezone.utc).isoformat()})
        print(f"FORMAL {manifest['benchmark']} task {eta_text(index,len(manifest['test']),started,sum(durations)/len(durations))}; results={completed}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "config/qwen3_8_27b_100_100.json")
    parser.add_argument("--source-run", required=True, help="Completed common no-memory run ID")
    parser.add_argument("--run", required=True, help="Separate formal baseline run ID")
    parser.add_argument("--stage", choices=("prepare", "build", "evaluate", "all"), default="all")
    parser.add_argument("--benchmarks", nargs="+", choices=("mathvision", "vision2code", "mpdocvqa"),
                        default=("mathvision", "vision2code"))
    parser.add_argument("--retry-failed", action="store_true")
    args = parser.parse_args()
    if args.source_run == args.run:
        raise ValueError("Formal results must use a separate run ID from common trajectories")
    config = load_config(args.config)
    lock_dir = ROOT / "artifacts/locks"
    lock_dir.mkdir(parents=True, exist_ok=True)
    lock = (lock_dir / f"formal_{args.run}.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    for benchmark in args.benchmarks:
        source, root = (ROOT / "artifacts" / benchmark / name for name in (args.source_run, args.run))
        if args.stage in ("prepare", "all"):
            manifest = prepare(source, root, benchmark, config)
        else:
            snapshot_run(root, config)
            manifest = read(root / "dataset_manifest.json")
        if args.stage in ("build", "all"):
            with heartbeat(f"BUILD {benchmark}: 100 common memory-source trajectories"):
                construct(root, manifest, config, resume=True)
        if args.stage in ("evaluate", "all"):
            run_evaluation(root, manifest, config, args.retry_failed)


if __name__ == "__main__":
    main()
