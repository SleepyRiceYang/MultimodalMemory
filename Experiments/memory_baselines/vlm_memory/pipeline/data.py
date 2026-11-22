"""Deterministic pilot splits. Scoring targets never enter TaskInput."""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

from ..prompts.base_tasks import VISION2CODE_TASK

from ..config import WORKSPACE, write_json
from ..models.local import image_hash


def math_task(row):
    prompt = row["question"]
    if row.get("options") and "".join(row["options"]) != "ABCDE":
        prompt += "\n" + "\n".join(f"({chr(65+i)}) {v}" for i, v in enumerate(row["options"]))
    return {"id": str(row["id"]), "benchmark": "mathvision", "split": "test",
            "prompt": prompt, "query": "visual mathematics " + row["subject"] + " " + prompt,
            "images": [str(WORKSPACE / "Benchmark/MATH-Vision" / row["image"])],
            "metadata": {"subject": row["subject"], "level": row["level"]}}


def vision_rows(split):
    repo = WORKSPACE / "Benchmark/Vision2Code"
    sys.path.insert(0, str(repo))
    from vision2code.benchmark.run_benchmark import selected_rows, source_image_path
    data = repo / "data/kaggle/vision2code"
    return [(row, source_image_path(data, row)) for row in selected_rows(data, split, 0)
            if row.get("image_path") and (data / row["image_path"]).is_file()]


def mp_rows(split):
    repo = WORKSPACE / "Benchmark/MP-DocVQA-Framework"
    return json.loads((repo / "data/MP-DocVQA/qas" / f"{split}.json").read_text())["data"]


def select(benchmark: str, config: dict, output: Path):
    """Select solely using public IDs/metadata; reject exact image/doc overlap."""
    fixed_split_dir = config.get("fixed_split_dir")
    if fixed_split_dir:
        source = WORKSPACE / fixed_split_dir / benchmark
        manifest = json.loads((source / "dataset_manifest.json").read_text())
        if len(manifest.get("memory", [])) != config["sampling"]["history_count"]:
            raise ValueError(f"Fixed memory split count mismatch for {benchmark}")
        if len(manifest.get("test", [])) != config["sampling"]["test_count"]:
            raise ValueError(f"Fixed test split count mismatch for {benchmark}")
        if {str(x["id"]) for x in manifest["memory"]} & {str(x["id"]) for x in manifest["test"]}:
            raise ValueError(f"Fixed split contains task ID overlap for {benchmark}")
        memory_hashes = {h for x in manifest["memory"] for h in x.get("image_hashes", [])}
        test_hashes = {h for x in manifest["test"] for h in x.get("image_hashes", [])}
        if memory_hashes & test_hashes:
            raise ValueError(f"Fixed split contains exact image overlap for {benchmark}")
        if benchmark == "mpdocvqa":
            memory_docs = {str(x["metadata"]["doc_id"]) for x in manifest["memory"]}
            test_docs = {str(x["metadata"]["doc_id"]) for x in manifest["test"]}
            if memory_docs & test_docs:
                raise ValueError("Fixed MP-DocVQA split contains document overlap")
        for task in manifest["memory"] + manifest["test"]:
            for image in task["images"]:
                if not Path(image).is_file():
                    raise FileNotFoundError(f"Missing image in fixed split: {image}")
        write_json(output / "dataset_manifest.json", manifest)
        # Keep evaluator-only labels outside TaskInput and flatten IDs for the
        # runner's existing EvaluationTarget lookup. IDs are disjoint by set.
        for role in ("memory", "test"):
            target_dir = source / "evaluation_targets" / role
            if not target_dir.exists():
                continue
            for target_path in target_dir.glob("*.json"):
                target = json.loads(target_path.read_text())
                write_json(output / "evaluation_targets" / target_path.name, target)
        return manifest

    rng = random.Random(config["sampling"]["seed"])
    nh, nt = config["sampling"]["history_count"], config["sampling"]["test_count"]
    chosen, hashes, ids, docs = [], set(), set(), set()
    if benchmark == "mathvision":
        rows = [json.loads(s) for s in (WORKSPACE / "Benchmark/MATH-Vision/data/test.jsonl").read_text().splitlines()]
        rng.shuffle(rows)
        pools = [(rows, nh + nt, "test")]
    elif benchmark == "vision2code":
        # The official mini split is a subset of test. Split the mini pool ourselves.
        rows = vision_rows("test_mini")
        rng.shuffle(rows)
        pools = [(rows, nh + nt, "test_mini")]
    else:
        pools = []
        for split, count in (("train", nh), ("val", nt)):
            rows = [r for r in mp_rows(split) if len(r["page_ids"]) <= config["sampling"]["mpdocvqa_max_pages"]]
            rng.shuffle(rows)
            pools.append((rows, count, split))
    for rows, count, split in pools:
        added = 0
        for row in rows:
            target = None
            if benchmark == "mathvision":
                task = math_task(row)
            elif benchmark == "vision2code":
                meta, image = row
                task = {"id": meta["question_folder"], "benchmark": benchmark, "split": split,
                        "prompt": VISION2CODE_TASK,
                        "query": "recreate image visual code " + meta["category"],
                        "images": [str(image)], "metadata": {k: meta[k] for k in ("category", "source_dataset")}}
            else:
                task = {"id": str(row["questionId"]), "benchmark": benchmark, "split": split,
                        "prompt": row["question"], "query": "document question " + row["question"],
                        "images": [str(WORKSPACE / "Benchmark/MP-DocVQA-Framework/data/MP-DocVQA/images" / f"{p}.jpg") for p in row["page_ids"]],
                        "metadata": {"doc_id": row["doc_id"], "page_ids": row["page_ids"]}}
                target = {k: row[k] for k in ("answers", "answer_page_idx")}
            image_hashes = [image_hash(Path(p)) for p in task["images"]]
            doc = task["metadata"].get("doc_id")
            if task["id"] in ids or set(image_hashes) & hashes or (doc and doc in docs):
                continue
            task["image_hashes"] = image_hashes
            chosen.append(task)
            if target is not None:
                write_json(output / "evaluation_targets" / f"{task['id']}.json", target)
            ids.add(task["id"])
            hashes.update(image_hashes)
            if doc:
                docs.add(doc)
            added += 1
            if added == count:
                break
        if added != count:
            raise ValueError(f"Insufficient eligible distinct tasks: {benchmark}/{split}")
    manifest = {"benchmark": benchmark, "seed": config["sampling"]["seed"],
                "memory": chosen[:nh], "test": chosen[nh:],
                "selection": "seeded shuffle, unique IDs/image SHA256; MP unique doc_id and <=6 pages; Vision2Code released local image assets only",
                "near_duplicate_policy": "exact image and document exclusion; semantic/template near duplicates not certified",
                "dev": [], "hyperparameters": "fixed before pilot; no tuning on these five test tasks"}
    write_json(output / "dataset_manifest.json", manifest)
    return manifest
