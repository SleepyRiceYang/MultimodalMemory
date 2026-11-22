"""Build the fixed 100-memory / 100-test multimodal reasoning split manifests.

Run from the repository root:
  python3 Experiments/memory_baselines/vlm_memory/pipeline/build_100_100_splits.py

The output is a custom benchmark-internal memory-generalization protocol. It
does not reproduce the official zero-shot leaderboard split for MathVision or
Vision2Code. MP-DocVQA uses train/validation because its local test labels are
not released; selected documents are explicitly disjoint across the two sets.
"""
from __future__ import annotations

import csv
import hashlib
import json
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path

from ..prompts.base_tasks import VISION2CODE_TASK
from typing import Any

ROOT = Path(__file__).resolve().parents[4]
OUT = ROOT / "Experiments/memory_baselines/splits/100_100_seed_20261023"
SEED = 20261023
N_PER_SET = 100
PREVIOUS_RUNS = {
    "mathvision": "Experiments/memory_baselines/artifacts/mathvision/pilot_10_5_v3/dataset_manifest.json",
    "mpdocvqa": "Experiments/memory_baselines/artifacts/mpdocvqa/pilot_10_5_v4/dataset_manifest.json",
    "vision2code": "Experiments/memory_baselines/artifacts/vision2code/pilot_10_5_v1/dataset_manifest.json",
}


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def allocate(counts: dict[Any, int], n: int, minimum: int = 0) -> dict[Any, int]:
    """Proportional largest-remainder allocation with an optional per-group floor."""
    keys = sorted(counts, key=str)
    if sum(counts.values()) < n or n < minimum * len(keys):
        raise ValueError(f"Cannot allocate {n} over {counts} with minimum {minimum}")
    result = {k: minimum for k in keys}
    remaining = n - minimum * len(keys)
    weights = {k: max(0, counts[k] - minimum) for k in keys}
    weight_sum = sum(weights.values())
    if remaining and not weight_sum:
        raise ValueError("No allocation capacity remains")
    raw = {k: remaining * weights[k] / weight_sum for k in keys} if remaining else {k: 0 for k in keys}
    for k in keys:
        result[k] += int(raw[k])
    left = n - sum(result.values())
    for k in sorted(keys, key=lambda x: (-(raw[x] - int(raw[x])), str(x)))[:left]:
        result[k] += 1
    if any(result[k] > counts[k] for k in keys):
        raise ValueError("Allocation exceeds an available stratum")
    return result


def old_pilot(benchmark: str) -> tuple[set[str], set[str]]:
    """Return all previously used IDs and image hashes (or doc IDs for MP-DocVQA)."""
    d = read_json(ROOT / PREVIOUS_RUNS[benchmark])
    tasks = d["memory"] + d["test"]
    ids = {str(x["id"]) for x in tasks}
    if benchmark == "mpdocvqa":
        protected = {str(x["metadata"]["doc_id"]) for x in tasks}
    else:
        protected = {str(h) for x in tasks for h in x.get("image_hashes", [])}
    return ids, protected


def task_record(*, task_id: str, split: str, prompt: str, query: str,
                images: list[Path], metadata: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": str(task_id), "benchmark": metadata.pop("benchmark"), "split": split,
        "prompt": prompt, "query": query, "images": [str(p) for p in images],
        "metadata": metadata, "image_hashes": [sha256(p) for p in images],
    }


def save_benchmark(name: str, memory: list[dict], test: list[dict],
                   memory_targets: dict, test_targets: dict, details: dict) -> None:
    out = OUT / name
    write_json(out / "dataset_manifest.json", {
        "benchmark": name, "seed": SEED,
        "memory": memory, "test": test, "dev": [],
        "selection": details,
        "near_duplicate_policy": "Exclude prior pilot IDs and exact image hashes; MP-DocVQA also excludes document IDs across and within sets. Semantic/template near duplicates are not certified.",
        "test_policy": "Static evaluation: bank is frozen; no test feedback or answer is written to memory.",
    })
    for task_id, target in memory_targets.items():
        write_json(out / "evaluation_targets/memory" / f"{task_id}.json", target)
    for task_id, target in test_targets.items():
        write_json(out / "evaluation_targets/test" / f"{task_id}.json", target)


def build_math() -> dict:
    data = ROOT / "Benchmark/MATH-Vision"
    rows = [json.loads(line) for line in (data / "data/test.jsonl").read_text().splitlines() if line]
    old_ids, old_hashes = old_pilot("mathvision")
    candidates = []
    seen_hashes = set(old_hashes)
    for row in rows:
        task_id = str(row["id"])
        image = data / row["image"]
        if task_id in old_ids or not image.is_file():
            continue
        h = sha256(image)
        if h in seen_hashes:
            continue
        seen_hashes.add(h)
        candidates.append((row, image, h))
    by_level: dict[int, list] = defaultdict(list)
    for item in candidates:
        by_level[int(item[0]["level"])].append(item)
    quotas = allocate({k: len(v) for k, v in by_level.items()}, N_PER_SET)

    # Reserve one example from every subject in each set, while preserving exact
    # level quotas. Backtracking assigns each subject to a level with two unique
    # examples, then the remaining quota is sampled proportionally within level.
    by_subject_level: dict[str, dict[int, list]] = defaultdict(lambda: defaultdict(list))
    for item in candidates:
        by_subject_level[item[0]["subject"]][int(item[0]["level"])].append(item)
    subjects = sorted(by_subject_level, key=lambda s: (sum(len(v) >= 2 for v in by_subject_level[s].values()), s))
    remaining = dict(quotas)
    assignment: dict[str, int] = {}

    def assign(i: int) -> bool:
        if i == len(subjects):
            return True
        subject = subjects[i]
        options = [lev for lev, values in by_subject_level[subject].items()
                   if len({x[2] for x in values}) >= 2 and remaining.get(lev, 0) >= 1]
        options.sort(key=lambda lev: (-remaining[lev], -len(by_subject_level[subject][lev]), lev))
        for lev in options:
            # A shared level slot can serve one subject in both sets.
            if remaining[lev] <= 0:
                continue
            assignment[subject] = lev
            remaining[lev] -= 1
            if assign(i + 1):
                return True
            remaining[lev] += 1
            assignment.pop(subject, None)
        return False

    if not assign(0):
        raise ValueError("Could not cover all MathVision subjects in both sets under level quotas")

    rng = random.Random(SEED)
    memory_items, test_items = [], []
    used_hashes = set(old_hashes)
    for subject, lev in assignment.items():
        pool = [x for x in by_subject_level[subject][lev] if x[2] not in used_hashes]
        rng.shuffle(pool)
        chosen = pool[:2]
        if len(chosen) != 2:
            raise ValueError(f"Insufficient unique MathVision images for {subject}, level {lev}")
        memory_items.append(chosen[0])
        test_items.append(chosen[1])
        used_hashes.update(x[2] for x in chosen)
    for lev, quota in quotas.items():
        already = sum(int(x[0]["level"]) == lev for x in memory_items)
        need = quota - already
        pool = [x for x in by_level[lev] if x[2] not in used_hashes]
        rng.shuffle(pool)
        chosen = pool[:2 * need]
        if len(chosen) != 2 * need:
            raise ValueError(f"Insufficient MathVision tasks at difficulty level {lev}")
        memory_items.extend(chosen[:need])
        test_items.extend(chosen[need:])
        used_hashes.update(x[2] for x in chosen)

    def convert(item):
        row, image, _ = item
        prompt = row["question"]
        if row.get("options") and "".join(row["options"]) != "ABCDE":
            prompt += "\n" + "\n".join(f"({chr(65+i)}) {v}" for i, v in enumerate(row["options"]))
        return task_record(task_id=row["id"], split="test",
                           prompt=prompt, query=f"visual mathematics {row['subject']} {prompt}",
                           images=[image], metadata={"benchmark": "mathvision", "subject": row["subject"], "level": row["level"]})

    memory = [convert(x) for x in memory_items]
    test = [convert(x) for x in test_items]
    targets = {str(item[0]["id"]): {"answer": item[0]["answer"]} for item in memory_items + test_items}
    save_benchmark("mathvision", memory, test,
                   {k: v for k, v in targets.items() if k in {x["id"] for x in memory}},
                   {k: v for k, v in targets.items() if k in {x["id"] for x in test}},
                   {"source": "official MATH-Vision test.jsonl; custom within-benchmark split",
                    "stratification": "100 tasks per set; exact per-level quotas proportional to the available pool; every one of the 16 subjects appears in each set",
                    "difficulty_quotas_per_set": {str(k): v for k, v in sorted(quotas.items())},
                    "prior_pilot_ids_and_images_excluded": True,
                    "note": "MATH-Vision has no official training split. Memory examples are removed from this custom 200-task evaluation protocol; do not report as the official full-test leaderboard score."})
    return {"difficulty_quotas_per_set": quotas,
            "memory_levels": Counter(x["metadata"]["level"] for x in memory),
            "test_levels": Counter(x["metadata"]["level"] for x in test),
            "memory_subjects": len({x["metadata"]["subject"] for x in memory}),
            "test_subjects": len({x["metadata"]["subject"] for x in test})}


def page_bin(n: int) -> str:
    if n <= 2:
        return "01-02"
    if n <= 5:
        return "03-05"
    if n <= 10:
        return "06-10"
    return "11-20"


def build_mpdocvqa() -> dict:
    qas = ROOT / "Benchmark/MP-DocVQA-Framework/data/MP-DocVQA/qas"
    images_dir = ROOT / "Benchmark/MP-DocVQA-Framework/data/MP-DocVQA/images"
    old_ids, old_docs = old_pilot("mpdocvqa")
    rng = random.Random(SEED)

    def doc_groups(split: str) -> dict[str, list[dict]]:
        rows = read_json(qas / f"{split}.json")["data"]
        grouped: dict[str, list] = defaultdict(list)
        for row in rows:
            if str(row["questionId"]) in old_ids or str(row["doc_id"]) in old_docs:
                continue
            n_pages = len(row["page_ids"])
            if not 1 <= n_pages <= 20:
                continue
            paths = [images_dir / f"{p}.jpg" for p in row["page_ids"]]
            if not all(p.is_file() for p in paths):
                continue
            grouped[str(row["doc_id"])].append(row)
        return grouped

    train_docs, val_docs = doc_groups("train"), doc_groups("val")
    # Draw at document level; choose one QA per document so neither set is
    # dominated by documents with many questions. Reuse the validation page-bin
    # distribution for the training memory pool.
    def pagecount(group: list[dict]) -> int:
        counts = Counter(len(r["page_ids"]) for r in group)
        return counts.most_common(1)[0][0]

    val_by_bin: dict[str, list[str]] = defaultdict(list)
    for doc, group in val_docs.items():
        val_by_bin[page_bin(pagecount(group))].append(doc)
    quotas = allocate({k: len(v) for k, v in val_by_bin.items()}, N_PER_SET)
    val_selected: dict[str, str] = {}
    train_selected: dict[str, str] = {}
    used_test_docs: set[str] = set()
    for b, quota in quotas.items():
        pool = val_by_bin[b][:]
        rng.shuffle(pool)
        test_docs = pool[:quota]
        if len(test_docs) != quota:
            raise ValueError(f"Insufficient MP-DocVQA validation docs in page bin {b}")
        val_selected.update({doc: b for doc in test_docs})
        used_test_docs.update(test_docs)
    if used_test_docs & old_docs:
        raise AssertionError("MP-DocVQA test overlaps a prior pilot document")
    train_by_bin: dict[str, list[str]] = defaultdict(list)
    for doc, group in train_docs.items():
        if doc in used_test_docs:
            continue
        train_by_bin[page_bin(pagecount(group))].append(doc)
    for b, quota in quotas.items():
        pool = train_by_bin[b][:]
        rng.shuffle(pool)
        chosen = pool[:quota]
        if len(chosen) != quota:
            raise ValueError(f"Insufficient MP-DocVQA train docs in page bin {b}")
        train_selected.update({doc: b for doc in chosen})

    def select_row(doc: str, group: list[dict], expected_bin: str) -> dict:
        rows = [r for r in group if page_bin(len(r["page_ids"])) == expected_bin]
        rng.shuffle(rows)
        for row in rows:
            if str(row["questionId"]) not in old_ids:
                return row
        raise ValueError(f"No unused question in page bin {expected_bin} for document {doc}")

    def convert(doc: str, split: str, group: list[dict], expected_bin: str) -> tuple[dict, dict]:
        row = select_row(doc, group, expected_bin)
        task_id = str(row["questionId"])
        page_ids = row["page_ids"]
        paths = [images_dir / f"{p}.jpg" for p in page_ids]
        task = task_record(task_id=task_id, split=split, prompt=row["question"],
                           query="document question " + row["question"], images=paths,
                           metadata={"benchmark": "mpdocvqa", "doc_id": doc,
                                     "page_ids": page_ids, "page_count": len(page_ids),
                                     "page_bin": page_bin(len(page_ids))})
        target = {k: row[k] for k in ("answers", "answer_page_idx") if k in row}
        return task, target

    memory_pairs = [convert(doc, "train", train_docs[doc], train_selected[doc]) for doc in sorted(train_selected)]
    test_pairs = [convert(doc, "val", val_docs[doc], val_selected[doc]) for doc in sorted(val_selected)]
    memory, memory_targets = [x[0] for x in memory_pairs], {x[0]["id"]: x[1] for x in memory_pairs}
    test, test_targets = [x[0] for x in test_pairs], {x[0]["id"]: x[1] for x in test_pairs}
    save_benchmark("mpdocvqa", memory, test, memory_targets, test_targets,
                   {"source": "MP-DocVQA official train for memory and validation for locally scoreable held-out tasks",
                    "stratification": "one question per document; same page-count-bin quotas in both sets",
                    "page_count_bins": {k: v for k, v in sorted(quotas.items())},
                    "maximum_pages": 20,
                    "isolation": "100 unique doc_ids per set; no doc_id overlap across sets; exclude every doc_id used in the 10/5 pilot",
                    "prior_pilot_ids_and_documents_excluded": True,
                    "note": "The official test split has no local reference answers. This initial locally-scored test uses validation tasks; treat it as a pilot held-out set, not a sealed official test set."})
    return {"page_bin_quotas_per_set": quotas,
            "memory_page_bins": Counter(x["metadata"]["page_bin"] for x in memory),
            "test_page_bins": Counter(x["metadata"]["page_bin"] for x in test),
            "unique_memory_docs": len({x["metadata"]["doc_id"] for x in memory}),
            "unique_test_docs": len({x["metadata"]["doc_id"] for x in test}),
            "max_memory_pages": max(x["metadata"]["page_count"] for x in memory),
            "max_test_pages": max(x["metadata"]["page_count"] for x in test)}


def build_vision2code() -> dict:
    data_dir = ROOT / "Benchmark/Vision2Code/data/kaggle/vision2code"
    old_ids, old_hashes = old_pilot("vision2code")
    rows = list(csv.DictReader((data_dir / "manifest.csv").open(encoding="utf-8", newline="")))
    eligible = []
    for row in rows:
        if row["split"] != "test" or row.get("image_available", "").lower() != "true":
            continue
        image = data_dir / row["image_path"]
        if not image.is_file() or row["question_folder"] in old_ids or row["image_sha256"] in old_hashes:
            continue
        eligible.append((row, image))
    by_source: dict[str, list] = defaultdict(list)
    for item in eligible:
        by_source[item[0]["source_dataset"]].append(item)
    quotas = allocate({k: len(v) for k, v in by_source.items()}, N_PER_SET, minimum=1)
    rng = random.Random(SEED)
    memory_items, test_items = [], []
    used_hashes = set(old_hashes)
    for source, quota in sorted(quotas.items()):
        pool = by_source[source][:]
        rng.shuffle(pool)
        chosen = []
        for item in pool:
            if item[0]["image_sha256"] in used_hashes:
                continue
            chosen.append(item)
            used_hashes.add(item[0]["image_sha256"])
            if len(chosen) == 2 * quota:
                break
        if len(chosen) != 2 * quota:
            raise ValueError(f"Insufficient unique Vision2Code images for {source}")
        memory_items.extend(chosen[:quota])
        test_items.extend(chosen[quota:])

    def convert(item):
        row, image = item
        category = row["category"]
        return task_record(task_id=row["question_folder"], split="test",
                           prompt=VISION2CODE_TASK,
                           query=f"recreate image visual code {category}", images=[image],
                           metadata={"benchmark": "vision2code", "category": category,
                                     "source_dataset": row["source_dataset"],
                                     "source_subset": row.get("source_subset", ""),
                                     "source_id": row.get("source_id", ""),
                                     "release_split": row["split"]})

    memory, test = [convert(x) for x in memory_items], [convert(x) for x in test_items]
    save_benchmark("vision2code", memory, test, {}, {},
                   {"source": "official Vision2Code test pool, restricted to released local image assets",
                    "stratification": "proportional stratification by source dataset (15 source datasets); same quotas in both sets, covering all six domains",
                    "source_dataset_quotas_per_set": quotas,
                    "available_local_test_images_before_pilot_exclusion": 1994,
                    "prior_pilot_ids_and_images_excluded": True,
                    "isolation": "disjoint task IDs and exact image SHA256 across the 100/100 sets",
                    "note": "The local release has no source training images. Therefore this is a custom split within the official evaluation pool; report as benchmark-internal memory generalization, not as the official Vision2Code leaderboard score."})
    return {"source_quotas_per_set": quotas,
            "memory_domains": Counter(x["metadata"]["category"] for x in memory),
            "test_domains": Counter(x["metadata"]["category"] for x in test),
            "available_local_test_images": len(eligible)}


def main() -> None:
    if OUT.exists():
        for child in OUT.iterdir():
            if child.name == "README.md":
                continue
            if child.is_dir():
                shutil.rmtree(child)
            else:
                child.unlink()
    OUT.mkdir(parents=True, exist_ok=True)
    summary = {
        "seed": SEED,
        "tasks_per_set": N_PER_SET,
        "dev": "not included in this initial experiment",
        "benchmarks": {
            "mathvision": build_math(),
            "mpdocvqa": build_mpdocvqa(),
            "vision2code": build_vision2code(),
        },
    }
    write_json(OUT / "selection_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
