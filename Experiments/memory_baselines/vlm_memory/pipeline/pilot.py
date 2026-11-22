"""Run real benchmark-internal 10-history/5-test experiments with audit traces.

python -m vlm_memory.pipeline.pilot --benchmark mathvision --stage all --run pilot_10_5_v1
Stages are immutable at the task boundary; --resume skips completed artifacts.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

from ..config import ROOT, WORKSPACE, load_config, write_json
from ..models import LocalModel
from ..models.local import image_hash, parse_object
from ..memory import MemoryBank
from ..memory.build import build_memories, construction_schema
from ..benchmarks.tasks import evaluate_math, evaluate_vision
from ..prompts import CONSTRUCTION, get_task_prompt
from ..prompts import ksi_style
from ..prompts.memory_context import CURRENT_IMAGE_LABEL, DOCUMENT_PAGE_LABEL
from .data import select


def read(path):
    return json.loads(path.read_text())


def call(model, folder, name, system, prompt, images, **kwargs):
    saved = folder / "calls" / f"{name}.json"
    if saved.exists():
        response = read(saved)
        expected_images = [(label, image_hash(Path(path))) for label, path in images]
        actual_images = [(image["label"], image["sha256"]) for image in response["images"]]
        if (response["system"] != system or response["prompt"] != prompt
                or actual_images != expected_images
                or response.get("requested_max_tokens", kwargs.get("max_tokens", model.config["max_tokens"]))
                    != kwargs.get("max_tokens", model.config["max_tokens"])):
            raise ValueError(f"Preserved generation request differs: {saved}")
        return response
    partial = saved.with_suffix(".partial.json")
    if partial.exists():
        archive = folder / "calls" / "partial_history" / f"{name}_{time.time_ns()}.json"
        archive.parent.mkdir(parents=True, exist_ok=True)
        partial.replace(archive)
    response = model.generate(system, prompt, images, progress_path=partial, **kwargs)
    write_json(folder / "calls" / f"{name}.json", response)
    return response


def document_schema(indices):
    common = {"evidence": {"type": "string"}}
    def variant(found, answer, page):
        return {"type": "object", "additionalProperties": False,
                "required": ["answer", "answer_page_idx", "evidence", "found"],
                "properties": {**common, "found": {"const": found}, "answer": answer, "answer_page_idx": page}}
    return {"anyOf": [variant(True, {"type": "string", "minLength": 1}, {"enum": indices}),
                       variant(False, {"const": ""}, {"type": "null"})]}


def execute(task, context, folder, config, dataset_root, prompt_version=None):
    """Generation sees TaskInput only; evaluation is invoked after generation."""
    folder.mkdir(parents=True, exist_ok=True)
    prompt_version = prompt_version or config.get("prompt_version")
    model = LocalModel(config["model"])
    started = time.monotonic()
    write_json(folder / "input.json", task)
    trace = {**context, "images": [(label, str(path)) for label, path in context["images"]]}
    write_json(folder / "memory_trace.json", trace)
    benchmark = task["benchmark"]
    history = "\n\n" + context["text"] if context["text"] else ""
    calls, steps = [], []
    if benchmark == "mpdocvqa":
        sys.path.insert(0, str(WORKSPACE / "Benchmark/MP-DocVQA-Framework"))
        READ_PAGES, read_user = get_task_prompt("mpdocvqa", prompt_version, "read_pages")
        AGGREGATE, aggregate_user = get_task_prompt("mpdocvqa", prompt_version, "aggregate")
        from local_eval.runner import validate_prediction
        readings = []
        invalid_document_output = None
        for start in range(0, len(task["images"]), 2):
            indices = list(range(start, min(start + 2, len(task["images"]))))
            images = [(DOCUMENT_PAGE_LABEL.format(index=i), Path(task["images"][i])) for i in indices]
            response = call(model, folder, f"pages_{start}", READ_PAGES,
                            read_user.format(question=task["prompt"], page_indices=indices) + history,
                            images + context["images"], max_tokens=768, json_schema=document_schema(indices))
            calls.append(response)
            try:
                prediction = validate_prediction(parse_object(response["text"]), indices)
            except (json.JSONDecodeError, ValueError) as exc:
                invalid_document_output = {"stage": f"pages_{start}",
                    "error": f"{type(exc).__name__}: {exc}", "response": response["text"],
                    "finish_reason": response.get("finish_reason")}
                steps.append({"kind": "invalid_page_reading", "text": response["text"],
                              "page_indices": indices, "error": invalid_document_output["error"]})
                break
            readings.append({"page_indices": indices, "reading": prediction})
            steps.append({"kind": "page_reading", "text": response["text"], "page_indices": indices})
        if invalid_document_output is None and len(readings) > 1:
            response = call(model, folder, "aggregate", AGGREGATE,
                            aggregate_user.format(question_json=json.dumps(task["prompt"], ensure_ascii=False),
                                                  readings_json=json.dumps(readings, ensure_ascii=False)) + history,
                            context["images"], max_tokens=768, json_schema=document_schema(list(range(len(task["images"])))))
            calls.append(response)
            try:
                prediction = validate_prediction(parse_object(response["text"]), list(range(len(task["images"]))))
            except (json.JSONDecodeError, ValueError) as exc:
                invalid_document_output = {"stage": "aggregate",
                    "error": f"{type(exc).__name__}: {exc}", "response": response["text"],
                    "finish_reason": response.get("finish_reason")}
                steps.append({"kind": "invalid_aggregation", "text": response["text"],
                              "error": invalid_document_output["error"]})
        write_json(folder / "page_readings.json", readings)
        if invalid_document_output is not None:
            write_json(folder / "invalid_prediction.json", invalid_document_output)
            prediction = {"answer": "", "answer_page_idx": None, "evidence": "", "found": False}
            evaluation = {"status": "invalid_json", "exact_match": False, "anls": 0.0,
                          "answer_page_accuracy": 0.0,
                          "failure_reason": "Invalid or schema-incompatible JSON from document reader."}
            primary, success = 0.0, False
        else:
            primary, success = None, None
        write_json(folder / "prediction.json", prediction)
        steps.append({"kind": "final_answer", "text": json.dumps(prediction, ensure_ascii=False)})
        # Only now load reference fields, in the separate scorer process.
        if invalid_document_output is None:
            code = ("import json,sys; from local_eval.runner import score; "
                "row=json.load(open(sys.argv[1])); pred=json.load(open(sys.argv[2])); "
                "print(json.dumps(score(row,pred)))")
            scored = subprocess.run(["/root/anaconda3/envs/mpdocvqa/bin/python", "-c", code,
                                 str(dataset_root / "evaluation_targets" / f"{task['id']}.json"),
                                 str(folder / "prediction.json")],
                                cwd=WORKSPACE / "Benchmark/MP-DocVQA-Framework", capture_output=True, text=True, timeout=90)
            (folder / "evaluator.log").write_text(scored.stdout + scored.stderr)
            if scored.returncode:
                raise RuntimeError(scored.stderr[-1500:])
            evaluation = json.loads(scored.stdout)
            primary = float(evaluation["anls"])
            success = bool(evaluation["exact_match"])
        text = json.dumps(prediction, ensure_ascii=False)
    else:
        system, user_template = get_task_prompt(benchmark, prompt_version)
        user_prompt = user_template.format(task=task["prompt"])
        response = call(model, folder, "generation", system,
                        user_prompt + history,
                        [(CURRENT_IMAGE_LABEL, Path(task["images"][0]))] + context["images"])
        calls.append(response)
        text = response["text"]
        (folder / "response.txt").write_text(text)
        steps.append({"kind": "model_response", "text": text})
        adapted = {**task, "image": Path(task["images"][0])}
        if benchmark == "mathvision":
            evaluation = evaluate_math(folder, adapted, text, "/root/anaconda3/envs/mathvision/bin/python")
            primary, success = float(evaluation["correct"]), bool(evaluation["correct"])
        else:
            evaluation = evaluate_vision(folder, adapted, text, config["model"], config["vision2code"])
            primary = evaluation.get("rating_0_to_5")
            success = bool(evaluation["render_success"] and primary is not None and primary >= 3)
    write_json(folder / "evaluation.json", evaluation)
    if evaluation.get("rating_status") == "error":
        raise RuntimeError("Vision2Code evaluation failed: " + evaluation.get("rating_error", "unknown error"))
    usage = {k: sum(c["usage"].get(k, 0) for c in calls) for k in ("prompt_tokens", "completion_tokens", "total_tokens")}
    result = {"status": "completed", "benchmark": benchmark, "task_id": task["id"],
              "split": task["split"], "prediction": text, "evaluation": evaluation,
              "score": primary, "success": success, "steps": steps, "usage": usage,
              "model_calls": len(calls), "seconds": time.monotonic() - started,
              "generation_truncated": any(c["finish_reason"] == "length" for c in calls),
              "memory_entries": len(context["selected"]), "memory_images": len(context["images"]),
              "memory_chars": len(context["text"]), "memory_truncated": context["truncated"]}
    write_json(folder / "result.json", result)
    return result


def empty_context():
    return {"text": "", "images": [], "selected": [], "truncated": False}


def collect(root, manifest, config, resume):
    for task in manifest["memory"]:
        folder = root / "histories" / task["id"]
        episode_path = folder / "episode.json"
        if resume and episode_path.exists():
            continue
        if episode_path.exists():
            raise FileExistsError(episode_path)
        print(f"COLLECT {task['benchmark']} {task['id']}", flush=True)
        result = execute(task, empty_context(), folder / "execution", config, root)
        images = []
        for i, source in enumerate(task["images"]):
            target = folder / "images" / f"page_{i}{Path(source).suffix}"
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
            images.append({"path": str(target.relative_to(folder)), "role": f"source_observation_{i}"})
        rendered = folder / "execution/rendered_image.png"
        if task["benchmark"] == "vision2code" and result["evaluation"].get("render_success"):
            images.append({"path": str(rendered.relative_to(folder)), "role": "render_after_action"})
        episode = {"id": task["id"], "benchmark": task["benchmark"], "task": task["prompt"],
                   "images": images, "steps": result["steps"], "feedback": result["evaluation"],
                   "success": result["success"], "provenance": {"split": task["split"],
                       "manifest_sha256": image_hash(root / "dataset_manifest.json"), "model": config["model"]["name"]}}
        episode['generation_truncated'] = result.get('generation_truncated', False)
        write_json(episode_path, episode)
        (folder / "history.jsonl").write_text(json.dumps(episode, ensure_ascii=False) + "\n")


def construct(root, manifest, config, resume):
    if (root / "bank.json").exists():
        if resume:
            MemoryBank(root / "bank.json").verify_frozen()
            return
        raise FileExistsError(root / "bank.json")
    if config.get('memory_construction', {}).get('protocol') == 'paired_ksi_magnet_v1':
        from ..memory.representations import build_bank
        episodes = []
        for task in manifest['memory']:
            folder = root / 'histories' / task['id']
            episode = read(folder / 'episode.json')
            if episode['id'] != task['id'] or episode['benchmark'] != task['benchmark']:
                raise ValueError('Construction episode differs from manifest')
            for asset in episode['images']:
                asset['path'] = str(folder.relative_to(root) / asset['path'])
            episodes.append(episode)
        build_bank(root, config, episodes)
        return
    methods = {m: [] for m in config["baselines"] if m != "no_memory"}
    sources, assets = [], {}
    for task in manifest["memory"]:
        folder = root / "histories" / task["id"]
        if not (folder / "bank.json").exists():
            print(f"BUILD {manifest['benchmark']} source {task['id']} ({len(sources)+1}/{len(manifest['memory'])})", flush=True)
            build_memories(folder, {**config, "benchmarks": [task["benchmark"]]})
        bank = read(folder / "bank.json")
        prefix = folder.relative_to(root)
        sources.extend(bank["sources"])
        for name, digest in bank["asset_hashes"].items():
            assets[str(prefix / name)] = digest
        for method, entries in bank["methods"].items():
            for entry in entries:
                entry["id"] = f"{task['id']}/{entry['id']}"
                for image in entry["images"]:
                    image["path"] = str(prefix / image["path"])
                methods[method].append(entry)
    # Cross-task curation within this benchmark only, based on all ten source tasks.
    ids = {s["id"] for s in sources}
    payload = {"knowledge_from_tasks": [{"source_ids": entry["source_ids"],
                "knowledge": json.loads(entry["text"])} for entry in methods["ksi_style"]]}
    path = root / "construction_calls/cross_task_distill.json"
    if path.exists() and resume:
        response = read(path)
    else:
        response = LocalModel(config["model"]).generate(ksi_style.DISTILL_SYSTEM + ksi_style.CROSS_TASK_SUFFIX.format(task_count=len(manifest["memory"])),
            json.dumps(payload, ensure_ascii=False), [], max_tokens=config["model"].get("construction_max_tokens", 32768), json_mode=True,
            json_schema=construction_schema("cross_ksi_distill", [{"id": s["id"]} for s in sources]))
        write_json(path, response)
    if response["finish_reason"] == "length":
        raise ValueError("Cross-task knowledge truncated")
    knowledge = parse_object(response["text"])["knowledge"]
    if not knowledge:
        raise ValueError("Empty cross-task knowledge")
    methods["ksi_style"] = []
    for i, item in enumerate(knowledge):
        if not item["evidence_ids"] or not set(item["evidence_ids"]) <= ids:
            raise ValueError("Cross-task curation cites unknown source")
        text = json.dumps(item, ensure_ascii=False)
        methods["ksi_style"].append({"id": f"{manifest['benchmark']}/ksi_style/cross_task_{i}",
            "benchmark": manifest["benchmark"], "source_ids": item["evidence_ids"],
            "text": text, "index_text": text, "images": []})
    write_json(root / "bank.json", {"schema_version": 1, "mode": config["data_mode"],
        "sources": sources, "methods": methods, "frozen": True, "asset_hashes": assets})


def evaluate(root, manifest, config, resume, task_ids=None, claim_pending=False):
    bank = MemoryBank(root / "bank.json", config)
    expected_ids = {t["id"] for t in manifest["memory"]}
    if task_ids is not None and not set(task_ids) <= {t["id"] for t in manifest["test"]}:
        raise ValueError("Evaluation shard contains task IDs outside the frozen test set")
    if ({s["benchmark"] for s in bank.data["sources"]} != {manifest["benchmark"]}
            or {s["id"] for s in bank.data["sources"]} != expected_ids):
        raise ValueError("Memory source benchmark/IDs differ from the frozen dataset manifest")
    test_hashes = {h for t in manifest["test"] for h in t["image_hashes"]}
    source_hashes = {h for s in bank.data["sources"] for h in s["image_hashes"]}
    if test_hashes & source_hashes:
        raise ValueError("History/test image overlap")
    for task in manifest["test"]:
        if task_ids is not None and task["id"] not in task_ids:
            continue
        for method in config["baselines"]:
            folder = root / "runs" / method / task["id"] / "seed_0" / "attempt_0"
            path = folder / "result.json"
            if resume and path.exists():
                if read(path)["status"] == "running":
                    continue  # reserved by a shard; completion is checked at the matrix level
                if read(path)["status"] != "completed":
                    raise RuntimeError(f"Prior failed attempt preserved: {path}; use a new run")
                continue
            if path.exists():
                raise FileExistsError(path)
            if claim_pending:
                path.parent.mkdir(parents=True, exist_ok=True)
                # O_EXCL prevents another scheduled worker from claiming this result.
                with path.open("x") as handle:
                    json.dump({"status": "running", "benchmark": task["benchmark"],
                               "method": method, "task_id": task["id"]}, handle)
            print(f"EVALUATE {task['benchmark']} {method} {task['id']}", flush=True)
            started = time.monotonic()
            try:
                context = bank.context(method, task["benchmark"], task["id"], task["query"],
                                       Path(task["images"][0]), config["memory"], task=task)
                result = execute(task, context, folder, config, root)
                result.update(method=method, identity="reference_control" if method in ("no_memory", "raw_multimodal") else "reasoning_domain_adaptation",
                              bank_sha256=bank.sha256, bank_frozen=bank.verify_frozen(), seed=0, attempt_id=0)
                write_json(path, result)
            except Exception as exc:
                write_json(path, {"status": "error", "benchmark": task["benchmark"], "method": method,
                    "task_id": task["id"], "error": f"{type(exc).__name__}: {exc}", "seconds": time.monotonic() - started})
                raise
    results = [read(p) for p in sorted((root / "runs").glob("*/*/seed_0/attempt_0/result.json"))
               if p.relative_to(root / 'runs').parts[0] in config['baselines']]
    write_json(root / "summary.json", results)
    validation = {"history_tasks": len(manifest["memory"]), "test_tasks": len(manifest["test"]),
        "completed_results": len(results), "expected_results": len(manifest["test"]) * len(config["baselines"]),
        "all_completed": all(r["status"] == "completed" for r in results),
        "bank_frozen": bank.verify_frozen(), "history_test_exact_image_overlap": 0,
        "source_benchmarks": sorted({s["benchmark"] for s in bank.data["sources"]})}
    write_json(root / ("validation.json" if len(results) == validation["expected_results"] and validation["all_completed"]
                       else "checkpoint.json"), validation)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", required=True, choices=("mathvision", "mpdocvqa", "vision2code"))
    parser.add_argument("--stage", choices=("prepare", "collect", "build", "evaluate", "all"), default="all")
    parser.add_argument("--run", required=True)
    parser.add_argument("--config", type=Path, default=ROOT / "config/pilot_10_5.json")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--task-ids", nargs="+", help="Disjoint evaluation shard; does not change the frozen dataset")
    parser.add_argument("--claim-pending", action="store_true", help="Reserve each pending result before issuing inference")
    args = parser.parse_args()
    config = load_config(args.config)
    root = ROOT / "artifacts" / args.benchmark / args.run
    snapshot = root / "config.snapshot.json"
    if snapshot.exists() and read(snapshot) != config:
        raise ValueError("Configuration changed; use a new run ID")
    write_json(snapshot, config)
    dataset = root / "dataset_manifest.json"
    manifest = read(dataset) if dataset.exists() else select(args.benchmark, config, root)
    if args.stage in ("all", "collect"):
        collect(root, manifest, config, args.resume)
    if args.stage in ("all", "build"):
        construct(root, manifest, config, args.resume)
    if args.stage in ("all", "evaluate"):
        evaluate(root, manifest, config, args.resume, args.task_ids, args.claim_pending)


if __name__ == "__main__":
    main()
