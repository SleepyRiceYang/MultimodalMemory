from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

from ..config import write_json
from ..models.local import LocalModel, image_hash, parse_object
from ..prompts import CONSTRUCTION, ksi_style, magnet_adapted, evoskill_adapted
from ..prompts.memory_context import SOURCE_IMAGE_LABEL
from .bank import inside


def _entry(method: str, benchmark: str, index: int, source_ids: list[str],
           text: str, index_text: str, images: list[dict] | None = None, **extra) -> dict:
    return {"id": f"{benchmark}/{method}/{index}", "benchmark": benchmark,
            "source_ids": source_ids, "text": text, "index_text": index_text,
            "images": images or [], **extra}


def _call(model, prompt, episodes, logs: Path, label: str, extra=None):
    payload = {"episodes": episodes, "extra": extra}
    assets = [(SOURCE_IMAGE_LABEL.format(source_id=episode["id"], index=i), inside(logs.parent, asset["path"]))
              for episode in episodes for i, asset in enumerate(episode["images"])]
    log_path = logs / f"{label}.json"
    user_prompt = json.dumps(payload, ensure_ascii=False)
    schema = construction_schema(label, episodes)
    if log_path.exists():
        response = json.loads(log_path.read_text())
        if (response["system"] != prompt or response["prompt"] != user_prompt
                or response.get("json_schema") != schema):
            raise ValueError("Construction request differs from preserved call; use a new run")
    else:
        response = model.generate(prompt, user_prompt, assets,
                                  max_tokens=model.config.get("construction_max_tokens", 1800),
                                  json_mode=True, json_schema=schema)
        write_json(log_path, response)
    if response["finish_reason"] == "length":
        raise ValueError(f"Construction response truncated: {label}")
    return parse_object(response["text"])


def construction_schema(label, episodes):
    """Constrain references to real input evidence, not model-invented filenames."""
    string = {"type": "string", "maxLength": 320}
    strings = {"type": "array", "items": {"type": "string", "maxLength": 200}, "maxItems": 2}
    def obj(fields):
        return {"type": "object", "properties": fields, "required": list(fields), "additionalProperties": False}
    def array(item, minimum=0, maximum=10):
        return {"type": "array", "items": item, "minItems": minimum, "maxItems": maximum}
    if label.endswith("_ksi_forum"):
        item = obj({"claim": {"type": "string", "maxLength": 600},
                    "evidence_ids": array({"enum": [e["id"] for e in episodes]}, 1),
                    "counterexample_or_limit": {"type": "string", "maxLength": 400}})
        return obj({"claims": array(item, 1, 3)})
    if label.endswith("_ksi_distill"):
        item = obj({"claim": string, "conditions": strings, "exclusions": strings,
                    "evidence_ids": array({"enum": [e["id"] for e in episodes]}, 1),
                    "rejected_hypotheses": strings, "checks": strings})
        return obj({"knowledge": array(item, 1, 3 if len(episodes) == 1 else 6)})
    if label.endswith("_magnet"):
        variants = []
        for episode in episodes:
            patch = obj({"image_index": {"enum": list(range(len(episode["images"])))},
                         "box": {"type": "array", "items": {"type": "number", "minimum": 0, "maximum": 1}, "minItems": 4, "maxItems": 4},
                         "object": string, "function": string})
            variants.append(obj({"source_id": {"const": episode["id"]},
                                 "workflow": array(string, 1, 8), "patches": array(patch, 1, 2)}))
        return obj({"entries": array({"anyOf": variants}, 1, max(1, len(episodes)))})
    if label.endswith("_darwinian"):
        variants = [obj({"source_id": {"const": e["id"]}, "precondition": string, "goal": string,
                         "step_indices": array({"enum": list(range(len(e["steps"])))}, 1, len(e["steps"])),
                         "verification": string}) for e in episodes]
        return obj({"units": array({"anyOf": variants}, 1, max(1, 2 * len(episodes)))})
    return None


def build_memories(root: Path, config: dict) -> Path:
    if config.get('memory_construction', {}).get('protocol') == 'paired_ksi_magnet_v1':
        from .representations import build_bank
        episodes = [json.loads(line) for line in (root / 'history.jsonl').read_text().splitlines() if line]
        return build_bank(root, config, episodes)
    output = root / "bank.json"
    if output.exists():
        raise FileExistsError(f"Memory already built: {output}; use a new artifact directory")
    episodes = [json.loads(line) for line in (root / "history.jsonl").read_text().splitlines() if line]
    if not episodes or len({e["id"] for e in episodes}) != len(episodes):
        raise ValueError("History must contain unique episodes")
    methods = {method: [] for method in config["baselines"] if method != "no_memory"}
    model = LocalModel(config["model"])
    sources = [{"id": e["id"], "benchmark": e["benchmark"], "provenance": e["provenance"],
                "image_hashes": [image_hash(inside(root, a["path"])) for a in e["images"]]}
               for e in episodes]
    for benchmark in config["benchmarks"]:
        group = [e for e in episodes if e["benchmark"] == benchmark]
        by_id = {e["id"]: e for e in group}
        for index, episode in enumerate(group):
            transcript = json.dumps({key: episode[key] for key in ("task", "steps", "feedback")}, ensure_ascii=False)
            for method in ("raw_multimodal", "raw_text", "raw_visual"):
                if method not in methods:
                    continue
                methods[method].append(_entry(method, benchmark, index, [episode["id"]],
                    "" if method == "raw_visual" else transcript, episode["task"],
                    [] if method == "raw_text" else episode["images"]))
        for method in ("ksi_style", "magnet_adapted", "darwinian_adapted", "evoskill_adapted"):
            if method not in methods:
                continue
            print(f"Constructing {benchmark}/{method}", flush=True)
            logs = root / "construction_calls"
            if method == "ksi_style":
                discussion = _call(model, ksi_style.REVIEW_SYSTEM, group, logs, benchmark + "_ksi_forum")
                data = _call(model, CONSTRUCTION[method], group, logs, benchmark + "_ksi_distill", discussion)
                for i, item in enumerate(data["knowledge"]):
                    ids = item["evidence_ids"]
                    if not ids or not set(ids) <= by_id.keys():
                        raise ValueError("KSI knowledge cites unknown evidence")
                    text = json.dumps(item, ensure_ascii=False)
                    methods[method].append(_entry(method, benchmark, i, ids, text,
                        " ".join(by_id[s]["task"] for s in ids) + " " + text))
            elif method == "magnet_adapted":
                data = _call(model, CONSTRUCTION[method], group, logs, benchmark + "_magnet")
                for i, item in enumerate(data["entries"]):
                    episode = by_id[item["source_id"]]
                    text = magnet_adapted.WORKFLOW_PREFIX + "\n".join(item["workflow"])
                    methods[method].append(_entry(method, benchmark, i * 100, [episode["id"]], text, episode["task"]))
                    for j, patch in enumerate(item["patches"][:2]):
                        coords = [float(x) for x in patch["box"]]
                        if len(coords) != 4 or not (0 <= coords[0] < coords[2] <= 1 and 0 <= coords[1] < coords[3] <= 1):
                            raise ValueError("Invalid visual-patch box")
                        source = inside(root, episode["images"][patch["image_index"]]["path"])
                        target = root / "patches" / f"{benchmark}_{i}_{j}.png"
                        target.parent.mkdir(exist_ok=True)
                        with Image.open(source) as image:
                            box = (int(coords[0] * image.width), int(coords[1] * image.height),
                                   int(coords[2] * image.width), int(coords[3] * image.height))
                            if box[2] <= box[0] or box[3] <= box[1]:
                                raise ValueError("Empty visual patch")
                            image.crop(box).save(target)
                        text = magnet_adapted.PATCH_TEMPLATE.format(object=patch["object"], function=patch["function"])
                        methods[method].append(_entry(method, benchmark, i * 100 + j + 1,
                            [episode["id"]], text, episode["task"] + " " + text,
                            [{"path": str(target.relative_to(root)), "role": "function_visual_patch"}]))
            elif method == "darwinian_adapted":
                data = _call(model, CONSTRUCTION[method], group, logs, benchmark + "_darwinian")
                for i, item in enumerate(data["units"]):
                    episode = by_id[item["source_id"]]
                    steps = [episode["steps"][int(j)] for j in item["step_indices"]]
                    if not steps:
                        raise ValueError("Empty subtask unit")
                    text = json.dumps({**item, "trajectory": steps}, ensure_ascii=False)
                    methods[method].append(_entry(method, benchmark, i, [episode["id"]], text,
                        episode["task"] + " " + item["precondition"] + " " + item["goal"], episode["images"],
                        statistics={"successes": int(episode["success"]), "failures": int(not episode["success"]),
                                    "uses": 1, "steps": len(steps)}))
            else:
                initial = _call(model, CONSTRUCTION[method], group, logs, benchmark + "_evoskill_initial")
                # The critic sees observed outcomes, not private solver reasoning or skill contents.
                public = [{"id": e["id"], "task": e["task"], "feedback": e["feedback"],
                           "images": e["images"]} for e in group]
                critique = _call(model, evoskill_adapted.CRITIC_SYSTEM, public, logs, benchmark + "_evoskill_critic")
                data = _call(model, CONSTRUCTION[method] + evoskill_adapted.REVISION_SUFFIX,
                    group, logs, benchmark + "_evoskill_revision", {"initial": initial, "critic": critique})
                for i, item in enumerate(data["skills"]):
                    ids = item["source_ids"]
                    if not ids or not set(ids) <= by_id.keys():
                        raise ValueError("Skill cites unknown source")
                    folder = root / "skills" / benchmark / f"skill_{i}"
                    folder.mkdir(parents=True, exist_ok=True)
                    files = {}
                    for name in ("plan", "backup", "recover"):
                        path = folder / f"{name}.md"
                        path.write_text(str(item[name]) + "\n")
                        files[name] = str(path.relative_to(root))
                    path = folder / "failure_cases.md"
                    path.write_text("\n".join(item["failure_cases"]) + "\n")
                    files["failure_cases"] = str(path.relative_to(root))
                    write_json(folder / "meta_info.json", {"intent": item["intent"], "source_ids": ids,
                               "revision": 1, "revision_validation": "not independently evaluated"})
                    methods[method].append(_entry(method, benchmark, i, ids, "", item["intent"] + " " +
                        " ".join(by_id[s]["task"] for s in ids), files=files))
            if not any(e["benchmark"] == benchmark for e in methods[method]):
                raise ValueError(f"No memory entries constructed: {benchmark}/{method}")
    asset_paths = {asset["path"] for episode in episodes for asset in episode["images"]}
    for entries in methods.values():
        for entry in entries:
            asset_paths.update(asset["path"] for asset in entry["images"])
            asset_paths.update(entry.get("files", {}).values())
    write_json(output, {"schema_version": 1, "mode": config["data_mode"], "sources": sources,
                        "methods": methods, "frozen": True,
                        "asset_hashes": {p: image_hash(inside(root, p)) for p in sorted(asset_paths)}})
    return output
