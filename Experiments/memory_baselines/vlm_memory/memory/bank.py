from __future__ import annotations

import json
import math
import re
from collections import Counter
from pathlib import Path

from ..models.local import image_hash
from ..prompts.memory_context import MEMORY_NOTICE, ENTRY_TEMPLATE, HISTORICAL_IMAGE_LABEL, SKILL_SECTION_TEMPLATE


def tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9_]+", text.lower())


def bm25(query: str, documents: list[str]) -> list[float]:
    """Small deterministic lexical index; no embedding model or training required."""
    docs = [Counter(tokens(doc)) for doc in documents]
    average = sum(sum(doc.values()) for doc in docs) / max(len(docs), 1) or 1
    scores = []
    for doc in docs:
        score = 0.0
        for token in set(tokens(query)):
            frequency = doc[token]
            df = sum(token in other for other in docs)
            idf = math.log(1 + (len(docs) - df + 0.5) / (df + 0.5))
            denominator = frequency + 1.5 * (0.25 + 0.75 * sum(doc.values()) / average)
            score += idf * frequency * 2.5 / denominator
        scores.append(score)
    return scores


def inside(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()) or not path.is_file():
        raise ValueError(f"Missing or out-of-root memory asset: {relative}")
    return path


class MemoryBank:
    def __init__(self, path: Path, config: dict | None = None):
        self.config = config
        self.retriever = None
        self.path = path
        self.root = path.parent
        self.data = json.loads(path.read_text())
        if 'ksi_visual' in self.data['methods'] and 'ksi_style' in self.data['methods']:
            textual = {e['id']: e for e in self.data['methods']['ksi_style']}
            visual = {e['id']: e for e in self.data['methods']['ksi_visual']}
            if (len(textual) != len(self.data['methods']['ksi_style'])
                    or len(visual) != len(self.data['methods']['ksi_visual']) or textual.keys() != visual.keys()):
                raise ValueError('Paired KSI node identities differ')
            for key in textual:
                if textual[key].get('images') or any(textual[key][field] != visual[key][field]
                        for field in ('text', 'index_text', 'source_ids', 'benchmark')):
                    raise ValueError('Paired KSI content differs beyond visual evidence')
        self.sha256 = image_hash(path)
        asset_names = set()
        for entries in self.data["methods"].values():
            for entry in entries:
                asset_names.update(asset["path"] for asset in entry.get("images", []))
                asset_names.update(entry.get("files", {}).values())
        self.asset_hashes = self.data.get("asset_hashes", {
            name: image_hash(inside(self.root, name)) for name in asset_names})
        self.verify_frozen()

    def verify_frozen(self) -> bool:
        if image_hash(self.path) != self.sha256:
            raise ValueError("Memory bank changed")
        for name, expected in self.asset_hashes.items():
            if image_hash(inside(self.root, name)) != expected:
                raise ValueError(f"Frozen memory asset changed: {name}")
        if self.retriever is not None and hasattr(self.retriever, "verify_frozen"):
            self.retriever.verify_frozen()
        return True

    def context(self, method: str, benchmark: str, task_id: str, query: str,
                current_image: Path, budget: dict, task: dict | None = None) -> dict:
        if method == "no_memory":
            return {"text": "", "images": [], "selected": [], "bank_sha256": self.sha256,
                    "retrieval_metadata": {"method": method}, "truncated": False}
        if method == 'ksi_style' and 'ksi_visual' in self.data['methods']:
            # Select and budget once against the visual arm, then remove images only.
            # This prevents different text/IDs when the image budget excludes a node.
            context = self.context('ksi_visual', benchmark, task_id, query, current_image, budget, task)
            context['retrieval_metadata']['paired_selection'] = 'ksi_style_shared_with_ksi_visual'
            context['retrieval_metadata']['reserved_visual_images'] = len(context['images'])
            context['images'] = []
            for selected in context['selected']:
                selected['images'] = []
            return context
        # Check the ENTIRE source bank, not just retrieval hits.
        current_hash = image_hash(current_image)
        for source in self.data["sources"]:
            if source["benchmark"] == benchmark and source["id"] == task_id:
                raise ValueError("Evaluation task ID occurs in memory sources")
            if current_hash in source["image_hashes"]:
                raise ValueError("Evaluation image occurs in memory sources")
        entries = [entry for entry in self.data["methods"][method]
                   if entry["benchmark"] == benchmark]
        dense = self.config and self.config.get("retrieval", {}).get("enabled", False)
        metadata = None
        if dense:
            if task is None:
                raise ValueError("Dense retrieval requires the full public task")
            if self.retriever is None:
                from .retrieval import DenseRetriever
                self.retriever = DenseRetriever(self, self.config)
            ranked, metadata = self.retriever.rank(method, task)
        scores = [] if dense else bm25(query, [entry["index_text"] for entry in entries])
        ranked = ranked if dense else []
        for entry, relevance in zip(entries, scores):
            adjusted = relevance
            if method == "darwinian_adapted":
                stats = entry["statistics"]
                risk = (stats["failures"] + 1) / (stats["successes"] + stats["failures"] + 2)
                if risk > budget["darwinian_risk_threshold"]:
                    continue
                # Smoothed reliability and step efficiency, frozen during evaluation.
                adjusted *= (1 - risk) * (1 + math.log1p(stats["uses"])) / math.sqrt(stats["steps"])
            if relevance > 0:
                ranked.append((adjusted, entry))
        if not dense:
            ranked.sort(key=lambda pair: (-pair[0], pair[1]["id"]))
        selected = ranked if dense else ranked[:budget["top_k"]]
        text_parts, images, traces = [], [], []
        remaining = budget["max_text_chars"] - len(MEMORY_NOTICE)
        truncated = False
        for score, entry in selected:
            if len(traces) >= budget["top_k"]:
                break
            if dense and len(images) + len(entry.get("images", [])) > budget["max_images"]:
                metadata.setdefault("budget_rejections", []).append({"id": entry["id"], "reason": "image_budget"})
                truncated = True
                continue
            text = entry["text"]
            if method == "evoskill_adapted":
                text = "\n".join(SKILL_SECTION_TEMPLATE.format(name=name, content=inside(self.root, path).read_text())
                                 for name, path in entry["files"].items())
            if method == "raw_visual":
                text = ""  # no historical task, answer, action, or transcript in visual-only input
            fragment = ENTRY_TEMPLATE.format(entry_id=entry["id"], text=text) if text else ""
            if len(fragment) > max(remaining, 0):
                truncated = True
                if dense:
                    metadata.setdefault("budget_rejections", []).append({"id": entry["id"], "reason": "text_budget"})
                continue  # do not cut a structured entry mid-way
            remaining -= len(fragment)
            text_parts.append(fragment)
            selected_images = []
            for asset in entry.get("images", []):
                if len(images) >= budget["max_images"]:
                    truncated = True
                    break
                path = inside(self.root, asset["path"])
                label = HISTORICAL_IMAGE_LABEL.format(number=len(images)+1, entry_id=entry["id"], role=asset["role"])
                images.append((label, path))
                selected_images.append({"path": str(path), "role": asset["role"]})
            traces.append({"id": entry["id"], "score": score,
                           "source_ids": entry["source_ids"], "images": selected_images,
                           "injected_text": text})
        historical_text = MEMORY_NOTICE + "".join(text_parts) if traces else ""
        return {"text": historical_text, "images": images, "selected": traces,
                "bank_sha256": self.sha256, "truncated": truncated,
                "retrieval_metadata": metadata or {"method": method, "index": "BM25",
                                       "query": query, "candidates": len(entries)}}
