from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

from .base import DocStatus


COUNTER_FILENAME = "memory_counter.json"
DEFAULT_WEIGHTS = (0.7, 0.3)


def load_global_counter(working_dir: str) -> int:
    path = Path(working_dir) / COUNTER_FILENAME
    if not path.exists():
        return 0
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return 0
    return int(payload.get("counter", 0))


def save_global_counter(working_dir: str, counter: int) -> None:
    path = Path(working_dir) / COUNTER_FILENAME
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"counter": int(counter)}), encoding="utf-8")


def reset_global_counter(working_dir: str) -> None:
    save_global_counter(working_dir, 0)


def _parse_timestamp(value: Any) -> float:
    if value is None:
        return 0.0
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value).timestamp()
        except ValueError:
            return 0.0
    return 0.0


def _normalize(values: List[float]) -> List[float]:
    if not values:
        return []
    min_val = min(values)
    max_val = max(values)
    if max_val == min_val:
        return [1.0 for _ in values]
    return [(val - min_val) / (max_val - min_val) for val in values]


def retention_score(counter: int, last_access: Any, access_count: Any) -> float:
    try:
        access_count = int(access_count)
    except (TypeError, ValueError):
        access_count = 0
    if access_count <= 0:
        return 0.0
    try:
        last_access = int(last_access)
    except (TypeError, ValueError):
        return 0.0
    gap = max(0, counter - last_access)
    return math.exp(-gap / max(1, access_count))


def ensure_doc_status(status: Dict[str, Any] | None, now_iso: str) -> Dict[str, Any]:
    if status is None:
        status = {
            "status": DocStatus.PROCESSED,
            "created_at": now_iso,
            "updated_at": now_iso,
            "metadata": {},
        }
    status.setdefault("created_at", now_iso)
    status.setdefault("updated_at", now_iso)
    metadata = status.get("metadata") or {}
    metadata.setdefault("last_access", 0)
    metadata.setdefault("access_count", 0)
    status["metadata"] = metadata
    return status


def update_doc_access(status: Dict[str, Any], counter: int, now_iso: str) -> Dict[str, Any]:
    metadata = status.get("metadata") or {}
    metadata["access_count"] = int(metadata.get("access_count", 0)) + 1
    metadata["last_access"] = int(counter)
    status["metadata"] = metadata
    status["updated_at"] = now_iso
    return status


def rank_candidates(
    candidates: List[Dict[str, Any]],
    status_map: Dict[str, Dict[str, Any] | None],
    counter: int,
    weights: Tuple[float, float] = DEFAULT_WEIGHTS,
) -> List[Dict[str, Any]]:
    now_iso = datetime.now().isoformat()
    created_times: List[float] = []
    prepared: List[Tuple[int, Dict[str, Any], Dict[str, Any]]] = []
    for idx, candidate in enumerate(candidates):
        doc_id = candidate.get("doc_id") or candidate.get("full_doc_id")
        status = ensure_doc_status(status_map.get(doc_id), now_iso)
        created_times.append(_parse_timestamp(status.get("created_at")))
        prepared.append((idx, candidate, status))

    recency_scores = _normalize(created_times)
    scored: List[Tuple[float, int, Dict[str, Any], Dict[str, Any]]] = []
    for idx, (original_idx, candidate, status) in enumerate(prepared):
        metadata = status.get("metadata") or {}
        retention = retention_score(
            counter,
            metadata.get("last_access"),
            metadata.get("access_count"),
        )
        score = weights[0] * retention + weights[1] * recency_scores[idx]
        scored.append((score, original_idx, candidate, status))

    scored.sort(key=lambda item: (-item[0], item[1]))
    return [item[2] for item in scored]


def normalize_image_entries(entries: Any, now_iso: str) -> List[Dict[str, Any]]:
    if not entries:
        return []
    if not isinstance(entries, list):
        return []
    normalized: List[Dict[str, Any]] = []
    for entry in entries:
        if isinstance(entry, str):
            normalized.append(
                {
                    "path": entry,
                    "created_at": now_iso,
                    "last_access": 0,
                    "access_count": 0,
                }
            )
        elif isinstance(entry, dict):
            path = entry.get("path")
            if path is None:
                continue
            normalized.append(
                {
                    "path": path,
                    "created_at": entry.get("created_at", now_iso),
                    "last_access": entry.get("last_access", 0),
                    "access_count": entry.get("access_count", 0),
                }
            )
    return normalized


def rank_image_entries(
    entries: List[Dict[str, Any]],
    counter: int,
    weights: Tuple[float, float] = DEFAULT_WEIGHTS,
) -> List[Dict[str, Any]]:
    created_times = [_parse_timestamp(entry.get("created_at")) for entry in entries]
    recency_scores = _normalize(created_times)
    scored: List[Tuple[float, int, Dict[str, Any]]] = []
    for idx, entry in enumerate(entries):
        retention = retention_score(
            counter,
            entry.get("last_access"),
            entry.get("access_count"),
        )
        score = weights[0] * retention + weights[1] * recency_scores[idx]
        scored.append((score, idx, entry))
    scored.sort(key=lambda item: (-item[0], item[1]))
    return [item[2] for item in scored]
