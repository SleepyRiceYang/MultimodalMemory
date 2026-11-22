#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Dict


AUX_SUFFIXES = (
    "_0observe.json",
    "_1think.json",
    "_1think_action.json",
    "_reflect.json",
)
AUX_ERROR_DIRS = {"action_desc_errors", "action_predict_errors"}


def should_skip(path: Path) -> bool:
    name = path.name
    if name.startswith("."):
        return True
    if any(parent in AUX_ERROR_DIRS for parent in path.parts):
        return True
    if any(name.endswith(suffix) for suffix in AUX_SUFFIXES):
        return True
    return False


def load_json(path: Path) -> Dict[str, object] | None:
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except Exception as exc:
        print(f"[report_field_counts] Failed to load {path}: {exc}")
        return None
    if not isinstance(payload, dict):
        print(f"[report_field_counts] Skipping {path} (not a JSON object).")
        return None
    return payload


def count_fields(root: Path) -> None:
    counter: Counter[str] = Counter()
    total_files = 0
    for json_path in root.rglob("*.json"):
        if should_skip(json_path):
            continue
        record = load_json(json_path)
        if record is None:
            continue
        total_files += 1
        for key, value in record.items():
            if value is None:
                continue
            if isinstance(value, str) and not value.strip():
                continue
            if isinstance(value, (list, dict)) and not value:
                continue
            counter[key] += 1

    if total_files == 0:
        print(f"[report_field_counts] No valid JSON artifacts found under {root}")
        return

    print(f"[report_field_counts] JSON records scanned: {total_files}")
    for key in sorted(counter):
        print(f"[report_field_counts]   field '{key}': {counter[key]}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Report field counts for saved COAT artifacts.")
    parser.add_argument(
        "path",
        type=Path,
        help="Path to the experiment directory (e.g., artifacts/runs/.../flows/<model>).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    target = args.path
    if not target.exists():
        raise FileNotFoundError(f"Target path {target} does not exist.")
    count_fields(target)


if __name__ == "__main__":
    main()
