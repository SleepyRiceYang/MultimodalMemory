#!/usr/bin/env python3

"""
Remove `action_predict` fields whose paired `action_desc.type` is not in the allowed set.

Usage:
    python scripts/clear_unwanted_predictions.py <pairs_dir> \
        --allowed CLICK_ELEMENT SCROLL STOP INPUT PRESS_BACK PRESS_ENTER PRESS_HOME
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, Set


def normalize_allowed(values: Iterable[str]) -> Set[str]:
    return {value.strip() for value in values if value.strip()}


def should_keep_prediction(action_desc, allowed: Set[str]) -> bool:
    action_type = None
    if isinstance(action_desc, dict):
        action_type = action_desc.get("type") or action_desc.get("TYPE")
    elif isinstance(action_desc, str):
        action_type = action_desc
    if not isinstance(action_type, str):
        return False
    return action_type.strip() in allowed


def clean_predictions(root: Path, allowed: Set[str]) -> tuple[int, int, int]:
    processed = modified = skipped = 0
    for json_path in root.rglob("*.json"):
        processed += 1
        try:
            data = json.loads(json_path.read_text(encoding="utf-8"))
        except Exception as exc:
            skipped += 1
            print(f"[skip] {json_path}: {exc}")
            continue
        if not isinstance(data, dict):
            skipped += 1
            continue
        if "action_predict" not in data:
            continue
        if should_keep_prediction(data.get("action_desc"), allowed):
            continue
        data.pop("action_predict", None)
        json_path.write_text(json.dumps(data, indent=4) + "\n", encoding="utf-8")
        modified += 1
    return processed, modified, skipped


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root_dir", type=Path, help="Root directory containing pairs JSON files.")
    parser.add_argument(
        "--allowed",
        nargs="+",
        required=True,
        help="List of allowed action_desc types whose predictions should be kept.",
    )
    args = parser.parse_args()
    allowed = normalize_allowed(args.allowed)
    processed, modified, skipped = clean_predictions(args.root_dir, allowed)
    print(
        f"Processed {processed} files. Removed predictions from {modified} files. "
        f"Skipped {skipped} unreadable/non-JSON entries."
    )


if __name__ == "__main__":
    main()
