#!/usr/bin/env python3

"""
Remove action_predict entries whose predicted action is a click operation.

Usage:
    python scripts/remove_click_predictions.py <root_dir>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

CLICK_ACTIONS = {"click_element", "click"}


def predict_contains_click(payload: Any) -> bool:
    """Return True if any nested dict contains ACTION matching CLICK_ACTIONS."""
    if isinstance(payload, dict):
        action = payload.get("ACTION") or payload.get("action")
        if isinstance(action, str) and action.lower() in CLICK_ACTIONS:
            return True
        return any(predict_contains_click(value) for value in payload.values())
    if isinstance(payload, list):
        return any(predict_contains_click(item) for item in payload)
    return False


def strip_click_predictions(root: Path) -> tuple[int, int, int]:
    total = modified = skipped = 0
    for json_path in root.rglob("*.json"):
        total += 1
        try:
            data = json.loads(json_path.read_text(encoding="utf-8"))
        except Exception as exc:  # pragma: no cover - guardrail for corrupt files
            skipped += 1
            print(f"[skip] {json_path}: {exc}")
            continue
        if not isinstance(data, dict):
            skipped += 1
            continue
        ap = data.get("action_predict")
        if predict_contains_click(ap):
            data.pop("action_predict", None)
            json_path.write_text(json.dumps(data, indent=4) + "\n", encoding="utf-8")
            modified += 1
    return total, modified, skipped


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "root_dir",
        type=Path,
        help="Directory tree to scan for prediction JSON files.",
    )
    args = parser.parse_args()
    total, modified, skipped = strip_click_predictions(args.root_dir)
    print(
        f"Processed {total} files under {args.root_dir}. "
        f"Modified {modified}, skipped {skipped} corrupt/unreadable files."
    )


if __name__ == "__main__":
    main()
