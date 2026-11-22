#!/usr/bin/env python3

"""
For samples whose action_desc type is outside the whitelist, drop every field
except screen_desc and action_result.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, Set

WHITELIST = {
    "CLICK_ELEMENT",
    "SCROLL",
    "STOP",
    "INPUT",
    "PRESS_BACK",
    "PRESS_ENTER",
    "PRESS_HOME",
}


def normalize_whitelist(values: Iterable[str]) -> Set[str]:
    return {v.strip() for v in values if v.strip()}


def action_type(payload) -> str | None:
    if isinstance(payload, dict):
        value = payload.get("type") or payload.get("TYPE")
    else:
        value = payload
    if isinstance(value, str):
        val = value.strip()
        return val
    return None


def prune_directory(root: Path, whitelist: Set[str]) -> tuple[int, int, int]:
    processed = pruned = skipped = 0
    for json_path in root.rglob("*.json"):
        processed += 1
        try:
            data = json.loads(json_path.read_text(encoding="utf-8"))
        except Exception as exc:  # pragma: no cover
            skipped += 1
            print(f"[skip] {json_path}: {exc}")
            continue
        if not isinstance(data, dict):
            skipped += 1
            continue
        desc_type = action_type(data.get("action_desc"))
        if desc_type and desc_type in whitelist:
            continue
        kept = {k: data.get(k) for k in ("screen_desc", "action_result") if k in data}
        if kept == data:
            continue
        json_path.write_text(json.dumps(kept, indent=4) + "\n", encoding="utf-8")
        pruned += 1
    return processed, pruned, skipped


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root_dir", type=Path, help="Directory tree to prune.")
    parser.add_argument(
        "--allowed",
        nargs="+",
        default=list(WHITELIST),
        help="Overrides for allowed action types.",
    )
    args = parser.parse_args()
    whitelist = normalize_whitelist(args.allowed)
    processed, pruned, skipped = prune_directory(args.root_dir, whitelist)
    print(
        f"Processed {processed} files; pruned {pruned}; skipped {skipped} unreadable entries."
    )


if __name__ == "__main__":
    main()
