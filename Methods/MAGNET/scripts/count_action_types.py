#!/usr/bin/env python3

"""
Count action types inside cached prediction JSONs.

By default this scans the `action_predict` field of every JSON file under
the provided directory and tallies how many times each `ACTION` token appears.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Sequence


def iter_actions(payload: Any, keys: Sequence[str]) -> Iterable[str]:
    """Yield every configured action key string found recursively."""
    if isinstance(payload, dict):
        for key in keys:
            value = payload.get(key)
            if isinstance(value, str):
                yield value
                break
        for value in payload.values():
            yield from iter_actions(value, keys)
    elif isinstance(payload, list):
        for item in payload:
                yield from iter_actions(item, keys)


def count_actions(root: Path, field: str, keys: Sequence[str]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for json_path in root.rglob("*.json"):
        try:
            data = json.loads(json_path.read_text(encoding="utf-8"))
        except Exception as exc:
            print(f"[skip] {json_path}: {exc}")
            continue
        if not isinstance(data, dict):
            continue
        payload = data.get(field)
        if payload is None:
            continue
        for action in iter_actions(payload, keys):
            counts[action] += 1
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "root_dir",
        type=Path,
        help="Root directory (e.g., the `pairs` folder) to scan for JSON files.",
    )
    parser.add_argument(
        "--field",
        nargs="+",
        default=["action_predict"],
        help="Top-level JSON field(s) to inspect for ACTION entries (default: action_predict).",
    )
    parser.add_argument(
        "--per-folder",
        action="store_true",
        help="Report counts for each first-level sub-folder under root (default aggregates everything).",
    )
    parser.add_argument(
        "--keys",
        nargs="+",
        default=["ACTION", "action", "type"],
        help="JSON keys whose string values should be tallied as actions.",
    )
    args = parser.parse_args()

    fields = args.field
    for field in fields:
        print(f"\n=== Field: {field} ===")
        if args.per_folder:
            overall = Counter()
            for subdir in sorted(p for p in args.root_dir.iterdir() if p.is_dir()):
                counts = count_actions(subdir, field, args.keys)
                overall.update(counts)
                total = sum(counts.values())
                print(f"[{subdir.name}] {total} instances, {len(counts)} action types")
                for action, cnt in counts.most_common():
                    print(f"  {action}\t{cnt}")
            print("[total]")
            total = sum(overall.values())
            print(f"{total} instances across {len(overall)} unique action types")
            for action, cnt in overall.most_common():
                print(f"{action}\t{cnt}")
        else:
            counts = count_actions(args.root_dir, field, args.keys)
            total = sum(counts.values())
            print(f"Scanned {args.root_dir} for field '{field}'.")
            print(f"Discovered {len(counts)} unique action types, {total} total instances.")
            for action, cnt in counts.most_common():
                print(f"{action}\t{cnt}")


if __name__ == "__main__":
    main()
