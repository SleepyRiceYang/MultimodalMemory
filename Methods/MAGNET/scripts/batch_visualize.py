"""Batch driver for generating visualization comparisons for many predictions."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, Iterable, Sequence


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch run the GUI action visualizer over many files.")
    parser.add_argument(
        "input_dir",
        type=Path,
        help="Directory containing prediction JSON files (recursively scanned).",
    )
    parser.add_argument(
        "output_dir",
        type=Path,
        help="Root directory where visualizations and summaries will be stored.",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="*.json",
        help="Glob pattern (relative) used to pick prediction files. Defaults to '*.json'.",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=None,
        help="Optional dataset root passed through to the visualize tool.",
    )
    parser.add_argument(
        "--demo-mode",
        type=str,
        default=None,
        help="Override demo mode passed to the visualize tool.",
    )
    parser.add_argument(
        "--screen-mode",
        type=str,
        default=None,
        help="Override screen mode passed to the visualize tool.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model name override for every visualization.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip files whose output PNG already exists in the destination tree.",
    )
    parser.add_argument(
        "--skip-missing",
        action="store_true",
        help="Skip prediction files that lack image paths or action_predict outputs instead of failing.",
    )
    return parser.parse_args()


def find_predictions(root: Path, pattern: str) -> Iterable[Path]:
    yield from sorted(root.rglob(pattern))


def build_command(prediction_file: Path, output_dir: Path, args: argparse.Namespace) -> Sequence[str]:
    cmd = [
        sys.executable,
        str(Path("src/agent/evaluation/visualize.py")),
        str(prediction_file),
        "--output",
        str(output_dir),
    ]
    if args.data_root:
        cmd.extend(["--data-root", str(args.data_root)])
    if args.demo_mode:
        cmd.extend(["--demo-mode", args.demo_mode])
    if args.screen_mode:
        cmd.extend(["--screen-mode", args.screen_mode])
    if args.model:
        cmd.extend(["--model", args.model])
    if args.skip_missing:
        cmd.append("--skip-missing")
    return cmd


def _load_prediction(prediction_file: Path) -> Dict:
    with prediction_file.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def should_skip(prediction_file: Path, args: argparse.Namespace) -> bool:
    if not args.skip_missing:
        return False
    try:
        # print(prediction_file)
        payload = _load_prediction(prediction_file)
    except json.JSONDecodeError:
        # print(f"[skip] Invalid JSON: {prediction_file}")
        return True
    # print(payload)
    image_path = payload.get("image_full_path") or payload.get("image_path")
    action_predict = payload.get("action_predict")
    if not image_path or not action_predict:
        # print(f"[skip] Missing fields (image/action) -> {prediction_file}")
        return True
    else:
        print(image_path)
    return False


def main() -> None:
    args = parse_args()
    input_dir = args.input_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    predictions = list(find_predictions(input_dir, args.pattern))
    if not predictions:
        print(f"No prediction files matched pattern {args.pattern!r} under {input_dir}")
        return
    print(f"Found {len(predictions)} prediction files. Generating visualizations...")
    skip_dirs = ["action_desc_errors","action_predict_errors","make_action_with_memory_prompts"]
    for idx, prediction_file in enumerate(predictions, start=1):
        if any(skip_dir in prediction_file.parts for skip_dir in skip_dirs):
            continue
        # print(prediction_file)

        rel = prediction_file.relative_to(input_dir)
        dest_dir = output_dir / rel.parent
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest_image = dest_dir / f"{prediction_file.stem}_viz.png"
        if args.skip_existing and dest_image.exists():
            print(f"[{idx}/{len(predictions)}] Skipping existing {dest_image}")
            continue
        if should_skip(prediction_file, args):
            # print(f"[{idx}/{len(predictions)}] Skipping missing prediction {prediction_file}")
            continue
        cmd = build_command(prediction_file, dest_dir, args)
        print(cmd)
        print(f"[{idx}/{len(predictions)}] Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, check=False)
        if result.returncode != 0:
            print(f"Visualization failed for {prediction_file} (exit code {result.returncode}).")
            break


if __name__ == "__main__":
    main()
