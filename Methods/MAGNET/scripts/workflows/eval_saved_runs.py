#!/usr/bin/env python
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "eval_base.yaml"
EVAL_SCRIPT = REPO_ROOT / "scripts" / "evaluate_predictions.py"

DATASET_ALIASES = {
    "aitz": {"dataset": "android-in-the-zoo", "data_dir": "/remote-home/lbsun/data/android_in_the_zoo"},
    "odyssey": {"dataset": "odyssey", "data_dir": "/remote-home/lbsun/data/gui_odyssey"},
    "amex": {"dataset": "amex", "data_dir": "/remote-home/lbsun/data/amex"},
}


def resolve_dataset(alias: str, dataset_dir: str | None) -> tuple[str, str | None]:
    key = alias.lower()
    info = DATASET_ALIASES.get(key)
    if info is None:
        raise ValueError(f"Unknown dataset alias '{alias}'. Known aliases: {', '.join(DATASET_ALIASES)}")
    name = info["dataset"]
    resolved_dir = dataset_dir or info.get("data_dir")
    if not resolved_dir:
        raise ValueError(f"Dataset directory for alias '{alias}' is not specified and no default is available.")
    return name, resolved_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a saved GUI-Memory run directory.")
    parser.add_argument("run_dir", type=Path, help="Directory containing prediction JSON files.")
    parser.add_argument("dataset", help="Dataset alias (aitz/odyssey/amex).")
    parser.add_argument("split", help="Dataset split name (e.g., test, val_id_app).")
    parser.add_argument("model_name", help="Logical model name used for reporting.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="Base config used for evaluation overrides.")
    parser.add_argument("--dataset-dir", help="Override dataset directory (otherwise taken from alias preset).")
    parser.add_argument("--demo-mode", default="COAT", help="Demo mode used when generating predictions (default COAT).")
    parser.add_argument("--screen-mode", default="tag", choices=["tag"], help="Screen mode to evaluate (default tag).")
    parser.add_argument("--click-mode", choices=["bbox", "point"], help="Override click evaluation mode.")
    parser.add_argument("--metrics-json", type=Path, help="Optional output JSON path (default: run_dir/metrics.json).")
    parser.add_argument("--html-report", type=Path, help="Optional HTML output path (default: run_dir/report.html).")
    parser.add_argument("--ratio", type=float, default=1.0)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--extra", nargs=argparse.REMAINDER, help="Additional args forwarded to evaluate_predictions.py.")
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory {run_dir} does not exist.")

    dataset_name, dataset_path = resolve_dataset(args.dataset, args.dataset_dir)
    if args.demo_mode == "COAT" and "ui-venus" in args.model_name.lower():
        args.demo_mode = "UI_VENUS"

    metrics_path = args.metrics_json or (run_dir / "metrics.json")
    html_path = args.html_report or (run_dir / "report.html")

    cmd = [
        sys.executable,
        str(EVAL_SCRIPT),
        "--config",
        str(args.config.resolve()),
        "--dataset-name",
        dataset_name,
        "--dataset-dir",
        dataset_path,
        "--split",
        args.split,
        "--pred-dir",
        str(run_dir),
        "--model-name",
        args.model_name,
        "--demo-mode",
        args.demo_mode,
        "--screen-mode",
        args.screen_mode,
        "--metrics-json",
        str(metrics_path),
        "--html-report",
        str(html_path),
        "--ratio",
        str(args.ratio),
        "--limit",
        str(args.limit),
    ]
    if args.click_mode:
        cmd.extend(["--click-mode", args.click_mode])
    if args.extra:
        cmd.extend(args.extra)
    subprocess.run(cmd, check=True, cwd=str(REPO_ROOT))


if __name__ == "__main__":
    main()
