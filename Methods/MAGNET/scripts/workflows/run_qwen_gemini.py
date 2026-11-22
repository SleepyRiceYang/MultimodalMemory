#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, Optional

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_ROOT = REPO_ROOT / "scripts"

FLOW_MEMORY = {
    "none": "none",
    "stationary": "none",
    "procedural": "procedural",
    "both": "procedural",
}

PREDICT_MEMORY = {
    "none": "none",
    "stationary": "stationary",
    "procedural": "none",
    "both": "stationary",
}

REUSE_RULES = {
    "stationary": {"base": "none", "reuse_clicks": False},
    "both": {"base": "procedural", "reuse_clicks": False},
}


def stage_marker(path: Path, stage: str) -> Path:
    return path / f".{stage}_complete"


def expected_marker(path: Path, stage: str) -> Path:
    return path / f".{stage}_expected"


def clear_marker(path: Path, stage: str) -> None:
    marker = stage_marker(path, stage)
    if marker.exists():
        marker.unlink()


def read_expected_count(path: Path, stage: str) -> Optional[int]:
    marker = expected_marker(path, stage)
    if not marker.exists():
        return None
    try:
        return int(marker.read_text(encoding="utf-8").strip())
    except ValueError:
        return None


def run_agent(
    config: Path,
    output_dir: Path,
    stage: str,
    memory_mode: str,
    limit: int,
    ratio: float,
    flow_dir: Optional[Path] = None,
    log_prompts: bool = False,
) -> None:
    cmd = [
        sys.executable,
        str(SCRIPT_ROOT / "run_agent.py"),
        "--config",
        str(config),
        "--output-dir",
        str(output_dir),
        "--stage",
        stage,
        "--memory-mode",
        memory_mode,
        "--limit",
        str(limit),
        "--ratio",
        str(ratio),
    ]
    if flow_dir is not None:
        cmd.extend(["--flow-dir", str(flow_dir)])
    if log_prompts:
        cmd.append("--log-prompts")
    print(f"[workflow] Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True, cwd=str(REPO_ROOT))


def ensure_flow_stage(
    config: Path,
    flow_dir: Path,
    flow_memory: str,
    limit: int,
    ratio: float,
    log_prompts: bool = False,
) -> None:
    clear_marker(flow_dir, "flow")
    if flow_dir.exists():
        if _flow_outputs_need_rebuild(flow_dir):
            print(f"[workflow] Flow outputs at {flow_dir} missing plan metadata; rerunning stage.")
        else:
            print(f"[workflow] Flow outputs already exist at {flow_dir}, skipping.")
            return
    run_agent(config, flow_dir, "flow", flow_memory, limit, ratio, log_prompts=log_prompts)


def is_click_step(payload: Dict[str, object]) -> bool:
    desc = payload.get("action_desc")
    candidate = None
    if isinstance(desc, dict):
        candidate = desc
    elif isinstance(desc, list) and desc:
        candidate = desc[0]
    if candidate is None:
        plan = payload.get("action_plan")
        if isinstance(plan, list) and plan:
            candidate = plan[0]
    if not isinstance(candidate, dict):
        return False
    action_type = candidate.get("type", "")
    if isinstance(action_type, str):
        return "click" in action_type.lower()
    return False


def copy_flow_artifacts(flow_src: Path, target_dir: Path) -> None:
    if not flow_src.exists():
        raise FileNotFoundError(f"Flow directory {flow_src} does not exist.")
    if target_dir.exists():
        shutil.rmtree(target_dir)
    shutil.copytree(flow_src, target_dir)


def reuse_predictions(reuse_src: Path, target_dir: Path, allow_click_copy: bool) -> None:
    if not reuse_src.exists():
        raise FileNotFoundError(f"Prediction source {reuse_src} does not exist.")
    for src_file in reuse_src.rglob("*.json"):
        rel = src_file.relative_to(reuse_src)
        dst_file = target_dir / rel
        if not dst_file.exists():
            continue
        with open(src_file, "r", encoding="utf-8") as handle:
            src_payload = json.load(handle)
        with open(dst_file, "r", encoding="utf-8") as handle:
            dst_payload = json.load(handle)
        src_predict = (
            src_payload.get("action_predict", {}).get("COAT") if isinstance(src_payload, dict) else None
        )
        if not isinstance(src_predict, dict):
            continue
        is_click = is_click_step(dst_payload)
        if is_click and not allow_click_copy:
            continue
        dst_payload.setdefault("action_predict", {})
        dst_payload["action_predict"]["COAT"] = src_predict
        with open(dst_file, "w", encoding="utf-8") as handle:
            json.dump(dst_payload, handle, indent=4, ensure_ascii=False)


def run_predict_stage(
    config: Path,
    predict_dir: Path,
    memory_mode: str,
    limit: int,
    ratio: float,
    flow_dir: Optional[Path],
    log_prompts: bool = False,
) -> None:
    clear_marker(predict_dir, "predict")
    if predict_dir.exists():
        if _predict_outputs_need_rebuild(predict_dir):
            print(f"[workflow] Predict artifacts at {predict_dir} missing actions; rerunning stage.")
        else:
            print(f"[workflow] Predict artifacts already at {predict_dir}, skipping.")
            return
    run_agent(config, predict_dir, "predict", memory_mode, limit, ratio, flow_dir=flow_dir, log_prompts=log_prompts)


_FLOW_AUX_SUFFIXES = (
    "_0observe.json",
    "_1think.json",
    "_1think_action.json",
    "_reflect.json",
)
_AUX_ERROR_DIRS = {"action_desc_errors", "action_predict_errors"}


def _flow_outputs_need_rebuild(flow_dir: Path) -> bool:
    if not flow_dir.exists():
        return True
    expected = read_expected_count(flow_dir, "flow")
    record_count = 0
    for json_file in flow_dir.rglob("*.json"):
        if _should_skip_action_json(json_file):
            continue
        record_count += 1
        payload = _safe_load_json(json_file)
        if payload is None:
            return True
        if not _has_plan_fields(payload):
            return True
    if expected is not None:
        return record_count < expected
    return record_count == 0


def _predict_outputs_need_rebuild(predict_dir: Path) -> bool:
    if not predict_dir.exists():
        return True
    expected = read_expected_count(predict_dir, "predict")
    record_count = 0
    for json_file in predict_dir.rglob("*.json"):
        if _should_skip_action_json(json_file):
            continue
        record_count += 1
        payload = _safe_load_json(json_file)
        if payload is None:
            return True
        if not _has_prediction(payload):
            return True
    if expected is not None:
        return record_count < expected
    return record_count == 0


def _should_skip_action_json(path: Path) -> bool:
    name = path.name
    if name.startswith("."):
        return True
    if any(parent in _AUX_ERROR_DIRS for parent in path.parts):
        return True
    if any(name.endswith(suffix) for suffix in _FLOW_AUX_SUFFIXES):
        return True
    return False


def _safe_load_json(path: Path) -> Dict[str, object] | None:
    try:
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def _has_plan_fields(record: Dict[str, object]) -> bool:
    action_think = record.get("action_think")
    if not action_think:
        return False
    action_desc = record.get("action_desc")
    if not isinstance(action_desc, dict) or not action_desc:
        return False
    desc_type = action_desc.get("type")
    desc_text = action_desc.get("description")
    return bool(desc_type) and bool(desc_text)


def _has_prediction(record: Dict[str, object]) -> bool:
    predictions = record.get("action_predict")
    if not isinstance(predictions, dict) or not predictions:
        return False
    for action_mode in predictions.values():
        if not isinstance(action_mode, dict):
            continue
        for screen_entry in action_mode.values():
            if not isinstance(screen_entry, dict):
                continue
            action_value = screen_entry.get("ACTION")
            if isinstance(action_value, str) and action_value.strip():
                return True
            if isinstance(action_value, dict):
                nested = action_value.get("ACTION")
                if isinstance(nested, str) and nested.strip():
                    return True
    return False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Qwen flow + Gemini predict with optional reuse.")
    parser.add_argument("--flow-config", type=Path, default=REPO_ROOT / "configs" / "presets" / "coat" / "qwenvl_coat.yaml")
    parser.add_argument("--predict-config", type=Path, default=REPO_ROOT / "configs" / "presets" / "coat" / "config_gemini_android.yaml")
    parser.add_argument("--output-root", type=Path, default=REPO_ROOT / "artifacts" / "runs" / "qwen_gemini_workflows")
    parser.add_argument("--memory-mode", choices=["none", "stationary", "procedural", "both"], required=True)
    parser.add_argument("--limit", type=int, default=0, help="Limit number of steps (0 = run all).")
    parser.add_argument("--ratio", type=float, default=1.0, help="Dataset sampling ratio.")
    parser.add_argument(
        "--log-prompts",
        action="store_true",
        help="Print prompts/responses for all VLM calls invoked by this workflow.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)

    flow_memory = FLOW_MEMORY[args.memory_mode]
    predict_memory = PREDICT_MEMORY[args.memory_mode]

    flow_dir = args.output_root / f"flow_{flow_memory}"
    # Ensure necessary flow artifacts exist.
    ensure_flow_stage(args.flow_config, flow_dir, flow_memory, args.limit, args.ratio, log_prompts=args.log_prompts)

    predict_dir = args.output_root / f"predict_{args.memory_mode}"

    reuse_info = REUSE_RULES.get(args.memory_mode)
    if reuse_info:
        base_mode = reuse_info["base"]
        base_predict_dir = args.output_root / f"predict_{base_mode}"
        print(f"[workflow] Preparing workspace for {args.memory_mode} using base results from {base_predict_dir}")
        flow_source = args.output_root / f"flow_{FLOW_MEMORY[base_mode]}"
        copy_flow_artifacts(flow_source, predict_dir)
        reuse_predictions(
            base_predict_dir,
            predict_dir,
            allow_click_copy=reuse_info.get("reuse_clicks", False),
        )
        flow_arg: Optional[Path] = None
    else:
        flow_arg = flow_dir

    run_predict_stage(
        args.predict_config,
        predict_dir,
        predict_memory,
        args.limit,
        args.ratio,
        flow_dir=flow_arg,
        log_prompts=args.log_prompts,
    )


if __name__ == "__main__":
    main()
