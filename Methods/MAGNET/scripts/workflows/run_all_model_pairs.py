#!/usr/bin/env python
from __future__ import annotations

import argparse
import copy
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from agent.config.schema import AgentConfig
from datasets.datasets import build_dataset
from workflow.runner import (
    record_has_plan_fields,
    record_has_prediction,
    safe_load_action_json,
    should_skip_action_json,
)

RUN_AGENT = REPO_ROOT / "scripts" / "run_agent.py"
RUN_EVAL = REPO_ROOT / "scripts" / "evaluate_predictions.py"
DEFAULT_CONFIG = REPO_ROOT / "configs" / "presets" / "coat" / "config_gemini_android.yaml"


def _slugify(text: str) -> str:
    return "".join(ch.lower() if ch.isalnum() else "-" for ch in text).strip("-")


MODEL_PRESETS: Dict[str, Dict[str, Any]] = {
    "qwen2.5-vl-32b": {
        "slug": "qwen25-vl32b",
        "model_name": "qwenvl-coat",
        "overrides": {"QWENVL_MODEL": "Qwen2.5-VL-32B"},
    },
    "atlas-base-7b": {
        "slug": "atlas-base-7b",
        "model_name": "atlas-base-coat",
        "overrides": {"QWENVL_MODEL": "Atlas-Base-7B"},
    },
    "atlas-pro-7b": {
        "slug": "atlas-pro-7b",
        "model_name": "atlas-pro-coat",
        "overrides": {"QWENVL_MODEL": "Atlas-Pro-7B"},
    },
    "gui-r1-7b": {
        "slug": "gui-r1-7b",
        "model_name": "atlas-guir1-coat",
        "overrides": {"QWENVL_MODEL": "GUI-R1-7B"},
    },
    "ui-venus-navi-7b": {
        "slug": "ui-venus-navi-7b",
        "model_name": "ui-venus-navi",
        "overrides": {"QWENVL_MODEL": "UI-Venus-Navi-7B"},
    },
    "infigui-r1-3b": {
        "slug": "infigui-r1-3b",
        "model_name": "infigui-r1t",
        "overrides": {"QWENVL_MODEL": "InfiGUI-R1-3B"},
    },
    "gemini-2.5-pro": {
        "slug": "gemini-25-pro",
        "model_name": "gemini-2.5-pro",
        "overrides": {"GEMINI_MODEL": "gemini-2.5-pro"},
    },
    "gpt-4o": {
        "slug": "gpt-4o",
        "model_name": "openai-coat",
        "overrides": {"OPENAI_MODEL": "gpt-4o"},
    },
}

# DATASET_PRESETS: Dict[str, Dict[str, str]] = {
#     "aitz": {"dataset": "android-in-the-zoo", "data_dir": "/attached/remote-home1/lbsun/data/android_in_the_zoo"},
#     "odyssey": {"dataset": "odyssey", "data_dir": "/attached/remote-home1/lbsun/data/gui_odyssey"},
#     "amex": {"dataset": "amex", "data_dir": "/attached/remote-home1/lbsun/data/amex"},
# }

DATASET_PRESETS: Dict[str, Dict[str, str]] = {
    "aitz": {"dataset": "android-in-the-zoo", "data_dir": "/remote-home/lbsun/data/android_in_the_zoo"},
    "odyssey": {"dataset": "odyssey", "data_dir": "/remote-home/lbsun/data/gui_odyssey"},
    "amex": {"dataset": "amex", "data_dir": "/remote-home/lbsun/data/amex"},
}

DATASET_SPLITS: Dict[str, tuple[str, ...]] = {
    "aitz": ("test",),
    "odyssey": ("test_sample", "val_sample"),
    "amex": ("val_id_app", "val_ood_app", "val_ood_region"),
}

DATASET_SCREEN_MODES: Dict[str, Dict[str, bool]] = {}

MEMORY_DEFAULTS: Dict[str, Dict[str, Path]] = {
    "aitz": {
        "stationary": REPO_ROOT / "artifacts" / "storage" / "aitz_stationary_omni",
        "procedural": REPO_ROOT / "artifacts" / "storage" / "aitz_procedural",
    },
    "odyssey": {
        "stationary": REPO_ROOT / "artifacts" / "storage" / "odyssey_stationary_omni",
        "procedural": REPO_ROOT / "artifacts" / "storage" / "odyssey_procedural",
    },
    "amex": {
        "stationary": REPO_ROOT / "artifacts" / "storage" / "amex_stationary_omni",
        "procedural": REPO_ROOT / "artifacts" / "storage" / "amex_procedural",
    },
}


def _memory_label(path: str | None, dataset_alias: str) -> str:
    if path:
        return Path(path).name.split("_")[0]
    return dataset_alias


def _dataset_default_memory(dataset_alias: str, key: str) -> str | None:
    defaults = MEMORY_DEFAULTS.get(dataset_alias, {})
    candidate = defaults.get(key)
    return str(candidate) if candidate else None


def stage_marker(path: Path, stage: str) -> Path:
    return path / f".{stage}_complete"


def expected_marker(path: Path, stage: str) -> Path:
    return path / f".{stage}_expected"


def has_marker(path: Path, stage: str) -> bool:
    return stage_marker(path, stage).exists()


def mark_stage(path: Path, stage: str) -> None:
    marker = stage_marker(path, stage)
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text("ok", encoding="utf-8")


def clear_marker(path: Path, stage: str) -> None:
    marker = stage_marker(path, stage)
    if marker.exists():
        marker.unlink()


def read_expected_count(path: Path, stage: str) -> int | None:
    marker = expected_marker(path, stage)
    if not marker.exists():
        return None
    try:
        return int(marker.read_text(encoding="utf-8").strip())
    except ValueError:
        return None


def load_config(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not data:
        return {}
    includes = data.get("INCLUDE", [])
    if isinstance(includes, str):
        includes = [includes]
    # Resolve includes to absolute paths so autogen configs work outside the repo config dir.
    data["INCLUDE"] = [str((path.parent / inc).resolve()) for inc in includes]
    return data


def compute_dataset_total_steps(cfg: Dict[str, Any], ratio: float, limit: int) -> int:
    agent_cfg = AgentConfig.from_mapping(copy.deepcopy(cfg))
    dataset = build_dataset(agent_cfg, ratio=ratio)
    total = len(dataset)
    if limit and limit > 0:
        total = min(total, limit)
    return total


def resolve_memory_path(value: str | None) -> str | None:
    if not value:
        return None
    candidate = Path(value).expanduser()
    if candidate.is_absolute():
        return str(candidate)
    repo_candidate = (REPO_ROOT / candidate).resolve()
    if repo_candidate.exists():
        return str(repo_candidate)
    storage_candidate = (REPO_ROOT / "artifacts" / "storage" / candidate).resolve()
    return str(storage_candidate)


def apply_dataset_settings(
    base_cfg: Dict[str, Any],
    dataset_alias: str,
    split: str,
    data_dir_override: str | None,
    stationary_override: str | None,
    stationary_click_override: str | None,
    procedural_override: str | None,
) -> Dict[str, Any]:
    cfg = copy.deepcopy(base_cfg)
    data_cfg = cfg.setdefault("DATA", {})
    preset = DATASET_PRESETS[dataset_alias]
    data_cfg["DATASET"] = preset["dataset"]
    data_cfg["SPLIT"] = split
    if data_dir_override:
        data_cfg["DATA_DIR"] = data_dir_override
    else:
        data_cfg["DATA_DIR"] = preset.get("data_dir", data_cfg.get("DATA_DIR", ""))

    screen_overrides = DATASET_SCREEN_MODES.get(dataset_alias, {})
    for key, value in screen_overrides.items():
        data_cfg[key] = value

    memory_cfg = cfg.setdefault("MEMORY", {})
    stationary_dirs = memory_cfg.setdefault("STATIONARY_RAG_DIRS", {})
    if stationary_override:
        stationary_dirs["default"] = stationary_override
    if stationary_click_override:
        stationary_dirs["click_to"] = stationary_click_override
    if procedural_override:
        memory_cfg["PROCEDURAL_RAG_DIR"] = procedural_override
    return cfg


def materialize_config(
    base_cfg: Dict[str, Any],
    preset: Dict[str, Any],
    output_dir: Path,
    stash_dir: Path,
    suffix: str,
    demo_mode: str,
    model_overrides: Dict[str, Any] | None = None,
) -> Path:
    cfg = copy.deepcopy(base_cfg)
    cfg["DEMO_MODE"] = demo_mode.upper()
    cfg["OUTPUT_DIR"] = str(output_dir)
    cfg.setdefault("MODEL", {})
    model_cfg = cfg["MODEL"]
    for key, value in preset.get("overrides", {}).items():
        model_cfg[key] = value
    if "NAME" not in model_cfg and preset.get("model_name"):
        model_cfg["NAME"] = preset["model_name"]
    if model_overrides:
        for key, value in model_overrides.items():
            model_cfg[key] = value
    stash_dir.mkdir(parents=True, exist_ok=True)
    config_path = stash_dir / f"{preset['slug']}-{_slugify(suffix)}.yaml"
    with open(config_path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(cfg, handle, sort_keys=False)
    return config_path


def run_agent_stage(
    config_path: Path,
    stage: str,
    memory_mode: str,
    limit: int,
    ratio: float,
    num_workers: int,
    flow_dir: Path | None = None,
    log_prompts: bool = False,
) -> None:
    cmd = [
        sys.executable,
        str(RUN_AGENT),
        "--config",
        str(config_path),
        "--stage",
        stage,
        "--memory-mode",
        memory_mode,
        "--limit",
        str(limit),
        "--ratio",
        str(ratio),
        "--num-workers",
        str(num_workers),
    ]
    if flow_dir is not None:
        cmd.extend(["--flow-dir", str(flow_dir)])
    if log_prompts:
        cmd.append("--log-prompts")
    print(f"[run_all_model_pairs] Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True, cwd=str(REPO_ROOT))


def run_eval_stage(
    config_path: Path,
    pred_dir: Path,
    demo_mode: str,
    model_name: str,
    click_mode: str,
    ratio: float,
    limit: int,
) -> None:
    metrics_dir = pred_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str(RUN_EVAL),
        "--config",
        str(config_path),
        "--pred-dir",
        str(pred_dir),
        "--demo-mode",
        demo_mode.upper(),
        "--model-name",
        model_name,
        "--screen-mode",
        "tag",
        "--click-mode",
        click_mode,
        "--ratio",
        str(ratio),
        "--html-report",
        str(metrics_dir / "report.html"),
        "--metrics-json",
        str(metrics_dir / "metrics.json"),
    ]
    if limit:
        cmd.extend(["--limit", str(limit)])
    print(f"[run_all_model_pairs] Evaluating: {' '.join(cmd)}")
    subprocess.run(cmd, check=True, cwd=str(REPO_ROOT))


def ensure_flow_stage(
    model_key: str,
    preset: Dict[str, Any],
    base_cfg: Dict[str, Any],
    flow_root: Path,
    config_root: Path,
    use_procedural: bool,
    procedural_mem: str,
    limit: int,
    ratio: float,
    num_workers: int,
    dataset_alias: str,
    split: str,
    flow_reuse_root: Path | None,
    allow_partial_flow: bool,
    expected_total: int | None = None,
    log_prompts: bool = False,
) -> Path:
    mem_suffix = f"__{procedural_mem}mem" if procedural_mem else ""
    flow_dir = flow_root / f"{preset['slug']}__proc{int(use_procedural)}{mem_suffix}"
    clear_marker(flow_dir, "flow")
    reuse_candidate = _find_reuse_flow_dir(flow_reuse_root, dataset_alias, split, flow_dir.name)
    if flow_dir.exists():
        if _flow_outputs_need_rebuild(flow_dir, expected_total):
            if reuse_candidate and _copy_flow_cache(reuse_candidate, flow_dir):
                if not _flow_outputs_need_rebuild(flow_dir, expected_total):
                    print(f"[run_all_model_pairs] Flow reused from {reuse_candidate} -> {flow_dir}")
                    return flow_dir
            if allow_partial_flow:
                print(
                    f"[run_all_model_pairs] Flow cache at {flow_dir} incomplete but continuing due to allow-partial-flow."
                )
                return flow_dir
            print(f"[run_all_model_pairs] Flow cache at {flow_dir} is missing plan metadata; rerunning stage.")
        else:
            print(f"[run_all_model_pairs] Flow cached for {model_key} -> {flow_dir}")
            return flow_dir
    else:
        if reuse_candidate and _copy_flow_cache(reuse_candidate, flow_dir):
            if not _flow_outputs_need_rebuild(flow_dir, expected_total):
                print(f"[run_all_model_pairs] Flow reused from {reuse_candidate} -> {flow_dir}")
                return flow_dir
            if allow_partial_flow and flow_dir.exists():
                print(
                    f"[run_all_model_pairs] Flow reused from {reuse_candidate} but incomplete; continuing due to allow-partial-flow."
                )
                return flow_dir

    flow_dir.mkdir(parents=True, exist_ok=True)
    config_path = materialize_config(
        base_cfg,
        preset,
        flow_dir,
        config_root / "flow",
        suffix=f"{preset['slug']}-proc{int(use_procedural)}{f'-{procedural_mem}' if procedural_mem else ''}",
        demo_mode="COAT",
    )
    flow_memory_mode = "procedural" if use_procedural else "none"
    run_agent_stage(
        config_path,
        "flow",
        flow_memory_mode,
        limit,
        ratio,
        num_workers=num_workers,
        log_prompts=log_prompts,
    )
    return flow_dir


def _find_reuse_flow_dir(reuse_root: Path | None, dataset_alias: str, split: str, flow_dir_name: str) -> Path | None:
    if not reuse_root:
        return None
    candidate = Path(reuse_root) / dataset_alias / split / "flows" / flow_dir_name
    return candidate if candidate.exists() else None


def _copy_flow_cache(src: Path, dst: Path) -> bool:
    if not src.exists():
        return False
    shutil.copytree(src, dst, dirs_exist_ok=True)
    return True


def _is_click_step(record: Dict[str, Any]) -> bool:
    action_desc = record.get("action_desc")
    candidate = None
    if isinstance(action_desc, dict):
        candidate = action_desc
    elif isinstance(action_desc, list) and action_desc:
        candidate = action_desc[0]
    if candidate is None:
        plan = record.get("action_plan")
        if isinstance(plan, list) and plan:
            candidate = plan[0]
    if not isinstance(candidate, dict):
        return False
    action_type = candidate.get("type")
    if isinstance(action_type, str):
        return "click" in action_type.lower()
    return False


def _strip_click_predictions(target_dir: Path) -> tuple[int, int]:
    cleared = 0
    reused = 0
    for json_path in target_dir.rglob("*.json"):
        if should_skip_action_json(json_path):
            continue
        record = safe_load_action_json(json_path)
        if record is None:
            continue
        if _is_click_step(record):
            if "action_predict" in record:
                record.pop("action_predict", None)
                with open(json_path, "w", encoding="utf-8") as handle:
                    json.dump(record, handle, ensure_ascii=False, indent=2)
            cleared += 1
        else:
            reused += 1
    return cleared, reused


def _flow_outputs_need_rebuild(flow_dir: Path, expected_total: int | None = None) -> bool:
    """Return True if any cached flow JSON lacks a populated action_think/action_desc."""
    if not flow_dir.exists():
        return True

    expected = expected_total if expected_total is not None else read_expected_count(flow_dir, "flow")
    record_count = 0
    for json_path in flow_dir.rglob("*.json"):
        if should_skip_action_json(json_path):
            continue
        record_count += 1
        record = safe_load_action_json(json_path)
        if record is None:
            return True
        if not record_has_plan_fields(record):
            return True
    if expected is not None:
        return record_count < expected
    return record_count == 0


def _predict_outputs_need_rebuild(pair_dir: Path, expected_total: int | None = None) -> bool:
    """Return True if any cached prediction JSON lacks a populated action_predict entry."""
    if not pair_dir.exists():
        return True

    expected = expected_total if expected_total is not None else read_expected_count(pair_dir, "predict")
    record_count = 0
    for json_path in pair_dir.rglob("*.json"):
        if should_skip_action_json(json_path):
            continue
        record_count += 1
        record = safe_load_action_json(json_path)
        if record is None:
            return True
        if not record_has_prediction(record):
            return True
    if expected is not None:
        return record_count < expected
    return record_count == 0


def run_predict_for_pair(
    flow_key: str,
    predict_key: str,
    flow_dir: Path,
    predict_preset: Dict[str, Any],
    base_cfg: Dict[str, Any],
    pair_root: Path,
    config_root: Path,
    use_stationary: bool,
    stationary_mem: str,
    predict_click_mode: str,
    flow_uses_procedural: bool,
    procedural_mem: str,
    limit: int,
    ratio: float,
    num_workers: int,
    reuse_stationary_base: bool,
    expected_total: int | None = None,
    combined_model_name: str | None = None,
    auto_eval: bool = False,
    log_prompts: bool = False,
) -> None:
    proc_suffix = f"{procedural_mem}mem" if procedural_mem else "nomem"
    stat_suffix = f"{stationary_mem}mem" if stationary_mem else "nomem"
    suffix_parts = [
        _slugify(flow_key),
        f"proc{int(flow_uses_procedural)}",
        proc_suffix,
        _slugify(predict_key),
        f"stat{int(use_stationary)}",
        stat_suffix,
        predict_click_mode,
    ]
    pair_dir = pair_root / "__".join(suffix_parts)
    def _materialize_pair_config() -> Path:
        return materialize_config(
            base_cfg,
            predict_preset,
            pair_dir,
            config_root / "predict",
            suffix=(
                f"{predict_preset['slug']}-from-{_slugify(flow_key)}-stat{int(use_stationary)}"
                f"{f'-{stationary_mem}' if stationary_mem else ''}-{predict_click_mode}"
            ),
            demo_mode="COAT",
            model_overrides={"CLICK_MODE": predict_click_mode},
        )
    clear_marker(pair_dir, "predict")
    if (
        use_stationary
        and reuse_stationary_base
        and not pair_dir.exists()
    ):
        base_suffix = list(suffix_parts)
        base_suffix[4] = "stat0"
        base_suffix[5] = "nomem"
        base_dir = pair_root / "__".join(base_suffix)
        if base_dir.exists():
            print(f"[run_all_model_pairs] Reusing non-click predictions from {base_dir} -> {pair_dir}")
            shutil.copytree(base_dir, pair_dir, dirs_exist_ok=True)
            cleared, reused = _strip_click_predictions(pair_dir)
            if cleared or reused:
                print(
                    f"[run_all_model_pairs] Preserved {reused} non-click predictions; "
                    f"{cleared} click step(s) will be re-run with stationary memory."
                )
            clear_marker(pair_dir, "predict")

    if pair_dir.exists():
        if _predict_outputs_need_rebuild(pair_dir, expected_total):
            print(f"[run_all_model_pairs] Predict cache at {pair_dir} missing actions; rerunning stage.")
        else:
            print(f"[run_all_model_pairs] Predict cached for {pair_dir}")
            if auto_eval:
                config_path = _materialize_pair_config()
                model_name = combined_model_name or predict_preset.get("model_name", predict_preset["slug"])
                run_eval_stage(
                    config_path,
                    pair_dir,
                    demo_mode="COAT",
                    model_name=model_name,
                    click_mode=predict_click_mode,
                    ratio=ratio,
                    limit=limit,
                )
            return

    pair_dir.mkdir(parents=True, exist_ok=True)
    config_path = _materialize_pair_config()
    memory_mode = "stationary" if use_stationary else "none"
    run_agent_stage(
        config_path,
        stage="predict",
        memory_mode=memory_mode,
        limit=limit,
        ratio=ratio,
        flow_dir=flow_dir,
        num_workers=num_workers,
        log_prompts=log_prompts,
    )
    if auto_eval:
        model_name = combined_model_name or predict_preset.get("model_name", predict_preset["slug"])
        run_eval_stage(
            config_path,
            pair_dir,
            demo_mode="COAT",
            model_name=model_name,
            click_mode=predict_click_mode,
            ratio=ratio,
            limit=limit,
        )


def run_e2e_model(
    demo_mode: str,
    model_key: str,
    preset: Dict[str, Any],
    base_cfg: Dict[str, Any],
    output_root: Path,
    config_root: Path,
    click_mode: str,
    limit: int,
    ratio: float,
    num_workers: int,
    expected_total: int | None = None,
    auto_eval: bool = False,
    log_prompts: bool = False,
) -> None:
    stage_name = f"{demo_mode.lower()}_{click_mode}"
    model_dir = output_root / preset["slug"]
    combined_name = f"{preset['slug']}"

    def _materialize_e2e_config() -> Path:
        return materialize_config(
            base_cfg,
            preset,
            model_dir,
            config_root,
            suffix=f"{preset['slug']}-{demo_mode.lower()}-{click_mode}",
            demo_mode=demo_mode,
            model_overrides={"CLICK_MODE": click_mode},
        )

    needs_rebuild = True
    if model_dir.exists():
        needs_rebuild = _predict_outputs_need_rebuild(model_dir, expected_total)
    print(needs_rebuild)
    if has_marker(model_dir, stage_name):
        if not needs_rebuild:
            print(f"[run_all_model_pairs] {demo_mode.upper()} cached for {model_key} -> {model_dir}")
            if auto_eval:
                config_path = _materialize_e2e_config()
                run_eval_stage(
                    config_path,
                    model_dir,
                    demo_mode=demo_mode,
                    model_name=combined_name or preset.get("model_name", preset["slug"]),
                    click_mode=click_mode,
                    ratio=ratio,
                    limit=limit,
                )
            return
        else:
            clear_marker(model_dir, stage_name)

    if model_dir.exists() and not needs_rebuild:
        print(f"[run_all_model_pairs] {demo_mode.upper()} artifacts already present for {model_key} -> {model_dir}")
        mark_stage(model_dir, stage_name)
        if auto_eval:
            config_path = _materialize_e2e_config()
            run_eval_stage(
                config_path,
                model_dir,
                demo_mode=demo_mode,
                model_name=combined_name or preset.get("model_name", preset["slug"]),
                click_mode=click_mode,
                ratio=ratio,
                limit=limit,
            )
        return

    model_dir.mkdir(parents=True, exist_ok=True)
    print(f"[run_all_model_pairs] combined_name: {combined_name}")
    config_path = _materialize_e2e_config()
    run_agent_stage(
        config_path,
        stage="predict",
        memory_mode="none",
        limit=limit,
        ratio=ratio,
        num_workers=num_workers,
        log_prompts=log_prompts,
    )
    mark_stage(model_dir, stage_name)
    if auto_eval:
        model_name = combined_name or preset.get("model_name", preset["slug"])
        run_eval_stage(
            config_path,
            model_dir,
            demo_mode=demo_mode,
            model_name=model_name,
            click_mode=click_mode,
            ratio=ratio,
            limit=limit,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Enumerate all flow/predict model combinations (COAT mode) and launch the pipeline."
    )
    parser.add_argument("--base-config", type=Path, default=DEFAULT_CONFIG, help="Template config (COAT mode).")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=REPO_ROOT / "artifacts" / "runs" / "all_model_pairs",
        help="Root directory for generated artifacts.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        help="Subset of models to include. Default = all.",
    )
    parser.add_argument(
        "--demo-mode",
        choices=["coat", "cot", "coa", "atlas", "guir1", "ui_venus", "infigui_r1"],
        default="coat",
        help="Reasoning mode. COAT runs flow+predict, COT/COA are direct e2e runs.",
    )
    parser.add_argument(
        "--flow-procedural",
        dest="flow_procedural",
        action="store_true",
        help="Enable procedural memory in the flow stage (default: enabled).",
    )
    parser.add_argument(
        "--no-flow-procedural",
        dest="flow_procedural",
        action="store_false",
        help="Disable procedural memory in the flow stage.",
    )
    parser.add_argument(
        "--predict-stationary",
        dest="predict_stationary",
        action="store_true",
        help="Enable stationary memory during predict (default: enabled).",
    )
    parser.add_argument(
        "--no-predict-stationary",
        dest="predict_stationary",
        action="store_false",
        help="Disable stationary memory during predict.",
    )
    parser.set_defaults(flow_procedural=True, predict_stationary=True)
    parser.add_argument(
        "--predict-click-mode",
        choices=["bbox", "point"],
        default="bbox",
        help="Click output representation for predict stage models.",
    )
    parser.add_argument(
        "--log-prompts",
        action="store_true",
        help="Print prompts/responses for all VLM calls.",
    )
    parser.add_argument(
        "--dataset",
        choices=sorted(DATASET_PRESETS.keys()),
        default="aitz",
        help="Dataset alias to run over.",
    )
    parser.add_argument(
        "--subsets",
        nargs="+",
        help="Dataset splits to evaluate (e.g., test val val_id_app ...). Defaults depend on dataset.",
    )
    parser.add_argument(
        "--data-dir",
        help="Optional override for DATA.DATA_DIR (defaults depend on dataset).",
    )
    parser.add_argument(
        "--stationary-rag",
        help="Override STATIONARY_RAG_DIRS.default (path or alias under artifacts/storage).",
    )
    parser.add_argument(
        "--stationary-rag-clickto",
        help="Override STATIONARY_RAG_DIRS.click_to.",
    )
    parser.add_argument(
        "--procedural-rag",
        help="Override PROCEDURAL_RAG_DIR.",
    )
    parser.add_argument("--ratio", type=float, default=1.0, help="Dataset sampling ratio.")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of processed steps (0 = all).")
    parser.add_argument("--num-workers", type=int, default=64, help="Number of parallel workers per stage.")
    parser.add_argument(
        "--flow-reuse-root",
        type=Path,
        help="Optional path to an existing run directory to reuse flow artifacts from.",
    )
    parser.add_argument(
        "--disable-stationary-reuse",
        action="store_true",
        help="Disable reusing non-click predictions from non-stationary runs.",
    )
    parser.add_argument(
        "--allow-partial-flow",
        action="store_true",
        help="Skip rebuilding flows even if caches appear incomplete (use with caution).",
    )
    parser.add_argument(
        "--auto-eval",
        action="store_true",
        help="Run evaluate_predictions.py after each predict/e2e stage completes.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    config_cache = load_config(args.base_config)

    if not args.models:
        raise ValueError("--models must be provided (flow/predict pair for COAT or single model for COA/COT).")
    selected_models: list[str] = []
    for key in args.models:
        normalized = key.lower()
        if normalized not in MODEL_PRESETS:
            raise ValueError(f"Unknown model '{key}'. Available: {', '.join(MODEL_PRESETS)}")
        selected_models.append(normalized)

    dataset_alias = args.dataset.lower()
    if dataset_alias not in DATASET_PRESETS:
        raise ValueError(f"Unknown dataset '{args.dataset}'. Choices: {', '.join(DATASET_PRESETS)}")
    if dataset_alias not in DATASET_SPLITS:
        valid_splits = "unknown"
    else:
        valid_splits = DATASET_SPLITS[dataset_alias]
    if args.subsets:
        subsets = args.subsets
    else:
        subsets = list(valid_splits)
    invalid = [split for split in subsets if split not in DATASET_SPLITS[dataset_alias]]
    if invalid:
        raise ValueError(
            f"Invalid subset(s) {invalid} for dataset '{dataset_alias}'. Valid options: {', '.join(DATASET_SPLITS[dataset_alias])}"
        )
    stationary_override = resolve_memory_path(args.stationary_rag)
    stationary_click_override = resolve_memory_path(args.stationary_rag_clickto)
    procedural_override = resolve_memory_path(args.procedural_rag)

    demo_mode = args.demo_mode.lower()
    if demo_mode == "coat":
        if len(selected_models) != 2:
            raise ValueError("--models must specify exactly two entries for COAT runs: <flow_model> <predict_model>.")
        flow_model_key, predict_model_key = selected_models
        combined_model_name = (
            f"{MODEL_PRESETS[flow_model_key]['slug']}_{MODEL_PRESETS[predict_model_key]['slug']}_coat"
        )
    else:
        if len(selected_models) != 1:
            raise ValueError("--models must specify exactly one model when using COA/COT demo modes.")
        flow_model_key = selected_models[0]
        predict_model_key = flow_model_key
        combined_model_name = None
    flow_reuse_root = args.flow_reuse_root
    reuse_stationary = not args.disable_stationary_reuse

    for split in subsets:
        subset_root = args.output_root / dataset_alias / split
        subset_root.mkdir(parents=True, exist_ok=True)
        stationary_dir = stationary_override or _dataset_default_memory(dataset_alias, "stationary")
        procedural_dir = procedural_override or _dataset_default_memory(dataset_alias, "procedural")
        stationary_mem_label = _memory_label(stationary_dir, dataset_alias)
        procedural_mem_label = _memory_label(procedural_dir, dataset_alias)

        base_cfg = apply_dataset_settings(
            config_cache,
            dataset_alias,
            split,
            args.data_dir,
            stationary_dir,
            stationary_click_override,
            procedural_dir,
        )
        print(
            f"[run_all_model_pairs] Dataset '{dataset_alias}' split '{split}' "
            f"using stationary_mem='{stationary_mem_label}', procedural_mem='{procedural_mem_label}'"
        )
        expected_total = compute_dataset_total_steps(base_cfg, args.ratio, args.limit)
        if demo_mode == "coat":
            flow_root = subset_root / "flows"
            pair_root = subset_root / "pairs"
            config_root = subset_root / "autogen_configs"

            flow_preset = MODEL_PRESETS[flow_model_key]
            flow_dir = ensure_flow_stage(
                flow_model_key,
                flow_preset,
                base_cfg,
                flow_root,
                config_root,
                use_procedural=args.flow_procedural,
                procedural_mem=procedural_mem_label,
                limit=args.limit,
                ratio=args.ratio,
                num_workers=args.num_workers,
                dataset_alias=dataset_alias,
                split=split,
                flow_reuse_root=flow_reuse_root,
                allow_partial_flow=args.allow_partial_flow,
                expected_total=expected_total,
                log_prompts=args.log_prompts,
            )
            run_predict_for_pair(
                flow_model_key,
                predict_model_key,
                flow_dir=flow_dir,
                predict_preset=MODEL_PRESETS[predict_model_key],
                base_cfg=base_cfg,
                pair_root=pair_root,
                config_root=config_root,
                use_stationary=args.predict_stationary,
                stationary_mem=stationary_mem_label,
                predict_click_mode=args.predict_click_mode,
                flow_uses_procedural=args.flow_procedural,
                procedural_mem=procedural_mem_label,
                limit=args.limit,
                ratio=args.ratio,
                num_workers=args.num_workers,
                reuse_stationary_base=reuse_stationary,
                expected_total=expected_total,
                combined_model_name=combined_model_name,
                auto_eval=args.auto_eval,
                log_prompts=args.log_prompts,
            )
        else:
            demo_root = subset_root / f"{demo_mode}_runs"
            config_root = subset_root / "autogen_configs"
            preset = MODEL_PRESETS[flow_model_key]
            run_e2e_model(
                demo_mode=demo_mode.upper(),
                model_key=flow_model_key,
                preset=preset,
                base_cfg=base_cfg,
                output_root=demo_root,
                config_root=config_root,
                click_mode=args.predict_click_mode,
                limit=args.limit,
                ratio=args.ratio,
                num_workers=args.num_workers,
                expected_total=expected_total,
                auto_eval=args.auto_eval,
                log_prompts=args.log_prompts,
            )


if __name__ == "__main__":
    main()
