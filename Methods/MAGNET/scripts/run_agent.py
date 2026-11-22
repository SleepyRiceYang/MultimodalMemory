#!/usr/bin/env python
from __future__ import annotations

import argparse
import copy
import json
import os
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List

def _ensure_local_no_proxy():
    hosts = ["localhost", "127.0.0.1"]
    for key in ("NO_PROXY", "no_proxy"):
        current_entries = [entry.strip() for entry in os.environ.get(key, "").split(",") if entry.strip()]
        existing = set(current_entries)
        for host in hosts:
            if host not in existing:
                current_entries.append(host)
                existing.add(host)
        os.environ[key] = ",".join(current_entries) if current_entries else ",".join(hosts)


_ensure_local_no_proxy()

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from agent.agents import AtlasAgent, GUIR1Agent, ScreenAgent, UI_Venus_Navi_Agent, InfiGUI_R1_Agent
from agent.config.loader import load_config
from datasets.datasets import build_dataset
from utils.model_traits import get_model_traits
from workflow.runner import WorkflowRunner, ensure_flow_dir

AGENT_REGISTRY = {
    "coat": ScreenAgent,
    "atlas": AtlasAgent,
    "guir1": GUIR1Agent,
    "ui_venus": UI_Venus_Navi_Agent,
    "infigui_r1": InfiGUI_R1_Agent,
}

STAGES = ("flow", "predict", "reflect", "all")
MEMORY_CHOICES = ("none", "stationary", "procedural", "both")


def parse_args():
    parser = argparse.ArgumentParser(description="Run the GUI agent pipeline with a single entry point.")
    parser.add_argument("--config", default=REPO_ROOT / "configs" / "presets" / "coat" / "config_gemini_android.yaml")
    parser.add_argument("--output-dir", help="Optional override for OUTPUT_DIR in the config.")
    parser.add_argument("--dataset-dir", help="Optional override for DATA.DATA_DIR.")
    parser.add_argument("--split", help="Optional override for DATA.SPLIT.")
    parser.add_argument("--demo-mode", help="Optional override for DEMO_MODE.")
    parser.add_argument("--ratio", type=float, default=1.0, help="Subset ratio for quick experiments.")
    parser.add_argument("--double-sample", action="store_true", help="Whether to double sample episodes.")
    parser.add_argument("--limit", type=int, default=0, help="Stop after processing N steps (0 = run all).")
    parser.add_argument("--num-workers", type=int, default=64, help="Number of parallel workers per stage.")
    parser.add_argument("--print-config", action="store_true", help="Dump the resolved configuration.")
    parser.add_argument("--stage", choices=STAGES, default="flow", help="Which stage(s) to execute.")
    parser.add_argument(
        "--memory-mode",
        choices=MEMORY_CHOICES,
        default="none",
        help="Inject memory for the selected stage (used for ablations).",
    )
    counter_group = parser.add_mutually_exclusive_group()
    counter_group.add_argument(
        "--inherit-memory-counter",
        action="store_true",
        help="Reuse memory ranking counters from the existing RAG directory.",
    )
    counter_group.add_argument(
        "--reset-memory-counter",
        action="store_true",
        help="Reset memory ranking counters for this run.",
    )
    parser.add_argument(
        "--flow-dir",
        type=Path,
        help="Reuse an existing flow directory (for running predict/reflect with a different config).",
    )
    parser.add_argument(
        "--log-prompts",
        action="store_true",
        help="Print every prompt/response pair sent to the VLM.",
    )
    return parser.parse_args()


def build_overrides(args) -> Dict[str, Any]:
    overrides: Dict[str, Any] = {}
    if args.output_dir:
        overrides["OUTPUT_DIR"] = args.output_dir
    if args.demo_mode:
        overrides["DEMO_MODE"] = args.demo_mode
    if getattr(args, "log_prompts", False):
        overrides["LOG_PROMPTS"] = True
    data_overrides: Dict[str, Any] = {}
    if args.dataset_dir:
        data_overrides["DATA_DIR"] = args.dataset_dir
    if args.split:
        data_overrides["SPLIT"] = args.split
    if data_overrides:
        overrides["DATA"] = data_overrides
    memory_overrides: Dict[str, Any] = {}
    if getattr(args, "inherit_memory_counter", False):
        memory_overrides["INHERIT_COUNTER"] = True
    if getattr(args, "reset_memory_counter", False):
        memory_overrides["INHERIT_COUNTER"] = False
    if memory_overrides:
        overrides.setdefault("MEMORY", {}).update(memory_overrides)
    return overrides


def apply_memory_mode(cfg, stage: str, memory_mode: str, traits):
    action_style = cfg.DEMO_MODE.upper()
    if action_style in ("COA", "COT"):
        cfg.MEMORY.STATIONARY = False
        cfg.MEMORY.PROCEDURAL = False
        return cfg

    wants_stationary = memory_mode in ("stationary", "both")
    wants_procedural = memory_mode in ("procedural", "both")

    if stage == "flow":
        cfg.MEMORY.PROCEDURAL = wants_procedural
        cfg.MEMORY.STATIONARY = False
    elif stage == "predict":
        if wants_stationary and not traits.supports_stationary:
            raise ValueError(f"Model {cfg.MODEL.NAME} does not support stationary memory.")
        cfg.MEMORY.STATIONARY = wants_stationary
        cfg.MEMORY.PROCEDURAL = False
    else:
        cfg.MEMORY.STATIONARY = False
        cfg.MEMORY.PROCEDURAL = False
    return cfg


def build_agent(cfg):
    agent_cls = AGENT_REGISTRY.get(cfg.DEMO_MODE.lower(), ScreenAgent)
    print(f"[run_agent] Using agent class: {agent_cls.__name__}")
    return agent_cls(cfg)


def compute_stages(stage_flag: str) -> List[str]:
    if stage_flag == "all":
        return ["flow", "predict", "reflect"]
    return [stage_flag]


def main():
    args = parse_args()
    print(args.config)
    base_cfg = load_config(args.config, overrides=build_overrides(args))
    if args.print_config:
        print(json.dumps(asdict(base_cfg), indent=2))

    dataset = build_dataset(base_cfg, ratio=args.ratio, double_sample=args.double_sample)
    work_dir = Path(args.output_dir or base_cfg.OUTPUT_DIR)
    work_dir.mkdir(parents=True, exist_ok=True)

    action_mode = base_cfg.DEMO_MODE.upper()
    e2e_mode = action_mode in ("COA", "COT")
    if e2e_mode:
        supported_model = any(token in base_cfg.MODEL.NAME.lower() for token in ("gemini", "qwenvl"))
        if not supported_model:
            raise ValueError(f"Demo mode {action_mode} currently supports only Gemini or Qwen-VL models.")

    traits = get_model_traits(base_cfg.MODEL.NAME, getattr(base_cfg.MODEL, "CLICK_MODE", None))
    memory_mode = args.memory_mode
    if e2e_mode and memory_mode != "none":
        print("[run_agent] End-to-end COA/COT does not support memory; forcing --memory-mode none.")
        memory_mode = "none"

    if not e2e_mode and args.stage in ("predict", "reflect") and args.flow_dir:
        ensure_flow_dir(args.flow_dir, work_dir)
    elif e2e_mode and args.flow_dir:
        print("[run_agent] Ignoring --flow-dir for COA/COT runs.")

    stages = compute_stages(args.stage)
    ran_stage = False
    for stage in stages:
        if e2e_mode and stage in ("flow", "reflect"):
            print(f"[run_agent] Skipping stage '{stage}' for {action_mode} (not applicable).")
            continue
        stage_cfg = apply_memory_mode(copy.deepcopy(base_cfg), stage, memory_mode, traits)

        def agent_factory(cfg=stage_cfg):
            return build_agent(copy.deepcopy(cfg))

        agent = agent_factory() if args.num_workers == 1 else None
        print(f"Dataset has {len(dataset)} episodes for stage '{stage}'")
        runner = WorkflowRunner(
            agent,
            dataset,
            work_dir,
            limit=args.limit,
            desc=stage.capitalize(),
            num_workers=args.num_workers,
            agent_factory=agent_factory,
        )
        if stage == "flow":
            runner.run_flow()
        elif stage == "predict":
            runner.run_predict()
        else:
            runner.run_reflect()
        ran_stage = True

    if ran_stage:
        print(f"[run_agent] Finished. Artifacts saved to {work_dir}")
    else:
        print("[run_agent] No stages were executed; please check --stage and --demo-mode combinations.")


if __name__ == "__main__":
    main()
