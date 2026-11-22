from __future__ import annotations

import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parents[1]
METHODS = ("no_memory", "raw_multimodal", "raw_text", "raw_visual", "ksi_style", "ksi_visual",
           "magnet_adapted", "darwinian_adapted", "evoskill_adapted")


def load_config(path: Path | None = None) -> dict:
    config = json.loads((path or ROOT / "config/smoke.json").read_text())
    if config["memory"]["online_updates"]:
        raise ValueError("The evaluation runner requires frozen memory; rebuild offline instead")
    if not set(config["baselines"]) <= set(METHODS):
        raise ValueError("Unknown baseline")
    if 'ksi_visual' in config['baselines'] and config.get('memory_construction', {}).get('protocol') != 'paired_ksi_magnet_v1':
        raise ValueError('ksi_visual requires the paired construction protocol')
    if config.get('memory_construction',{}).get('magnet_protocol') == 'described_complete_link_v2':
        threshold=config['memory_construction'].get('magnet_v2_threshold')
        if type(threshold) not in (int,float) or not 0<=threshold<=1:
            raise ValueError('MAGNET v2 requires a construction-calibrated threshold in [0, 1]')
    for key in ("top_k", "max_images", "max_text_chars"):
        if config["memory"][key] < 0:
            raise ValueError(f"Negative memory budget: {key}")
    return config


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)
