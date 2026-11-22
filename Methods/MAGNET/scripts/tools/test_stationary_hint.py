#!/usr/bin/env python
from __future__ import annotations

import ast
import json
from pathlib import Path

from PIL import Image

from utils.image_ops import template_match_center, nearest_bboxes, enclosing_bbox, draw_hint_bbox


def main():
    repo_root = Path(__file__).resolve().parents[2]
    sample_root = repo_root / "sample_data" / "android_in_the_zoo" / "test"
    episode_dir = next((sample_root / "general").iterdir())
    episode = json.load((episode_dir / f"{episode_dir.name}.json").open("r", encoding="utf-8"))
    step = episode[0]
    image_path = sample_root / step["image_path"]

    if isinstance(step["ui_positions"], str):
        ui_positions = ast.literal_eval(step["ui_positions"])
    else:
        ui_positions = step["ui_positions"]
    if not ui_positions:
        raise RuntimeError("No ui_positions in sample data.")

    ymin, xmin, h, w = ui_positions[0]
    screenshot = Image.open(image_path)
    patch = screenshot.crop((xmin, ymin, xmin + w, ymin + h))
    patch_path = repo_root / "artifacts" / "runtime" / "hint_patch.png"
    patch_path.parent.mkdir(parents=True, exist_ok=True)
    patch.save(patch_path)

    center, score = template_match_center(str(image_path), str(patch_path))
    if center is None:
        raise RuntimeError("Template match failed.")

    nearest = nearest_bboxes(center, ui_positions, k=5)
    bbox = enclosing_bbox(nearest)
    out_path = repo_root / "artifacts" / "runtime" / "hint_overlay.png"
    draw_hint_bbox(str(image_path), bbox, str(out_path))

    print("Template match score:", f"{score:.3f}")
    print("Hint image:", out_path)


if __name__ == "__main__":
    main()
