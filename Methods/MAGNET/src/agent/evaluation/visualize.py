"""Visualize predicted GUI actions against ground truth annotations."""

from __future__ import annotations

import argparse
import json
import math
import textwrap
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import imagesize
from PIL import Image, ImageDraw, ImageFont

try:  # pragma: no cover - fallback for direct script execution
    from .action import ActionEvaluator, infer_episode_and_step, infer_model_from_path
except ImportError:  # pragma: no cover - script mode
    import sys

    REPO_SRC = Path(__file__).resolve().parents[2]
    if str(REPO_SRC) not in sys.path:
        sys.path.append(str(REPO_SRC))
    from agent.evaluation.action import (  # type: ignore
        ActionEvaluator,
        infer_episode_and_step,
        infer_model_from_path,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize prediction vs. ground truth actions.")
    parser.add_argument("prediction_file", type=Path, help="Path to the saved prediction JSON file.")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=None,
        help="Optional dataset root used to resolve image/json files when relative paths are stored.",
    )
    parser.add_argument("--output", type=Path, default=None, help="Where to write the visualization image.")
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Optional model name override. Defaults to inferring the model from the prediction path.",
    )
    parser.add_argument("--demo-mode", type=str, default=None, help="Override the demo mode key when needed.")
    parser.add_argument("--screen-mode", type=str, default=None, help="Override the screen mode key when needed.")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging for debugging.")
    parser.add_argument(
        "--skip-missing",
        action="store_true",
        help="Skip files that lack image paths or predictions instead of raising errors.",
    )
    return parser.parse_args()


def load_prediction(prediction_path: Path) -> Dict[str, Any]:
    with prediction_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def resolve_image_path(prediction: Dict[str, Any], data_root: Optional[Path]) -> Path:
    raw_image = prediction.get("image_path") or prediction.get("image_full_path")
    if not raw_image:
        raise ValueError("Prediction JSON does not contain `image_path` or `image_full_path` fields.")
    image_path = Path(raw_image).expanduser()
    if not image_path.is_absolute() and data_root:
        image_path = (data_root / image_path).resolve()
    if not image_path.exists():
        raise FileNotFoundError(f"Screenshot not found: {image_path}")
    return image_path


def load_ground_truth(
    prediction_file: Path,
    prediction: Dict[str, Any],
    screenshot_path: Path,
    data_root: Optional[Path],
) -> Dict[str, Any]:
    episode_from_name, step_id = infer_episode_and_step(str(prediction_file))
    screenshot_dir = screenshot_path.parent

    def _candidate_names() -> Iterable[str]:
        """Yield plausible GT basenames (dir name, suffix, inferred episode, etc)."""
        seen: set[str] = set()
        queue: List[str] = []

        def enqueue(name: Optional[str]) -> None:
            if not name:
                return
            cleaned = name.strip()
            if not cleaned or cleaned in seen:
                return
            seen.add(cleaned)
            queue.append(cleaned)

        enqueue(screenshot_dir.name)
        if "_" in screenshot_dir.name:
            enqueue(screenshot_dir.name.split("_")[-1])
        enqueue(episode_from_name)
        if episode_from_name and "_" in episode_from_name:
            enqueue(episode_from_name.split("_")[-1])
        while queue:
            yield queue.pop(0)

    def _search(base_dir: Path) -> Optional[Path]:
        for name in _candidate_names():
            candidate = base_dir / f"{name}.json"
            if candidate.exists():
                return candidate
        return None

    gt_json = _search(screenshot_dir)
    if not gt_json and data_root:
        resolved_root = data_root.expanduser().resolve()
        candidate_dirs: List[Path] = []
        try:
            relative = screenshot_dir.resolve().relative_to(resolved_root)
            candidate_dirs.append(resolved_root / relative)
        except ValueError:
            candidate_dirs.append(resolved_root / screenshot_dir.name)
            parent = screenshot_dir.parent.name if screenshot_dir.parent else None
            if parent:
                candidate_dirs.append(resolved_root / parent / screenshot_dir.name)
        for directory in candidate_dirs:
            gt_json = _search(directory)
            if gt_json:
                break

    if not gt_json:
        raise FileNotFoundError(f"Ground truth JSON not found near {screenshot_path}")
    with gt_json.open("r", encoding="utf-8") as handle:
        records: List[Dict[str, Any]] = json.load(handle)
    target = None
    if step_id is not None:
        for item in records:
            if int(item.get("step_id", -1)) == step_id:
                target = item
                break
    if target is None:
        target = records[0]
    target = dict(target)  # copy so we can mutate safely
    target["image_full_path"] = str(screenshot_path)
    target.setdefault("episode_id", screenshot_dir.name)
    target.setdefault("step_id", step_id if step_id is not None else target.get("step_id", 0))
    target.setdefault("subset", screenshot_path.parent.parent.name if screenshot_path.parent.parent else "")
    target.setdefault("instruction", prediction.get("instruction", ""))
    return target


def draw_click_marker(
    draw: ImageDraw.ImageDraw,
    coords: Optional[Iterable[float]],
    color: Tuple[int, int, int],
    radius: int,
) -> None:
    if coords is None:
        return
    if isinstance(coords, dict):
        coords = [
            coords.get("xmin"),
            coords.get("ymin"),
            coords.get("xmax"),
            coords.get("ymax"),
        ]
    try:
        values = list(coords)
    except TypeError:
        return
    center_x: Optional[float] = None
    center_y: Optional[float] = None
    if len(values) >= 4 and all(v is not None for v in values[:4]):
        xmin, ymin, xmax, ymax = (int(round(float(v))) for v in values[:4])
        draw.rectangle((xmin, ymin, xmax, ymax), outline=color, width=3)
        center_x = (xmin + xmax) / 2
        center_y = (ymin + ymax) / 2
    elif len(values) >= 2 and all(v is not None for v in values[:2]):
        center_x = float(values[0])
        center_y = float(values[1])
    if center_x is None or center_y is None:
        return
    x = int(round(center_x))
    y = int(round(center_y))
    draw.ellipse((x - radius, y - radius, x + radius, y + radius), outline=color, width=3)
    draw.line((x - radius * 2, y, x + radius * 2, y), fill=color, width=2)
    draw.line((x, y - radius * 2, x, y + radius * 2), fill=color, width=2)


def draw_bbox(draw: ImageDraw.ImageDraw, bbox: Optional[Iterable[float]], color: Tuple[int, int, int]) -> None:
    if not bbox:
        return
    xmin, ymin, xmax, ymax = bbox
    draw.rectangle((xmin, ymin, xmax, ymax), outline=color, width=3)


def draw_direction_arrow(
    draw: ImageDraw.ImageDraw,
    direction: Optional[str],
    canvas_size: Tuple[int, int],
    color: Tuple[int, int, int],
) -> None:
    if not direction:
        return
    width, height = canvas_size
    cx, cy = width // 2, height // 2
    length = min(width, height) // 5
    direction = direction.lower()
    dx = 0
    dy = 0
    if "left" in direction:
        dx -= 1
    if "right" in direction:
        dx += 1
    if "up" in direction:
        dy -= 1
    if "down" in direction:
        dy += 1
    if dx == 0 and dy == 0:
        return
    norm = math.sqrt(dx * dx + dy * dy)
    dx = dx / norm
    dy = dy / norm
    end_x = cx + dx * length
    end_y = cy + dy * length
    draw.line((cx, cy, end_x, end_y), fill=color, width=5)
    angle = math.atan2(dy, dx)
    head_len = 24
    left = (
        end_x - head_len * math.cos(angle - math.pi / 6),
        end_y - head_len * math.sin(angle - math.pi / 6),
    )
    right = (
        end_x - head_len * math.cos(angle + math.pi / 6),
        end_y - head_len * math.sin(angle + math.pi / 6),
    )
    draw.polygon([(end_x, end_y), left, right], fill=color)


def wrap_lines(text: str, width: int) -> List[str]:
    if not text:
        return []
    return textwrap.wrap(text, width=width)


def resolve_output_paths(prediction_file: Path, requested: Optional[Path]) -> tuple[Path, Path]:
    """Return the final (image_path, summary_path)."""
    default_base = prediction_file.with_name(prediction_file.stem + "_viz")
    if requested:
        if requested.suffix:
            image_path = requested
        else:
            image_path = requested / f"{prediction_file.stem}_viz.png"
    else:
        image_path = default_base.with_suffix(".png")
    summary_path = image_path.with_suffix(".json")
    return image_path, summary_path


def build_panel(
    base_image: Image.Image,
    label: str,
    color: Tuple[int, int, int],
    details: Dict[str, Any],
    radius: int,
    font: ImageFont.ImageFont,
) -> Image.Image:
    panel = base_image.copy()
    draw = ImageDraw.Draw(panel)
    action_type = details.get("type")
    action_text = None
    if action_type in {"click", "tap", "long"}:
        coords = details.get("coordinates")
        draw_click_marker(draw, coords, color, radius)
        if coords:
            action_text = f"({int(coords[0])}, {int(coords[1])})"
    elif action_type == "scroll":
        draw_direction_arrow(draw, details.get("direction"), panel.size, color)
        action_text = details.get("direction")
    elif action_type == "type":
        action_text = details.get("text")
    elif action_type == "press":
        action_text = details.get("button")
    header = f"{label}: {action_type.upper() if action_type else 'UNKNOWN'}"
    draw.text((16, 16), header, fill=color, font=font)
    if action_text:
        draw.text((16, 48), f"detail: {action_text}", fill=color, font=font)
    return panel


def create_visualization(
    screenshot_path: Path,
    gt_details: Dict[str, Any],
    pred_details: Dict[str, Any],
    metadata_lines: List[Tuple[str, str]],
    output_path: Path,
) -> None:
    base_image = Image.open(screenshot_path).convert("RGB")
    width, height = base_image.size
    font = ImageFont.load_default()
    radius = max(8, min(width, height) // 40)
    gt_panel = build_panel(base_image, "GT", (79, 195, 247), gt_details, radius, font)
    pred_panel = build_panel(base_image, "PRED", (255, 99, 71), pred_details, radius, font)
    gap = 32
    text_height = 200
    canvas = Image.new("RGB", (width * 2 + gap, height + text_height), (18, 18, 18))
    canvas.paste(gt_panel, (0, 0))
    canvas.paste(pred_panel, (width + gap, 0))
    draw = ImageDraw.Draw(canvas)
    text_y = height + 12
    for title, content in metadata_lines:
        draw.text((12, text_y), f"{title}:", fill=(230, 230, 230), font=font)
        text_y += 18
        for line in wrap_lines(content, width=110):
            draw.text((24, text_y), line, fill=(200, 200, 200), font=font)
            text_y += 16
        text_y += 6
    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(output_path)
    canvas.save(output_path)


def save_summary(summary_path: Path, payload: Dict[str, Any]) -> None:
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def main() -> None:
    args = parse_args()
    try:
        prediction = load_prediction(args.prediction_file)
        # print(prediction)
    except json.JSONDecodeError as exc:
        if args.skip_missing:
            # print(f"[skip] Invalid prediction JSON {args.prediction_file}: {exc}")
            return
        raise
    try:
        screenshot_path = resolve_image_path(prediction, args.data_root)
    except (FileNotFoundError, ValueError) as exc:
        if args.skip_missing:
            # print(f"[skip] {exc}; file={args.prediction_file}")
            return
        raise
    try:
        gt_record = load_ground_truth(args.prediction_file, prediction, screenshot_path, args.data_root)
        print(gt_record)
    except FileNotFoundError as exc:
        if args.skip_missing:
            # print(f"[skip] {exc}")
            return
        raise
    if "action_predict" not in prediction:
        if args.skip_missing:
            # print(f"[skip] Missing action_predict: {args.prediction_file}")
            return
        raise ValueError(f"Prediction missing action_predict: {args.prediction_file}")
    model_name = args.model or infer_model_from_path(str(args.prediction_file)) or ""
    print(model_name)
    width, height = imagesize.get(str(screenshot_path))
    evaluator = ActionEvaluator(
        demo_mode=args.demo_mode,
        screen_mode=args.screen_mode,
        model=model_name,
        verbose=args.verbose,
    )
    print(prediction)
    (
        pd_action_type,
        pd_action_coordinates,
        _pd_action_idx,
        pd_action_direction,
        pd_action_text,
        pd_action_button,
    ) = evaluator.parse_action(prediction, width, height)
    print(pd_action_type, pd_action_coordinates)
    if pd_action_type is None and args.skip_missing:
        # print(f"[skip] Unable to parse action for {args.prediction_file}")
        return
    (
        gt_action_type,
        gt_action_xy,
        _gt_cand_nodes,
        gt_action_text,
        gt_action_button,
        gt_action_direction,
    ) = evaluator._parse_answer_(gt_record)
    gt_details = {
        "type": gt_action_type,
        "coordinates": gt_action_xy,
        "direction": gt_action_direction,
        "text": gt_action_text,
        "button": gt_action_button,
        "bbox": None,
    }
    pred_details = {
        "type": pd_action_type,
        "coordinates": pd_action_coordinates,
        "direction": pd_action_direction,
        "text": pd_action_text,
        "button": pd_action_button,
    }
    print(pred_details)
    instruction = gt_record.get("instruction") or ""
    thought = prediction.get("action_think") or ""
    episode_info, step_id = infer_episode_and_step(str(args.prediction_file))
    metadata_lines = [
        ("Prediction file", str(args.prediction_file)),
        ("Model", model_name or "unknown"),
        (
            "Episode",
            f"{episode_info or gt_record.get('episode_id')} | step {step_id if step_id is not None else gt_record.get('step_id')}",
        ),
        ("Instruction", instruction),
        ("Model thought", thought),
    ]
    evaluation_result = evaluator(gt_record, prediction)
    metadata_lines.extend(
        [
            ("Type match", "yes" if evaluation_result.get("type_match") else "no"),
            ("Exact match", "yes" if evaluation_result.get("exact_match") else "no"),
        ]
    )
    image_path, summary_path = resolve_output_paths(args.prediction_file, args.output)
    print(image_path)
    create_visualization(screenshot_path, gt_details, pred_details, metadata_lines, image_path)
    summary_payload = {
        "prediction_file": str(args.prediction_file),
        "image_path": str(image_path),
        "summary_path": str(summary_path),
        "model": model_name or "unknown",
        "episode_id": episode_info or gt_record.get("episode_id"),
        "step_id": step_id if step_id is not None else gt_record.get("step_id"),
        "instruction": instruction,
        "model_thought": thought,
        "ground_truth": {
            "parsed": gt_details,
            "record": gt_record,
        },
        "prediction": {
            "parsed": pred_details,
            "raw": prediction,
        },
        "metadata": {key: value for key, value in metadata_lines},
        "evaluation": {
            "type_match": evaluation_result.get("type_match"),
            "exact_match": evaluation_result.get("exact_match"),
            "format_hit": evaluation_result.get("format_hit"),
            "text_similarity": evaluation_result.get("text_dist"),
            "answer": evaluation_result.get("answer"),
            "prediction": evaluation_result.get("pred"),
        },
    }
    save_summary(summary_path, summary_payload)
    print(f"Saved visualization to {image_path}\nSaved summary to {summary_path}")


if __name__ == "__main__":
    main()
