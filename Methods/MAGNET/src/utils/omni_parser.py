from __future__ import annotations

import json
import os
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

from PIL import Image


def _try_add_omniparser_path(root: str | None) -> None:
    env_path = root or os.environ.get("OMNIPARSER_ROOT", "")
    if env_path and env_path not in sys.path:
        sys.path.insert(0, env_path)


def _parse_config(config: str | Dict[str, Any] | None) -> Dict[str, Any]:
    if config is None:
        return {}
    if isinstance(config, dict):
        return dict(config)
    if isinstance(config, str) and config.strip():
        path = Path(config)
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
    return {}


def _ratio_to_absolute(
    bbox_ratio: Sequence[float],
    width: int,
    height: int,
) -> Tuple[int, int, int, int]:
    xmin = int(max(min(bbox_ratio[0] * width, width), 0))
    ymin = int(max(min(bbox_ratio[1] * height, height), 0))
    xmax = int(max(min(bbox_ratio[2] * width, width), 0))
    ymax = int(max(min(bbox_ratio[3] * height, height), 0))
    return xmin, ymin, xmax, ymax


@lru_cache(maxsize=1)
def _load_omniparser_models(weights_dir: str):
    from util.utils import (  # type: ignore import-not-found
        get_caption_model_processor,
        get_yolo_model,
    )

    weights_root = Path(weights_dir).resolve()
    som_model_path = weights_root / "icon_detect" / "model.pt"
    caption_dir = weights_root / "icon_caption_florence"
    if not som_model_path.exists():
        raise FileNotFoundError(f"Missing detector weights: {som_model_path}")
    if not caption_dir.exists():
        raise FileNotFoundError(f"Missing caption weights: {caption_dir}")

    yolo_model = get_yolo_model(model_path=str(som_model_path))
    caption_model_processor = get_caption_model_processor(
        model_name="florence2",
        model_name_or_path=str(caption_dir),
        device=None,
    )
    return yolo_model, caption_model_processor


def parse_ui_bboxes(image_path: str, config: str | Dict[str, Any] | None = None) -> List[list]:
    """Return UI bounding boxes as [y, x, h, w]; empty if OmniParser unavailable."""
    cfg = _parse_config(config)
    _try_add_omniparser_path(cfg.get("root"))

    try:
        from util.utils import (  # type: ignore import-not-found
            check_ocr_box,
            get_som_labeled_img,
        )
    except Exception:
        return []

    weights_dir = cfg.get("weights_dir") or os.environ.get("OMNIPARSER_WEIGHTS", "weights")
    box_threshold = float(cfg.get("box_threshold", 0.05))
    iou_threshold = float(cfg.get("iou_threshold", 0.1))
    imgsz = int(cfg.get("imgsz", 640))
    batch_size = int(cfg.get("batch_size", 128))
    easyocr_threshold = float(cfg.get("easyocr_threshold", 0.9))
    use_paddleocr = bool(cfg.get("use_paddleocr", False))

    yolo_model, caption_model_processor = _load_omniparser_models(weights_dir)
    pil_image = Image.open(image_path).convert("RGB")
    width, height = pil_image.size

    easyocr_args = {"paragraph": False, "text_threshold": easyocr_threshold}
    ocr_bbox_rslt, _ = check_ocr_box(
        pil_image,
        display_img=False,
        output_bb_format="xyxy",
        goal_filtering=None,
        easyocr_args=easyocr_args,
        use_paddleocr=use_paddleocr,
    )
    text, ocr_bbox = ocr_bbox_rslt
    if text is None:
        text = []
    if ocr_bbox is None:
        ocr_bbox = []

    _, _, parsed_elements = get_som_labeled_img(
        pil_image,
        yolo_model,
        BOX_TRESHOLD=box_threshold,
        output_coord_in_ratio=True,
        ocr_bbox=ocr_bbox,
        draw_bbox_config=None,
        caption_model_processor=caption_model_processor,
        ocr_text=text,
        use_local_semantics=True,
        iou_threshold=iou_threshold,
        scale_img=False,
        imgsz=imgsz,
        batch_size=batch_size,
    )

    ui_positions: List[list] = []
    for elem in parsed_elements:
        bbox_ratio = tuple(float(v) for v in elem["bbox"])
        xmin, ymin, xmax, ymax = _ratio_to_absolute(bbox_ratio, width, height)
        ui_positions.append([ymin, xmin, ymax - ymin, xmax - xmin])
    return ui_positions
