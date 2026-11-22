import argparse
import sys
from pathlib import Path
from typing import Iterable, Sequence

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from utils.image_ops import smart_resize


DEFAULT_IMAGE = "/remote-home/lbsun/data/amex/val_id_app/00b131dc132946e186a38730c90f0d58/00b131dc132946e186a38730c90f0d58_13.png"
DEFAULT_BBOX = [
        475,
        2196,
        720,
        2406
      ]
DEFAULT_MODEL = "qwen2-vl-7b-instruct"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize raw model bbox vs. smart-resized bbox (reference: qwenvl_bbox_resize.py).",
    )
    parser.add_argument("--image-path", default=DEFAULT_IMAGE, help="Target image path.")
    parser.add_argument(
        "--bbox",
        type=float,
        nargs=4,
        default=DEFAULT_BBOX,
        metavar=("XMIN", "YMIN", "XMAX", "YMAX"),
        help="Raw bbox from model prediction (assumed in smart-resized coordinate space).",
    )
    parser.add_argument("--model-name", default=DEFAULT_MODEL, help="Model name hint for smart_resize params.")
    parser.add_argument(
        "--output-dir",
        default="scripts/outputs",
        help="Directory to store visualization PNGs.",
    )
    return parser.parse_args()


def get_resized_shape(width: int, height: int, model_name: str) -> tuple[int, int]:
    lower = (model_name or "").lower()
    if "ui-venus-navi-7b" in lower:
        resized_h, resized_w = smart_resize(
            height,
            width,
            factor=28,
            min_pixels=830000,
            max_pixels=937664,
        )
    else:
        resized_h, resized_w = smart_resize(
            height,
            width,
            factor=28,
            min_pixels=28 * 28,
            max_pixels=28*28*2000,
        )
    return resized_w, resized_h


def model_bbox_to_original(
    bbox: Sequence[float],
    width: int,
    height: int,
    model_name: str,
) -> list[int]:
    """Convert bbox from smart-resized space back to the original resolution."""
    resized_w, resized_h = get_resized_shape(width, height, model_name)
    scale_x = width / resized_w
    scale_y = height / resized_h
    xmin, ymin, xmax, ymax = bbox
    return [
        round(xmin * scale_x),
        round(ymin * scale_y),
        round(xmax * scale_x),
        round(ymax * scale_y),
    ]


def resize_image(image: Image.Image, model_name: str) -> Image.Image:
    """Resize image following the smart_resize policy for comparison."""
    resized_w, resized_h = get_resized_shape(image.width, image.height, model_name)
    if (image.width, image.height) == (resized_w, resized_h):
        return image.copy()
    return image.resize((resized_w, resized_h))


def draw_bbox(image: Image.Image, bbox: Iterable[float], color: str, width: int = 6) -> Image.Image:
    out = image.copy()
    draw = ImageDraw.Draw(out)
    draw.rectangle(bbox, outline=color, width=width)
    return out


def visualize(image_path: str, bbox: Sequence[float], model_name: str, output_dir: str) -> None:
    image = Image.open(image_path).convert("RGB")
    resized_image = resize_image(image, model_name)

    original_bbox = model_bbox_to_original(bbox, image.width, image.height, model_name)
    resized_bbox = [round(v) for v in bbox]

    print(f"Image path: {image_path}")
    print(f"Model: {model_name}")
    print(f"Original size: {image.size}, resized size: {resized_image.size}")
    print(f"Raw bbox: {bbox}")
    print(f"Original-space bbox: {original_bbox}")

    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    original_with_box = draw_bbox(image, original_bbox, color="red")
    resized_with_box = draw_bbox(resized_image, resized_bbox, color="blue")

    original_out = output_root / "original_bbox.png"
    resized_out = output_root / "resized_bbox.png"
    original_with_box.save(original_out)
    resized_with_box.save(resized_out)

    print(f"Saved original visualization to: {original_out}")
    print(f"Saved resized visualization to: {resized_out}")


if __name__ == "__main__":
    args = parse_args()
    visualize(args.image_path, args.bbox, args.model_name, args.output_dir)
