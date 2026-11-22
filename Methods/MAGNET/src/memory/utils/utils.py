from typing import TYPE_CHECKING, Optional, Tuple, Union, Callable, List, Any, Generator, Dict

import os, sys
import json

from PIL import Image, ImageDraw, ImageFont
from colorama import Fore, Style


dir_name = os.path.abspath(os.path.dirname(__file__))
FONT_PATH = os.path.join(dir_name, "SimHei.ttf")




def print_with_color(*values: object, color="", end="\n", reset=True):
    if color == "red":
        print(Fore.RED, *values, sep="", end=end)
    elif color == "green":
        print(Fore.GREEN, *values, sep="", end=end)
    elif color == "yellow":
        print(Fore.YELLOW, *values, sep="", end=end)
    elif color == "blue":
        print(Fore.BLUE, *values, sep="", end=end)
    elif color == "magenta":
        print(Fore.MAGENTA, *values, sep="", end=end)
    elif color == "cyan":
        print(Fore.CYAN, *values, sep="", end=end)
    elif color == "white":
        print(Fore.WHITE, *values, sep="", end=end)
    elif color == "black":
        print(Fore.BLACK, *values, sep="", end=end)
    else:
        print(*values, end=end)
    
    if reset: print(Style.RESET_ALL)
    sys.stdout.flush()




def draw_arrow(image_obj_or_path:Union[Image.Image, str], touch_yx:Tuple[int, int], lift_yx:Tuple[int, int], color=(255, 0, 0)):
    """ Add a red arrow on the image between two points """
    # Open the image
    if not isinstance(image_obj_or_path, Image.Image):
        image = Image.open(image_obj_or_path).convert('RGB')
    else: image = image_obj_or_path.convert('RGB')
    w, h = image.size

    draw = ImageDraw.Draw(image)

    touch_y, touch_x = touch_yx
    lift_y, lift_x = lift_yx

    # Draw a red arrow from lift to touch (optional: add some size to the arrow)
    arrow_w_size = max(int(w * 0.1), 20)  # Horizonal size of the arrowhead
    arrow_h_size = min(max(int(abs(touch_y-lift_y) * 0.1), 20), 40)
    draw.line([lift_x, lift_y, touch_x, touch_y], fill=color, width=5)
    if lift_y < touch_y:  # Arrow goes upwards
        draw.polygon([
            (lift_x, lift_y),
            (lift_x - arrow_w_size, lift_y + arrow_h_size),
            (lift_x + arrow_w_size, lift_y + arrow_h_size)
        ], fill=color)
    else: # Arrow goes downwards
        draw.polygon([
            (lift_x, lift_y),
            (lift_x - arrow_w_size, lift_y - arrow_h_size),
            (lift_x + arrow_w_size, lift_y - arrow_h_size)
        ], fill=color)
    
    return image




def draw_bbox(image_obj_or_path:Union[Image.Image, str], bboxes:List[Tuple[float, float, float, float]], texts=None, rgba_color=(0, 0, 255, 0), thickness=1):
    """ Draw the bounding boxes (x1, y1, x2, y2) with corresponding texts """
    if not isinstance(image_obj_or_path, Image.Image):
        image = Image.open(image_obj_or_path).convert('RGBA')
    else: image = image_obj_or_path.convert('RGBA')
    w, h = image.size

    text_coords = []

    with image as base:
        tmp = Image.new("RGBA", base.size, (0, 0, 0, 0))
        # get a drawing context
        draw = ImageDraw.Draw(tmp)
        for idx, bbox in enumerate(bboxes):
            xmin, ymin, xmax, ymax = bbox[:4]
            xmin = min(max(0, xmin), w)
            xmax = min(max(xmin, xmax), w)
            ymin = min(max(0, ymin), h)
            ymax = min(max(ymin, ymax), h)
            # draw the boudning box
            draw.rectangle((xmin, ymin, xmax, ymax), outline=rgba_color, width=thickness)
        
            # draw text if any
            if texts:
                text = texts[idx]
                box_height = ymax - ymin
                font_size = int(min(max(int(box_height * 0.7), 12), 36))

                font = ImageFont.truetype(FONT_PATH, font_size, encoding="utf-8")
                left, top, right, bot = font.getbbox(text)
                coords = [xmin, ymin,
                        xmin + right*1.1, ymin,
                        xmin + right*1.1, ymin - bot*1.1,
                        xmin, ymin - bot*1.1]
                draw.polygon(coords, fill=rgba_color)
                draw.text((xmin, ymin - bot*1.05), text, (255,255,255,255), font=font)
                text_coords.append([coords[0], coords[5], coords[2], coords[1]])
        
        out = Image.alpha_composite(base, tmp)

        return out




def read_jsonl(file_path: str):
    """
    Reads a JSON Lines (jsonl) file and returns a list of dictionaries/objects.
    
    Each line in the file is parsed as a separate JSON object.
    """
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            line = line.strip()
            data.append(json.loads(line))
    
    return data


def save_to_jsonl(data:list, file_path:str):
    """
    Saves a list of Python objects to a JSON Lines (jsonl) file.
    
    Each element in `data` is written as a separate JSON line.
    
    :param data: List of Python objects (e.g., dicts) to save.
    :param file_path: Path to the output .jsonl file.
    """
    with open(file_path, 'w', encoding='utf-8') as f:
        for record in data:
            # Convert each record to a JSON string and write it as a new line
            f.write(json.dumps(record) + '\n')
    
