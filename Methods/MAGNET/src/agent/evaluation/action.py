import json
import math
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Optional, Tuple

import Levenshtein
import imagesize
import numpy as np
import pandas as pd
from tqdm import tqdm

from utils.model_traits import get_model_traits
from utils.screen import check_inside, enlarge_bbox, intersect_iou, row_col_sort
from utils.image_ops import smart_resize

class ActionEvaluator(object):
    BBOX_PATTERN = re.compile(r'\[ *(\d+) *, *(\d+) *, *(\d+) *, *(\d+) *\]')

    def __init__(self, demo_mode, screen_mode, model=None, click_mode_override: str | None = None, verbose: bool = False) -> None:
        self.demo_mode = demo_mode
        self.screen_mode = screen_mode
        self.model = model.lower() if model else None
        self.traits = get_model_traits(model or "", click_mode_override)
        self.verbose = verbose

        self._all_action_types_ = [
            "click",
            "tap", # convert to click 
            "scroll", 
            "swipe", # convert ot scroll
            "type", 
            "press",
            "stop",
            "long" # long press
        ]
    
    def action_map(self, action_api:str):
# Map action API to action type
        if not action_api: return None
        action_api = action_api.lower()
        if action_api == "input": return "type"
        for act_type in self._all_action_types_:
            if act_type in action_api: return act_type
        return None

    def _parse_action_(self, pred, w, h):
# Parse predicted action
        # print(f"pred", pred)
        pr = pred.get('action_predict', {})
        # print(pr)
        if self.demo_mode not in pr:
            if "UI_VENUS" in pr:
                action = pr["UI_VENUS"].get(self.screen_mode, {})
            
            else:
                action = None
        else:
            action = pr[self.demo_mode].get(self.screen_mode, {})
        # print(pr.keys(),self.demo_mode)
        if not action:
            return (None,) * 6
        print(action)
        pd_action_type = self.action_map(action.get('ACTION', None))
        if pd_action_type == "swipe":
            pd_action_type = "scroll"
        if pd_action_type is None and self.verbose:
            print(action)
        print(pd_action_type)
        pd_action_args = action.get('ARGS', {})
        if not isinstance(pd_action_args, dict):
            pd_action_args = {}
        raw_bbox = pd_action_args.get('bbox')
        raw_xy = pd_action_args.get('xy')
        pd_action_xy = None
        if isinstance(raw_bbox, dict):
            raw_bbox = [
                raw_bbox.get("xmin"),
                raw_bbox.get("ymin"),
                raw_bbox.get("xmax"),
                raw_bbox.get("ymax"),
            ]
        if isinstance(raw_bbox, (list, tuple)) and len(raw_bbox) >= 4:
            try:
                coords = [float(raw_bbox[i]) for i in range(4)]
            except (ValueError, TypeError):
                coords = None
        else:
            coords = None
        lower_model_name = (self.model or "").lower()
        print(f"self.model: {self.model}")
        needs_qwen_resize = any(token in lower_model_name for token in ("qwen", "qwen", "ui-venus", "infigui-r1"))
        print(f"needs_qwen_resize: {needs_qwen_resize}")
        pd_action_bbox = None
        if coords:
            xmin, ymin, xmax, ymax = coords
            norm_max = max(abs(v) for v in coords)
            if needs_qwen_resize:
                xmin, ymin, xmax, ymax = self._resize_qwen_coordinates((xmin, ymin, xmax, ymax), w, h, mode="bbox")
            else:
                xmin, ymin, xmax, ymax = map(lambda val: round(val), (xmin, ymin, xmax, ymax))
            pd_action_bbox = [xmin, ymin, xmax, ymax]
        if pd_action_bbox:
            pd_action_xy = ((pd_action_bbox[0] + pd_action_bbox[2]) / 2, (pd_action_bbox[1] + pd_action_bbox[3]) / 2)
        elif isinstance(raw_xy, (list, tuple)) and len(raw_xy) >= 2:
            try:
                point = [float(raw_xy[0]), float(raw_xy[1])]
            except (TypeError, ValueError):
                point = None
            if point is not None:
                if needs_qwen_resize:
                    px, py = self._resize_qwen_coordinates(point, w, h, mode="point")
                else:
                    norm_max = max(abs(point[0]), abs(point[1]))
                    if norm_max <= 1.3:
                        px = round(point[0] * w)
                        py = round(point[1] * h)
                    elif norm_max <= 1000 and "infigui-r1" not in lower_model_name:
                        px = round(point[0] / 1000 * w)
                        py = round(point[1] / 1000 * h)
                    else:
                        px = round(point[0])
                        py = round(point[1])
                pd_action_xy = [px, py]

        pd_action_idx = pd_action_args.get('idx', None)
        if pd_action_idx is not None: 
            try: pd_action_idx = int(pd_action_idx)
            except: pd_action_idx = None
        pd_action_direction = pd_action_args.get('direction', None)
        # if pd_action_type=="scroll" and pd_action_direction is None:
        #     print(64, action)
        pd_action_text = pd_action_args.get('text', "")
        action['ACTION'] = action['ACTION'].replace(" ", "_")
        pd_action_button = None if pd_action_type != "press" else \
            action['ACTION'].split("_")[1].lower()
        if pd_action_bbox is not None:
            pd_action_coordinates = pd_action_bbox
        elif pd_action_xy is not None:
            pd_action_coordinates = pd_action_xy
        else:
            pd_action_coordinates = None
        print(f"pd_action_coordinates: {pd_action_coordinates}")
        return (
            pd_action_type,
            pd_action_coordinates,
            pd_action_idx,
            pd_action_direction,
            pd_action_text,
            pd_action_button
        )

    def parse_action(self, pred, width, height):
        """Public wrapper used by visualization utilities."""
        return self._parse_action_(pred, width, height)

    def _parse_answer_(self, gt):
# Parse ground truth action
        gt_words = gt['coat_action_desc'].split(' ')

        gt_action_type = self.action_map(gt_words[0])
        if gt_action_type is None and self.verbose:
            print(gt['subset'], gt['episode_id'])
        gt_action_text = gt['result_action_text']
        gt_action_direction = "" if gt_action_type != "scroll" else gt_words[1].strip()
        gt_action_button = ""
        if gt_action_type == "press":
            for button in ['enter', 'back', 'home', 'switch']: 
                if button in gt['coat_action_desc']: 
                    gt_action_button = button
                    break

        w, h = imagesize.get(gt['image_full_path'])
        gt_action_xy = [0, 0]
        if gt_action_type in ["scroll", "click", "tap", "long", "swipe"]:
            if gt_action_type == "swipe": gt_action_type = "scroll"
            if gt_action_type == "tap": gt_action_type = "click"
            rel_y, rel_x = json.loads(gt['result_touch_yx']) if isinstance(gt['result_touch_yx'], str) else gt['result_touch_yx']
            abs_y, abs_x = int(rel_y*h), int(rel_x*w)
            gt_action_xy = [abs_x, abs_y]
        
        gt_positions = gt.get('ui_positions') or []
        gt_texts = gt.get('ui_text') or []
        gt_types = gt.get('ui_types')
        if isinstance(gt_positions, str):
            gt_positions = eval(gt_positions)
        if isinstance(gt_texts, str):
            gt_texts = eval(gt_texts)
        if isinstance(gt_types, str):
            gt_types = eval(gt_types)
        
        if gt_types is None:
            gt_types = ["unknown"] * len(gt_positions)
        gt_cand_nodes = []
        # print(gt_positions)
        for org_bbox, txt, ui_class in zip(gt_positions, gt_texts, gt_types):
            print(org_bbox)
            ymin, xmin, h, w  = org_bbox
            bbox = [xmin, ymin, xmin+w, ymin+h]
            gt_cand_nodes.append({"bounds": bbox, "text": txt, "type": ui_class})
        gt_cand_nodes = row_col_sort(gt_cand_nodes)
        
        return gt_action_type, gt_action_xy, gt_cand_nodes, \
               gt_action_text, gt_action_button, gt_action_direction

    # def _check_click_(self, pred_bbox, gt_xy, gt_nodes):
    #     # gt_xy is within pred_bbox
    #     if not pred_bbox: return False
    #     print(pred_bbox)
    #     pred_bbox = enlarge_bbox([pred_bbox])[0]
    #     xmin, ymin, xmax, ymax = pred_bbox
    #     gt_x, gt_y = gt_xy
    #     is_correct = (xmin <= gt_x <= xmax and ymin <= gt_y <= ymax)
    #     if is_correct: return True

    #     # gt_xy is within any bbox
    #     if len(gt_nodes) > 0:
    #         bbox_array = enlarge_bbox([x['bounds'] for x in gt_nodes], scale_factor=1.2)
    #         is_inside, bbox_inside = check_inside(gt_x, gt_y, bbox_array)
    #         if is_inside:
    #             ious = intersect_iou(pred_bbox, bbox_inside)
    #             if np.any(ious > 0.5): return True

    #     return False

    def _check_click_(self, image_size, pred_xy, gt_xy, gt_nodes):
        # gt_xy is within pred_bbox
        if not pred_xy: return False
        if self.verbose:
            print(pred_xy)
        # pred_bbox = enlarge_bbox([pred_bbox])[0]
        pred_x, pred_y = pred_xy
        gt_x, gt_y = gt_xy
        width, height = image_size
        if self.model and "atlas" in self.model:
            pred_x = pred_x / 1000 * width
            pred_y = pred_y / 1000 * height
        pred_rlv_x, pred_rlv_y = pred_x / width * 1000, pred_y / height * 1000 
        gt_rlv_x, gt_rlv_y = gt_x / width * 1000, gt_y / height * 1000 

        if math.sqrt((pred_rlv_x - gt_rlv_x)**2 + (pred_rlv_y - gt_rlv_y)**2) <=0.14*1000:
            is_correct = True
        else:
            is_correct = False

        if is_correct: return True

        # # gt_xy is within any bbox
        # if len(gt_nodes) > 0:
        #     bbox_array = enlarge_bbox([x['bounds'] for x in gt_nodes], scale_factor=1.2)
        #     is_inside, bbox_inside = check_inside(gt_x, gt_y, bbox_array)
        #     if is_inside:
        #         ious = intersect_iou(pred_bbox, bbox_inside)
        #         if np.any(ious > 0.5): return True

        return False

    def _check_click_bbox(self, pred_bbox, gt_xy, gt_nodes):
        if not pred_bbox:
            return False
        bbox = enlarge_bbox([pred_bbox], scale_factor=1.4)[0]
        xmin, ymin, xmax, ymax = bbox
        gt_x, gt_y = gt_xy
        is_correct = xmin <= gt_x <= xmax and ymin <= gt_y <= ymax
        if is_correct:
            return True

        if gt_nodes:
            bbox_array = enlarge_bbox([x["bounds"] for x in gt_nodes], scale_factor=1.2)
            is_inside, bbox_inside = check_inside(gt_x, gt_y, bbox_array)
            if is_inside:
                ious = intersect_iou(bbox, bbox_inside)
                if np.any(ious > 0.5):
                    return True
        return False

    def _resize_qwen_coordinates(self, coords, width, height, mode="bbox"):
        print(self.model)
        if "ui-venus-navi-7b" in self.model.lower():
            resized_h, resized_w = smart_resize(
                height,
                width,
                factor=28,
                min_pixels=830000,
                max_pixels=937664,
            )
        if "infigui-r1-3b" in self.model.lower():
            resized_h, resized_w = smart_resize(
                height,
                width,
                factor=28,
                min_pixels=3136,
                max_pixels=1568000,
            )
        else:
            resized_h, resized_w = smart_resize(
                height,
                width,
                factor=28,
                min_pixels=28*28,
                max_pixels=28*28*2000,
            )
        print(f"Raw coords {coords}")
        print(f"Original ({width}, {height})")
        print(f"Resized ({resized_w}, {resized_h})")
        scale_x = width / resized_w
        scale_y = height / resized_h
        if mode == "bbox":
            xmin, ymin, xmax, ymax = coords
            resized_coords = (
                round(xmin * scale_x),
                round(ymin * scale_y),
                round(xmax * scale_x),
                round(ymax * scale_y),
            )
            print(f"Resized coords: {resized_coords}")
            return resized_coords
        else:
            x, y = coords
            return (
                round(x * scale_x),
                round(y * scale_y),
            )

    def __call__(self, gt, pred):
        """ eval_single_step """
        subset, episode_id, step_id = gt['subset'], gt['episode_id'], gt['step_id']
        w, h = imagesize.get(gt['image_full_path'])
        print(f"Image Path: {gt['image_full_path']}")
        # get ground truth information
        gt_action_type, gt_action_xy, gt_cand_nodes, \
            gt_action_text, gt_action_button, gt_action_direction = self._parse_answer_(gt)

        if not gt_action_type and self.verbose:
            print(gt['coat_action_desc'])
        gt_action_detail = {
            "click": gt_action_xy, "long": gt_action_xy, "scroll": gt_action_direction, 
            "type": gt_action_text, "press": gt_action_button, "stop": "stop"
        }.get(gt_action_type, None)
        # print(gt['image_full_path'])
        # print(f"GT action type: {gt_action_type}, detail: {gt_action_detail}")

        # print(pred)
        # get predict action information
        pd_action_type, pd_action_coordinates, pd_action_idx, \
            pd_action_direction, pd_action_text, pd_action_button = self._parse_action_(pred, w, h)

        # print(f"PD action type: {pd_action_type}, detail: {pd_action_coordinates if pd_action_coordinates else pd_action_bbox if pd_action_bbox else pd_action_idx if pd_action_idx is not None else pd_action_direction if pd_action_direction else pd_action_text if pd_action_text else pd_action_button if pd_action_button else 'N/A'}")

        hit_format = True if pd_action_type is not None else False
        type_match = (pd_action_type is not None and gt_action_type == pd_action_type)
        # if gt_action_type in ["scroll", "stop"] and type_match==False:
        #     print({"gt":gt_action_type, "pd":pd_action_type})

        exact_match = False
        pd_action_detail = None
        text_dist = None
        if type_match and pd_action_type == "click":
            if self.screen_mode == "tag" and pd_action_idx is not None: # transform idx into bbox
                if 0 <= pd_action_idx < len(gt_cand_nodes):
                    pd_action_coordinates = gt_cand_nodes[pd_action_idx]['bounds']
            
            use_bbox_eval = self.traits.click_mode == "bbox" and pd_action_coordinates is not None
            pd_action_detail = pd_action_coordinates if use_bbox_eval else pd_action_coordinates
            if use_bbox_eval:
                exact_match = self._check_click_bbox(pd_action_coordinates, gt_action_xy, gt_cand_nodes)
            else:
                exact_match = self._check_click_((w, h), pd_action_coordinates, gt_action_xy, gt_cand_nodes)

        if pd_action_type == "scroll":
            pd_action_detail = pd_action_direction
            if type_match:
                exact_match = (pd_action_direction == gt_action_direction)

        if type_match and pd_action_type == "type":
            pd_action_detail = pd_action_text
            text_dist = Levenshtein.ratio(pd_action_text, gt_action_text)
            exact_match = (pd_action_text in gt_action_text or \
                           gt_action_text in pd_action_text or \
                           text_dist > 0.8)

        if type_match and pd_action_type == "press":
            pd_action_detail = pd_action_button
            exact_match = (pd_action_button == gt_action_button)

        if type_match and pd_action_type == "stop":
            pd_action_detail = "stop"
            exact_match = True
        # if pd_action_type:
        #     print(
        #         {
        #             "image": gt["image_full_path"],
        #             "answer": {"action_type": gt_action_type, "action_detail": gt_action_detail},
        #             "pred": {"action_type": pd_action_type, "action_detail": pd_action_detail},
        #         }
        #     )

        # if gt_action_type in ["scroll"] and type_match==True and exact_match==False:
        #     print("scroll exact false", {"gt":gt_action_detail, "pd":pd_action_detail})

        return {
            "subset": subset, "episode_id": episode_id, "step_id": step_id,
            "answer": {"action_type": gt_action_type, "action_detail": gt_action_detail},
            "pred": {"action_type": pd_action_type, "action_detail": pd_action_detail},
            "type_match": type_match, "exact_match": exact_match, 
            "text_dist": text_dist, "format_hit": hit_format
        }

    def compute_episode_metrics(self, episode_results):
# Compute metrics for an episode
        success, progress = [], []
        for __, eplist in episode_results.items():
            ep_success, ep_progress = True, 0
            for ex in eplist:
                if ex['exact_match'] is True: ep_progress += 1
                else: ep_success = False
                if not ep_success: break
            success.append(ep_success)
            progress.append(ep_progress/len(eplist)*1.0)
        
        return {"success_rate": round(sum(success) / len(success), 4),
                "goal_progress": round(sum(progress) / len(progress), 4)}

    def compute_atomic_metrics(self, step_results):
# Compute atomic metrics for each step
        recorder = {
            'total':  {'count':0, 'type_match':0, 'exact_match':0, "hit": 0}, 
            # -------------------------------------------
            'CLICK':  {'count':0, 'type_match':0, 'exact_match':0},  
            'LONG':  {'count':0, 'type_match':0, 'exact_match':0},  
            'TYPE':   {'count':0, 'type_match':0, 'exact_match':0, 'text_dist': []},
            'SCROLL': {'count':0, 'type_match':0, 'exact_match':0},  
            'PRESS':  {'count':0, 'type_match':0, 'exact_match':0},  
            'STOP':   {'count':0, 'type_match':0, 'exact_match':0}, 
        }
        atomic_metrics_df = []
        for step in tqdm(step_results):
            
            recorder['total']['count'] += 1
            recorder['total']['hit'] += step['format_hit']
            if self.verbose:
                print(step)
                print(step['answer'])
            if step['answer']['action_type'] is None:
                continue
            action_type = step['answer']['action_type'].upper()
            recorder[action_type]['count'] += 1
            recorder[action_type]['type_match'] += step['type_match']
            recorder['total']['type_match'] += step['type_match']
            recorder[action_type]['exact_match'] += step['exact_match']
            recorder['total']['exact_match'] += step['exact_match']
            if 'text_dist' in recorder[action_type] and step['text_dist'] is not None:
                recorder[action_type]['text_dist'].append(step['text_dist'])

        scores = {metric_key:{} for metric_key in ['total', 'CLICK', 'SCROLL', 'PRESS', 'STOP', 'TYPE']}
        scores['total']['hit_rate'] = round(recorder['total']['hit']/recorder['total']['count'], 4)
        for metric_key in ['total', 'CLICK', 'SCROLL', 'PRESS', 'STOP', "TYPE"]:
            if recorder[metric_key]['count']==0:
                scores[metric_key]['type_acc'] = 0
                scores[metric_key]['exact_acc'] = 0
                continue
            scores[metric_key]['type_acc'] = round(recorder[metric_key]['type_match']/recorder[metric_key]['count'], 4)
            scores[metric_key]['exact_acc'] = round(recorder[metric_key]['exact_match']/recorder[metric_key]['count'], 4)
        if recorder['TYPE']['text_dist']:
            scores['TYPE']['text_dist'] = round(sum(recorder['TYPE']['text_dist'])/len(recorder['TYPE']['text_dist']), 4)
        return scores


_STEP_PATTERN = re.compile(r"_(?P<step>\d+)(?:_|$)")
_MODEL_HINTS: Tuple[Tuple[str, str], ...] = (
    ("infigui", "infigui-r1"),
    ("ui-venus", "ui-venus"),
    ("ui_venus", "ui-venus"),
    ("guir1", "guir1"),
    ("atlas", "atlas"),
    ("gemini", "gemini"),
    ("qwen2.5", "qwen2.5"),
    ("qwen2-5", "qwen2.5"),
    ("qwen25", "qwen2.5"),
    ("qwenvl", "qwenvl"),
    ("qwen", "qwen"),
)


def infer_episode_and_step(path: str) -> Tuple[Optional[str], Optional[int]]:
    """Infer episode id and step id from a prediction file path."""
    file_path = Path(path)
    stem = file_path.stem
    if stem.endswith("_viz"):
        stem = stem[:-4]
    episode = None
    remainder = stem
    if "-" in stem:
        episode, remainder = stem.split("-", 1)
    if not episode:
        parent = file_path.parent.name
        episode = parent or None
    step: Optional[int] = None
    match = _STEP_PATTERN.search(remainder)
    if match:
        try:
            step = int(match.group("step"))
        except ValueError:
            step = None
    return episode, step


def infer_model_from_path(path: str) -> Optional[str]:
    """Guess the model name from a prediction path."""
    lower = path.lower()
    for token, label in _MODEL_HINTS:
        if token in lower:
            return label
    return None
