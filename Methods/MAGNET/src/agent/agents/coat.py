from __future__ import annotations

import base64
import json
import os
import re
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import imagesize
import numpy as np
import requests
from http import HTTPStatus
from PIL import Image, ImageDraw

from memory.core.paths import storage_path

from utils.parsing import extract_json_block
from utils.screen import row_col_sort, draw_bbox
from utils.image_ops import (
    template_match_center,
    nearest_bboxes,
    enclosing_bbox,
    draw_hint_bbox,
)
from utils.omni_parser import parse_ui_bboxes
from .base import BaseAgent

SRC_ROOT = Path(__file__).resolve().parents[2]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

class ScreenAgent(BaseAgent):

    def __init__(self, config) -> None:
        super().__init__(config)

        self.use_screen_tag = bool(self.cfg.DATA.USE_SCREEN_TAG)
        if not self.use_screen_tag:
            raise ValueError("SCREEN_TAG prompting must stay enabled; text-only screens are unsupported.")
        self.screen_mode = "tag"

    def _strip_code_fences(self, text: str | None) -> str:
        if not text:
            return ""
        cleaned = re.sub(r"```json", "", text, flags=re.IGNORECASE)
        cleaned = cleaned.replace("```", "")
        return cleaned.strip()

    def _try_load_json(self, payload: str) -> dict | None:
        if not payload:
            return None
        try:
            return json.loads(payload)
        except json.JSONDecodeError:
            return None

    def _parse_action_response(self, asst_head: str | None, asst_res: str) -> dict | None:
        cleaned_res = self._strip_code_fences(asst_res)
        candidates = [cleaned_res, "{"+cleaned_res, cleaned_res+"}", cleaned_res[:-1], "{"+cleaned_res[:-1]]
        if asst_head:
            candidates.append(self._strip_code_fences(asst_head) + cleaned_res)
            candidates.append(self._strip_code_fences(asst_head) + cleaned_res + "}")
        for candidate in candidates:
            # print(candidate)
            parsed = self._try_load_json(candidate)
            if parsed is not None:
                return parsed
        for candidate in candidates:
            # print(candidate)
            parsed = extract_json_block(candidate)
            if parsed is not None:
                return parsed
        return None

    def add_screen_tag(self, step_data, save_dir, update=False):
        """ Set of Mark Prompting """
        src_image_path = step_data['image_full_path']
        base_name, ext = os.path.splitext(src_image_path)
        dst_image_path = base_name + "_tag" + ext
        # image_name = os.path.basename(src_image_path)
        # dst_image_path = os.path.join(save_dir, image_name)

        if not os.path.exists(dst_image_path) or update:
            ui_bboxes = []
            for ui_node in step_data['ui_elements']:
                ui_bboxes.append(ui_node['bounds'])
            
            tag_image = draw_bbox(src_image_path, bboxes=ui_bboxes, 
                                texts=list(map(str, range(len(step_data['ui_positions'])))),
                                rgba_color=(255, 0, 0, 180), thickness=5)
            tag_image = tag_image.convert('RGB')
            tag_image.save(dst_image_path)
        return dst_image_path

    # make action with memory
    def make_action(self, step_data, usr_prompt, save_dir, asst_prompt=None):
        # Determine action mode and image path
        action_mode = self.cfg.DEMO_MODE.upper()
        image_path = step_data['image_full_path']

        # Prepare UI elements
        ui_elements = []
        ui_positions = step_data.get('ui_positions') or []
        ui_texts = step_data.get('ui_text') or []
        ui_types = step_data.get('ui_types')
        if ui_types is None:
            ui_types = ["unknown"] * len(ui_positions)
        if ui_positions and ui_texts and ui_types:
            for org_bbox, txt, ui_class in zip(ui_positions, ui_texts, ui_types):
                ymin, xmin, h, w  = org_bbox
                bbox = [xmin, ymin, xmin+w, ymin+h]
                ui_elements.append({"bounds": bbox, "text": txt, "type": ui_class})
            ui_elements = row_col_sort(ui_elements)
        step_data['ui_elements'] = ui_elements

        # Handle screen tag mode
        if self.use_screen_tag:
            sys_prompt = self.prompts['ACTION_PREDICT'][action_mode]['SYSTEM']['SCREEN_TAG']
            action_space = self.prompts['ACTION_PREDICT']['ACTION_SPACE']['SCREEN_TAG']
            if self.cfg.MODEL.NAME not in ["qwenvl32_cot"]:
                action_space_list = action_space.split("\n\n")
                pattern = r'"NEXT"\s*:\s*\{\s*"type"\s*:\s*"([^"]+)"'
                asst_act_type = re.search(pattern, asst_prompt).group(1)
                for action_space in action_space_list:
                    act_type = action_space.split(':\n')[0]
                    if act_type == asst_act_type:
                        # print(act_type, act_type in "".join(usr_prompt))
                        # print(action_space)
                        break
            sys_prompt = sys_prompt.replace("{action_space}", action_space)
            tag_image_path = self.add_screen_tag(step_data, save_dir=save_dir)
            
            if self.cfg.MODEL.NAME == "openai":
                image_path = self.compress_image(image_path, max_height=1440, mode="jpg")
                tag_image_path = self.compress_image(tag_image_path, max_height=1440, mode="scale")

            prompt = [
                ("system", sys_prompt),
                ("user", {"text": usr_prompt[0], "img_index": 0}),
            ]
            for txt in usr_prompt[1:]: prompt.append(("user", {"text": txt}))
            if asst_prompt:
                prompt.append(("assistant", {"text": asst_prompt}))
                prompt.append(("user", {"text": 'Complete the above response: '}))
            episode_id, step_id = step_data['episode_id'], step_data['step_id']
            prompt_save_path = f"/root/autodl-tmp/agent/agent/results/odyssey/prompts/val_wo/{episode_id}_{step_id}.json"
            # for item in prompt:
            #     if "text" not in item[1]:
            #         print(item[1])
            #     else:
            #         print(item[1]['text'])
            self._log_prompt(prompt)
            asst_res, state = self.vlm.get_response([image_path], prompt)
            self._log_response(asst_res)

        # Return assistant response
        asst_head = asst_prompt if asst_prompt else ""         
        return asst_head, asst_res, state
    
    def make_action_with_memory(
        self,
        step_data,
        usr_prompt,
        save_dir,
        asst_prompt=None,
        ui_path=None,
        hint_image_path=None,
    ):
        # Determine action mode and image path
        action_mode = self.cfg.DEMO_MODE.upper()
        image_path = hint_image_path or step_data['image_full_path']
        step_id = step_data['step_id']
        episode_id = step_data['episode_id']
        # Prepare UI elements
        ui_elements = []
        ui_positions = step_data.get('ui_positions') or []
        ui_texts = step_data.get('ui_text') or []
        ui_types = step_data.get('ui_types')
        if ui_types is None:
            ui_types = ["unknown"] * len(ui_positions)
        if ui_positions and ui_texts and ui_types:
            for org_bbox, txt, ui_class in zip(ui_positions, ui_texts, ui_types):
                ymin, xmin, h, w  = org_bbox
                bbox = [xmin, ymin, xmin+w, ymin+h]
                ui_elements.append({"bounds": bbox, "text": txt, "type": ui_class})
            ui_elements = row_col_sort(ui_elements)
        step_data['ui_elements'] = ui_elements

        # Handle screen tag mode
        if self.use_screen_tag:
            if "bbox" in self.cfg.METRICS_DIR and "wotag" in self.cfg.OUTPUT_DIR:
                sys_prompt = self.prompts['ACTION_PREDICT'][action_mode]['SYSTEM']['SCREEN_TAG_WITH_MEMORY']
                action_space = self.prompts['ACTION_PREDICT']['ACTION_SPACE']['SCREEN_TAG']
                action_space_list = action_space.split("\n\n")
                pattern = r'"NEXT"\s*:\s*\{\s*"type"\s*:\s*"([^"]+)"'
                asst_act_type = re.search(pattern, asst_prompt).group(1)
                for action_space in action_space_list:
                    act_type = action_space.split(':\n')[0]
                    if act_type == asst_act_type:
                        # print(act_type, act_type in "".join(usr_prompt))
                        # print(action_space)
                        break
                sys_prompt = sys_prompt.replace("{action_space}", action_space)
                tag_image_path = self.add_screen_tag(step_data, save_dir=save_dir)
                
                if self.cfg.MODEL.NAME == "openai":
                    image_path = self.compress_image(image_path, max_height=1440, mode="jpg")
                    tag_image_path = self.compress_image(tag_image_path, max_height=1440, mode="scale")
                if ui_path:
                    prompt = [
                        ("system", sys_prompt),
                        ("user", {"text": usr_prompt[0], "img_index": 0}),
                        # ("user", {"text": "", "img_index": 1}),
                        ("user", {"text": usr_prompt[1], "img_index": 1}),
                        ("user", {"text": usr_prompt[2]})
                    ]
                    if asst_prompt:
                        prompt.append(("assistant", {"text": asst_prompt}))
                        prompt.append(("user", {"text": 'Complete the above response: '}))
                    episode_id, step_id = step_data['episode_id'], step_data['step_id']
                    # prompt_save_path = f"/root/autodl-tmp/agent/agent/results/odyssey/prompts/test_w_bbox/{episode_id}_{step_id}.json"
                    # print(f"Saving prompt to {prompt_save_path} ...")
                    # with open(prompt_save_path, "w", encoding="utf-8") as f:
                    #     json.dump({"prompt":prompt, "images":[image_path, ui_path]}, f, indent=4, ensure_ascii=False)
                    
                    self._log_prompt(prompt)
                    asst_res, state = self.vlm.get_response([image_path, ui_path], prompt)
                    self._log_response(asst_res)
                else:
                    prompt = [
                        ("system", sys_prompt),
                        ("user", {"text": usr_prompt[0], "img_index": 0}),
                        # ("user", {"text": "", "img_index": 1})
                    ]
                    for txt in usr_prompt[1:]: prompt.append(("user", {"text": txt}))
                    if asst_prompt:
                        prompt.append(("assistant", {"text": asst_prompt}))
                        prompt.append(("user", {"text": 'Complete the above response: '}))
                        episode_id, step_id = step_data['episode_id'], step_data['step_id']
                        prompt_save_path = f"/root/autodl-tmp/agent/agent/results/odyssey/prompts/val_w/{episode_id}_{step_id}.json"
                        # print(f"Saving prompt to {prompt_save_path} ...")
                        # with open(prompt_save_path, "w", encoding="utf-8") as f:
                            # json.dump({"prompt":prompt, "images":[image_path, tag_image_path]}, f, indent=4, ensure_ascii=False)
                        
                        self._log_prompt(prompt)
                        asst_res, state = self.vlm.get_response([image_path], prompt)
                        self._log_response(asst_res)
            else:
                sys_prompt = self.prompts['ACTION_PREDICT'][action_mode]['SYSTEM']['SCREEN_TAG_WITH_MEMORY']
                action_space = self.prompts['ACTION_PREDICT']['ACTION_SPACE']['SCREEN_TAG']
                action_space_list = action_space.split("\n\n")
                pattern = r'"NEXT"\s*:\s*\{\s*"type"\s*:\s*"([^"]+)"'
                asst_act_type = re.search(pattern, asst_prompt).group(1)
                for action_space in action_space_list:
                    act_type = action_space.split(':\n')[0]
                    if act_type == asst_act_type:
                        # print(act_type, act_type in "".join(usr_prompt))
                        # print(action_space)
                        break
                sys_prompt = sys_prompt.replace("{action_space}", action_space)
                tag_image_path = self.add_screen_tag(step_data, save_dir=save_dir)
                
                if self.cfg.MODEL.NAME == "openai":
                    image_path = self.compress_image(image_path, max_height=1440, mode="jpg")
                    tag_image_path = self.compress_image(tag_image_path, max_height=1440, mode="scale")
                if ui_path:
                    prompt = [
                        ("system", sys_prompt),
                        ("user", {"text": usr_prompt[0], "img_index": 0}),
                        ("user", {"text": "", "img_index": 1}),
                        ("user", {"text": usr_prompt[1], "img_index": 2}),
                        ("user", {"text": usr_prompt[2]})
                    ]
                    if asst_prompt:
                        prompt.append(("assistant", {"text": asst_prompt}))
                        prompt.append(("user", {"text": 'Complete the above response: '}))
                    episode_id, step_id = step_data['episode_id'], step_data['step_id']
                    prompt_save_path = Path("/root/autodl-tmp/agent/agent/results/odyssey/prompts/test_w_bbox") / f"{episode_id}_{step_id}.json"
                    prompt_save_path.parent.mkdir(parents=True, exist_ok=True)
                    print(f"Saving prompt to {prompt_save_path} ...")
                    with open(prompt_save_path, "w", encoding="utf-8") as f:
                        json.dump({"prompt":prompt, "images":[image_path, tag_image_path, ui_path]}, f, indent=4, ensure_ascii=False)
                    
                    self._log_prompt(prompt)
                    asst_res, state = self.vlm.get_response([image_path, tag_image_path, ui_path], prompt)
                    self._log_response(asst_res)
                else:
                    prompt = [
                        ("system", sys_prompt),
                        ("user", {"text": usr_prompt[0], "img_index": 0}),
                        ("user", {"text": "", "img_index": 1})
                    ]
                    for txt in usr_prompt[1:]: prompt.append(("user", {"text": txt}))
                    if asst_prompt:
                        prompt.append(("assistant", {"text": asst_prompt}))
                        prompt.append(("user", {"text": 'Complete the above response: '}))
                        episode_id, step_id = step_data['episode_id'], step_data['step_id']
                        prompt_save_path = f"/root/autodl-tmp/agent/agent/results/odyssey/prompts/val_w/{episode_id}_{step_id}.json"
                        # print(f"Saving prompt to {prompt_save_path} ...")
                        # with open(prompt_save_path, "w", encoding="utf-8") as f:
                            # json.dump({"prompt":prompt, "images":[image_path, tag_image_path]}, f, indent=4, ensure_ascii=False)
                        
                        self._log_prompt(prompt)
                        asst_res, state = self.vlm.get_response([image_path, tag_image_path], prompt)
                        self._log_response(asst_res)
            prompts_dir = "/".join(save_dir.split("/")[:-1]) + "/make_action_with_memory_prompts"
            os.makedirs(prompts_dir, exist_ok=True)
            with open(f"{prompts_dir}/{episode_id}_{step_id}.json","w") as f:
                json.dump(prompt, f)
            print("save to", f"{prompts_dir}/{episode_id}_{step_id}.json")
        
        # Return assistant response
        asst_head = asst_prompt if asst_prompt else ""         
        return asst_head, asst_res, state, ui_path

    def build_stationary_hint(self, step_data, patch_path, save_dir):
        if not patch_path or not getattr(self.cfg.MEMORY, "HINT_ENABLE", False):
            return None
        image_path = step_data['image_full_path']
        center, score = template_match_center(image_path, patch_path)
        tm_threshold = getattr(self.cfg.MEMORY, "TM_THRESHOLD", 0.3)
        if center is None or score < tm_threshold:
            print(f"[hint] template match skipped (score={score:.3f}, threshold={tm_threshold})")
            return None

        ui_positions = step_data.get('ui_positions') or []
        if not ui_positions:
            ui_positions = parse_ui_bboxes(
                image_path,
                getattr(self.cfg.MEMORY, "OMNIPARSER_CONFIG", None),
            )
            if ui_positions:
                step_data['ui_positions'] = ui_positions
        if not ui_positions:
            print("[hint] no ui_positions available")
            return None

        k = getattr(self.cfg.MEMORY, "HINT_K", 5)
        nearest = nearest_bboxes(center, ui_positions, k=k)
        bbox = enclosing_bbox(nearest)
        if not bbox:
            print("[hint] failed to build enclosing bbox")
            return None

        hints_dir = Path(save_dir) / "hint_images"
        hints_dir.mkdir(parents=True, exist_ok=True)
        hint_path = hints_dir / f"{step_data['subset']}-{step_data['episode_id']}_{step_data['step_id']}_hint.png"
        draw_hint_bbox(image_path, bbox, str(hint_path))
        print(f"[hint] saved: {hint_path}")
        return str(hint_path)

    def coa_action(self, step_data, *args, **kwargs):
        """ Chain of Action """
        user_request, image_path = step_data['instruction'], step_data['image_full_path']
        history_actions = step_data['history_actions']

        if history_actions: history_actions = ", ".join(history_actions)
        else: history_actions = "None"
        history_actions = history_actions + "."

        usr_prompt = self.prompts['ACTION_PREDICT']['COA']['USER']
        usr_prompt = usr_prompt.replace("{history_actions}", history_actions)
        usr_prompt = usr_prompt.replace("{user_request}", user_request)
        usr_prompt = [x for x in usr_prompt.split("{screenshot}")]

        return self.make_action(step_data, usr_prompt, kwargs['save_dir'])
 
    # def cot_action(self, step_data, *args, **kwargs):
    #     """ Chain of Thought """
    #     user_request, image_path = step_data['instruction'], step_data['image_full_path']
    #     acton_think = kwargs['action_think']
        
    #     usr_prompt = self.prompts['ACTION_PREDICT']['COT']['USER']
    #     usr_prompt = usr_prompt.replace("{user_request}", user_request)
    #     usr_prompt = [x for x in usr_prompt.split("{screenshot}")]

    #     if self.cfg.MODEL.NAME == "qwenvl":
    #         asst_prompt = json.dumps({"THINK": acton_think}, indent=4)
    #     else:
    #         asst_prompt = self.prompts['ACTION_PREDICT']['COT']['ASST']
    #         asst_prompt = asst_prompt.replace("{action_thought}", json.dumps(acton_think))

    #     return self.make_action(step_data, usr_prompt, kwargs['save_dir'], asst_prompt=asst_prompt)

    def cot_action(self, step_data, *args, **kwargs):
        """ Chain of Thought """
        user_request, image_path = step_data['instruction'], step_data['image_full_path']
        
        usr_prompt = self.prompts['ACTION_PREDICT']['COT']['USER']
        usr_prompt = usr_prompt.replace("{user_request}", user_request)
        usr_prompt = [x for x in usr_prompt.split("{screenshot}")]

        asst_prompt = self.prompts['ACTION_PREDICT']['COT']['ASST']

        return self.make_action(step_data, usr_prompt, kwargs['save_dir'], asst_prompt=asst_prompt)


    def coat_action(self, step_data, *args, **kwargs):
        """ Chain of Action Thought """
        user_request, image_path = step_data['instruction'], step_data['image_full_path']
        history_actions = step_data['history_coat_actions']
        
        screen_desc = kwargs.get('screen_desc', '')
        prev_action_result = kwargs.get('prev_action_result', None)
        acton_think = kwargs.get('action_think', {})
        next_action_desc = kwargs.get('action_desc', {})

        usr_prompt = self.prompts['ACTION_PREDICT']['COAT']['USER']
        usr_prompt = usr_prompt.split("\n")
        if not history_actions: usr_prompt = [x for x in usr_prompt if 'history_actions' not in x]
        if not prev_action_result: usr_prompt = [x for x in usr_prompt if 'prev_action_result' not in x]
        usr_prompt = "\n".join(usr_prompt)

        usr_prompt = usr_prompt.replace("{screen_desc}", screen_desc)
        if history_actions: history_actions = ", ".join(history_actions)
        else: history_actions = "None"
        history_actions = history_actions + "."
        usr_prompt = usr_prompt.replace("{history_actions}", history_actions)
        if prev_action_result:
            usr_prompt = usr_prompt.replace("{prev_action_result}", prev_action_result)
        usr_prompt = usr_prompt.replace("{user_request}", user_request)
        usr_prompt = [x for x in usr_prompt.split("{screenshot}")]

        if self.cfg.MODEL.NAME == "qwenvl":
            asst_prompt = json.dumps({"THINK": acton_think, "NEXT": next_action_desc,}, indent=4)
        else:
            asst_prompt = self.prompts['ACTION_PREDICT']['COAT']['ASST']
            asst_prompt = asst_prompt.replace("{action_thought}", json.dumps(acton_think))
            asst_prompt = asst_prompt.replace("{next_single_action}", json.dumps(next_action_desc))

        return self.make_action(step_data, usr_prompt, kwargs['save_dir'], asst_prompt=asst_prompt)

    # def coat_action_with_memory(self, step_data, *args, **kwargs):
    #     """ Chain of Action Thought """
    #     user_request, image_path = step_data['instruction'], step_data['image_full_path']
    #     history_actions = step_data['history_coat_actions']
        
    #     screen_desc = kwargs['screen_desc']
    #     prev_action_result = kwargs.get('prev_action_result', None)
    #     acton_think, next_action_desc = kwargs['action_think'], kwargs['action_desc']

    #     action_type, query = next_action_desc["type"], next_action_desc["description"]

    #     if action_type in ["CLICK_ELEMENT"] and self.stationary_memory:
    #         doc = self.stationary_memory.rag.query(query, param=self.stationary_query_param)
    #     else:
    #         doc = None
            
    #     if doc is None:
    #         ui_path = None
    #     else:
    #         ui_path = doc[0]['image']

    #     # save doc
    #     # query_path = f"{save_dir}/{subset}-{episode_id}_{step_id}_2query.json"
    #     # with open(query_path, "w", encoding="utf-8") as f:
    #     #     json.dump(
    #     #         {
    #     #             "query":query, 
    #     #             "result":doc[0] if doc is not None else None, 
    #     #         }, f)
    #     kwargs['relevant_ui_path'] = ui_path
    #     print("ui_path", ui_path)
    #     if ui_path:
    #         usr_prompt = self.prompts['ACTION_PREDICT']['COAT']['USER_WITH_MEMORY']
    #     else:
    #         usr_prompt = self.prompts['ACTION_PREDICT']['COAT']['USER']
    #     usr_prompt = usr_prompt.split("\n")
    #     if not history_actions: usr_prompt = [x for x in usr_prompt if 'history_actions' not in x]
    #     if not prev_action_result: usr_prompt = [x for x in usr_prompt if 'prev_action_result' not in x]
    #     usr_prompt = "\n".join(usr_prompt)

    #     usr_prompt = usr_prompt.replace("{screen_desc}", screen_desc)
    #     if history_actions: history_actions = ", ".join(history_actions)
    #     else: history_actions = "None"
    #     history_actions = history_actions + "."
    #     usr_prompt = usr_prompt.replace("{history_actions}", history_actions)
    #     if prev_action_result:
    #         usr_prompt = usr_prompt.replace("{prev_action_result}", prev_action_result)
    #     usr_prompt = usr_prompt.replace("{user_request}", user_request)
    #     usr_prompt = [x for x in usr_prompt.split("{screenshot}")]

    #     if self.cfg.MODEL.NAME == "qwenvl":
    #         asst_prompt = json.dumps({"THINK": acton_think, "NEXT": next_action_desc,}, indent=4)
    #     else:
    #         if ui_path:
                
    #             # asst_prompt = self.prompts['ACTION_PREDICT']['COAT']['ASST_WITH_MEMORY']
    #             usr_prompt[-1] = usr_prompt[-1].replace("{action_thought}", json.dumps(acton_think))
    #             usr_prompt[-1] = usr_prompt[-1].replace("{next_single_action}", json.dumps(next_action_desc))
    #             usr_prompt = usr_prompt[:-1] + [x for x in usr_prompt[-1].split("{revelant_ui}")]
    #             asst_prompt = None
    #         else:
    #             asst_prompt = self.prompts['ACTION_PREDICT']['COAT']['ASST']
    #             asst_prompt = asst_prompt.replace("{action_thought}", json.dumps(acton_think))
    #             asst_prompt = asst_prompt.replace("{next_single_action}", json.dumps(next_action_desc))

    #     return self.make_action_with_memory(step_data, usr_prompt, kwargs['save_dir'], asst_prompt=asst_prompt, ui_path=ui_path)

    def coat_action_with_memory(self, step_data, *args, **kwargs):
        """ Chain of Action Thought """
        user_request, image_path = step_data['instruction'], step_data['image_full_path']
        history_actions = step_data['history_coat_actions']
        
        screen_desc = kwargs.get('screen_desc', '')
        prev_action_result = kwargs.get('prev_action_result', None)
        acton_think = kwargs.get('action_think', {})
        next_action_desc = kwargs.get('action_desc', {})

        action_type, query = next_action_desc.get("type", ""), next_action_desc.get("description", "")

        if action_type in ["CLICK_ELEMENT"] and self.stationary_memory:
            doc = self.stationary_memory.rag.query(query, param=self.stationary_query_param)
        else:
            doc = None
            
        if doc is None:
            ui_path = None
        else:
            print("Stationary memory retrived!")
            ui_path = doc[0]['image']
            if "/home/slb/data" in ui_path:
                ui_path = ui_path.replace("/home/slb/data", "/root/autodl-tmp/agent/agent/data")
            elif "/mnt/shanghai1/sunlibo/gui-odyssey" in ui_path:
                ui_path = ui_path.replace("/mnt/shanghai1/sunlibo/gui-odyssey", "/root/autodl-tmp/agent/agent/data/gui_odyssey")
            elif "/root/autodl-tmp/data/gui_memory" in ui_path:
                ui_path = ui_path.replace("/root/autodl-tmp/data/gui_memory", "/remote-home/lbsun/data")
            print("ui_path:", ui_path)
            assert os.path.exists(ui_path)

        hint_image_path = None
        if ui_path:
            hint_image_path = self.build_stationary_hint(step_data, ui_path, kwargs.get('save_dir', ''))
            if hint_image_path:
                ui_path = None

        # save doc
        # query_path = f"{save_dir}/{subset}-{episode_id}_{step_id}_2query.json"
        # with open(query_path, "w", encoding="utf-8") as f:
        #     json.dump(
        #         {
        #             "query":query, 
        #             "result":doc[0] if doc is not None else None, 
        #         }, f)
        kwargs['relevant_ui_path'] = ui_path
        if hint_image_path:
            kwargs['hint_image_path'] = hint_image_path
        print("ui_path", ui_path)
        if ui_path:
            # ui_path = ui_path.replace("/home/slb/data", "/mnt/shanghai1/slb/data")
            # assert os.path.exists(ui_path)
            usr_prompt = self.prompts['ACTION_PREDICT']['COAT']['USER_WITH_MEMORY']
        else:
            usr_prompt = self.prompts['ACTION_PREDICT']['COAT']['USER']
        usr_prompt = usr_prompt.split("\n")
        if not history_actions: usr_prompt = [x for x in usr_prompt if 'history_actions' not in x]
        if not prev_action_result: usr_prompt = [x for x in usr_prompt if 'prev_action_result' not in x]
        usr_prompt = "\n".join(usr_prompt)

        usr_prompt = usr_prompt.replace("{screen_desc}", screen_desc)
        if history_actions: history_actions = ", ".join(history_actions)
        else: history_actions = "None"
        history_actions = history_actions + "."
        usr_prompt = usr_prompt.replace("{history_actions}", history_actions)
        if prev_action_result:
            usr_prompt = usr_prompt.replace("{prev_action_result}", prev_action_result)
        usr_prompt = usr_prompt.replace("{user_request}", user_request)
        usr_prompt = [x for x in usr_prompt.split("{screenshot}")]

        if self.cfg.MODEL.NAME == "qwenvl":
            asst_prompt = json.dumps({"THINK": acton_think, "NEXT": next_action_desc,}, indent=4)
        else:
            if ui_path:
                usr_prompt = usr_prompt[:-1] + [x for x in usr_prompt[-1].split("{revelant_ui}")]
            asst_prompt = self.prompts['ACTION_PREDICT']['COAT']['ASST']
            asst_prompt = asst_prompt.replace("{action_thought}", json.dumps(acton_think))
            asst_prompt = asst_prompt.replace("{next_single_action}", json.dumps(next_action_desc))

        if ui_path or hint_image_path:
            return self.make_action_with_memory(
                step_data,
                usr_prompt,
                kwargs['save_dir'],
                asst_prompt=asst_prompt,
                ui_path=ui_path,
                hint_image_path=hint_image_path,
            )
        asst_head, asst_res, state = self.make_action(step_data, usr_prompt, kwargs['save_dir'], asst_prompt=asst_prompt)
        return asst_head, asst_res, state, ui_path



    def predict(self, step_data, save_dir, max_trials=5): 
        subset, episode_id = step_data['subset'], step_data['episode_id']
        prev_step_id, step_id = step_data['prev_step_id'], step_data['step_id']
        action_predict_error_dir = os.path.join(save_dir, "action_predict_errors")
        os.makedirs(action_predict_error_dir, exist_ok=True)
        if subset==episode_id:
            save_dir = os.path.join(save_dir, episode_id)
        else:
            save_dir = os.path.join(save_dir, f"{subset}-{episode_id}")
        # 创建save_dir
        os.makedirs(save_dir, exist_ok=True)
        cur_save_path = os.path.join(save_dir, f"{subset}-{episode_id}_{step_id}.json")
        try:
            cur_anno = json.load(open(cur_save_path, "r"))
        except Exception as e:
            cur_anno = {}
                        
        action_mode = self.cfg.DEMO_MODE.upper()
        print(f"[Predict] Action Mode: {action_mode}")
        requires_plan = (action_mode == "COAT")
        if prev_step_id and requires_plan:
            prev_save_path = os.path.join(save_dir, f"{subset}-{episode_id}_{prev_step_id}.json")
            if os.path.exists(prev_save_path):
                prev_anno = json.load(open(prev_save_path, "r"))
                cur_anno['prev_action_result'] = prev_anno.get('action_result')
        if requires_plan and ("action_think" not in cur_anno or "action_desc" not in cur_anno):
            raise ValueError("Missing action_think/action_desc in flow outputs; rerun flow before predict. {}".format(cur_save_path))

        if action_mode == "COA":    func_handler = self.coa_action
        elif action_mode == "COT":  func_handler = self.cot_action
        elif action_mode == "COAT": func_handler = self.coat_action
        else: raise NotImplementedError

        if 'action_predict' not in cur_anno: 
            cur_anno['action_predict'] = {}
            # print(save_dir)
        if action_mode not in cur_anno['action_predict']: cur_anno['action_predict'][action_mode] = {}
        # if action_mode in cur_anno['action_predict']: 
        #     anno = cur_anno['action_predict'][action_mode]
        #     cur_anno['action_predict'][action_mode] = {}
        #     cur_anno['action_predict'][action_mode][self.screen_mode] = anno
        #     json.dump(cur_anno, open(cur_save_path, "w", encoding="utf-8"), indent=4)
        
        needs_prediction = (
            self.screen_mode not in cur_anno['action_predict'][action_mode]
            or cur_anno['action_predict'][action_mode][self.screen_mode].get('ACTION', "") == ""
        )
        print(cur_save_path)
        print(self.screen_mode)
        print("cur_anno['action_predict'][action_mode]:", cur_anno['action_predict'][action_mode])
        print("needs_prediction:", needs_prediction)
        # exit()
        if needs_prediction:
        # if True:
            print(f"[Mode {action_mode}][{self.screen_mode}] Gnerating action ... ({cur_save_path})")
            try_num = 0
            while try_num < max_trials:
                asst_head, asst_res, state = func_handler(step_data, **cur_anno, save_dir=save_dir)
                if state == HTTPStatus.OK: 
                    try:
                        response = self._parse_action_response(asst_head, asst_res)
                        if response is None:
                            raise json.decoder.JSONDecodeError(
                                "Unable to parse assistant response",
                                self._strip_code_fences(asst_res),
                                0,
                            )
                        resp_action = response.get("ACTION")
                        if resp_action is None:
                            print("[predict] Missing ACTION field, retrying...", response)
                            try_num += 1
                            continue
                        if isinstance(resp_action, str):
                            response["ACTION"] = resp_action.lower()
                        elif isinstance(resp_action, dict):
                            response["ARGS"] = resp_action.get("ARGS", {})
                            response["ACTION"] = resp_action.get("ACTION", "").lower()
                        if 'action_desc' in cur_anno:
                            print("Predicted Action Plan:", cur_anno['action_desc'])
                        print("Predicted Action:", response)

                        if "ACTION" not in response:
                            print("[predict] Missing ACTION field, retrying...", response)
                            continue
                        resp_args = response.get("ARGS", {})
                        if not isinstance(resp_args, dict):
                            resp_args = {}
                        action_name = response.get("ACTION", "")
                        if isinstance(action_name, dict):
                            resp_args = action_name.get("ARGS", resp_args)
                            action_name = action_name.get("ACTION", "")
                        action_name = action_name.lower() if isinstance(action_name, str) else action_name
                        if 'bbox' in resp_args:
                            resp_args['bbox'] = self._preserve_model_bbox(resp_args['bbox'])

                        expected_type = cur_anno['action_desc']['type'].lower() if cur_anno.get('action_desc') else ""
                        action_name = self._enforce_action_match(
                            cur_anno,
                            action_name,
                            response,
                            action_predict_error_dir,
                            subset,
                            episode_id,
                            step_id,
                        )
                        action_entry = {
                            "ACTION": action_name,
                            "ARGS": resp_args
                        }
                        if "THINK" in response:
                            action_entry["THINK"] = response["THINK"]
                        if "NEXT" in response:
                            action_entry["NEXT"] = response["NEXT"]

                        cur_anno['action_predict'][action_mode][self.screen_mode] = action_entry

                        print(f"Updating [{action_mode}][{self.screen_mode}] {cur_save_path} ...")
                        cur_anno["image_path"] = step_data["image_full_path"]
                        json.dump(cur_anno, open(cur_save_path, "w", encoding="utf-8"), indent=4)
                        # exit()
                    except json.decoder.JSONDecodeError as e:
                        print('-'*50 + "\n", asst_head, sep=" ")
                        print('-'*50 + "\n", asst_res, sep=" ")
                    break
                elif state == HTTPStatus.REQUEST_ENTITY_TOO_LARGE: 
                    print(f"Trial {try_num} failured! -- Status {state}")
                    break
                print(f"Trial {try_num} failured! -- Status {state}")
                try_num += 1
                time.sleep(10)

        return None

    def predict_with_memory(self, step_data, save_dir, max_trials=5): 
        subset, episode_id = step_data['subset'], step_data['episode_id']
        prev_step_id, step_id = step_data['prev_step_id'], step_data['step_id']
        action_predict_error_dir = os.path.join(save_dir, "action_predict_errors")
        os.makedirs(action_predict_error_dir, exist_ok=True)
        if subset!=episode_id:
            save_dir = os.path.join(save_dir, f"{subset}-{episode_id}")
        else:
            save_dir = os.path.join(save_dir, f"{subset}")

        cur_save_path = os.path.join(save_dir, f"{subset}-{episode_id}_{step_id}.json")
        cur_anno = json.load(open(cur_save_path, "r"))
        if prev_step_id:
            prev_save_path = os.path.join(save_dir, f"{subset}-{episode_id}_{prev_step_id}.json")
            prev_anno = json.load(open(prev_save_path, "r"))
            if 'action_result' not in prev_anno:
                print(prev_save_path)
            cur_anno['prev_action_result'] = prev_anno['action_result']

        action_mode = self.cfg.DEMO_MODE.upper()
        if action_mode == "COAT" and ("action_think" not in cur_anno or "action_desc" not in cur_anno):
            raise ValueError("Missing action_think/action_desc in flow outputs; rerun flow before predict.")
        # if action_mode == "COA":    func_handler = self.coa_action
        # elif action_mode == "COT":  func_handler = self.cot_action
        # elif action_mode == "COAT": func_handler = self.coat_action
        # else: raise NotImplementedError
        func_handler = self.coat_action_with_memory

        if 'action_predict' not in cur_anno: 
            cur_anno['action_predict'] = {}
            # print(save_dir)
        if action_mode not in cur_anno['action_predict']: cur_anno['action_predict'][action_mode] = {}
        # if action_mode in cur_anno['action_predict']: 
        #     anno = cur_anno['action_predict'][action_mode]
        #     cur_anno['action_predict'][action_mode] = {}
        #     cur_anno['action_predict'][action_mode][self.screen_mode] = anno
        #     json.dump(cur_anno, open(cur_save_path, "w", encoding="utf-8"), indent=4)
        
        if self.screen_mode not in cur_anno['action_predict'][action_mode]:
        # if True:
            print(f"[Mode {action_mode}][{self.screen_mode}] Gnerating action ... ({cur_save_path})")
            try_num = 0
            while try_num < max_trials:
                asst_head, asst_res, state, ui_path = func_handler(step_data, **cur_anno, save_dir=save_dir)
                if state == HTTPStatus.OK: 
                    try:
                        response = self._parse_action_response(asst_head, asst_res)
                        if response is None:
                            raise json.decoder.JSONDecodeError(
                                "Unable to parse assistant response",
                                self._strip_code_fences(asst_res),
                                0,
                            )
                        print("ACTION", response["ACTION"])
                        if isinstance(response["ACTION"], str):
                            response["ACTION"] = response["ACTION"].lower()
                        elif isinstance(response["ACTION"], dict):
                            response["ARGS"] = response["ACTION"]["ARGS"]
                            response["ACTION"] = response["ACTION"]["ACTION"].lower()
                            
                        normalized_action = self._enforce_action_match(
                            cur_anno,
                            response["ACTION"],
                            response,
                            action_predict_error_dir,
                            subset,
                            episode_id,
                            step_id,
                        )
                        response["ACTION"] = normalized_action
                        cur_anno['action_predict'][action_mode][self.screen_mode] = {
                            "ACTION": response["ACTION"], "ARGS": response["ARGS"]}
                        cur_anno['relevant_ui_path'] = ui_path
                        
                        if cur_anno['action_desc']['type'].lower() == "click":
                            cur_anno['action_desc']['type'] = "click_element"

                        predicted_action = response.get("ACTION")
                        if not predicted_action:
                            print("[predict_with_memory] Missing ACTION field, retrying...", response)
                            continue
                        print(f"Updating [{action_mode}][{self.screen_mode}] {cur_save_path} ...")
                        cur_anno["image_path"] = step_data["image_full_path"]
                        if predicted_action == "click_element" and "bbox" in response['ARGS']:
                            response['ARGS']['bbox'] = self._preserve_model_bbox(response['ARGS']['bbox'])
                                
                                # assert 0 <= bbox[0] < bbox[2] <= width and 0 <= bbox[1] < bbox[3] <= height, f"Invalid bbox {bbox} in {cur_save_path}"
                        json.dump(cur_anno, open(cur_save_path, "w", encoding="utf-8"), indent=4)
                    except json.decoder.JSONDecodeError as e:
                        print('-'*50 + "\n", asst_head, sep=" ")
                        print('-'*50 + "\n", asst_res, sep=" ")
                        continue
                    break
                elif state == HTTPStatus.REQUEST_ENTITY_TOO_LARGE: 
                    print(f"Trial {try_num} failured! -- Status {state}")
                    break
                print(f"Trial {try_num} failured! -- Status {state}")
                try_num += 1
                time.sleep(10)

        return None

class AtlasAgent(BaseAgent):
    def __init__(self, config):
        super().__init__(config)
        assert "atlas" in self.cfg.MODEL.NAME.lower(), "Only support atlas model!"

        self.use_screen_tag = bool(self.cfg.DATA.USE_SCREEN_TAG)
        if not self.use_screen_tag:
            raise ValueError("AtlasAgent expects SCREEN_TAG inputs.")
        self.screen_mode = "tag"
        self.focus_dir = (
            f"/remote-home/lbsun/memory/0721/focus_images/{self.cfg.DATA.DATASET}/"
            f"statinoary{self.cfg.MEMORY.STATIONARY}_procedural{self.cfg.MEMORY.PROCEDURAL}"
        )


    def format_query(self, image_paths, prompt):
        # print(image_paths)
        # print(len(prompt))
        # print(prompt)
        
        if isinstance(prompt, str):
            image = Image.open(image_paths)
            return [prompt, image]
        
        if isinstance(image_paths, str): image_paths = [image_paths]

        if isinstance(prompt, list):
            user_message = {
                "role": "user",
                "content": [],
            }
            for role, msg in prompt:
                if isinstance(msg, dict) and "img_index" in msg:
                    img_index = msg.pop('img_index')
                    image = Image.open(image_paths[img_index])
                    # user_message['content'] += [
                    #     {"type": "text", "text": msg['text']},
                    #     {
                    #         "type": "image",
                    #         "image": image_paths[img_index]
                    #     }
                    # ]
                    with open(image_paths[img_index], "rb") as f:
                        image_b64 = base64.b64encode(f.read()).decode("utf-8")
                    user_message['content'] += [
                        {"type": "text", "text": msg['text']},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{image_b64}"}
                        }
                    ]
                else:
                    if isinstance(msg, dict): user_message['content'].append(
                        {"type": "text", "text": msg['text']}
                    )
                    else: user_message['content'].append(
                        {"type": "text", "text": msg}
                    )
            return [user_message]
        
    def inference(self, image_paths, prompt):
        messages = self.format_query(image_paths, prompt)
        print(self.cfg.MODEL)
        # 转成 OpenAI Chat 格式
        payload = {
            # "model": "Qwen2.5-VL-32B",
            # "model": "Atlas-Base-7B",
            "model": self.cfg.MODEL.NAME,
            "messages": messages,
            "max_tokens": 1280,
        }

        # 如果有图片，需要转成 base64 放进 messages 里
        # for image_path in image_paths:
        #     with open(image_path, "rb") as f:
        #         image_b64 = base64.b64encode(f.read()).decode("utf-8")
        #     payload["messages"][0]["content"].append({
        #         "type": "image_url",
        #         "image_url": {"url": f"data:image/png;base64,{image_b64}"}
        #     })

        # 调用本地 vLLM API
        resp = requests.post(
            "http://localhost:8182/v1/chat/completions",
            headers={"Content-Type": "application/json"},
            json=payload,
            timeout=600
        )

        if resp.status_code != 200:
            return f"Error: {resp.text}", resp.status_code

        output_text = resp.json()["choices"][0]["message"]["content"]
        return output_text, 200

    # def inference(self, image_paths, prompt):
    #     messages = self.format_query(image_paths, prompt)
    #     # print(image_paths)
    #     # print(prompt)
    #     # print(messages)
    #     # Preparation for inference
    #     text = self.processor.apply_chat_template(
    #         messages, tokenize=False, add_generation_prompt=True
    #     )
    #     # print(messages)
    #     image_inputs, video_inputs = process_vision_info(messages)
    #     inputs = self.processor(
    #         text=[text],
    #         images=image_inputs,
    #         videos=video_inputs,
    #         padding=True,
    #         return_tensors="pt",
    #     )
    #     inputs = inputs.to("cuda")

    #     # Inference: Generation of the output
    #     generated_ids = self.model.generate(**inputs, max_new_tokens=128)

    #     generated_ids_trimmed = [
    #         out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    #     ]

    #     output_text = self.processor.batch_decode(
    #         generated_ids_trimmed, skip_special_tokens=False, clean_up_tokenization_spaces=False
    #     )[0]
    #     return output_text, 200

    def think_action(self, step_data, *args, **kwargs):
        """ Chain of Action Thought """
        user_request, image_path = step_data['instruction'], step_data['image_full_path']
        history_actions = step_data['history_coat_actions']
        
        usr_prompt = self.prompts['ATLAS_PROMPT_EXECUTOR']
        usr_prompt = usr_prompt.split("\n")
        if not history_actions: usr_prompt = [x for x in usr_prompt if 'history_actions' not in x]
        usr_prompt = "\n".join(usr_prompt)

        if history_actions: 
            history_action_string = ""
            for idx, history_action in enumerate(history_actions):
                step_idx = idx + 1
                history_action = f"Step {step_idx}: {history_action}\n"
                history_action_string += history_action
        else: history_action_string = "None"

        usr_prompt = usr_prompt.replace("{history_actions}", history_action_string)
        usr_prompt = usr_prompt.replace("{user_request}", user_request)
        action_space = self.prompts['ACTION_SPACE'][self.cfg['DATA']['DATASET']]
        usr_prompt = usr_prompt.replace("{action_space}", action_space)
        usr_prompt = [x for x in usr_prompt.split("{screenshot}")]

        # asst_prompt = self.prompts['ACTION_PREDICT']['SYSTEM']
        # return self.make_action(step_data, usr_prompt, kwargs['save_dir'], asst_prompt=asst_prompt)

        # Determine action mode and image path
        action_mode = self.cfg.DEMO_MODE.upper()
        image_path = step_data['image_full_path']

        # Prepare UI elements
        ui_elements = []
        ui_positions = step_data.get('ui_positions') or []
        ui_texts = step_data.get('ui_text') or []
        ui_types = step_data.get('ui_types')
        if ui_types is None:
            ui_types = ["unknown"] * len(ui_positions)
        if ui_positions and ui_texts and ui_types:
            for org_bbox, txt, ui_class in zip(ui_positions, ui_texts, ui_types):
                ymin, xmin, h, w = org_bbox
                bbox = [xmin, ymin, xmin + w, ymin + h]
                ui_elements.append({"bounds": bbox, "text": txt, "type": ui_class})
            ui_elements = row_col_sort(ui_elements)
        step_data['ui_elements'] = ui_elements

        # Handle screen tag mode
        if self.use_screen_tag:
            prompt = [
                ("user", {"text": usr_prompt[0], "img_index": 0}),
            ]
            for txt in usr_prompt[1:]:
                prompt.append(("user", {"text": txt}))
            asst_res, state = self.inference([image_path], prompt)
        else:
            raise ValueError("AtlasAgent requires SCREEN_TAG inputs.")

        # Return assistant response
        # asst_head = asst_prompt if asst_prompt else ""         
        return None, asst_res, state
    
    
    def draw_focus(self, image_path, predict_point, max_radius_ratio=0.5, mask_color=(0, 0, 0, 150)):
        """
        根据图片和坐标点，绘制一个聚焦的圆形区域。

        Args:
            image_path (str): 输入图片的路径。
            predict_point (tuple): 聚焦点在图片上的坐标 (x, y)。
            max_radius_ratio (float, optional): 最大半径与图片最小边长的比例。默认为 0.5。
            mask_color (tuple, optional): 蒙版的颜色和透明度 (R, G, B, Alpha)。
                                        默认为半透明的黑色。

        Returns:
            PIL.Image.Image: 处理后带有聚焦效果的图片对象，失败时返回 None。
        """
        # 1. 检查文件是否存在
        if not os.path.exists(image_path):
            print(f"错误：找不到文件 {image_path}")
            return None
            
        # 2. 打开图片并转换为 RGBA 模式，以支持透明度处理
        try:
            base_image = Image.open(image_path).convert("RGBA")
        except Exception as e:
            print(f"错误：无法打开图片 {image_path}，错误信息：{e}")
            return None

        # 3. 获取图片尺寸和坐标点
        img_width, img_height = base_image.size
        cx, cy = int(predict_point[0]), int(predict_point[1])
        
        # 4. 检查坐标点是否在图片内
        if not (0 <= cx < img_width and 0 <= cy < img_height):
            print(f"错误：坐标点 {predict_point} 超出图片边界 ({img_width}x{img_height})。")
            return None  # 返回 None 而不是原图，因为这是一个错误情况

        # 5. 验证 max_radius_ratio 参数
        if not (0 < max_radius_ratio <= 1.0):
            print(f"错误：max_radius_ratio 必须在 0 到 1 之间，当前值：{max_radius_ratio}")
            return None

        # 6. 计算最大可能的半径，确保圆圈不超过图片边界
        # 计算中心点到四条边的距离
        dist_to_left = cx
        dist_to_right = img_width - cx
        dist_to_top = cy
        dist_to_bottom = img_height - cy
        
        # 半径不能超过到任何边界的距离
        max_radius_by_boundary = min(dist_to_left, dist_to_right, dist_to_top, dist_to_bottom)
        
        # 根据比例计算推荐半径
        recommended_radius = int(min(img_width, img_height) * max_radius_ratio)
        
        # 取两者中的较小值作为最终半径
        radius = min(max_radius_by_boundary, recommended_radius)
        
        # 确保半径至少为1像素
        if radius <= 0:
            radius = 1
            print(f"警告：聚焦点太靠近边缘，使用最小半径 {radius}")
        
        print(f"计算得出的聚焦半径: {radius} 像素")

        # 7. 创建一个全新的、与原图大小相同的透明图层作为蒙版
        mask_layer = Image.new("RGBA", base_image.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(mask_layer)

        # 8. 在蒙版上绘制半透明的蒙皮，并"挖"出一个完全透明的圆形区域
        # 首先，整个蒙版填充为半透明的颜色
        draw.rectangle([(0, 0), base_image.size], fill=mask_color)
        
        # 然后，在蒙版上绘制一个完全透明的圆形，这就是我们的"聚焦区域"
        # 计算圆形的边界框
        left = cx - radius
        top = cy - radius
        right = cx + radius
        bottom = cy + radius
        
        # 绘制透明圆形（聚焦区域）
        draw.ellipse((left, top, right, bottom), fill=(0, 0, 0, 0))

        # 9. 将处理好的蒙版层与原始图片合并
        result_image = Image.alpha_composite(base_image, mask_layer)

        return result_image.convert("RGB")  # 返回 RGB 格式的图片
    
    def get_focus_image(self, image_path, ui_path):
        os.makedirs(self.focus_dir, exist_ok=True)
        save_image_path = os.path.join(self.focus_dir, image_path.split('/')[-1])
        if os.path.exists(save_image_path):
            return save_image_path
        # 根据ui_path image_path得到匹配中心点
        import sys
        sys.path.append("/remote-home/lbsun/memory/0721/code/template_match/methods/self_tm")
        from generate_center_point import generate_center_point
        predict_center_point = generate_center_point(image_path, ui_path)
        print(imagesize.get(image_path))
        print(predict_center_point)
        # 根据中心点绘制蒙皮
        image_with_focus = self.draw_focus(image_path, predict_center_point[1])
        # 保存路径
        image_with_focus.save(save_image_path)
        print(f"focus image saved in {save_image_path}")
        return save_image_path

    def action_wo_history(self, step_data, *args, **kwargs):
        """ Chain of Action Thought """
        user_request, image_path = step_data['instruction'], step_data['image_full_path']
        history_actions = step_data['history_coat_actions']
        
        usr_prompt = self.prompts['OPENAI_ATLAS_PROMPT_WO_HISTORY']
        usr_prompt = usr_prompt.split("\n")
        if not history_actions: usr_prompt = [x for x in usr_prompt if 'history_actions' not in x]
        usr_prompt = "\n".join(usr_prompt)


        usr_prompt = usr_prompt.replace("{user_request}", kwargs['action_desc']['description'])
        action_space = self.prompts['ACTION_SPACE'][self.cfg['DATA']['DATASET']]
        usr_prompt = usr_prompt.replace("{action_space}", action_space)
        usr_prompt = [x for x in usr_prompt.split("{screenshot}")]

        # asst_prompt = self.prompts['ACTION_PREDICT']['SYSTEM']
        # return self.make_action(step_data, usr_prompt, kwargs['save_dir'], asst_prompt=asst_prompt)

        # Determine action mode and image path
        action_mode = self.cfg.DEMO_MODE.upper()
        image_path = step_data['image_full_path']

        # Prepare UI elements
        ui_elements = []
        ui_positions = step_data.get('ui_positions') or []
        ui_texts = step_data.get('ui_text') or []
        ui_types = step_data.get('ui_types')
        if ui_types is None:
            ui_types = ["unknown"] * len(ui_positions)
        if ui_positions and ui_texts and ui_types:
            for org_bbox, txt, ui_class in zip(ui_positions, ui_texts, ui_types):
                ymin, xmin, h, w = org_bbox
                bbox = [xmin, ymin, xmin + w, ymin + h]
                ui_elements.append({"bounds": bbox, "text": txt, "type": ui_class})
            ui_elements = row_col_sort(ui_elements)
        step_data['ui_elements'] = ui_elements

        # Handle screen tag mode
        if self.use_screen_tag:
            prompt = [
                ("user", {"text": usr_prompt[0], "img_index": 0}),
            ]
            for txt in usr_prompt[1:]:
                prompt.append(("user", {"text": txt}))
            asst_res, state = self.inference([image_path], prompt)
        else:
            raise ValueError("AtlasAgent requires SCREEN_TAG inputs.")

        # Return assistant response
        # asst_head = asst_prompt if asst_prompt else ""         
        return None, asst_res, state


    def action_wo_history_w_focus(self, step_data, *args, **kwargs):
        """ Chain of Action Thought """
        user_request, image_path = step_data['instruction'], step_data['image_full_path']

        history_actions = step_data['history_coat_actions']
        
        usr_prompt = self.prompts['OPENAI_ATLAS_PROMPT_WO_HISTORY']
        usr_prompt = usr_prompt.split("\n")
        if not history_actions: usr_prompt = [x for x in usr_prompt if 'history_actions' not in x]
        usr_prompt = "\n".join(usr_prompt)

        # print(kwargs)
        # exit()
        usr_prompt = usr_prompt.replace("{user_request}", kwargs['action_desc']['description'])

        if kwargs['action_desc']['type'].lower() == "click_element" and self.stationary_memory:
            # retrieve from memory
            doc = self.stationary_memory.rag.query(kwargs['action_desc']['description'], param=self.stationary_query_param)
            
            if doc is None:
                ui_path = None
            else:
                print("Stationary memory retrived!")
                ui_path = doc[0]['image']
                if "/home/slb/data" in ui_path:
                    ui_path = ui_path.replace("/home/slb/data", "/root/autodl-tmp/agent/agent/data")
                elif "/mnt/shanghai1/sunlibo/gui-odyssey" in ui_path:
                    ui_path = ui_path.replace("/mnt/shanghai1/sunlibo/gui-odyssey", "/root/autodl-tmp/agent/agent/data/gui_odyssey")
                elif "/root/autodl-tmp/data/gui_memory" in ui_path:
                    ui_path = ui_path.replace("/root/autodl-tmp/data/gui_memory", "/remote-home/lbsun/data")
                print("ui_path:", ui_path)
                assert os.path.exists(ui_path)

                image_path = self.get_focus_image(image_path, ui_path)
                kwargs['relevant_ui_path'] = ui_path
                kwargs['focus_image_path'] = image_path
                print("Image with focus:", image_path)

        action_space = self.prompts['ACTION_SPACE'][self.cfg['DATA']['DATASET']]
        usr_prompt = usr_prompt.replace("{action_space}", action_space)
        usr_prompt = [x for x in usr_prompt.split("{screenshot}")]

        # asst_prompt = self.prompts['ACTION_PREDICT']['SYSTEM']
        # return self.make_action(step_data, usr_prompt, kwargs['save_dir'], asst_prompt=asst_prompt)

        # Determine action mode and image path
        action_mode = self.cfg.DEMO_MODE.upper()
        # image_path = step_data['image_full_path']

        # Prepare UI elements
        ui_elements = []
        ui_positions = step_data.get('ui_positions') or []
        ui_texts = step_data.get('ui_text') or []
        ui_types = step_data.get('ui_types')
        if ui_types is None:
            ui_types = ["unknown"] * len(ui_positions)
        if ui_positions and ui_texts and ui_types:
            for org_bbox, txt, ui_class in zip(ui_positions, ui_texts, ui_types):
                ymin, xmin, h, w = org_bbox
                bbox = [xmin, ymin, xmin + w, ymin + h]
                ui_elements.append({"bounds": bbox, "text": txt, "type": ui_class})
            ui_elements = row_col_sort(ui_elements)
        step_data['ui_elements'] = ui_elements

        # Handle screen tag mode
        if self.use_screen_tag:
            prompt = [
                ("user", {"text": usr_prompt[0], "img_index": 0}),
            ]
            for txt in usr_prompt[1:]:
                prompt.append(("user", {"text": txt}))
            asst_res, state = self.inference([image_path], prompt)
        else:
            raise ValueError("AtlasAgent requires SCREEN_TAG inputs.")

        # Return assistant response
        # asst_head = asst_prompt if asst_prompt else ""         
        return kwargs, asst_res, state


    def parse_action(self, action_string):

        if "CLICK" in action_string:
            action_type = "click_element"
            pattern = r"\[\[(\d+),\s*(\d+)\]\]"
            action_args = {"xy": [int(num) for num in re.findall(pattern, action_string)[0]]}
        
        elif "SCROLL" in action_string:
            action_type = "scroll"
            if "DOWN" in action_string:
                action_args = {"direction":"down"}
            elif "UP" in action_string:
                action_args = {"direction":"up"}
            elif "LEFT" in action_string:
                action_args = {"direction":"left"}
            elif "RIGHT" in action_string:
                action_args = {"direction":"right"}
            else:
                raise ValueError("Invalid SCROLL action direction.")
            
        elif "TYPE" in action_string:
            action_type = "input"
            action_args = {"text":action_string.split("[")[-1].split("]")[0]}

        elif "press" in action_string.lower() or "PRESS" in action_string or "ENTER" in action_string:
            if "enter" in action_string.lower():
                action_type = "press_enter"
            elif "home" in action_string.lower():
                action_type = "press_home"
            elif "back" in action_string.lower():
                action_type = "press_back"
            else:
                raise ValueError("Invalid PRESS action.")
            action_args = {}

        elif "STOP" in action_string:
            action_type = "stop"
            if "success" in action_string:
                action_args = {"task_status": "success"}
            else:
                action_args = {"task_status": "failure"}

        elif "COMPLETE" in action_string or "Done" in action_string or "No further" in action_string:
            action_type = "stop"
            action_args = {"task_status": "success"}

        # print(action_string)
        # print(action_type, action_args)

        return action_type, action_args

    # def parse_res(self, asst_res):
    #     if "actions:" not in asst_res:
    #         raise ValueError("No actions found in the response.")
    #     else:
    #         if "thoughts:" in asst_res:
    #             action_string = asst_res.replace("<|im_end|>", "").split("actions:")[-1]
    #             think_string = asst_res.split("actions:")[0].split("thoughts:")[-1]
    #         else:
    #             action_string = asst_res.replace("<|im_end|>", "").split("actions:")[-1]
    #             think_string = None
    #         action_type, action_args = self.parse_action(action_string)
            
    #         return {"ACTION":action_type, "ARGS":action_args}
    #         # return action_type, action_args

    def parse_res(self, asst_res):
        if "Actions:" in asst_res:
            asst_res = asst_res.replace("Actions:", "actions:")
        if "Thoughts:" in asst_res:
            asst_res = asst_res.replace("Thoughts:", "thoughts:")
        if "actions:" not in asst_res:
            raise ValueError("No actions found in the response.")
        else:
            if "thoughts:" in asst_res:
                action_string = asst_res.replace("<|im_end|>", "").split("actions:")[-1]
                think_string = asst_res.split("actions:")[0].split("thoughts:")[-1]
            else:
                action_string = asst_res.replace("<|im_end|>", "").split("actions:")[-1]
                think_string = None
            action_type, action_args = self.parse_action(action_string)
            
            return {"ACTION":action_type, "ARGS":action_args}

    def predict(self, step_data, save_dir, max_trials=5):
        subset, episode_id = step_data['subset'], step_data['episode_id']
        prev_step_id, step_id = step_data['prev_step_id'], step_data['step_id']
        action_predict_error_dir = os.path.join(save_dir, "action_predict_errors")
        os.makedirs(action_predict_error_dir, exist_ok=True)
        if subset==episode_id:
            save_dir = os.path.join(save_dir, episode_id)
        else:
            save_dir = os.path.join(save_dir, f"{subset}-{episode_id}")
        os.makedirs(save_dir, exist_ok=True)
        cur_save_path = os.path.join(save_dir, f"{subset}-{episode_id}_{step_id}.json")
        if os.path.exists(cur_save_path):
            cur_anno = json.load(open(cur_save_path, "r"))
        else:
            cur_anno = {}

        # if prev_step_id:
        #     prev_save_path = os.path.join(save_dir, f"{subset}-{episode_id}_{prev_step_id}.json")
        #     prev_anno = json.load(open(prev_save_path, "r"))
        #     cur_anno['prev_action_result'] = prev_anno['action_result']

        action_mode = self.cfg.DEMO_MODE.upper()
        if self.cfg['MODEL']['NAME'] in ["openai_atlas", "openai_atlas_base"]:
            if self.cfg['MEMORY']['STATIONARY']==False:
                func_handler = self.action_wo_history
            else:
                func_handler = self.action_wo_history_w_focus
        else:
            func_handler = self.think_action

        if 'action_predict' not in cur_anno: 
            cur_anno['action_predict'] = {}
            # print(save_dir)
        if action_mode not in cur_anno['action_predict']: cur_anno['action_predict'][action_mode] = {}
        # if action_mode in cur_anno['action_predict']: 
        #     anno = cur_anno['action_predict'][action_mode]
        #     cur_anno['action_predict'][action_mode] = {}
        #     cur_anno['action_predict'][action_mode][self.screen_mode] = anno
        #     json.dump(cur_anno, open(cur_save_path, "w", encoding="utf-8"), indent=4)
        
        if self.screen_mode not in cur_anno['action_predict'][action_mode] or cur_anno['action_predict'][action_mode][self.screen_mode]['ACTION']=="":
        # if True:
            print(f"[Mode {action_mode}][{self.screen_mode}] Gnerating action ... ({cur_save_path})")
            try_num = 0
            while try_num < max_trials:
                asst_head, asst_res, state = func_handler(step_data, **cur_anno, save_dir=save_dir)
                # print(asst_res)
                if state == HTTPStatus.OK: 
                    try:
                        try:
                            response = self.parse_res(asst_res)
                            print(f"ACTION: {response['ACTION']}, ARGS: {response['ARGS']}")
                        except json.decoder.JSONDecodeError:
                            response = self._parse_action_response(asst_head, asst_res)
                            if response is None:
                                raise json.decoder.JSONDecodeError(
                                    "Unable to parse assistant response",
                                    self._strip_code_fences(asst_res),
                                    0,
                                )
                        if isinstance(response["ACTION"], str):
                            response["ACTION"] = response["ACTION"].lower()
                        elif isinstance(response["ACTION"], dict):
                            response["ARGS"] = response["ACTION"]["ARGS"]
                            response["ACTION"] = response["ACTION"]["ACTION"].lower()
                        if 'bbox' in response["ARGS"]:
                            response["ARGS"]['bbox'] = self._preserve_model_bbox(response["ARGS"]['bbox'])
                        cur_anno['action_predict'][action_mode][self.screen_mode] = {
                            "ACTION": response["ACTION"], "ARGS": response["ARGS"]}
                        
                        json.dump(cur_anno, open(cur_save_path, "w", encoding="utf-8"), indent=4)
                    
                    except json.decoder.JSONDecodeError as e:
                        print('-'*50 + "\n", asst_head, sep=" ")
                        print('-'*50 + "\n", asst_res, sep=" ")
                    break
                elif state == HTTPStatus.REQUEST_ENTITY_TOO_LARGE: 
                    print(f"Trial {try_num} failured! -- Status {state}")
                    break
                print(f"Trial {try_num} failured! -- Status {state}")
                try_num += 1
                time.sleep(10)

        return None    
    
    def predict_with_memory(self, step_data, save_dir, max_trials=5):
        subset, episode_id = step_data['subset'], step_data['episode_id']
        prev_step_id, step_id = step_data['prev_step_id'], step_data['step_id']
        action_predict_error_dir = os.path.join(save_dir, "action_predict_errors")
        os.makedirs(action_predict_error_dir, exist_ok=True)
        if subset==episode_id:
            save_dir = os.path.join(save_dir, episode_id)
        else:
            save_dir = os.path.join(save_dir, f"{subset}-{episode_id}")
        os.makedirs(save_dir, exist_ok=True)
        cur_save_path = os.path.join(save_dir, f"{subset}-{episode_id}_{step_id}.json")
        if os.path.exists(cur_save_path):
            cur_anno = json.load(open(cur_save_path, "r"))
        else:
            cur_anno = {}

        # if prev_step_id:
        #     prev_save_path = os.path.join(save_dir, f"{subset}-{episode_id}_{prev_step_id}.json")
        #     prev_anno = json.load(open(prev_save_path, "r"))
        #     cur_anno['prev_action_result'] = prev_anno['action_result']

        action_mode = self.cfg.DEMO_MODE.upper()
        if self.cfg['MODEL']['NAME'] in ["openai_atlas", "openai_atlas_base"]:
            if self.cfg['MEMORY']['STATIONARY']==False:
                func_handler = self.action_wo_history
            else:
                func_handler = self.action_wo_history_w_focus
        else:
            func_handler = self.think_action

        if 'action_predict' not in cur_anno: 
            cur_anno['action_predict'] = {}
            # print(save_dir)
        if action_mode not in cur_anno['action_predict']: cur_anno['action_predict'][action_mode] = {}
        # if action_mode in cur_anno['action_predict']: 
        #     anno = cur_anno['action_predict'][action_mode]
        #     cur_anno['action_predict'][action_mode] = {}
        #     cur_anno['action_predict'][action_mode][self.screen_mode] = anno
        #     json.dump(cur_anno, open(cur_save_path, "w", encoding="utf-8"), indent=4)
        
        if self.screen_mode not in cur_anno['action_predict'][action_mode] or cur_anno['action_predict'][action_mode][self.screen_mode]['ACTION']=="":
        # if True:
            print(f"[Mode {action_mode}][{self.screen_mode}] Gnerating action ... ({cur_save_path})")
            try_num = 0
            while try_num < max_trials:
                asst_head, asst_res, state = func_handler(step_data, **cur_anno, save_dir=save_dir)
                print(asst_res)
                if state == HTTPStatus.OK: 
                    try:
                        try:
                            response = self.parse_res(asst_res)
                        except json.decoder.JSONDecodeError:
                            response = self._parse_action_response(asst_head, asst_res)
                            if response is None:
                                raise json.decoder.JSONDecodeError(
                                    "Unable to parse assistant response",
                                    self._strip_code_fences(asst_res),
                                    0,
                                )
                        if isinstance(response["ACTION"], str):
                            response["ACTION"] = response["ACTION"].lower()
                        elif isinstance(response["ACTION"], dict):
                            response["ARGS"] = response["ACTION"]["ARGS"]
                            response["ACTION"] = response["ACTION"]["ACTION"].lower()
                        if 'bbox' in response["ARGS"]:
                            response["ARGS"]['bbox'] = self._preserve_model_bbox(response["ARGS"]['bbox'])
                        cur_anno['action_predict'][action_mode][self.screen_mode] = {
                            "ACTION": response["ACTION"], "ARGS": response["ARGS"]}
                        if asst_head is not None:
                            if "relevant_ui_path" in asst_head:
                                cur_anno["relevant_ui_path"] = asst_head["relevant_ui_path"]
                            if "focus_image_path" in asst_head:
                                cur_anno["focus_image_path"] = asst_head["focus_image_path"]
                        
                        print(f"Updating [action_predict] {cur_save_path} ...")
                        cur_anno["image_path"] = step_data["image_full_path"]
                        json.dump(cur_anno, open(cur_save_path, "w", encoding="utf-8"), indent=4)

                    
                    except json.decoder.JSONDecodeError as e:
                        print('-'*50 + "\n", asst_head, sep=" ")
                        print('-'*50 + "\n", asst_res, sep=" ")
                    break
                elif state == HTTPStatus.REQUEST_ENTITY_TOO_LARGE: 
                    print(f"Trial {try_num} failured! -- Status {state}")
                    break
                print(f"Trial {try_num} failured! -- Status {state}")
                try_num += 1
                time.sleep(10)

        return None    
    


class GUIR1Agent(BaseAgent):
    def __init__(self, config):
        super().__init__(config)
        assert "guir1" in self.cfg.MODEL.NAME.lower(), "Only support guir1 model!"

        self.use_screen_tag = bool(self.cfg.DATA.USE_SCREEN_TAG)
        if not self.use_screen_tag:
            raise ValueError("GUIR1Agent expects SCREEN_TAG inputs.")
        self.screen_mode = "tag"
        self.focus_dir = (
            f"/remote-home/lbsun/memory/0721/focus_images/{self.cfg.DATA.DATASET}/"
            f"statinoary{self.cfg.MEMORY.STATIONARY}_procedural{self.cfg.MEMORY.PROCEDURAL}"
        )

    def format_query(self, image_paths, prompt):
        # print(image_paths)
        # print(len(prompt))
        # print(prompt)
        
        if isinstance(prompt, str):
            image = Image.open(image_paths)
            return [prompt, image]
        
        if isinstance(image_paths, str): image_paths = [image_paths]

        if isinstance(prompt, list):
            user_message = {
                "role": "user",
                "content": [],
            }
            for role, msg in prompt:
                if isinstance(msg, dict) and "img_index" in msg:
                    img_index = msg.pop('img_index')
                    image = Image.open(image_paths[img_index])
                    # user_message['content'] += [
                    #     {"type": "text", "text": msg['text']},
                    #     {
                    #         "type": "image",
                    #         "image": image_paths[img_index]
                    #     }
                    # ]
                    with open(image_paths[img_index], "rb") as f:
                        image_b64 = base64.b64encode(f.read()).decode("utf-8")
                    user_message['content'] += [
                        {"type": "text", "text": msg['text']},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{image_b64}"}
                        }
                    ]
                else:
                    if isinstance(msg, dict): user_message['content'].append(
                        {"type": "text", "text": msg['text']}
                    )
                    else: user_message['content'].append(
                        {"type": "text", "text": msg}
                    )
            # print(user_message)
            return [user_message]
        
    def inference(self, image_paths, prompt):
        messages = self.format_query(image_paths, prompt)

        # 转成 OpenAI Chat 格式
        payload = {
            # "model": "Qwen2.5-VL-32B",
            "model": "GUI-R1-7B",
            "messages": messages,
            "max_tokens": 1280,
        }

        # 如果有图片，需要转成 base64 放进 messages 里
        # for image_path in image_paths:
        #     with open(image_path, "rb") as f:
        #         image_b64 = base64.b64encode(f.read()).decode("utf-8")
        #     payload["messages"][0]["content"].append({
        #         "type": "image_url",
        #         "image_url": {"url": f"data:image/png;base64,{image_b64}"}
        #     })

        # 调用本地 vLLM API
        resp = requests.post(
            "http://localhost:8182/v1/chat/completions",
            headers={"Content-Type": "application/json"},
            json=payload,
            timeout=600
        )

        if resp.status_code != 200:
            return f"Error: {resp.text}", resp.status_code

        output_text = resp.json()["choices"][0]["message"]["content"]
        return output_text, 200

    # def inference(self, image_paths, prompt):
    #     messages = self.format_query(image_paths, prompt)
    #     # print(image_paths)
    #     # print(prompt)
    #     # print(messages)
    #     # Preparation for inference
    #     text = self.processor.apply_chat_template(
    #         messages, tokenize=False, add_generation_prompt=True
    #     )
    #     # print(messages)
    #     image_inputs, video_inputs = process_vision_info(messages)
    #     inputs = self.processor(
    #         text=[text],
    #         images=image_inputs,
    #         videos=video_inputs,
    #         padding=True,
    #         return_tensors="pt",
    #     )
    #     inputs = inputs.to("cuda")

    #     # Inference: Generation of the output
    #     generated_ids = self.model.generate(**inputs, max_new_tokens=128)

    #     generated_ids_trimmed = [
    #         out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    #     ]

    #     output_text = self.processor.batch_decode(
    #         generated_ids_trimmed, skip_special_tokens=False, clean_up_tokenization_spaces=False
    #     )[0]
    #     return output_text, 200

    def think_action(self, step_data, *args, **kwargs):
        """ Chain of Action Thought """
        user_request, image_path = step_data['instruction'], step_data['image_full_path']
        history_actions = step_data['history_coat_actions']
        
        usr_prompt = self.prompts['GUIR1_PROMPT']
        # usr_prompt = usr_prompt.split("\n")
        # if not history_actions: usr_prompt = [x for x in usr_prompt if 'history_actions' not in x]
        # usr_prompt = "\n".join(usr_prompt)

        if history_actions: 
            history_action_string = ""
            for idx, history_action in enumerate(history_actions):
                step_idx = idx + 1
                history_action = f"Step {step_idx}: {history_action}\n"
                history_action_string += history_action
        else: history_action_string = "None"

        usr_prompt = usr_prompt.replace("{history_actions}", history_action_string)
        usr_prompt = usr_prompt.replace("{user_request}", user_request)
        # action_space = self.prompts['ACTION_SPACE'][self.cfg['DATA']['DATASET']]
        # usr_prompt = usr_prompt.replace("{action_space}", action_space)
        # usr_prompt = [x for x in usr_prompt.split("{screenshot}")]

        # asst_prompt = self.prompts['ACTION_PREDICT']['SYSTEM']
        # return self.make_action(step_data, usr_prompt, kwargs['save_dir'], asst_prompt=asst_prompt)

        # Determine action mode and image path
        action_mode = self.cfg.DEMO_MODE.upper()
        image_path = step_data['image_full_path']

        # Prepare UI elements
        ui_elements = []
        ui_positions = step_data.get('ui_positions') or []
        ui_texts = step_data.get('ui_text') or []
        ui_types = step_data.get('ui_types')
        if ui_types is None:
            ui_types = ["unknown"] * len(ui_positions)
        if ui_positions and ui_texts and ui_types:
            for org_bbox, txt, ui_class in zip(ui_positions, ui_texts, ui_types):
                ymin, xmin, h, w = org_bbox
                bbox = [xmin, ymin, xmin + w, ymin + h]
                ui_elements.append({"bounds": bbox, "text": txt, "type": ui_class})
            ui_elements = row_col_sort(ui_elements)
        step_data['ui_elements'] = ui_elements

        # Handle screen tag mode
        if self.use_screen_tag:
            prompt = [
                ("user", {"text": "", "img_index": 0}),
                ("user", {"text": usr_prompt}),
            ]
            asst_res, state = self.inference([image_path], prompt)
        else:
            raise ValueError("AtlasAgent requires SCREEN_TAG inputs.")

        # Return assistant response
        # asst_head = asst_prompt if asst_prompt else ""
        return None, asst_res, state
    
    
    def draw_focus(self, image_path, predict_point, max_radius_ratio=0.5, mask_color=(0, 0, 0, 150)):
        """
        根据图片和坐标点，绘制一个聚焦的圆形区域。

        Args:
            image_path (str): 输入图片的路径。
            predict_point (tuple): 聚焦点在图片上的坐标 (x, y)。
            max_radius_ratio (float, optional): 最大半径与图片最小边长的比例。默认为 0.5。
            mask_color (tuple, optional): 蒙版的颜色和透明度 (R, G, B, Alpha)。
                                        默认为半透明的黑色。

        Returns:
            PIL.Image.Image: 处理后带有聚焦效果的图片对象，失败时返回 None。
        """
        # 1. 检查文件是否存在
        if not os.path.exists(image_path):
            print(f"错误：找不到文件 {image_path}")
            return None
            
        # 2. 打开图片并转换为 RGBA 模式，以支持透明度处理
        try:
            base_image = Image.open(image_path).convert("RGBA")
        except Exception as e:
            print(f"错误：无法打开图片 {image_path}，错误信息：{e}")
            return None

        # 3. 获取图片尺寸和坐标点
        img_width, img_height = base_image.size
        cx, cy = int(predict_point[0]), int(predict_point[1])
        
        # 4. 检查坐标点是否在图片内
        if not (0 <= cx < img_width and 0 <= cy < img_height):
            print(f"错误：坐标点 {predict_point} 超出图片边界 ({img_width}x{img_height})。")
            return None  # 返回 None 而不是原图，因为这是一个错误情况

        # 5. 验证 max_radius_ratio 参数
        if not (0 < max_radius_ratio <= 1.0):
            print(f"错误：max_radius_ratio 必须在 0 到 1 之间，当前值：{max_radius_ratio}")
            return None

        # 6. 计算最大可能的半径，确保圆圈不超过图片边界
        # 计算中心点到四条边的距离
        dist_to_left = cx
        dist_to_right = img_width - cx
        dist_to_top = cy
        dist_to_bottom = img_height - cy
        
        # 半径不能超过到任何边界的距离
        max_radius_by_boundary = min(dist_to_left, dist_to_right, dist_to_top, dist_to_bottom)
        
        # 根据比例计算推荐半径
        recommended_radius = int(min(img_width, img_height) * max_radius_ratio)
        
        # 取两者中的较小值作为最终半径
        radius = min(max_radius_by_boundary, recommended_radius)
        
        # 确保半径至少为1像素
        if radius <= 0:
            radius = 1
            print(f"警告：聚焦点太靠近边缘，使用最小半径 {radius}")
        
        print(f"计算得出的聚焦半径: {radius} 像素")

        # 7. 创建一个全新的、与原图大小相同的透明图层作为蒙版
        mask_layer = Image.new("RGBA", base_image.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(mask_layer)

        # 8. 在蒙版上绘制半透明的蒙皮，并"挖"出一个完全透明的圆形区域
        # 首先，整个蒙版填充为半透明的颜色
        draw.rectangle([(0, 0), base_image.size], fill=mask_color)
        
        # 然后，在蒙版上绘制一个完全透明的圆形，这就是我们的"聚焦区域"
        # 计算圆形的边界框
        left = cx - radius
        top = cy - radius
        right = cx + radius
        bottom = cy + radius
        
        # 绘制透明圆形（聚焦区域）
        draw.ellipse((left, top, right, bottom), fill=(0, 0, 0, 0))

        # 9. 将处理好的蒙版层与原始图片合并
        result_image = Image.alpha_composite(base_image, mask_layer)

        return result_image.convert("RGB")  # 返回 RGB 格式的图片
    
    def get_focus_image(self, image_path, ui_path):
        os.makedirs(self.focus_dir, exist_ok=True)
        save_image_path = os.path.join(self.focus_dir, image_path.split('/')[-1])
        if os.path.exists(save_image_path):
            return save_image_path
        # 根据ui_path image_path得到匹配中心点
        import sys
        sys.path.append("/remote-home/lbsun/memory/0721/code/template_match/methods/self_tm")
        from generate_center_point import generate_center_point
        predict_center_point = generate_center_point(image_path, ui_path)
        print(imagesize.get(image_path))
        print(predict_center_point)
        # 根据中心点绘制蒙皮
        image_with_focus = self.draw_focus(image_path, predict_center_point[1])
        # 保存路径
        image_with_focus.save(save_image_path)
        print(f"focus image saved in {save_image_path}")
        return save_image_path

    def action_wo_history(self, step_data, *args, **kwargs):
        """ Chain of Action Thought """
        user_request, image_path = step_data['instruction'], step_data['image_full_path']
        history_actions = step_data['history_coat_actions']
        
        usr_prompt = self.prompts['OPENAI_ATLAS_PROMPT_WO_HISTORY']
        usr_prompt = usr_prompt.split("\n")
        if not history_actions: usr_prompt = [x for x in usr_prompt if 'history_actions' not in x]
        usr_prompt = "\n".join(usr_prompt)


        usr_prompt = usr_prompt.replace("{user_request}", kwargs['action_desc']['description'])
        action_space = self.prompts['ACTION_SPACE'][self.cfg['DATA']['DATASET']]
        usr_prompt = usr_prompt.replace("{action_space}", action_space)
        usr_prompt = [x for x in usr_prompt.split("{screenshot}")]

        # asst_prompt = self.prompts['ACTION_PREDICT']['SYSTEM']
        # return self.make_action(step_data, usr_prompt, kwargs['save_dir'], asst_prompt=asst_prompt)

        # Determine action mode and image path
        action_mode = self.cfg.DEMO_MODE.upper()
        image_path = step_data['image_full_path']

        # Prepare UI elements
        ui_elements = []
        ui_positions = step_data.get('ui_positions') or []
        ui_texts = step_data.get('ui_text') or []
        ui_types = step_data.get('ui_types')
        if ui_types is None:
            ui_types = ["unknown"] * len(ui_positions)
        if ui_positions and ui_texts and ui_types:
            for org_bbox, txt, ui_class in zip(ui_positions, ui_texts, ui_types):
                ymin, xmin, h, w = org_bbox
                bbox = [xmin, ymin, xmin + w, ymin + h]
                ui_elements.append({"bounds": bbox, "text": txt, "type": ui_class})
            ui_elements = row_col_sort(ui_elements)
        step_data['ui_elements'] = ui_elements

        # Handle screen tag mode
        if self.use_screen_tag:
            prompt = [
                ("user", {"text": usr_prompt[0], "img_index": 0}),
            ]
            for txt in usr_prompt[1:]:
                prompt.append(("user", {"text": txt}))
            asst_res, state = self.inference([image_path], prompt)
        else:
            raise ValueError("AtlasAgent requires SCREEN_TAG inputs.")

        # Return assistant response
        # asst_head = asst_prompt if asst_prompt else ""         
        return None, asst_res, state


    def action_wo_history_w_focus(self, step_data, *args, **kwargs):
        """ Chain of Action Thought """
        user_request, image_path = step_data['instruction'], step_data['image_full_path']

        history_actions = step_data['history_coat_actions']
        
        usr_prompt = self.prompts['OPENAI_ATLAS_PROMPT_WO_HISTORY']
        usr_prompt = usr_prompt.split("\n")
        if not history_actions: usr_prompt = [x for x in usr_prompt if 'history_actions' not in x]
        usr_prompt = "\n".join(usr_prompt)

        # print(kwargs)
        # exit()
        usr_prompt = usr_prompt.replace("{user_request}", kwargs['action_desc']['description'])

        if kwargs['action_desc']['type'].lower() == "click_element" and self.stationary_memory:
            # retrieve from memory
            doc = self.stationary_memory.rag.query(kwargs['action_desc']['description'], param=self.stationary_query_param)
            
            if doc is None:
                ui_path = None
            else:
                print("Stationary memory retrived!")
                ui_path = doc[0]['image']
                if "/home/slb/data" in ui_path:
                    ui_path = ui_path.replace("/home/slb/data", "/root/autodl-tmp/agent/agent/data")
                elif "/mnt/shanghai1/sunlibo/gui-odyssey" in ui_path:
                    ui_path = ui_path.replace("/mnt/shanghai1/sunlibo/gui-odyssey", "/root/autodl-tmp/agent/agent/data/gui_odyssey")
                elif "/root/autodl-tmp/data/gui_memory" in ui_path:
                    ui_path = ui_path.replace("/root/autodl-tmp/data/gui_memory", "/remote-home/lbsun/data")
                print("ui_path:", ui_path)
                assert os.path.exists(ui_path)

                image_path = self.get_focus_image(image_path, ui_path)
                kwargs['relevant_ui_path'] = ui_path
                kwargs['focus_image_path'] = image_path
                print("Image with focus:", image_path)

        action_space = self.prompts['ACTION_SPACE'][self.cfg['DATA']['DATASET']]
        usr_prompt = usr_prompt.replace("{action_space}", action_space)
        usr_prompt = [x for x in usr_prompt.split("{screenshot}")]

        # asst_prompt = self.prompts['ACTION_PREDICT']['SYSTEM']
        # return self.make_action(step_data, usr_prompt, kwargs['save_dir'], asst_prompt=asst_prompt)

        # Determine action mode and image path
        action_mode = self.cfg.DEMO_MODE.upper()
        # image_path = step_data['image_full_path']

        # Prepare UI elements
        ui_elements = []
        ui_positions = step_data.get('ui_positions') or []
        ui_texts = step_data.get('ui_text') or []
        ui_types = step_data.get('ui_types')
        if ui_types is None:
            ui_types = ["unknown"] * len(ui_positions)
        if ui_positions and ui_texts and ui_types:
            for org_bbox, txt, ui_class in zip(ui_positions, ui_texts, ui_types):
                ymin, xmin, h, w = org_bbox
                bbox = [xmin, ymin, xmin + w, ymin + h]
                ui_elements.append({"bounds": bbox, "text": txt, "type": ui_class})
            ui_elements = row_col_sort(ui_elements)
        step_data['ui_elements'] = ui_elements

        # Handle screen tag mode
        if self.use_screen_tag:
            if "bbox" in self.cfg.METRICS_DIR and "wotag" in self.cfg.OUTPUT_DIR:
                prompt = [
                    ("user", {"text": usr_prompt[0], "img_index": 0}),
                ]
                for txt in usr_prompt[1:]: prompt.append(("user", {"text": txt}))
                episode_id, step_id = step_data['episode_id'], step_data['step_id']
                prompt_save_path = f"/root/autodl-tmp/agent/agent/results/odyssey/prompts/val_wo/{episode_id}_{step_id}.json"
                # print(f"Saving prompt to {prompt_save_path} ...")
                # with open(prompt_save_path, "w", encoding="utf-8") as f:
                    # json.dump({"prompt":prompt, "images":[image_path, tag_image_path]}, f, indent=4, ensure_ascii=False)
                asst_res, state = self.inference([image_path], prompt)
        

        # Return assistant response
        # asst_head = asst_prompt if asst_prompt else ""         
        return kwargs, asst_res, state


    def parse_action(self, action_string):

        if "CLICK" in action_string:
            action_type = "click_element"
            pattern = r"\[\[(\d+),\s*(\d+)\]\]"
            action_args = {"xy": [int(num) for num in re.findall(pattern, action_string)[0]]}
        
        elif "SCROLL" in action_string:
            action_type = "scroll"
            if "DOWN" in action_string:
                action_args = {"direction":"down"}
            elif "UP" in action_string:
                action_args = {"direction":"up"}
            elif "LEFT" in action_string:
                action_args = {"direction":"left"}
            elif "RIGHT" in action_string:
                action_args = {"direction":"right"}
            else:
                raise ValueError("Invalid SCROLL action direction.")
            
        elif "TYPE" in action_string:
            action_type = "input"
            action_args = {"text":action_string.split("[")[-1].split("]")[0]}

        elif "press" in action_string.lower() or "PRESS" in action_string or "ENTER" in action_string:
            if "enter" in action_string.lower():
                action_type = "press_enter"
            elif "home" in action_string.lower():
                action_type = "press_home"
            elif "back" in action_string.lower():
                action_type = "press_back"
            else:
                raise ValueError("Invalid PRESS action.")
            action_args = {}

        elif "STOP" in action_string:
            action_type = "stop"
            if "success" in action_string:
                action_args = {"task_status": "success"}
            else:
                action_args = {"task_status": "failure"}

        elif "COMPLETE" in action_string or "Done" in action_string or "No further" in action_string:
            action_type = "stop"
            action_args = {"task_status": "success"}

        print(action_string)
        print(action_type, action_args)

        return action_type, action_args

    # def parse_res(self, asst_res):
    #     if "actions:" not in asst_res:
    #         raise ValueError("No actions found in the response.")
    #     else:
    #         if "thoughts:" in asst_res:
    #             action_string = asst_res.replace("<|im_end|>", "").split("actions:")[-1]
    #             think_string = asst_res.split("actions:")[0].split("thoughts:")[-1]
    #         else:
    #             action_string = asst_res.replace("<|im_end|>", "").split("actions:")[-1]
    #             think_string = None
    #         action_type, action_args = self.parse_action(action_string)
            
    #         return {"ACTION":action_type, "ARGS":action_args}
    #         # return action_type, action_args

    def parse_res(self, asst_res):
        print(asst_res)
        think_match = re.search(r"<think>(.*?)</think>", asst_res, re.DOTALL)
        answer_match = re.search(r"<answer>(.*?)</answer>", asst_res, re.DOTALL)
        # print(think_match)
        # print(answer_match)
        if not think_match or not answer_match:
            raise ValueError("Input string is missing <think> or <answer> tags.")

        thinking_text = think_match.group(1).strip()
        answer_str = answer_match.group(1).strip()

        # 2. 使用 ast.literal_eval 安全地将 answer 字符串解析为 Python 对象
        action_list = eval(answer_str)
        print(action_list)
        if not isinstance(action_list, list) or not action_list:
            raise ValueError("Answer content is not a valid list or is empty.")
        
        # 通常我们只处理第一个动作
        action_data = action_list[0]
        
        action_type = action_data.get('action')
        point = action_data.get('point')
        input_text = action_data.get('input_text')
        print(
            action_type, point, input_text
        )
        # 3. 根据规则转换动作格式
        action_json = {}
        if action_type in ['press_back', 'press_enter', 'press_home']:
            action_json['ACTION'] = action_type
            action_json['ARGS'] = ''
        elif action_type in ['stop']:
            action_json['ACTION'] = 'stop'
            action_json['ARGS'] = {'task_status': 'success'}
        elif action_type in ['click', 'long_press']:
            action_json['ACTION'] = 'click_element'
            action_json['ARGS'] = {'xy': point}
        elif action_type in ['type']:
            action_json['ACTION'] = 'input'
            action_json['ARGS'] = {'text': input_text}
        elif action_type == 'scroll':
            action_json['ACTION'] = 'scroll'
            action_json['ARGS'] = {'direction': input_text}
        else:
            raise ValueError(f"Unknown action type: {action_type}")
        print(action_json)
        # 4. 构建最终的字典
        final_structure = {
            "THINKING": thinking_text,
            "ACTION": action_json.get("ACTION"),
            "ARGS": action_json.get("ARGS")
        }

        # 将字典转换为JSON格式的字符串
        return final_structure
    
    def predict(self, step_data, save_dir, max_trials=5):
        subset, episode_id = step_data['subset'], step_data['episode_id']
        prev_step_id, step_id = step_data['prev_step_id'], step_data['step_id']
        action_predict_error_dir = os.path.join(save_dir, "action_predict_errors")
        os.makedirs(action_predict_error_dir, exist_ok=True)
        if subset==episode_id:
            save_dir = os.path.join(save_dir, episode_id)
        else:
            save_dir = os.path.join(save_dir, f"{subset}-{episode_id}")
        os.makedirs(save_dir, exist_ok=True)
        cur_save_path = os.path.join(save_dir, f"{subset}-{episode_id}_{step_id}.json")
        if os.path.exists(cur_save_path):
            cur_anno = json.load(open(cur_save_path, "r"))
        else:
            cur_anno = {}

        # if prev_step_id:
        #     prev_save_path = os.path.join(save_dir, f"{subset}-{episode_id}_{prev_step_id}.json")
        #     prev_anno = json.load(open(prev_save_path, "r"))
        #     cur_anno['prev_action_result'] = prev_anno['action_result']

        action_mode = self.cfg.DEMO_MODE.upper()
        if self.cfg['MODEL']['NAME'] in ["openai_atlas", "openai_atlas_base"]:
            if self.cfg['MEMORY']['STATIONARY']==False:
                func_handler = self.action_wo_history
            else:
                func_handler = self.action_wo_history_w_focus
        else:
            func_handler = self.think_action

        if 'action_predict' not in cur_anno: 
            cur_anno['action_predict'] = {}
            # print(save_dir)
        if action_mode not in cur_anno['action_predict']: cur_anno['action_predict'][action_mode] = {}
        # if action_mode in cur_anno['action_predict']: 
        #     anno = cur_anno['action_predict'][action_mode]
        #     cur_anno['action_predict'][action_mode] = {}
        #     cur_anno['action_predict'][action_mode][self.screen_mode] = anno
        #     json.dump(cur_anno, open(cur_save_path, "w", encoding="utf-8"), indent=4)
        
        if self.screen_mode not in cur_anno['action_predict'][action_mode] or cur_anno['action_predict'][action_mode][self.screen_mode]['ACTION']=="":
        # if True:
            print(f"[Mode {action_mode}][{self.screen_mode}] Gnerating action ... ({cur_save_path})")
            try_num = 0
            while try_num < max_trials:
                asst_head, asst_res, state = func_handler(step_data, **cur_anno, save_dir=save_dir)
                print(asst_res)
                if state == HTTPStatus.OK: 
                    try: 
                        try:
                            response = self.parse_res(asst_res)
                        except json.decoder.JSONDecodeError:
                            response = self._parse_action_response(asst_head, asst_res)
                            if response is None:
                                raise json.decoder.JSONDecodeError(
                                    "Unable to parse assistant response",
                                    self._strip_code_fences(asst_res),
                                    0,
                                )
                        if 'THINKING' in response:
                            cur_anno['action_predict'][action_mode]['THINKING'] = response['THINKING']
                        if isinstance(response["ACTION"], str):
                            response["ACTION"] = response["ACTION"].lower()
                        elif isinstance(response["ACTION"], dict):
                            response["ARGS"] = response["ACTION"]["ARGS"]
                            response["ACTION"] = response["ACTION"]["ACTION"].lower()
                        if 'bbox' in response["ARGS"]:
                            response["ARGS"]['bbox'] = self._preserve_model_bbox(response["ARGS"]['bbox'])
                        cur_anno['action_predict'][action_mode][self.screen_mode] = {
                            "ACTION": response["ACTION"], "ARGS": response["ARGS"]}
                        
                        json.dump(cur_anno, open(cur_save_path, "w", encoding="utf-8"), indent=4)
                    
                    except json.decoder.JSONDecodeError as e:
                        print('-'*50 + "\n", asst_head, sep=" ")
                        print('-'*50 + "\n", asst_res, sep=" ")
                    break
                elif state == HTTPStatus.REQUEST_ENTITY_TOO_LARGE: 
                    print(f"Trial {try_num} failured! -- Status {state}")
                    break
                print(f"Trial {try_num} failured! -- Status {state}")
                try_num += 1
                time.sleep(10)

        return None    
    
    def predict_with_memory(self, step_data, save_dir, max_trials=5):
        subset, episode_id = step_data['subset'], step_data['episode_id']
        prev_step_id, step_id = step_data['prev_step_id'], step_data['step_id']
        action_predict_error_dir = os.path.join(save_dir, "action_predict_errors")
        os.makedirs(action_predict_error_dir, exist_ok=True)
        if subset==episode_id:
            save_dir = os.path.join(save_dir, episode_id)
        else:
            save_dir = os.path.join(save_dir, f"{subset}-{episode_id}")
        os.makedirs(save_dir, exist_ok=True)
        cur_save_path = os.path.join(save_dir, f"{subset}-{episode_id}_{step_id}.json")
        if os.path.exists(cur_save_path):
            cur_anno = json.load(open(cur_save_path, "r"))
        else:
            cur_anno = {}

        # if prev_step_id:
        #     prev_save_path = os.path.join(save_dir, f"{subset}-{episode_id}_{prev_step_id}.json")
        #     prev_anno = json.load(open(prev_save_path, "r"))
        #     cur_anno['prev_action_result'] = prev_anno['action_result']

        action_mode = self.cfg.DEMO_MODE.upper()
        if self.cfg['MODEL']['NAME'] in ["openai_atlas", "openai_atlas_base"]:
            if self.cfg['MEMORY']['STATIONARY']==False:
                func_handler = self.action_wo_history
            else:
                func_handler = self.action_wo_history_w_focus
        else:
            func_handler = self.think_action

        if 'action_predict' not in cur_anno: 
            cur_anno['action_predict'] = {}
            # print(save_dir)
        if action_mode not in cur_anno['action_predict']: cur_anno['action_predict'][action_mode] = {}
        # if action_mode in cur_anno['action_predict']: 
        #     anno = cur_anno['action_predict'][action_mode]
        #     cur_anno['action_predict'][action_mode] = {}
        #     cur_anno['action_predict'][action_mode][self.screen_mode] = anno
        #     json.dump(cur_anno, open(cur_save_path, "w", encoding="utf-8"), indent=4)
        
        if self.screen_mode not in cur_anno['action_predict'][action_mode] or cur_anno['action_predict'][action_mode][self.screen_mode]['ACTION']=="":
        # if True:
            print(f"[Mode {action_mode}][{self.screen_mode}] Gnerating action ... ({cur_save_path})")
            try_num = 0
            while try_num < max_trials:
                asst_head, asst_res, state = func_handler(step_data, **cur_anno, save_dir=save_dir)
                print(asst_res)
                if state == HTTPStatus.OK: 
                    try: 
                        try:
                            response = self.parse_res(asst_res)
                        except json.decoder.JSONDecodeError:
                            response = self._parse_action_response(asst_head, asst_res)
                            if response is None:
                                raise json.decoder.JSONDecodeError(
                                    "Unable to parse assistant response",
                                    self._strip_code_fences(asst_res),
                                    0,
                                )
                        if isinstance(response["ACTION"], str):
                            response["ACTION"] = response["ACTION"].lower()
                        elif isinstance(response["ACTION"], dict):
                            response["ARGS"] = response["ACTION"]["ARGS"]
                            response["ACTION"] = response["ACTION"]["ACTION"].lower()
                        if 'bbox' in response["ARGS"]:
                            response["ARGS"]['bbox'] = self._preserve_model_bbox(response["ARGS"]['bbox'])
                        cur_anno['action_predict'][action_mode][self.screen_mode] = {
                            "ACTION": response["ACTION"], "ARGS": response["ARGS"]}
                        if asst_head is not None:
                            if "relevant_ui_path" in asst_head:
                                cur_anno["relevant_ui_path"] = asst_head["relevant_ui_path"]
                            if "focus_image_path" in asst_head:
                                cur_anno["focus_image_path"] = asst_head["focus_image_path"]
                        
                        print(f"Updating [action_predict] {cur_save_path} ...")
                        json.dump(cur_anno, open(cur_save_path, "w", encoding="utf-8"), indent=4)
                        cur["image_path"] = step_data["image_full_path"]


                    except json.decoder.JSONDecodeError as e:
                        print('-'*50 + "\n", asst_head, sep=" ")
                        print('-'*50 + "\n", asst_res, sep=" ")
                    break
                elif state == HTTPStatus.REQUEST_ENTITY_TOO_LARGE: 
                    print(f"Trial {try_num} failured! -- Status {state}")
                    break
                print(f"Trial {try_num} failured! -- Status {state}")
                try_num += 1
                time.sleep(10)

        return None    
    
class UI_Venus_Navi_Agent(BaseAgent):
    def __init__(self, config):
        super().__init__(config)
        assert "ui-venus" in self.cfg.MODEL.NAME.lower(), "Only support ui_venus model!"

        self.use_screen_tag = bool(self.cfg.DATA.USE_SCREEN_TAG)
        if not self.use_screen_tag:
            raise ValueError("UI_Venus_Navi_Agent expects SCREEN_TAG inputs.")
        self.screen_mode = "tag"
        self.focus_dir = (
            f"/remote-home/lbsun/memory/0721/focus_images/{self.cfg.DATA.DATASET}/"
            f"statinoary{self.cfg.MEMORY.STATIONARY}_procedural{self.cfg.MEMORY.PROCEDURAL}"
        )


    def format_query(self, image_paths, prompt):
        # print(image_paths)
        # print(len(prompt))
        # print(prompt)
        
        if isinstance(prompt, str):
            image = Image.open(image_paths)
            return [prompt, image]
        
        if isinstance(image_paths, str): image_paths = [image_paths]

        if isinstance(prompt, list):
            user_message = {
                "role": "user",
                "content": [],
            }
            for role, msg in prompt:
                if isinstance(msg, dict) and "img_index" in msg:
                    img_index = msg.pop('img_index')
                    image = Image.open(image_paths[img_index])
                    # user_message['content'] += [
                    #     {"type": "text", "text": msg['text']},
                    #     {
                    #         "type": "image",
                    #         "image": image_paths[img_index]
                    #     }
                    # ]
                    with open(image_paths[img_index], "rb") as f:
                        image_b64 = base64.b64encode(f.read()).decode("utf-8")
                    user_message['content'] += [
                        {"type": "text", "text": msg['text']},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{image_b64}"}
                        }
                    ]
                else:
                    if isinstance(msg, dict): user_message['content'].append(
                        {"type": "text", "text": msg['text']}
                    )
                    else: user_message['content'].append(
                        {"type": "text", "text": msg}
                    )
            return [user_message]
        
    def inference(self, image_paths, prompt):
        messages = self.format_query(image_paths, prompt)
        # print(self.cfg.MODEL)
        # 转成 OpenAI Chat 格式
        payload = {
            # "model": "Qwen2.5-VL-32B",
            # "model": "Atlas-Base-7B",
            "model": self.cfg.MODEL.QWENVL_MODEL,
            "messages": messages,
            "max_tokens": 1280,
        }

        # 如果有图片，需要转成 base64 放进 messages 里
        # for image_path in image_paths:
        #     with open(image_path, "rb") as f:
        #         image_b64 = base64.b64encode(f.read()).decode("utf-8")
        #     payload["messages"][0]["content"].append({
        #         "type": "image_url",
        #         "image_url": {"url": f"data:image/png;base64,{image_b64}"}
        #     })

        # 调用本地 vLLM API
        resp = requests.post(
            "http://localhost:8182/v1/chat/completions",
            headers={"Content-Type": "application/json"},
            json=payload,
            timeout=600
        )

        if resp.status_code != 200:
            return f"Error: {resp.text}", resp.status_code

        output_text = resp.json()["choices"][0]["message"]["content"]
        return output_text, 200

    # def inference(self, image_paths, prompt):
    #     messages = self.format_query(image_paths, prompt)
    #     # print(image_paths)
    #     # print(prompt)
    #     # print(messages)
    #     # Preparation for inference
    #     text = self.processor.apply_chat_template(
    #         messages, tokenize=False, add_generation_prompt=True
    #     )
    #     # print(messages)
    #     image_inputs, video_inputs = process_vision_info(messages)
    #     inputs = self.processor(
    #         text=[text],
    #         images=image_inputs,
    #         videos=video_inputs,
    #         padding=True,
    #         return_tensors="pt",
    #     )
    #     inputs = inputs.to("cuda")

    #     # Inference: Generation of the output
    #     generated_ids = self.model.generate(**inputs, max_new_tokens=128)

    #     generated_ids_trimmed = [
    #         out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    #     ]

    #     output_text = self.processor.batch_decode(
    #         generated_ids_trimmed, skip_special_tokens=False, clean_up_tokenization_spaces=False
    #     )[0]
    #     return output_text, 200

    def think_action(self, step_data, *args, **kwargs):
        """ Chain of Action Thought """
        user_request, image_path = step_data['instruction'], step_data['image_full_path']
        history_actions = step_data['history_coat_actions']
        
        usr_prompt = self.prompts['UI_VENUS_PROMPT_E2E']
        usr_prompt = usr_prompt.split("\n")
        if not history_actions: usr_prompt = [x for x in usr_prompt if 'previous_actions' not in x]
        usr_prompt = "\n".join(usr_prompt)

        if history_actions: 
            history_action_string = ""
            for idx, history_action in enumerate(history_actions):
                step_idx = idx + 1
                history_action = f"Step {step_idx}: {history_action}\n"
                history_action_string += history_action
        else: history_action_string = "None"

        usr_prompt = usr_prompt.replace("{previous_actions}", history_action_string)
        usr_prompt = usr_prompt.replace("{user_task}", user_request)

        # asst_prompt = self.prompts['ACTION_PREDICT']['SYSTEM']
        # return self.make_action(step_data, usr_prompt, kwargs['save_dir'], asst_prompt=asst_prompt)

        # Determine action mode and image path
        action_mode = self.cfg.DEMO_MODE.upper()
        image_path = step_data['image_full_path']

        # Prepare UI elements
        ui_elements = []
        ui_positions = step_data.get('ui_positions') or []
        ui_texts = step_data.get('ui_text') or []
        ui_types = step_data.get('ui_types')
        if ui_types is None:
            ui_types = ["unknown"] * len(ui_positions)
        if ui_positions and ui_texts and ui_types:
            for org_bbox, txt, ui_class in zip(ui_positions, ui_texts, ui_types):
                ymin, xmin, h, w = org_bbox
                bbox = [xmin, ymin, xmin + w, ymin + h]
                ui_elements.append({"bounds": bbox, "text": txt, "type": ui_class})
            ui_elements = row_col_sort(ui_elements)
        step_data['ui_elements'] = ui_elements

        # Handle screen tag mode
        if self.use_screen_tag:
            prompt = [
                ("user", {"text": usr_prompt, "img_index": 0}),
            ]
            asst_res, state = self.inference([image_path], prompt)
        else:
            raise ValueError("UI_Venus_Navi_Agent expects SCREEN_TAG inputs.")

        # Return assistant response
        # asst_head = asst_prompt if asst_prompt else ""         
        return None, asst_res, state
    
    
    def draw_focus(self, image_path, predict_point, max_radius_ratio=0.5, mask_color=(0, 0, 0, 150)):
        """
        根据图片和坐标点，绘制一个聚焦的圆形区域。

        Args:
            image_path (str): 输入图片的路径。
            predict_point (tuple): 聚焦点在图片上的坐标 (x, y)。
            max_radius_ratio (float, optional): 最大半径与图片最小边长的比例。默认为 0.5。
            mask_color (tuple, optional): 蒙版的颜色和透明度 (R, G, B, Alpha)。
                                        默认为半透明的黑色。

        Returns:
            PIL.Image.Image: 处理后带有聚焦效果的图片对象，失败时返回 None。
        """
        # 1. 检查文件是否存在
        if not os.path.exists(image_path):
            print(f"错误：找不到文件 {image_path}")
            return None
            
        # 2. 打开图片并转换为 RGBA 模式，以支持透明度处理
        try:
            base_image = Image.open(image_path).convert("RGBA")
        except Exception as e:
            print(f"错误：无法打开图片 {image_path}，错误信息：{e}")
            return None

        # 3. 获取图片尺寸和坐标点
        img_width, img_height = base_image.size
        cx, cy = int(predict_point[0]), int(predict_point[1])
        
        # 4. 检查坐标点是否在图片内
        if not (0 <= cx < img_width and 0 <= cy < img_height):
            print(f"错误：坐标点 {predict_point} 超出图片边界 ({img_width}x{img_height})。")
            return None  # 返回 None 而不是原图，因为这是一个错误情况

        # 5. 验证 max_radius_ratio 参数
        if not (0 < max_radius_ratio <= 1.0):
            print(f"错误：max_radius_ratio 必须在 0 到 1 之间，当前值：{max_radius_ratio}")
            return None

        # 6. 计算最大可能的半径，确保圆圈不超过图片边界
        # 计算中心点到四条边的距离
        dist_to_left = cx
        dist_to_right = img_width - cx
        dist_to_top = cy
        dist_to_bottom = img_height - cy
        
        # 半径不能超过到任何边界的距离
        max_radius_by_boundary = min(dist_to_left, dist_to_right, dist_to_top, dist_to_bottom)
        
        # 根据比例计算推荐半径
        recommended_radius = int(min(img_width, img_height) * max_radius_ratio)
        
        # 取两者中的较小值作为最终半径
        radius = min(max_radius_by_boundary, recommended_radius)
        
        # 确保半径至少为1像素
        if radius <= 0:
            radius = 1
            print(f"警告：聚焦点太靠近边缘，使用最小半径 {radius}")
        
        print(f"计算得出的聚焦半径: {radius} 像素")

        # 7. 创建一个全新的、与原图大小相同的透明图层作为蒙版
        mask_layer = Image.new("RGBA", base_image.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(mask_layer)

        # 8. 在蒙版上绘制半透明的蒙皮，并"挖"出一个完全透明的圆形区域
        # 首先，整个蒙版填充为半透明的颜色
        draw.rectangle([(0, 0), base_image.size], fill=mask_color)
        
        # 然后，在蒙版上绘制一个完全透明的圆形，这就是我们的"聚焦区域"
        # 计算圆形的边界框
        left = cx - radius
        top = cy - radius
        right = cx + radius
        bottom = cy + radius
        
        # 绘制透明圆形（聚焦区域）
        draw.ellipse((left, top, right, bottom), fill=(0, 0, 0, 0))

        # 9. 将处理好的蒙版层与原始图片合并
        result_image = Image.alpha_composite(base_image, mask_layer)

        return result_image.convert("RGB")  # 返回 RGB 格式的图片
    
    def get_focus_image(self, image_path, ui_path):
        os.makedirs(self.focus_dir, exist_ok=True)
        save_image_path = os.path.join(self.focus_dir, image_path.split('/')[-1])
        if os.path.exists(save_image_path):
            return save_image_path
        # 根据ui_path image_path得到匹配中心点
        import sys
        sys.path.append("/remote-home/lbsun/memory/0721/code/template_match/methods/self_tm")
        from generate_center_point import generate_center_point
        predict_center_point = generate_center_point(image_path, ui_path)
        print(imagesize.get(image_path))
        print(predict_center_point)
        # 根据中心点绘制蒙皮
        image_with_focus = self.draw_focus(image_path, predict_center_point[1])
        # 保存路径
        image_with_focus.save(save_image_path)
        print(f"focus image saved in {save_image_path}")
        return save_image_path

    def action_wo_history(self, step_data, *args, **kwargs):
        """ Chain of Action Thought """
        user_request, image_path = step_data['instruction'], step_data['image_full_path']
        history_actions = step_data['history_coat_actions']
        
        usr_prompt = self.prompts['OPENAI_ATLAS_PROMPT_WO_HISTORY']
        usr_prompt = usr_prompt.split("\n")
        if not history_actions: usr_prompt = [x for x in usr_prompt if 'history_actions' not in x]
        usr_prompt = "\n".join(usr_prompt)


        usr_prompt = usr_prompt.replace("{user_request}", kwargs['action_desc']['description'])
        action_space = self.prompts['ACTION_SPACE'][self.cfg['DATA']['DATASET']]
        usr_prompt = usr_prompt.replace("{action_space}", action_space)
        usr_prompt = [x for x in usr_prompt.split("{screenshot}")]

        # asst_prompt = self.prompts['ACTION_PREDICT']['SYSTEM']
        # return self.make_action(step_data, usr_prompt, kwargs['save_dir'], asst_prompt=asst_prompt)

        # Determine action mode and image path
        action_mode = self.cfg.DEMO_MODE.upper()
        image_path = step_data['image_full_path']

        # Prepare UI elements
        ui_elements = []
        ui_positions = step_data.get('ui_positions') or []
        ui_texts = step_data.get('ui_text') or []
        ui_types = step_data.get('ui_types')
        if ui_types is None:
            ui_types = ["unknown"] * len(ui_positions)
        if ui_positions and ui_texts and ui_types:
            for org_bbox, txt, ui_class in zip(ui_positions, ui_texts, ui_types):
                ymin, xmin, h, w = org_bbox
                bbox = [xmin, ymin, xmin + w, ymin + h]
                ui_elements.append({"bounds": bbox, "text": txt, "type": ui_class})
            ui_elements = row_col_sort(ui_elements)
        step_data['ui_elements'] = ui_elements

        # Handle screen tag mode
        if self.use_screen_tag:
            prompt = [
                ("user", {"text": usr_prompt[0], "img_index": 0}),
            ]
            for txt in usr_prompt[1:]:
                prompt.append(("user", {"text": txt}))
            asst_res, state = self.inference([image_path], prompt)
        else:
            raise ValueError("AtlasAgent requires SCREEN_TAG inputs.")

        # Return assistant response
        # asst_head = asst_prompt if asst_prompt else ""         
        return None, asst_res, state


    def action_wo_history_w_focus(self, step_data, *args, **kwargs):
        """ Chain of Action Thought """
        user_request, image_path = step_data['instruction'], step_data['image_full_path']

        history_actions = step_data['history_coat_actions']
        
        usr_prompt = self.prompts['OPENAI_ATLAS_PROMPT_WO_HISTORY']
        usr_prompt = usr_prompt.split("\n")
        if not history_actions: usr_prompt = [x for x in usr_prompt if 'history_actions' not in x]
        usr_prompt = "\n".join(usr_prompt)

        # print(kwargs)
        # exit()
        usr_prompt = usr_prompt.replace("{user_request}", kwargs['action_desc']['description'])

        if kwargs['action_desc']['type'].lower() == "click_element" and self.stationary_memory:
            # retrieve from memory
            doc = self.stationary_memory.rag.query(kwargs['action_desc']['description'], param=self.stationary_query_param)
            
            if doc is None:
                ui_path = None
            else:
                print("Stationary memory retrived!")
                ui_path = doc[0]['image']
                if "/home/slb/data" in ui_path:
                    ui_path = ui_path.replace("/home/slb/data", "/root/autodl-tmp/agent/agent/data")
                elif "/mnt/shanghai1/sunlibo/gui-odyssey" in ui_path:
                    ui_path = ui_path.replace("/mnt/shanghai1/sunlibo/gui-odyssey", "/root/autodl-tmp/agent/agent/data/gui_odyssey")
                elif "/root/autodl-tmp/data/gui_memory" in ui_path:
                    ui_path = ui_path.replace("/root/autodl-tmp/data/gui_memory", "/remote-home/lbsun/data")
                print("ui_path:", ui_path)
                assert os.path.exists(ui_path)

                image_path = self.get_focus_image(image_path, ui_path)
                kwargs['relevant_ui_path'] = ui_path
                kwargs['focus_image_path'] = image_path
                print("Image with focus:", image_path)

        action_space = self.prompts['ACTION_SPACE'][self.cfg['DATA']['DATASET']]
        usr_prompt = usr_prompt.replace("{action_space}", action_space)
        usr_prompt = [x for x in usr_prompt.split("{screenshot}")]

        # asst_prompt = self.prompts['ACTION_PREDICT']['SYSTEM']
        # return self.make_action(step_data, usr_prompt, kwargs['save_dir'], asst_prompt=asst_prompt)

        # Determine action mode and image path
        action_mode = self.cfg.DEMO_MODE.upper()
        # image_path = step_data['image_full_path']

        # Prepare UI elements
        ui_elements = []
        ui_positions = step_data.get('ui_positions') or []
        ui_texts = step_data.get('ui_text') or []
        ui_types = step_data.get('ui_types')
        if ui_types is None:
            ui_types = ["unknown"] * len(ui_positions)
        if ui_positions and ui_texts and ui_types:
            for org_bbox, txt, ui_class in zip(ui_positions, ui_texts, ui_types):
                ymin, xmin, h, w = org_bbox
                bbox = [xmin, ymin, xmin + w, ymin + h]
                ui_elements.append({"bounds": bbox, "text": txt, "type": ui_class})
            ui_elements = row_col_sort(ui_elements)
        step_data['ui_elements'] = ui_elements

        # Handle screen tag mode
        if self.use_screen_tag:
            prompt = [
                ("user", {"text": usr_prompt[0], "img_index": 0}),
            ]
            for txt in usr_prompt[1:]:
                prompt.append(("user", {"text": txt}))
            asst_res, state = self.inference([image_path], prompt)
        else:
            raise ValueError("AtlasAgent requires SCREEN_TAG inputs.")

        # Return assistant response
        # asst_head = asst_prompt if asst_prompt else ""         
        return kwargs, asst_res, state


    def parse_action(self, action_string):
        if not isinstance(action_string, str):
            raise ValueError("Action string must be a string.")

        action_string = action_string.strip()
        pattern_xy = re.compile(r"\(([-\d\.]+),\s*([-\d\.]+)\)")
        action_type = None
        action_args: dict[str, object] = {}

        def extract_xy(text: str):
            match = pattern_xy.search(text)
            if not match:
                return None
            return [int(float(match.group(1))), int(float(match.group(2)))]

        if action_string.lower().startswith("click"):
            action_type = "click_element"
            coords = extract_xy(action_string)
            if coords:
                action_args["xy"] = coords
        elif action_string.lower().startswith("longpress"):
            action_type = "long"
            coords = extract_xy(action_string)
            if coords:
                action_args["xy"] = coords
        elif action_string.lower().startswith("drag"):
            action_type = "scroll"
            start = extract_xy(action_string)
            end_match = re.search(r"end=\(([-\d\.]+),\s*([-\d\.]+)\)", action_string)
            if start and end_match:
                end = [int(float(end_match.group(1))), int(float(end_match.group(2)))]
                dx = end[0] - start[0]
                dy = end[1] - start[1]
                if abs(dx) > abs(dy):
                    direction = "right" if dx > 0 else "left"
                else:
                    direction = "down" if dy > 0 else "up"
                action_args = {"direction": direction, "start": start, "end": end}
        elif action_string.lower().startswith("scroll"):
            action_type = "scroll"
            direction_match = re.search(r"direction=['\"]?([a-zA-Z]+)['\"]?", action_string)
            direction = direction_match.group(1).lower() if direction_match else "down"
            action_args = {"direction": direction}
        elif action_string.lower().startswith("type"):
            action_type = "input"
            content_match = re.search(r"content=['\"](.*?)['\"]", action_string)
            action_args = {"text": content_match.group(1) if content_match else ""}
        elif action_string.lower().startswith("finished") or "complete" in action_string.lower() or "no further" in action_string.lower():
            action_type = "stop"
            action_args = {"task_status": "success"}
        elif action_string.lower().startswith("wait"):
            action_type = "stop"
            action_args = {"task_status": "pending"}
        elif action_string.lower().startswith("pressback"):
            action_type = "press_back"
        elif action_string.lower().startswith("presshome"):
            action_type = "press_home"
        elif action_string.lower().startswith("pressenter"):
            action_type = "press_enter"
        else:
            raise ValueError(f"Unsupported action string: {action_string}")

        return action_type, action_args

    # def parse_res(self, asst_res):
    #     if "actions:" not in asst_res:
    #         raise ValueError("No actions found in the response.")
    #     else:
    #         if "thoughts:" in asst_res:
    #             action_string = asst_res.replace("<|im_end|>", "").split("actions:")[-1]
    #             think_string = asst_res.split("actions:")[0].split("thoughts:")[-1]
    #         else:
    #             action_string = asst_res.replace("<|im_end|>", "").split("actions:")[-1]
    #             think_string = None
    #         action_type, action_args = self.parse_action(action_string)
            
    #         return {"ACTION":action_type, "ARGS":action_args}
    #         # return action_type, action_args

    def parse_res(self, asst_res):
        if "<think>" in asst_res and "</think>" in asst_res:
            think_string = asst_res.split("<think>")[1].split("</think>")[0].strip()
        if "<action>" in asst_res and "</action>" in asst_res:
            action_string = asst_res.split("<action>")[1].split("</action>")[0].strip()
            action_type, action_args = self.parse_action(action_string)
        
        return {"ACTION": action_type, "ARGS": action_args, "THINKING": think_string}
        
    def predict(self, step_data, save_dir, max_trials=5):
        subset, episode_id = step_data['subset'], step_data['episode_id']
        prev_step_id, step_id = step_data['prev_step_id'], step_data['step_id']
        action_predict_error_dir = os.path.join(save_dir, "action_predict_errors")
        os.makedirs(action_predict_error_dir, exist_ok=True)
        if subset==episode_id:
            save_dir = os.path.join(save_dir, episode_id)
        else:
            save_dir = os.path.join(save_dir, f"{subset}-{episode_id}")
        os.makedirs(save_dir, exist_ok=True)
        cur_save_path = os.path.join(save_dir, f"{subset}-{episode_id}_{step_id}.json")
        if os.path.exists(cur_save_path):
            cur_anno = json.load(open(cur_save_path, "r"))
        else:
            cur_anno = {}

        # if prev_step_id:
        #     prev_save_path = os.path.join(save_dir, f"{subset}-{episode_id}_{prev_step_id}.json")
        #     prev_anno = json.load(open(prev_save_path, "r"))
        #     cur_anno['prev_action_result'] = prev_anno['action_result']

        action_mode = self.cfg.DEMO_MODE.upper()
        if self.cfg['MODEL']['NAME'] in ["openai_atlas", "openai_atlas_base"]:
            if self.cfg['MEMORY']['STATIONARY']==False:
                func_handler = self.action_wo_history
            else:
                func_handler = self.action_wo_history_w_focus
        else:
            func_handler = self.think_action

        if 'action_predict' not in cur_anno: 
            cur_anno['action_predict'] = {}
            # print(save_dir)
        if action_mode not in cur_anno['action_predict']: cur_anno['action_predict'][action_mode] = {}
        # if action_mode in cur_anno['action_predict']: 
        #     anno = cur_anno['action_predict'][action_mode]
        #     cur_anno['action_predict'][action_mode] = {}
        #     cur_anno['action_predict'][action_mode][self.screen_mode] = anno
        #     json.dump(cur_anno, open(cur_save_path, "w", encoding="utf-8"), indent=4)
        
        if self.screen_mode not in cur_anno['action_predict'][action_mode] or cur_anno['action_predict'][action_mode][self.screen_mode]['ACTION']=="":
        # if True:
            print(f"[Mode {action_mode}][{self.screen_mode}] Gnerating action ... ({cur_save_path})")
            try_num = 0
            while try_num < max_trials:
                asst_head, asst_res, state = func_handler(step_data, **cur_anno, save_dir=save_dir)
                print(f"Assistant Response: {asst_res}")
                if state == HTTPStatus.OK: 
                    try:
                        try:
                            response = self.parse_res(asst_res)
                        except json.decoder.JSONDecodeError:
                            response = self._parse_action_response(asst_head, asst_res)
                            if response is None:
                                raise json.decoder.JSONDecodeError(
                                    "Unable to parse assistant response",
                                    self._strip_code_fences(asst_res),
                                    0,
                                )
                        if isinstance(response["ACTION"], str):
                            response["ACTION"] = response["ACTION"].lower()
                        elif isinstance(response["ACTION"], dict):
                            response["ARGS"] = response["ACTION"]["ARGS"]
                            response["ACTION"] = response["ACTION"]["ACTION"].lower()
                        if isinstance(response["THINKING"], str):
                            cur_anno["action_think"] = response["THINKING"].strip()
                        if 'bbox' in response["ARGS"]:
                            response["ARGS"]['bbox'] = self._preserve_model_bbox(response["ARGS"]['bbox'])
                        cur_anno['action_predict'][action_mode][self.screen_mode] = {
                            "ACTION": response["ACTION"], "ARGS": response["ARGS"]}
                        cur_anno['image_path'] = step_data['image_full_path']
                        json.dump(cur_anno, open(cur_save_path, "w", encoding="utf-8"), indent=4)
                    
                    except json.decoder.JSONDecodeError as e:
                        print('-'*50 + "\n", asst_head, sep=" ")
                        print('-'*50 + "\n", asst_res, sep=" ")
                    break
                elif state == HTTPStatus.REQUEST_ENTITY_TOO_LARGE: 
                    print(f"Trial {try_num} failured! -- Status {state}")
                    break
                print(f"Trial {try_num} failured! -- Status {state}")
                try_num += 1
                time.sleep(10)

        return None    
    
    def predict_with_memory(self, step_data, save_dir, max_trials=5):
        subset, episode_id = step_data['subset'], step_data['episode_id']
        prev_step_id, step_id = step_data['prev_step_id'], step_data['step_id']
        action_predict_error_dir = os.path.join(save_dir, "action_predict_errors")
        os.makedirs(action_predict_error_dir, exist_ok=True)
        if subset==episode_id:
            save_dir = os.path.join(save_dir, episode_id)
        else:
            save_dir = os.path.join(save_dir, f"{subset}-{episode_id}")
        os.makedirs(save_dir, exist_ok=True)
        cur_save_path = os.path.join(save_dir, f"{subset}-{episode_id}_{step_id}.json")
        if os.path.exists(cur_save_path):
            cur_anno = json.load(open(cur_save_path, "r"))
        else:
            cur_anno = {}

        # if prev_step_id:
        #     prev_save_path = os.path.join(save_dir, f"{subset}-{episode_id}_{prev_step_id}.json")
        #     prev_anno = json.load(open(prev_save_path, "r"))
        #     cur_anno['prev_action_result'] = prev_anno['action_result']

        action_mode = self.cfg.DEMO_MODE.upper()
        if self.cfg['MODEL']['NAME'] in ["openai_atlas", "openai_atlas_base"]:
            if self.cfg['MEMORY']['STATIONARY']==False:
                func_handler = self.action_wo_history
            else:
                func_handler = self.action_wo_history_w_focus
        else:
            func_handler = self.think_action

        if 'action_predict' not in cur_anno: 
            cur_anno['action_predict'] = {}
            # print(save_dir)
        if action_mode not in cur_anno['action_predict']: cur_anno['action_predict'][action_mode] = {}
        # if action_mode in cur_anno['action_predict']: 
        #     anno = cur_anno['action_predict'][action_mode]
        #     cur_anno['action_predict'][action_mode] = {}
        #     cur_anno['action_predict'][action_mode][self.screen_mode] = anno
        #     json.dump(cur_anno, open(cur_save_path, "w", encoding="utf-8"), indent=4)
        
        if self.screen_mode not in cur_anno['action_predict'][action_mode] or cur_anno['action_predict'][action_mode][self.screen_mode]['ACTION']=="":
        # if True:
            print(f"[Mode {action_mode}][{self.screen_mode}] Gnerating action ... ({cur_save_path})")
            try_num = 0
            while try_num < max_trials:
                asst_head, asst_res, state = func_handler(step_data, **cur_anno, save_dir=save_dir)
                print(asst_res)
                if state == HTTPStatus.OK: 
                    try:
                        try:
                            response = self.parse_res(asst_res)
                        except json.decoder.JSONDecodeError:
                            response = self._parse_action_response(asst_head, asst_res)
                            if response is None:
                                raise json.decoder.JSONDecodeError(
                                    "Unable to parse assistant response",
                                    self._strip_code_fences(asst_res),
                                    0,
                                )
                        if isinstance(response["ACTION"], str):
                            response["ACTION"] = response["ACTION"].lower()
                        elif isinstance(response["ACTION"], dict):
                            response["ARGS"] = response["ACTION"]["ARGS"]
                            response["ACTION"] = response["ACTION"]["ACTION"].lower()
                        if 'bbox' in response["ARGS"]:
                            response["ARGS"]['bbox'] = self._preserve_model_bbox(response["ARGS"]['bbox'])
                        cur_anno['action_predict'][action_mode][self.screen_mode] = {
                            "ACTION": response["ACTION"], "ARGS": response["ARGS"]}
                        if asst_head is not None:
                            if "relevant_ui_path" in asst_head:
                                cur_anno["relevant_ui_path"] = asst_head["relevant_ui_path"]
                            if "focus_image_path" in asst_head:
                                cur_anno["focus_image_path"] = asst_head["focus_image_path"]
                        
                        print(f"Updating [action_predict] {cur_save_path} ...")
                        json.dump(cur_anno, open(cur_save_path, "w", encoding="utf-8"), indent=4)
                        cur_anno["image_path"] = step_data["image_full_path"]


                    except json.decoder.JSONDecodeError as e:
                        print('-'*50 + "\n", asst_head, sep=" ")
                        print('-'*50 + "\n", asst_res, sep=" ")
                    break
                elif state == HTTPStatus.REQUEST_ENTITY_TOO_LARGE: 
                    print(f"Trial {try_num} failured! -- Status {state}")
                    break
                print(f"Trial {try_num} failured! -- Status {state}")
                try_num += 1
                time.sleep(10)

        return None    
 
    
class InfiGUI_R1_Agent(BaseAgent):
    def __init__(self, config):
        super().__init__(config)
        assert "infigui-r1" in self.cfg.MODEL.NAME.lower(), "Only support infigui_r1 model!"

        self.use_screen_tag = bool(self.cfg.DATA.USE_SCREEN_TAG)
        if not self.use_screen_tag:
            raise ValueError("InfiGUI_R1_Agent expects SCREEN_TAG inputs.")
        self.screen_mode = "tag"
        self.focus_dir = (
            f"/remote-home/lbsun/memory/0721/focus_images/{self.cfg.DATA.DATASET}/"
            f"statinoary{self.cfg.MEMORY.STATIONARY}_procedural{self.cfg.MEMORY.PROCEDURAL}"
        )


    def format_query(self, image_paths, prompt):
        # print(image_paths)
        # print(len(prompt))
        # print(prompt)
        
        if isinstance(prompt, str):
            image = Image.open(image_paths)
            return [prompt, image]
        
        if isinstance(image_paths, str): image_paths = [image_paths]

        if isinstance(prompt, list):
            user_message = {
                "role": "user",
                "content": [],
            }
            for role, msg in prompt:
                if isinstance(msg, dict) and "img_index" in msg:
                    img_index = msg.pop('img_index')
                    image = Image.open(image_paths[img_index])
                    # user_message['content'] += [
                    #     {"type": "text", "text": msg['text']},
                    #     {
                    #         "type": "image",
                    #         "image": image_paths[img_index]
                    #     }
                    # ]
                    with open(image_paths[img_index], "rb") as f:
                        image_b64 = base64.b64encode(f.read()).decode("utf-8")
                    user_message['content'] += [
                        {"type": "text", "text": msg['text']},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{image_b64}"}
                        }
                    ]
                else:
                    if isinstance(msg, dict): user_message['content'].append(
                        {"type": "text", "text": msg['text']}
                    )
                    else: user_message['content'].append(
                        {"type": "text", "text": msg}
                    )
            return [user_message]
        
    def inference(self, image_paths, prompt):
        messages = self.format_query(image_paths, prompt)
        # print(self.cfg.MODEL)
        # 转成 OpenAI Chat 格式
        payload = {
            # "model": "Qwen2.5-VL-32B",
            # "model": "Atlas-Base-7B",
            "model": self.cfg.MODEL.QWENVL_MODEL,
            "messages": messages,
            "max_tokens": 1280,
        }

        # 如果有图片，需要转成 base64 放进 messages 里
        # for image_path in image_paths:
        #     with open(image_path, "rb") as f:
        #         image_b64 = base64.b64encode(f.read()).decode("utf-8")
        #     payload["messages"][0]["content"].append({
        #         "type": "image_url",
        #         "image_url": {"url": f"data:image/png;base64,{image_b64}"}
        #     })

        # 调用本地 vLLM API
        resp = requests.post(
            "http://localhost:8182/v1/chat/completions",
            headers={"Content-Type": "application/json"},
            json=payload,
            timeout=600
        )

        if resp.status_code != 200:
            return f"Error: {resp.text}", resp.status_code

        output_text = resp.json()["choices"][0]["message"]["content"]
        return output_text, 200

    # def inference(self, image_paths, prompt):
    #     messages = self.format_query(image_paths, prompt)
    #     # print(image_paths)
    #     # print(prompt)
    #     # print(messages)
    #     # Preparation for inference
    #     text = self.processor.apply_chat_template(
    #         messages, tokenize=False, add_generation_prompt=True
    #     )
    #     # print(messages)
    #     image_inputs, video_inputs = process_vision_info(messages)
    #     inputs = self.processor(
    #         text=[text],
    #         images=image_inputs,
    #         videos=video_inputs,
    #         padding=True,
    #         return_tensors="pt",
    #     )
    #     inputs = inputs.to("cuda")

    #     # Inference: Generation of the output
    #     generated_ids = self.model.generate(**inputs, max_new_tokens=128)

    #     generated_ids_trimmed = [
    #         out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    #     ]

    #     output_text = self.processor.batch_decode(
    #         generated_ids_trimmed, skip_special_tokens=False, clean_up_tokenization_spaces=False
    #     )[0]
    #     return output_text, 200

    def think_action(self, step_data, *args, **kwargs):
        """ Chain of Action Thought """
        user_request, image_path = step_data['instruction'], step_data['image_full_path']
        history_actions = step_data['history_coat_actions']
        
        usr_prompt = self.prompts['INFIGUI_R1_PROMPT_E2E']
        usr_prompt = usr_prompt.split("\n")
        if not history_actions: usr_prompt = [x for x in usr_prompt if 'previous_actions' not in x]
        usr_prompt = "\n".join(usr_prompt)

        if history_actions: 
            history_action_string = ""
            for idx, history_action in enumerate(history_actions):
                step_idx = idx + 1
                history_action = f"Step {step_idx}: {history_action}\n"
                history_action_string += history_action
        else: history_action_string = "None"
        # print("usr_prompt", usr_prompt)
        usr_prompt = usr_prompt.replace("{history_actions}", history_action_string)
        usr_prompt = usr_prompt.replace("{user_query}", user_request)

        # asst_prompt = self.prompts['ACTION_PREDICT']['SYSTEM']
        # return self.make_action(step_data, usr_prompt, kwargs['save_dir'], asst_prompt=asst_prompt)

        # Determine action mode and image path
        action_mode = self.cfg.DEMO_MODE.upper()
        image_path = step_data['image_full_path']

        # Prepare UI elements
        ui_elements = []
        ui_positions = step_data.get('ui_positions') or []
        ui_texts = step_data.get('ui_text') or []
        ui_types = step_data.get('ui_types')
        if ui_types is None:
            ui_types = ["unknown"] * len(ui_positions)
        if ui_positions and ui_texts and ui_types:
            for org_bbox, txt, ui_class in zip(ui_positions, ui_texts, ui_types):
                ymin, xmin, h, w = org_bbox
                bbox = [xmin, ymin, xmin + w, ymin + h]
                ui_elements.append({"bounds": bbox, "text": txt, "type": ui_class})
            ui_elements = row_col_sort(ui_elements)
        step_data['ui_elements'] = ui_elements

        # Handle screen tag mode
        if self.use_screen_tag:
            prompt = [
                ("user", {"text": usr_prompt, "img_index": 0}),
            ]
            # print(f"prompt\n", prompt)
            # print("\n"*5)
            asst_res, state = self.inference([image_path], prompt)
        else:
            raise ValueError("InfiGUI_R1_Agent expects SCREEN_TAG inputs.")

        # Return assistant response
        # asst_head = asst_prompt if asst_prompt else ""         
        return None, asst_res, state
    

    def get_focus_image(self, image_path, ui_path):
        os.makedirs(self.focus_dir, exist_ok=True)
        save_image_path = os.path.join(self.focus_dir, image_path.split('/')[-1])
        if os.path.exists(save_image_path):
            return save_image_path
        # 根据ui_path image_path得到匹配中心点
        import sys
        sys.path.append("/remote-home/lbsun/memory/0721/code/template_match/methods/self_tm")
        from generate_center_point import generate_center_point
        predict_center_point = generate_center_point(image_path, ui_path)
        print(imagesize.get(image_path))
        print(predict_center_point)
        # 根据中心点绘制蒙皮
        image_with_focus = self.draw_focus(image_path, predict_center_point[1])
        # 保存路径
        image_with_focus.save(save_image_path)
        print(f"focus image saved in {save_image_path}")
        return save_image_path

    def parse_action(self, action_string):
        """
        Parse the tool_call JSON returned by InfiGUI-R1.

        Examples:
            {"name": "mobile_use", "arguments": {"action": "click", "coordinate": [964, 1308]}}
            {"name": "mobile_use", "arguments": {"action": "swipe", "coordinate": [413, 1211], "coordinate2": [415, 193]}}
            {"name": "mobile_use", "arguments": {"action": "type", "text": "ABC"}}
            {"name": "mobile_use", "arguments": {"action": "system_button", "button": "Back"}}
            {"name": "mobile_use", "arguments": {"action": "terminate", "status": "success"}}
        """
        if not isinstance(action_string, str):
            raise ValueError("Action string must be a string.")

        action_string = action_string.strip()
        try:
            tool_call = json.loads(action_string)
        except json.JSONDecodeError as exc:  # noqa: B904
            raise ValueError(f"Failed to parse tool call JSON: {action_string}") from exc

        arguments = tool_call.get("arguments", {})
        action_name = (arguments.get("action") or "").lower()
        if not action_name:
            raise ValueError(f"No 'action' field in tool call: {action_string}")

        def to_point(value: Any) -> list[int] | None:
            if isinstance(value, (list, tuple)) and len(value) >= 2:
                return [int(float(value[0])), int(float(value[1]))]
            return None

        action_type: str | None = None
        action_args: dict[str, object] = {}

        if action_name == "click":
            action_type = "click_element"
            coords = to_point(arguments.get("coordinate"))
            if coords:
                action_args["xy"] = coords
        elif action_name == "long_press":
            action_type = "long"
            coords = to_point(arguments.get("coordinate"))
            if coords:
                action_args["xy"] = coords
        elif action_name == "swipe":
            action_type = "scroll"
            start = to_point(arguments.get("coordinate"))
            end = to_point(arguments.get("coordinate2"))
            if start and end:
                dx = end[0] - start[0]
                dy = end[1] - start[1]
                if abs(dx) > abs(dy):
                    direction = "right" if dx > 0 else "left"
                else:
                    direction = "down" if dy > 0 else "up"
                action_args = {"direction": direction, "start": start, "end": end}
        elif action_name == "scroll":
            action_type = "scroll"
            direction = (arguments.get("direction") or "down").lower()
            action_args = {"direction": direction}
        elif action_name == "type":
            action_type = "input"
            action_args = {"text": arguments.get("text", "")}
        elif action_name == "key":
            key_text = (arguments.get("text") or "").lower()
            if "enter" in key_text:
                action_type = "press_enter"
            elif "home" in key_text:
                action_type = "press_home"
            elif "back" in key_text:
                action_type = "press_back"
        elif action_name == "system_button":
            button = (arguments.get("button") or "").lower()
            if button == "enter":
                action_type = "press_enter"
            elif button == "home":
                action_type = "press_home"
            elif button == "back":
                action_type = "press_back"
        elif action_name == "wait":
            action_type = "stop"
            action_args = {"task_status": "pending"}
        elif action_name == "terminate":
            action_type = "stop"
            action_args = {"task_status": arguments.get("status", "success")}

        if not action_type:
            raise ValueError(f"Unsupported action: {action_string}")

        return action_type, action_args

    # def parse_res(self, asst_res):
    #     if "actions:" not in asst_res:
    #         raise ValueError("No actions found in the response.")
    #     else:
    #         if "thoughts:" in asst_res:
    #             action_string = asst_res.replace("<|im_end|>", "").split("actions:")[-1]
    #             think_string = asst_res.split("actions:")[0].split("thoughts:")[-1]
    #         else:
    #             action_string = asst_res.replace("<|im_end|>", "").split("actions:")[-1]
    #             think_string = None
    #         action_type, action_args = self.parse_action(action_string)
            
    #         return {"ACTION":action_type, "ARGS":action_args}
    #         # return action_type, action_args

    def parse_res(self, asst_res):
        if "<think>" in asst_res and "</think>" in asst_res:
            think_string = asst_res.split("<think>")[1].split("</think>")[0].strip()
        if "<tool_call>" in asst_res and "</tool_call>" in asst_res:
            action_string = asst_res.split("<tool_call>")[1].split("</tool_call>")[0].strip()
            action_type, action_args = self.parse_action(action_string)
        
        return {"ACTION": action_type, "ARGS": action_args, "THINKING": think_string}
        
    def predict(self, step_data, save_dir, max_trials=5):
        subset, episode_id = step_data['subset'], step_data['episode_id']
        prev_step_id, step_id = step_data['prev_step_id'], step_data['step_id']
        action_predict_error_dir = os.path.join(save_dir, "action_predict_errors")
        os.makedirs(action_predict_error_dir, exist_ok=True)
        if subset==episode_id:
            save_dir = os.path.join(save_dir, episode_id)
        else:
            save_dir = os.path.join(save_dir, f"{subset}-{episode_id}")
        os.makedirs(save_dir, exist_ok=True)
        cur_save_path = os.path.join(save_dir, f"{subset}-{episode_id}_{step_id}.json")
        # print(f"cur_save_path", cur_save_path)
        if os.path.exists(cur_save_path):
            cur_anno = json.load(open(cur_save_path, "r"))
        else:
            cur_anno = {}

        # if prev_step_id:
        #     prev_save_path = os.path.join(save_dir, f"{subset}-{episode_id}_{prev_step_id}.json")
        #     prev_anno = json.load(open(prev_save_path, "r"))
        #     cur_anno['prev_action_result'] = prev_anno['action_result']

        action_mode = self.cfg.DEMO_MODE.upper()
        if self.cfg['MODEL']['NAME'] in ["openai_atlas", "openai_atlas_base"]:
            if self.cfg['MEMORY']['STATIONARY']==False:
                func_handler = self.action_wo_history
            else:
                func_handler = self.action_wo_history_w_focus
        else:
            func_handler = self.think_action

        if 'action_predict' not in cur_anno: 
            cur_anno['action_predict'] = {}
            # print(save_dir)
        if action_mode not in cur_anno['action_predict']: cur_anno['action_predict'][action_mode] = {}
        # if action_mode in cur_anno['action_predict']: 
        #     anno = cur_anno['action_predict'][action_mode]
        #     cur_anno['action_predict'][action_mode] = {}
        #     cur_anno['action_predict'][action_mode][self.screen_mode] = anno
        #     json.dump(cur_anno, open(cur_save_path, "w", encoding="utf-8"), indent=4)
        
        if self.screen_mode not in cur_anno['action_predict'][action_mode] or cur_anno['action_predict'][action_mode][self.screen_mode]['ACTION']=="":
        # if True:
            print(f"[Mode {action_mode}][{self.screen_mode}] Gnerating action ... ({cur_save_path})")
            try_num = 0
            while try_num < max_trials:
                asst_head, asst_res, state = func_handler(step_data, **cur_anno, save_dir=save_dir)
                print(f"Assistant Response: {asst_res}")
                if state == HTTPStatus.OK: 
                    try:
                        try:
                            response = self.parse_res(asst_res)
                        except json.decoder.JSONDecodeError:
                            response = self._parse_action_response(asst_head, asst_res)
                            if response is None:
                                raise json.decoder.JSONDecodeError(
                                    "Unable to parse assistant response",
                                    self._strip_code_fences(asst_res),
                                    0,
                                )
                        if isinstance(response["ACTION"], str):
                            response["ACTION"] = response["ACTION"].lower()
                        elif isinstance(response["ACTION"], dict):
                            response["ARGS"] = response["ACTION"]["ARGS"]
                            response["ACTION"] = response["ACTION"]["ACTION"].lower()
                        if isinstance(response["THINKING"], str):
                            cur_anno["action_think"] = response["THINKING"].strip()
                        if 'bbox' in response["ARGS"]:
                            response["ARGS"]['bbox'] = self._preserve_model_bbox(response["ARGS"]['bbox'])
                        cur_anno['action_predict'][action_mode][self.screen_mode] = {
                            "ACTION": response["ACTION"], "ARGS": response["ARGS"]}
                        cur_anno['image_path'] = step_data['image_full_path']
                        json.dump(cur_anno, open(cur_save_path, "w", encoding="utf-8"), indent=4)
                    
                    except json.decoder.JSONDecodeError as e:
                        print('-'*50 + "\n", asst_head, sep=" ")
                        print('-'*50 + "\n", asst_res, sep=" ")
                    break
                elif state == HTTPStatus.REQUEST_ENTITY_TOO_LARGE: 
                    print(f"Trial {try_num} failured! -- Status {state}")
                    break
                print(f"Trial {try_num} failured! -- Status {state}")
                try_num += 1
                time.sleep(10)

        return None    
    
    def predict_with_memory(self, step_data, save_dir, max_trials=5):
        subset, episode_id = step_data['subset'], step_data['episode_id']
        prev_step_id, step_id = step_data['prev_step_id'], step_data['step_id']
        action_predict_error_dir = os.path.join(save_dir, "action_predict_errors")
        os.makedirs(action_predict_error_dir, exist_ok=True)
        if subset==episode_id:
            save_dir = os.path.join(save_dir, episode_id)
        else:
            save_dir = os.path.join(save_dir, f"{subset}-{episode_id}")
        os.makedirs(save_dir, exist_ok=True)
        cur_save_path = os.path.join(save_dir, f"{subset}-{episode_id}_{step_id}.json")
        if os.path.exists(cur_save_path):
            cur_anno = json.load(open(cur_save_path, "r"))
        else:
            cur_anno = {}

        # if prev_step_id:
        #     prev_save_path = os.path.join(save_dir, f"{subset}-{episode_id}_{prev_step_id}.json")
        #     prev_anno = json.load(open(prev_save_path, "r"))
        #     cur_anno['prev_action_result'] = prev_anno['action_result']

        action_mode = self.cfg.DEMO_MODE.upper()
        if self.cfg['MODEL']['NAME'] in ["openai_atlas", "openai_atlas_base"]:
            if self.cfg['MEMORY']['STATIONARY']==False:
                func_handler = self.action_wo_history
            else:
                func_handler = self.action_wo_history_w_focus
        else:
            func_handler = self.think_action

        if 'action_predict' not in cur_anno: 
            cur_anno['action_predict'] = {}
            # print(save_dir)
        if action_mode not in cur_anno['action_predict']: cur_anno['action_predict'][action_mode] = {}
        # if action_mode in cur_anno['action_predict']: 
        #     anno = cur_anno['action_predict'][action_mode]
        #     cur_anno['action_predict'][action_mode] = {}
        #     cur_anno['action_predict'][action_mode][self.screen_mode] = anno
        #     json.dump(cur_anno, open(cur_save_path, "w", encoding="utf-8"), indent=4)
        
        if self.screen_mode not in cur_anno['action_predict'][action_mode] or cur_anno['action_predict'][action_mode][self.screen_mode]['ACTION']=="":
        # if True:
            print(f"[Mode {action_mode}][{self.screen_mode}] Gnerating action ... ({cur_save_path})")
            try_num = 0
            while try_num < max_trials:
                asst_head, asst_res, state = func_handler(step_data, **cur_anno, save_dir=save_dir)
                print("asst_res",asst_res)
                if state == HTTPStatus.OK: 
                    try:
                        try:
                            response = self.parse_res(asst_res)
                            print("response", response)
                        except json.decoder.JSONDecodeError:
                            response = self._parse_action_response(asst_head, asst_res)
                            if response is None:
                                raise json.decoder.JSONDecodeError(
                                    "Unable to parse assistant response",
                                    self._strip_code_fences(asst_res),
                                    0,
                                )
                        if isinstance(response["ACTION"], str):
                            response["ACTION"] = response["ACTION"].lower()
                        elif isinstance(response["ACTION"], dict):
                            response["ARGS"] = response["ACTION"]["ARGS"]
                            response["ACTION"] = response["ACTION"]["ACTION"].lower()
                        if 'bbox' in response["ARGS"]:
                            response["ARGS"]['bbox'] = self._preserve_model_bbox(response["ARGS"]['bbox'])
                        cur_anno['action_predict'][action_mode][self.screen_mode] = {
                            "ACTION": response["ACTION"], "ARGS": response["ARGS"]}
                        if asst_head is not None:
                            if "relevant_ui_path" in asst_head:
                                cur_anno["relevant_ui_path"] = asst_head["relevant_ui_path"]
                            if "focus_image_path" in asst_head:
                                cur_anno["focus_image_path"] = asst_head["focus_image_path"]
                        
                        print(f"Updating [action_predict] {cur_save_path} ...")
                        json.dump(cur_anno, open(cur_save_path, "w", encoding="utf-8"), indent=4)
                        cur_anno["image_path"] = step_data["image_full_path"]
                    
                    except json.decoder.JSONDecodeError as e:
                        print('-'*50 + "\n", asst_head, sep=" ")
                        print('-'*50 + "\n", asst_res, sep=" ")
                    break
                elif state == HTTPStatus.REQUEST_ENTITY_TOO_LARGE: 
                    print(f"Trial {try_num} failured! -- Status {state}")
                    break
                print(f"Trial {try_num} failured! -- Status {state}")
                try_num += 1
                time.sleep(10)

        return None    
 
