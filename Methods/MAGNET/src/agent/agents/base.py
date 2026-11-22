from __future__ import annotations

import json
import os
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path

import imagesize
from http import HTTPStatus
from PIL import Image

from memory.lightrag import GUIQueryParam, QueryParam

from ..memory.factory import build_procedural_memory, build_stationary_memory
from models import get_model
from utils.parsing import extract_json_block

SRC_ROOT = Path(__file__).resolve().parents[2]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


@dataclass(slots=True)
class EmbeddingConfig:
    model_name: str
    embed_dim: int
    max_token_size: int


class BaseAgent(object):
    ALLOWED_ACTION_TYPES = {
        "CLICK_ELEMENT",
        "SCROLL",
        "STOP",
        "INPUT",
        "PRESS_BACK",
        "PRESS_ENTER",
        "PRESS_HOME",
    }
    
    def __init__(self, config, *args, **kwargs) -> None:
        self.cfg = config
        self.prompts = config['PROMPTS']
        model_cfg = config["MODEL"]
        def _cfg_lookup(key: str):
            if isinstance(model_cfg, dict):
                return model_cfg.get(key)
            return getattr(model_cfg, key, None)
        model_log = {
            key: _cfg_lookup(key)
            for key in ("NAME", "OPENAI_API_URL", "OPENAI_API_KEY", "GEMINI_MODEL", "QWENVL_MODEL")
            if _cfg_lookup(key) is not None
        }
        print(f"[BaseAgent] Initializing VLM with config fields: {model_log}")
        self.vlm = get_model(model_cfg)
        self._log_dialog = bool(getattr(config, "LOG_PROMPTS", False))

        self.embedding_config = EmbeddingConfig(
            model_name="MiniCPM-Embedding",
            embed_dim=2304,
            max_token_size=128,
        )

        self.stationary_memory = build_stationary_memory(config, self.embedding_config)
        self.procedural_memory = build_procedural_memory(config, self.embedding_config)

        self.stationary_query_param = GUIQueryParam(mode="naive", top_n=20, top_k=20)
        self.procedural_query_param = QueryParam(mode="naive", top_n=5, top_k=5)

    def _log_prompt(self, prompt):
        if not self._log_dialog:
            return
        if prompt is None:
            print("[PROMPT] None")
            return
        if not isinstance(prompt, list):
            print("[PROMPT]", prompt)
            return
        print("=== Prompt Start ===")
        for idx, entry in enumerate(prompt):
            if not isinstance(entry, (list, tuple)) or len(entry) != 2:
                print(f"[{idx}] {entry}")
                continue
            role, payload = entry
            text = payload
            if isinstance(payload, dict):
                text = payload.get("text", payload)
            print(f"[{role}] {text}")
        print("=== Prompt End ===")

    def _log_response(self, response):
        if not self._log_dialog:
            return
        print("=== Response Start ===")
        print(response)
        print("=== Response End ===")

    def _preserve_model_bbox(self, bbox):
        """Keep model-provided coordinates untouched (swap order only for Gemini variants)."""
        if not isinstance(bbox, list) or len(bbox) < 4:
            return bbox
        model_name = (self.cfg['MODEL']['NAME'] or '').lower()
        if model_name == 'gemini-2.5-pro':
            return [bbox[1], bbox[0], bbox[3], bbox[2]]
        return bbox

    def _normalize_click_action(self, action_desc):
        """
        Model may emit type 'CLICK'; map it to the expected 'CLICK_ELEMENT'.
        """
        if not isinstance(action_desc, dict):
            return action_desc
        action_type = action_desc.get("type")
        if isinstance(action_type, str) and action_type.upper() == "CLICK":
            normalized = dict(action_desc)
            normalized["type"] = "CLICK_ELEMENT"
            return normalized
        return action_desc

    def _normalize_expected_action(self, action_desc):
        if not isinstance(action_desc, dict):
            return None
        action_type = action_desc.get("type")
        if not isinstance(action_type, str):
            return None
        expected = action_type.lower()
        if expected == "click":
            expected = "click_element"
        return expected

    def _normalize_predicted_action(self, predicted, expected=None):
        if not isinstance(predicted, str):
            return predicted
        normalized = predicted.lower()
        if expected == "click_element" and normalized == "tracker_results":
            normalized = "click_element"
        return normalized

    def _enforce_action_match(self, cur_anno, predicted, response, error_dir, subset, episode_id, step_id):
        expected = self._normalize_expected_action(cur_anno.get("action_desc"))
        if not expected:
            return predicted
        normalized = self._normalize_predicted_action(predicted, expected)
        if normalized == expected:
            return normalized
        error_path = os.path.join(error_dir, f"{subset}-{episode_id}_{step_id}.json")
        with open(error_path, "w") as fh:
            json.dump({"cur_anno": cur_anno, "response": response}, fh)
            print("save to", error_path)
        print(f"Predict action type mismatch: expected '{expected}', got '{normalized}'")
        raise ValueError(f"Action type mismatch: expected '{expected}', got '{normalized}'")

    def observe(self, user_request, screen_path, **kwargs):
        sys_rompt = self.prompts['SCREEN_DESC']['SYSTEM']
        usr_prompt = self.prompts['SCREEN_DESC']['USER']
        usr_prompt = [x.strip() for x in usr_prompt.split("{screenshot}")]

        prompt = [
            ("system", sys_rompt),
            ("user", {"text": usr_prompt[0], "img_index": 0})
        ]
        for txt in usr_prompt[1:]: prompt.append(("user", {"text": txt}))
        
        self._log_prompt(prompt)
        res, state = self.vlm.get_response(screen_path, prompt)
        self._log_response(res)
        return res, state

    def think_action(self, user_request, screen_path, screen_desc, history_actions, **kwargs):
        prompt_key = 'ACTION_THINK_DESC'
        sys_rompt = self.prompts[prompt_key]['SYSTEM']

        if history_actions: history_actions = ", ".join(history_actions)
        else: history_actions = "None"
        history_actions = history_actions + "."
        
        usr_prompt = self.prompts[prompt_key]['USER']
        usr_prompt = usr_prompt.replace("{screen_desc}", screen_desc)
        usr_prompt = usr_prompt.replace("{history_actions}", history_actions)
        usr_prompt = usr_prompt.replace("{user_request}", user_request)
        usr_prompt = [x.strip() for x in usr_prompt.split("{screenshot}")]

        prompt = [
            ("system", sys_rompt),
            ("user", {"text": usr_prompt[0], "img_index": 0})
        ]
        for txt in usr_prompt[1:]: prompt.append(("user", {"text": txt}))

        self._log_prompt(prompt)
        res, state = self.vlm.get_response(screen_path, prompt)
        self._log_response(res)
        return res, state

    def think(self, user_request, screen_path, screen_desc, history_actions, **kwargs):
        prompt_key = 'THINK_DESC'
        sys_rompt = self.prompts[prompt_key]['SYSTEM']

        if history_actions: history_actions = ", ".join(history_actions)
        else: history_actions = "None"
        history_actions = history_actions + "."
        
        usr_prompt = self.prompts[prompt_key]['USER']
        usr_prompt = usr_prompt.replace("{screen_desc}", screen_desc)
        usr_prompt = usr_prompt.replace("{history_actions}", history_actions)
        usr_prompt = usr_prompt.replace("{user_request}", user_request)
        usr_prompt = [x.strip() for x in usr_prompt.split("{screenshot}")]

        prompt = [
            ("system", sys_rompt),
            ("user", {"text": usr_prompt[0], "img_index": 0})
        ]
        for txt in usr_prompt[1:]: prompt.append(("user", {"text": txt}))
        # print("THINK")
        # print(prompt)
        self._log_prompt(prompt)
        res, state = self.vlm.get_response(screen_path, prompt)
        self._log_response(res)
        # print("THINK RES", res)
        return res, state
    
    def think_with_memory(self, user_request, screen_path, screen_desc, history_actions, retrival_results, **kwargs):
        prompt_key = 'THINK_DESC_WITH_MEMORY'
        sys_rompt = self.prompts[prompt_key]['SYSTEM']

        if history_actions: history_actions = ", ".join(history_actions)
        else: history_actions = "None"
        history_actions = history_actions + "."

        # if isinstance(retrival_results, list):
        #     retrival_result = retrival_results[0]
        # elif isinstance(retrival_results, dict):
        #     retrival_result = retrival_results
        # elif isinstance(retrival_results, str):
        #     retrival_result = {
        #         "instruction": "",
        #         "actions": []
        #     }
        # else:
        #     print("retrival_results", retrival_results, type(retrival_results))
        #     raise ValueError("retrival_results should be a list or dict")
        if isinstance(retrival_results, list):
            retrival_results = [
                {
                    "title":result['title'],
                    "actions":result['actions'],
                } for result in retrival_results
            ]
        elif isinstance(retrival_results, str):
            retrival_results = []
        else:
            print("retrival_results", retrival_results, type(retrival_results))
            raise ValueError("retrival_results should be a list or str")

        usr_prompt = self.prompts[prompt_key]['USER']
        usr_prompt = usr_prompt.replace("{screen_desc}", screen_desc)
        usr_prompt = usr_prompt.replace("{history_actions}", history_actions)
        usr_prompt = usr_prompt.replace("{user_request}", user_request)
        usr_prompt = usr_prompt.replace("{relevant_workflows}", json.dumps(retrival_results))
        # usr_prompt = usr_prompt.replace("{relevant_task}", retrival_result['instruction'])
        # usr_prompt = usr_prompt.replace("{relevant_task_solution}", str(retrival_result['actions']) if isinstance(retrival_result['actions'], list) else str(retrival_result['actions'].split("\n")))
        usr_prompt = [x.strip() for x in usr_prompt.split("{screenshot}")]

        prompt = [
            ("system", sys_rompt),
            ("user", {"text": usr_prompt[0], "img_index": 0})
        ]
        for txt in usr_prompt[1:]: prompt.append(("user", {"text": txt}))
        # print("THINK")
        # print(prompt)
        self._log_prompt(prompt)
        res, state = self.vlm.get_response(screen_path, prompt)
        self._log_response(res)
        # print("THINK RES", res)
        return res, state

    def reflect_result(self, user_request, screen_path, next_screen_path, last_action, **kwargs):
        sys_rompt = self.prompts['ACTION_RESULT']['SYSTEM']
        
        usr_prompt = self.prompts['ACTION_RESULT']['USER']
        usr_prompt = usr_prompt.replace("{last_action}", last_action)
        usr_prompt = usr_prompt.replace("{user_request}", user_request)

        usr_prompt = [x.strip() for x in usr_prompt.split("{before_screenshot}")]
        usr_prompt = [usr_prompt[0]] + [x.strip() for x in usr_prompt[1].split("{after_screenshot}")]

        prompt = [
            ("system", sys_rompt),
            ("user", {"text": usr_prompt[0], "img_index": 0}),
            ("user", {"text": usr_prompt[1], "img_index": 1})
        ]
        for txt in usr_prompt[2:]: prompt.append(("user", {"text": txt}))
        
        self._log_prompt(prompt)
        res, state = self.vlm.get_response([screen_path, next_screen_path], prompt)
        self._log_response(res)
        return res, state

    def _normalize_bbox(self, bbox, image_path):
        """
        Normalize bounding box coordinates to absolute pixel values.
        Supports both relative (0-1) and absolute (0-1000) formats.
        """
        assert len(bbox) >=4, f"Bounding box {bbox} is invalid."
        width, height = imagesize.get(image_path)
        xmin, ymin, xmax, ymax = bbox
        norm_max = max(bbox)
        if norm_max <= 1.5:
            xmin = int(xmin * width)
            xmax = int(xmax * width)
            ymin = int(ymin * height)
            ymax = int(ymax * height)
        else:
            xmin = int(xmin / 1000 * width)
            xmax = int(xmax / 1000 * width)
            ymin = int(ymin / 1000 * height)
            ymax = int(ymax / 1000 * height)
        xmin = max(0, min(width, xmin))
        xmax = max(0, min(width, xmax))
        ymin = max(0, min(height, ymin))
        ymax = max(0, min(height, ymax))
        if xmin >= xmax or ymin >= ymax:
            normalized = [xmin, ymin, xmax, ymax]
            raise ValueError(
                f"Bounding box {bbox} becomes invalid (non-positive width/height) after clamping to image size ({width}, {height}); normalized={normalized}"
            )
        return [xmin, ymin, xmax, ymax]

    def flow(self, step_data, save_dir, max_trials=5):
        subset, episode_id, step_id = step_data['subset'], step_data['episode_id'], step_data['step_id']
        user_request, image_path = step_data['instruction'], step_data['image_full_path']
        base_save_dir = save_dir
        action_desc_error_dir = os.path.join(base_save_dir, f"action_desc_errors")
        os.makedirs(action_desc_error_dir, exist_ok=True)
        if subset != episode_id:
            save_dir = os.path.join(save_dir, f"{subset}-{episode_id}")
        else:
            save_dir = os.path.join(save_dir, episode_id)
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"{subset}-{episode_id}_{step_id}.json")
        if not os.path.exists(save_path): json.dump({}, open(save_path, "w", encoding="utf-8"))

        prev_anno = json.load(open(save_path, "r"))
        # if 'action_result' not in prev_anno: return
        # del prev_anno['action_result']
        # json.dump(prev_anno, open(save_path, "w", encoding="utf-8"), indent=4)
        # return

        update = False
        if self.cfg.MODEL.NAME == "openai":
            image_path = self.compress_image(image_path, max_height=1440, mode="jpg")
        if 'screen_desc' not in prev_anno: 
            print(f"Genearting [screen_desc] -> img_path {image_path}")
            try_num = 0
            while try_num < max_trials:
                res, state = self.observe(user_request, image_path)
                # save the observe 
                observe_path = f"{save_dir}/{subset}-{episode_id}_{step_id}_0observe.json"
                with open(observe_path, "w", encoding="utf-8") as f:
                    json.dump(
                        {
                            "user_request":user_request, 
                            "image_path":image_path, 
                            "screen_desc": res
                        }, f)
                if state == HTTPStatus.OK: 
                    prev_anno['screen_desc'], update = res.strip(), True
                    break
                try_num += 1
                time.sleep(5)
        if update: print(f"Updating [screen_desc] {save_path} ...")
        if update: json.dump(prev_anno, open(save_path, "w", encoding="utf-8"), indent=4)
        
        update = False
        if 'action_think' not in prev_anno or 'action_desc' not in prev_anno:
            print(f"Genearting [action_think] -> img_path {image_path}")
            try_num = 0
            while try_num < max_trials:
                observe_path = f"{save_dir}/{subset}-{episode_id}_{step_id}_0observe.json"
                with open(observe_path, "r", encoding="utf-8") as f:
                    observe = json.load(f)
                prev_anno['screen_desc'] = observe['screen_desc']
                res, state = self.think_action(user_request, image_path, 
                                            screen_desc=prev_anno['screen_desc'], 
                                            history_actions=step_data['history_coat_actions'])
                if state == HTTPStatus.OK: 
                    try: 
                        response = extract_json_block(res)
                        # print(res)
                        # print(response)
                        # save the think action
                        think_action_path = f"{save_dir}/{subset}-{episode_id}_{step_id}_1think_action.json"
                        with open(think_action_path, "w", encoding="utf-8") as f:
                            json.dump(
                                {
                                    "user_request":user_request, 
                                    "image_path":image_path, 
                                    "screen_desc":prev_anno['screen_desc'],
                                    "history_actions":step_data['history_coat_actions'],
                                    "think_action": response
                                }, f)
                        
                        action_think = response.get('Thought')
                        action_plan = response.get('Future Action Plan')
                        action_desc = response.get('Next Single Step Action')
                        action_plan = [self._normalize_click_action(act) for act in action_plan] if action_plan else []
                        action_desc = self._normalize_click_action(action_desc)
                        # if not (action_think and action_plan and action_desc):
                        #     raise KeyError("Missing action_think/action_plan/action_desc fields.")
                        prev_anno['action_think'] = action_think
                        prev_anno['action_plan'] = action_plan if len(action_plan) > 0 else [action_desc]
                        prev_anno['action_desc'] = prev_anno['action_plan'][0]
                        act_type = action_desc.get("type")
                        if act_type not in self.ALLOWED_ACTION_TYPES:
                            raise ValueError(f"Unsupported action type '{act_type}' in response.")
                        if prev_anno['action_desc']['type']=='CLICK_ELEMENT':
                            print("action_desc", prev_anno['action_desc']['description'])
                        if action_desc['type'] in ['CLICK_ELEMENT', 'SCROLL', 'INPUT', 'PRESS_ENTER', 'PRESS_HOME', 'PRESS_BACK', 'STOP']:
                            update = True
                            break
                        action_desc_error_path = os.path.join(action_desc_error_dir, f"{subset}-{episode_id}_{step_id}_think.json")
                        with open(action_desc_error_path, "w") as f:
                            json.dump(prev_anno, f, indent=4)
                        update = False
                    except Exception as e: 
                        if not isinstance(e, json.decoder.JSONDecodeError): 
                            print(traceback.format_exc())
                        else: 
                            print(res,"\n", save_path, "\n", sep="")
                    try_num += 1
                    time.sleep(5)
                    continue
                print(f"Trial {try_num} failured! -- Status {state}")
                try_num += 1
                time.sleep(5)
        if update: print(f"Updating [action_think] {save_path} ...")
        if update: json.dump(prev_anno, open(save_path, "w", encoding="utf-8"), indent=4)
        
        if 'action_result' not in prev_anno:
            self.reflect(step_data, base_save_dir, max_trials=max_trials)

    def reflect(self, step_data, save_dir, max_trials=5):
        subset, episode_id, step_id = step_data['subset'], step_data['episode_id'], step_data['step_id']
        user_request, image_path = step_data['instruction'], step_data['image_full_path']
        base_dir = Path(save_dir)
        nested_name = f"{subset}-{episode_id}" if subset != episode_id else episode_id
        if base_dir.name == nested_name:
            target_dir = base_dir
        elif base_dir.name == episode_id and subset != episode_id:
            target_dir = base_dir
        else:
            target_dir = base_dir / nested_name
        target_dir.mkdir(parents=True, exist_ok=True)
        save_path = target_dir / f"{subset}-{episode_id}_{step_id}.json"
        if not os.path.exists(save_path):
            print(f"[reflect] Missing flow result for {save_path}, skipping.")
            return
        prev_anno = json.load(open(save_path, "r"))
        if 'action_result' in prev_anno:
            return

        print(f"Genearting [action_result] -> img_path {image_path}")
        update = False
        if step_data['result_action_type'] not in [10, 11] and step_data['next_image_full_path'] is not None:
            try_num = 0
            cur_image_path, next_image_path = image_path, step_data['next_image_full_path']
            while try_num < max_trials:
                cur_image_height = imagesize.get(cur_image_path)[1]
                next_image_height = imagesize.get(next_image_path)[1]
                res, state = self.reflect_result(
                    user_request,
                    screen_path=cur_image_path,
                    next_screen_path=next_image_path,
                    last_action=step_data['coat_action_desc'],
                )
                reflect_path = target_dir / f"{subset}-{episode_id}_{step_id}_reflect.jsonl"
                with open(reflect_path, "a+", encoding="utf-8") as f:
                    json.dump(
                        {
                            "user_request": user_request,
                            "screen_path": cur_image_path,
                            "next_screen_path": next_image_path,
                            "last_action": step_data['coat_action_desc'],
                            "reflect_result": res,
                        },
                        f,
                    )
                    f.write("\n")
                if state == HTTPStatus.OK:
                    prev_anno['action_result'], update = res, True
                    break
                elif state == HTTPStatus.REQUEST_ENTITY_TOO_LARGE:
                    cur_image_path = self.compress_image(image_path, max_height=int(0.8 * cur_image_height))
                    next_image_path = self.compress_image(
                        step_data['next_image_full_path'], max_height=int(0.8 * next_image_height)
                    )
                else:
                    print(res)
                print(f"Trial {try_num} failured! -- Status {state}")
                try_num += 1
                time.sleep(5)
        else:
            prev_anno['action_result'] = "The execution of user request is stopped."
            update = True
        if update:
            print(f"Updating [action_result] {save_path} ...")
            json.dump(prev_anno, open(save_path, "w", encoding="utf-8"), indent=4)

    # def flow_with_memory(self, step_data, save_dir, max_trials=5):
    #     subset, episode_id, step_id = step_data['subset'], step_data['episode_id'], step_data['step_id']
    #     user_request, image_path = step_data['instruction'], step_data['image_full_path']

    #     save_dir = os.path.join(save_dir, f"{subset}-{episode_id}")
    #     if not os.path.exists(save_dir): os.makedirs(save_dir)
    #     save_path = os.path.join(save_dir, f"{subset}-{episode_id}_{step_id}.json")
    #     if not os.path.exists(save_path): json.dump({}, open(save_path, "w", encoding="utf-8"))

    #     prev_anno = json.load(open(save_path, "r"))
    #     # if 'action_result' not in prev_anno: return
    #     # del prev_anno['action_result']
    #     # json.dump(prev_anno, open(save_path, "w", encoding="utf-8"), indent=4)
    #     # return

    #     update = False
    #     if 'screen_desc' not in prev_anno: 
    #         print(f"Genearting [screen_desc] -> img_path {image_path}")
    #         try_num = 0
    #         while try_num < max_trials:
    #             res, state = self.observe(user_request, image_path)
    #             # save the observe 
    #             observe_path = f"{save_dir}/{subset}-{episode_id}_{step_id}_0observe.json"
    #             with open(observe_path, "w", encoding="utf-8") as f:
    #                 json.dump(
    #                     {
    #                         "user_request":user_request, 
    #                         "image_path":image_path, 
    #                         "screen_desc": res
    #                     }, f)
    #             if state == HTTPStatus.OK: 
    #                 prev_anno['screen_desc'], update = res.strip(), True
    #                 break
    #             try_num += 1
    #             time.sleep(5)
    #     if update: print(f"Updating [screen_desc] {save_path} ...")
    #     if update: json.dump(prev_anno, open(save_path, "w", encoding="utf-8"), indent=4)
        
    #     update = False
    #     if 'action_think' not in prev_anno:
    #         print(f"Genearting [action_think] -> img_path {image_path}")
    #         try_num = 0
    #         while try_num < max_trials:
    #             res, state = self.think(user_request, image_path, 
    #                                     screen_desc=prev_anno['screen_desc'], 
    #                                     history_actions=step_data['history_coat_actions'])
    #             if state == HTTPStatus.OK: 
    #                 try: 
    #                     response = extract_json_block(res)
    #                     # save the think action
    #                     think_path = f"{save_dir}/{subset}-{episode_id}_{step_id}_1think.json"
    #                     with open(think_path, "w", encoding="utf-8") as f:
    #                         json.dump(
    #                             {
    #                                 "user_request":user_request, 
    #                                 "image_path":image_path, 
    #                                 "screen_desc":prev_anno['screen_desc'],
    #                                 "history_actions":step_data['history_coat_actions'],
    #                                 "think": response
    #                             }, f)
    #                     action_think = response['Thought']
    #                     action_plan = response['Future Action Plan']
    #                     # action_desc = response['Next Single Step Action']
    #                     prev_anno['action_think'] = action_think
    #                     prev_anno['action_plan'] = action_plan
    #                     # prev_anno['action_desc'] = action_desc
    #                     update = True
    #                     break
    #                 except Exception as e: 
    #                     if not isinstance(e, json.decoder.JSONDecodeError): print(traceback.format_exc())
    #                     else: print(res,"\n", save_path, "\n", sep="")
    #                 break
    #             print(f"Trial {try_num} failured! -- Status {state}")
    #             try_num += 1
    #             time.sleep(5)
    #     if update: print(f"Updating [action_think] {save_path} ...")
    #     if update: json.dump(prev_anno, open(save_path, "w", encoding="utf-8"), indent=4)
        
    #     update = False
    #     if 'action_desc' not in prev_anno:
    #         print(f"Genearting [action_desc] -> img_path {image_path}")
    #         try_num = 0
    #         while try_num < max_trials:
    #             action_type, query = prev_anno['action_plan'][0]["type"], prev_anno['action_plan'][0]["description"]
    #             if action_type in ["CLICK_ELEMENT"]:
    #                 doc = self.memory.rag.query(query, param=self.query_param)
    #             else:
    #                 doc = None
                    
    #             if doc is None:
    #                 ui_path = None
    #             else:
    #                 ui_path = doc[0]['image']
    #             # save doc
    #             query_path = f"{save_dir}/{subset}-{episode_id}_{step_id}_2query.json"
    #             with open(query_path, "w", encoding="utf-8") as f:
    #                 json.dump(
    #                     {
    #                         "query":query, 
    def flow_with_memory(self, step_data, save_dir, max_trials=5):
        subset, episode_id, step_id = step_data['subset'], step_data['episode_id'], step_data['step_id']
        user_request, image_path = step_data['instruction'], step_data['image_full_path']

        base_save_dir = save_dir
        action_desc_error_dir = os.path.join(base_save_dir, f"action_desc_errors")
        os.makedirs(action_desc_error_dir, exist_ok=True)
        if subset != episode_id:
            save_dir = os.path.join(save_dir, f"{subset}-{episode_id}")
        else:
            save_dir = os.path.join(save_dir, episode_id)
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"{subset}-{episode_id}_{step_id}.json")
        if not os.path.exists(save_path): json.dump({}, open(save_path, "w", encoding="utf-8"))

        prev_anno = json.load(open(save_path, "r"))
        # if 'action_result' not in prev_anno: return
        # del prev_anno['action_result']
        # json.dump(prev_anno, open(save_path, "w", encoding="utf-8"), indent=4)
        # return

        update = False
        if self.cfg.MODEL.NAME == "openai":
            image_path = self.compress_image(image_path, max_height=1440, mode="jpg")
        if 'screen_desc' not in prev_anno: 
            print(f"Genearting [screen_desc] -> img_path {image_path}")
            try_num = 0
            while try_num < max_trials:
                res, state = self.observe(user_request, image_path)
                # save the observe 
                observe_path = f"{save_dir}/{subset}-{episode_id}_{step_id}_0observe.json"
                with open(observe_path, "w", encoding="utf-8") as f:
                    json.dump(
                        {
                            "user_request":user_request, 
                            "image_path":image_path, 
                            "screen_desc": res
                        }, f)
                if state == HTTPStatus.OK: 
                    prev_anno['screen_desc'], update = res.strip(), True
                    break
                try_num += 1
                time.sleep(5)
        if update: print(f"Updating [screen_desc] {save_path} ...")
        if update: json.dump(prev_anno, open(save_path, "w", encoding="utf-8"), indent=4)
        
        update = False
        if 'action_think' not in prev_anno or 'action_desc' not in prev_anno:
            print(f"Genearting [action_think] -> img_path {image_path}")
            try_num = 0
            while try_num < max_trials:
                observe_path = f"{save_dir}/{subset}-{episode_id}_{step_id}_0observe.json"
                with open(observe_path, "r", encoding="utf-8") as f:
                    observe = json.load(f)
                prev_anno['screen_desc'] = observe['screen_desc']
                retrival_results = None
                if self.procedural_memory:
                    # retrival_task_results = self.procedural_memory_task.rag.query(user_request, param=self.procedural_query_param)
                    # retrival_screen_results = self.procedural_memory_screen.rag.query(prev_anno['screen_desc'], param=self.procedural_query_param)
                    # retrival_results = self.merge_retrival_results(retrival_task_results, retrival_screen_results)
                    # retrival_results = self.procedural_memory.rag.query(
                    #     user_request, prev_anno['screen_desc'], param=self.procedural_query_param
                    # )
                    print("Retrival results from procedural memory...")
                    retrival_results = self.procedural_memory.rag.query(
                        user_request, param=self.procedural_query_param
                    )
                    # print(f"\n\n\nuser_request\n{user_request}\nretrival_results\n{retrival_results}\n\n\n")
                    if retrival_results:
                        print("Procedural memory retrived!")
                        res, state = self.think_with_memory(user_request, image_path, 
                                                screen_desc=prev_anno['screen_desc'], 
                                                history_actions=step_data['history_coat_actions'],
                                                retrival_results=retrival_results)
                    else:
                        print("No procedural memory retrived!")
                        res, state = self.think(user_request, image_path, 
                                                screen_desc=prev_anno['screen_desc'], 
                                                history_actions=step_data['history_coat_actions'])
                else:
                    res, state = self.think(user_request, image_path, 
                                            screen_desc=prev_anno['screen_desc'], 
                                            history_actions=step_data['history_coat_actions'])
                if state == HTTPStatus.OK: 
                    try: 
                        response = extract_json_block(res)
                        # save the think action
                        think_path = f"{save_dir}/{subset}-{episode_id}_{step_id}_1think.json"
                        with open(think_path, "w", encoding="utf-8") as f:
                            json.dump(
                                {
                                    "user_request":user_request, 
                                    "image_path":image_path, 
                                    "screen_desc":prev_anno['screen_desc'],
                                    "history_actions":step_data['history_coat_actions'],
                                    "retrival_result":retrival_results,
                                    "think": response
                                }, f)
                        action_think = response['Thought']
                        action_plan = response['Future Action Plan']
                        action_plan = [self._normalize_click_action(act) for act in action_plan] if action_plan else []
                        # action_desc = response['Next Single Step Action']
                        prev_anno['action_think'] = action_think
                        prev_anno['action_plan'] = action_plan
                        prev_anno['action_desc'] = action_plan[0]
                        act_type = prev_anno['action_desc'].get("type")
                        if act_type not in self.ALLOWED_ACTION_TYPES:
                            raise ValueError(f"Unsupported action type '{act_type}' in response.")
                        if prev_anno['action_desc']['type'] in ['CLICK_ELEMENT', 'SCROLL', 'INPUT', 'PRESS_ENTER', 'PRESS_HOME', 'PRESS_BACK', 'STOP']:
                            update = True
                            break
                        else:
                            update = False
                            action_desc_error_path = os.path.join(action_desc_error_dir, f"{subset}-{episode_id}_{step_id}_think.json")
                            with open(action_desc_error_path, "w") as f:
                                json.dump(prev_anno, f, indent=4)
                    except Exception as e: 
                        if not isinstance(e, json.decoder.JSONDecodeError): print(traceback.format_exc())
                        else: print(res,"\n", save_path, "\n", sep="")
                    break
                print(f"Trial {try_num} failured! -- Status {state}")
                try_num += 1
                time.sleep(5)
        if update: print(f"Updating [action_think] {save_path} ...")
        if update: json.dump(prev_anno, open(save_path, "w", encoding="utf-8"), indent=4)

        if 'action_result' not in prev_anno:
            self.reflect(step_data, base_save_dir, max_trials=max_trials)


    def compress_image(self, image_path, max_height=1024, mode="scale"):
        org_img_path, ext = os.path.splitext(image_path)

        screen_img = Image.open(image_path)
        image_width, image_height = screen_img.size

        if image_height > max_height:
            scale_percent = max_height / image_height
            height = int(scale_percent * image_height)
            width = int(scale_percent * image_width)
            screen_img = screen_img.resize((width, height))
        
        if mode == "scale":
            save_img_path = org_img_path+"_tmp"+ext
            screen_img.save(save_img_path)
        elif mode == "jpg":
            screen_img = screen_img.convert("RGB")
            save_img_path = org_img_path+"_tmp"+".jpeg"
            screen_img.save(save_img_path)
        else:
            save_img_path = image_path

        return save_img_path
