from memory.core.paths import dataset_path
from typing import TYPE_CHECKING, Optional, Tuple, Union, Callable, List, Any, Generator

import os
import random
import json
import numpy as np
import math
import imagesize

from torch.utils.data import Dataset
from collections import defaultdict
from tqdm import tqdm
from .action_utils import extract_gt_action, match_ui_bbox


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]



class AITZDataset(Dataset):
    """
    Creating a custom dataset for reading the processed AITW data
    with CoAT labeled detail semantic annotations.
    """
    DATASET_DIR = {
    'general': '{}/general',
    'google_apps': '{}/google_apps',
    'install': '{}/install',
    'single': '{}/single',
    'web_shopping': '{}/web_shopping',
    }

    def __init__(self, split="test", data_dir="data/android_in_the_zoo", ratio=1.0, chunks=1, chunk_idx=0, click_only=False, swipe_only=False) -> None:
        super().__init__()
        
        self.ratio = ratio
        self.num_chunks, self.chunk_idx = chunks, chunk_idx
        self.click_only, self.swipe_only = click_only, swipe_only
        
        self.data_dir = os.path.join(data_dir, split)
        self.episode_data = self._load_data_(ratio)
        
        raw_data = self._split_to_steps_(self.episode_data)
        assert not (click_only and swipe_only), "Cannot set both click_only and scroll_only to True!"
        if click_only: data = self.remove_non_click_steps(raw_data)
        elif swipe_only: data = self.remove_non_swipe_steps(raw_data)
        else: data = raw_data
        
        self.data = get_chunk(data, chunks, chunk_idx)

    def _load_data_(self, ratio=1.0): 
        valid_paths = defaultdict(list)
        for subset in self.DATASET_DIR:
            subdata_dir = self.DATASET_DIR[subset].format(self.data_dir)
            if os.path.exists(subdata_dir):
                sequence_names = os.listdir(subdata_dir)
                for seq_name in sequence_names:
                    seq_dir = os.path.join(subdata_dir, seq_name)
                    if not os.path.isdir(seq_dir): continue
                    episode_path = os.path.join(seq_dir, f"{seq_name}.json")
                    valid_paths[subset].append(episode_path)
        
        sampled_paths = []
        for subset, v_paths in valid_paths.items():
            N = len(v_paths)
            k = int(ratio * N)
            sampled_paths += random.sample(v_paths, k) if ratio < 1.0 else v_paths

        ep_data = []
        for episode_path in sampled_paths:
            episode_data = json.load(open(episode_path, "r"))                        
            ep_data.append(episode_data)
        return ep_data
    
    def _split_to_steps_(self, episode_data):
        data = []
        for edx, episode in enumerate(tqdm(episode_data)):
            history_plain_actions, history_coat_actions, history_action_res = [], [], []
            for idx, step in enumerate(episode):
                step['subset'] = step['image_path'].split('/')[0]
                step['image_full_path'] = os.path.join(self.data_dir, step['image_path'])
                step['prev_step_id'] = episode[idx-1]['step_id'] if idx > 0 else None
                next_img_path = os.path.join(self.data_dir, episode[idx+1]['image_path']) \
                    if idx + 1 < len(episode) else None
                step['next_image_full_path'] = next_img_path

                step['history_actions'] = history_plain_actions[:]
                step['history_coat_actions'] = history_coat_actions[:]
                step['history_action_results'] = history_action_res[:]

                if step["coat_action_desc"].split()[0].lower() == "scroll":
                    step["coat_action_desc"] = step["coat_action_desc"].replace("scroll", "swipe")

                for ui_key in ['result_lift_yx', 'result_touch_yx', 'ui_positions', 'ui_text']:
                    if 'ui_text' in ui_key:
                        step[ui_key] = step[ui_key][2:-2].split("', '")
                        # print(step[ui_key])
                        # print(type(step[ui_key]))
                        # step[ui_key] = step[ui_key][2:-2].split("', '")
                    else:
                        step[ui_key] = json.loads(step[ui_key])
                step['result_action'] = extract_gt_action(step)
                step['matched_ui_bbox'] = self.get_matched_ui_bbox(step)
                data.append(step)
                
                history_plain_actions.append(step['result_action'])
                history_coat_actions.append(step['coat_action_desc'])
                history_action_res.append(step["coat_action_result"])

        return data

    def __len__(self, ): return len(self.data)

    def __getitem__(self, index): return self.data[index]

    @staticmethod
    def format_gui_action(action_type, action_detail):
        if action_type == 'click':
            click_detail = np.array(action_detail, dtype=np.float32)
            if len(click_detail) == 2: 
                click_str = "x={:.2f}, y={:.2f}".format(*click_detail)
            elif len(click_detail) == 4: 
                click_str = 'bbox=[{:.2f}, {:.2f}, {:.2f}, {:.2f}]'.format(*click_detail)
            action_str = f'click({click_str})'
        elif action_type == 'swipe_finger':
            action_str = f'swipe(direction="{action_detail}")'
        elif action_type == 'set_text':
            action_str = f'set_text(text="{action_detail}")'
        elif action_type in {"done", "impossible"}:
            action_str = "done()"
        elif action_type == "press":
            action_str = f'press(system_button="{action_detail}")'
        else: 
            print(action_type)
            raise NotImplementedError

        return action_str

    def remove_non_click_steps(self, data):
        new_data = []
        for step in data:
            if step["coat_action_desc"].split()[0].lower() != "click":
                continue
            new_data.append(step)
        
        return new_data
    
    def remove_non_swipe_steps(self, data):
        new_data = []
        for step in data:
            if step["coat_action_desc"].split()[0].lower() != "swipe":
                continue
            new_data.append(step)
            print(dataset_path('android_in_the_zoo', 'train', step['image_path']), step['coat_action_desc'])
        
        return new_data

    def get_matched_ui_bbox(self, step) -> Optional[Tuple[int, int, int, int]]:
        width, height = imagesize.get(step['image_full_path'])

        if step['result_action'][0] == "click":
            ui_bbox = match_ui_bbox(step["result_lift_yx"], width, height, 
                                    annotation_abs_positions=step["ui_positions"])
            ui_bbox = None if ui_bbox[0] == -1 else ui_bbox
        else: ui_bbox = None

        return ui_bbox
