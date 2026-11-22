from typing import TYPE_CHECKING, Optional, Tuple, Union, Callable, List, Any, Generator

import os
import random
import json

import numpy as np
from torch.utils.data import Dataset
from tqdm import tqdm
from collections import defaultdict



class GUIOdysseyDataset(Dataset):
    IMAGE_DIR = "{data_dir}/screenshots"
    ANNO_DIR = "{data_dir}/annotations_with_bbox"
    SPLIT_DIR = "{data_dir}/splits"

    def __init__(self, split_method, split="test", data_dir="data/GUI-Odyssey", ratio=1.0):
        super().__init__()

        self.split_dir = self.SPLIT_DIR.format(data_dir=data_dir)
        self.image_dir = self.IMAGE_DIR.format(data_dir=data_dir)
        self.anno_dir = self.ANNO_DIR.format(data_dir=data_dir)

        self.episode_data = self._load_data_(split_method, split)
        self.data = self._split_to_steps_(self.episode_data)
        
        self.ratio = ratio

    def _load_data_(self, split_method, split):
        data_split = json.load(open(f"{self.split_dir}/{split_method}_split.json"))[split]
        
        valid_episodes = defaultdict(list)
        for seq_file in tqdm(data_split):
            episode_data = json.load(open(os.path.join(self.anno_dir, seq_file)))
            task_category = episode_data['task_info']['category']
            
            valid_episodes[task_category].append(episode_data)
        
        sampled_episodes = []
        for category, eps in valid_episodes.items():
            N = len(eps)
            k=int(self.ratio*N)
            sampled_episodes += random.sample(eps, k) if self.ratio < 1.0 else eps

        return sampled_episodes

    def _split_to_steps_(self, episode_data, max_history_len=4):
        all_step_data = []
        for edx, episode in enumerate(tqdm(episode_data)):
            instruction = episode['task_info']['instruction']
            steps = episode['steps']

            history_screenshot, history_action = [], []
            for step in steps:
                step['instruction'] = instruction
                step['screenshot'] = os.path.join(self.image_dir, step['screenshot'])

                step["history_screenshot"] = history_screenshot[-max_history_len:]
                step["history_action"] = history_action[-max_history_len:]
                
                if step['action_plus']['action_type'] == "CLICK":
                    xy = (step['info'][0][0]/1000.0, step['info'][0][1]/1000.0) # x_center, y_center
                    step['action_plus']["action_point"] = xy
                
                all_step_data.append(step)

                history_screenshot.append(step['screenshot'])
                history_action.append(step['action_plus'])

        return all_step_data

    def __len__(self, ): return len(self.data)

    def __getitem__(self, index): return self.data[index]

    @staticmethod
    def format_gui_action(action, w, h):
        action_type, action_detail, true_or_fake = \
            action['action_type'].lower(), action['action_detail'], action['true_or_fake']
        
        if action_type == 'click':
            box = np.array(action_detail, dtype=np.float32)
            if len(box) == 4:  # [x1, y1, x2, y2]
                box = box / np.array([w, h, w, h])
                click_str = 'bbox=[{:.2f}, {:.2f}, {:.2f}, {:.2f}]'.format(*box)
            elif len(box) == 2: # (x_center, y_center)
                click_str = 'x={:.2f}, y={:.2f}'.format(*box)
            action_str = f'click({click_str})'
        elif action_type == 'swipe_screen':
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

