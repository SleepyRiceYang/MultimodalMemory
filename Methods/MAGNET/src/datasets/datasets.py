from __future__ import annotations

import ast
import json
import os
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List
from tqdm import tqdm
from agent.config.schema import AgentConfig


@dataclass(slots=True)
class StepData:
    subset: str
    episode_id: str
    step_id: int
    instruction: str
    image_full_path: str
    prev_step_id: int | None
    next_image_full_path: str | None
    history_actions: list
    history_coat_actions: list
    result_action: str
    coat_action_desc: str
    ui_positions: list
    ui_text: list
    ui_types: list
    result_action_type: int | None = None
    result_action_text: str | None = None
    result_touch_yx: str | None = None
    result_lift_yx: str | None = None


class _BaseDataset:
    def __init__(self, split: str, data_dir: Path, ratio: float = 1.0, double_sample: bool = False) -> None:
        self.split = split
        self.data_dir = data_dir / split
        print("data_dir", self.data_dir)
        self.ratio = ratio
        self.double_sample = double_sample
        self.episode_data = self._load_data()
        self.data = self._split_to_steps(self.episode_data)

    def _load_data(self) -> List[dict]:
        raise NotImplementedError

    def _split_to_steps(self, episode_data: Iterable[dict]) -> List[dict]:
        raise NotImplementedError

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class AndroidInTheZooDataset(_BaseDataset):
    DATASET_DIR = {
        "general": "{}/general",
        "google_apps": "{}/google_apps",
        "install": "{}/install",
        "single": "{}/single",
        "web_shopping": "{}/web_shopping",
    }

    def _load_data(self):
        valid_paths: Dict[str, List[str]] = defaultdict(list)
        for subset, pattern in self.DATASET_DIR.items():
            subdata_dir = Path(pattern.format(self.data_dir))
            if not subdata_dir.exists():
                continue
            for seq_name in os.listdir(subdata_dir):
                seq_dir = subdata_dir / seq_name
                if not seq_dir.is_dir():
                    continue
                episode_path = seq_dir / f"{seq_name}.json"
                if episode_path.exists():
                    valid_paths[subset].append(str(episode_path))

        sampled_paths: List[str] = []
        for _, v_paths in valid_paths.items():
            n = len(v_paths)
            k = int(self.ratio * n)
            sampled_paths += random.sample(v_paths, k) if self.ratio < 1.0 else v_paths

        if self.double_sample:
            sampled_paths += sampled_paths

        return [json.load(open(p, "r")) for p in sampled_paths]

    def _split_to_steps(self, episodes):
        data = []
        for episode in episodes:
            history_plain_actions, history_coat_actions = [], []
            for idx, step in enumerate(episode):
                step["subset"] = step["image_path"].split("/")[0]
                step["image_full_path"] = str(self.data_dir / step["image_path"])
                step["prev_step_id"] = episode[idx - 1]["step_id"] if idx > 0 else None
                next_img_path = self.data_dir / episode[idx + 1]["image_path"] if idx + 1 < len(episode) else None
                step["next_image_full_path"] = str(next_img_path) if next_img_path else None
                step["history_actions"] = history_plain_actions[:]
                step["history_coat_actions"] = history_coat_actions[:]
                step["result_action"] = step.get("coat_action_desc")
                step["ui_positions"] = ast.literal_eval(step["ui_positions"])
                step["ui_text"] = ast.literal_eval(step["ui_text"])
                step["ui_types"] = ast.literal_eval(step["ui_types"])
                data.append(step)
                history_plain_actions.append(step["result_action"])
                history_coat_actions.append(step["coat_action_desc"])
        return data


class OdysseyDataset(_BaseDataset):
    def _load_data(self):
        valid_paths = []
        if self.data_dir.exists():
            for seq_name in os.listdir(self.data_dir):
                seq_dir = self.data_dir / seq_name
                if not seq_dir.is_dir():
                    continue
                episode_path = seq_dir / f"{seq_name}.json"
                valid_paths.append(str(episode_path))

        sampled_paths = random.sample(valid_paths, int(self.ratio * len(valid_paths))) if self.ratio < 1.0 else valid_paths
        return [json.load(open(p, "r")) for p in sampled_paths]

    def _split_to_steps(self, episodes):
        data = []
        for episode in episodes:
            history_plain_actions, history_coat_actions = [], []
            for idx, step in enumerate(episode):
                step["subset"] = step["image_path"].split("_")[0]
                step["image_full_path"] = str(self.data_dir / step["subset"] / step["image_path"])
                step["prev_step_id"] = episode[idx - 1]["step_id"] if idx > 0 else None
                next_img_path = (
                    self.data_dir / step["subset"] / episode[idx + 1]["image_path"] if idx + 1 < len(episode) else None
                )
                step["next_image_full_path"] = str(next_img_path) if next_img_path else None
                step["history_actions"] = history_plain_actions[:]
                step["history_coat_actions"] = history_coat_actions[:]
                step["result_action"] = step.get("coat_action_desc")
                step["ui_positions"] = (
                    json.loads(step["ui_positions"]) if isinstance(step["ui_positions"], str) else step["ui_positions"]
                )
                step["ui_text"] = (
                    step["ui_text"][2:-2].split("', '") if isinstance(step["ui_text"], str) else step["ui_text"]
                )
                step["ui_types"] = (
                    step["ui_type"][2:-2].split("', '") if isinstance(step.get("ui_type"), str) else step.get("ui_type")
                )
                data.append(step)
                history_plain_actions.append(step["result_action"])
                history_coat_actions.append(step["coat_action_desc"])
        return data


class AmexDataset(_BaseDataset):
    def _load_data(self):
        valid_paths = []
        if self.data_dir.exists():
            for seq_name in tqdm(os.listdir(self.data_dir)):
                seq_dir = self.data_dir / seq_name
                if not seq_dir.is_dir():
                    continue
                episode_path = seq_dir / f"{seq_name}.json"
                valid_paths.append(str(episode_path))

        sampled_paths = random.sample(valid_paths, int(self.ratio * len(valid_paths))) if self.ratio < 1.0 else valid_paths
        return [json.load(open(p, "r")) for p in tqdm(sampled_paths)]

    def _split_to_steps(self, episodes):
        data = []
        for episode in episodes:
            history_plain_actions, history_coat_actions = [], []
            for idx, step in enumerate(episode):
                step["subset"] = step["image_path"].split("_")[0]
                step["image_full_path"] = str(self.data_dir / step["subset"] / step["image_path"])
                step["prev_step_id"] = episode[idx - 1]["step_id"] if idx > 0 else None
                next_img_path = (
                    self.data_dir / step["subset"] / episode[idx + 1]["image_path"] if idx + 1 < len(episode) else None
                )
                step["next_image_full_path"] = str(next_img_path) if next_img_path else None
                step["history_actions"] = history_plain_actions[:]
                step["history_coat_actions"] = history_coat_actions[:]
                step["result_action"] = step.get("coat_action_desc")
                step["ui_positions"] = (
                    json.loads(step["ui_positions"]) if isinstance(step["ui_positions"], str) else step["ui_positions"]
                )
                data.append(step)
                history_plain_actions.append(step["result_action"])
                history_coat_actions.append(step["coat_action_desc"])
        return data


DATASET_REGISTRY = {
    "android-in-the-zoo": AndroidInTheZooDataset,
    "odyssey": OdysseyDataset,
    "amex": AmexDataset,
}


def build_dataset(cfg: AgentConfig, ratio: float = 1.0, double_sample: bool = False):
    dataset_name = cfg.DATA.DATASET.lower()
    dataset_cls = DATASET_REGISTRY.get(dataset_name)
    if dataset_cls is None:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    print(cfg.DATA)
    print(dataset_cls)
    return dataset_cls(
        split=cfg.DATA.SPLIT,
        data_dir=Path(cfg.DATA.DATA_DIR),
        ratio=ratio,
        double_sample=double_sample,
    )
