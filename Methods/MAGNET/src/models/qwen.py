from __future__ import annotations

import base64
import io
import os
import random
import tempfile
from functools import partial
from typing import Any, Dict, List, Union

import numpy as np
from http import HTTPStatus
from openai import OpenAI
from PIL import Image

from .base import BaseModel
from utils.image_ops import smart_resize


def _ensure_local_no_proxy():
    """Ensure local vLLM endpoints bypass any global HTTP proxy."""
    hosts = ["localhost", "127.0.0.1"]
    for key in ("NO_PROXY", "no_proxy"):
        current_entries = [entry.strip() for entry in os.environ.get(key, "").split(",") if entry.strip()]
        existing = set(current_entries)
        for host in hosts:
            if host not in existing:
                current_entries.append(host)
                existing.add(host)
        os.environ[key] = ",".join(current_entries) if current_entries else ",".join(hosts)


_ensure_local_no_proxy()

qwenvl_special_str = """
Your response should be strictly structured in JSON format, consisting of the following keys and corresponding content:
{
    "THINK": <Analyze the logic behind your the next single-step action.>, 
    "NEXT": <Describe the next single-step action in words, e.g. "click on the .... located at ...".>,
    "ACTION": <Specify the precise API function name without arguments to be called to complete the user request, e.g., click_element. Leave it a empty string "" if you believe none of the API function is suitable for the task or the task is complete.>,
    "ARGS": <Specify the precise arguments in a dictionary format of the selected API function to be called to complete the user request, e.g., {"bbox": [0, 1, 2, 3]}. Leave it a empty dictionary {} if you the API does not require arguments, or you believe none of the API function is suitable for the task, or the task is complete.>
}
You should output only one next single-step action.

<YOUR RESPONSE>: 
"""


class QwenVLModel(BaseModel):
    """Connector for Qwen-VL style endpoints."""

    def __init__(self, configs: Dict[str, str], seed: int = 2024, **kwargs) -> None:
        super().__init__()
        self.api_key = configs["QWENVL_API_KEY"]
        self.model = configs.get("QWENVL_MODEL", "qwen2.5-vl-72b-instruct")
        self.seed = seed
        # vLLM multi-port endpoints for basic load balancing
        self.api_url_candidates = [
            "http://localhost:8182/v1",
            # "http://localhost:8183/v1",
        ]
        self.clients = {
            url: OpenAI(api_key=self.api_key, base_url=url) for url in self.api_url_candidates
        }
        self.max_image_side = configs.get("MAX_IMAGE_SIDE", 1024)

    def encode_image(self, image=None, image_path=None):
        assert image is not None or image_path is not None, (
            "Please specify at least an image array or a path to image."
        )
        if image is not None:
            with io.BytesIO() as output_bytes:
                if isinstance(image, Image.Image):
                    pil_image = image
                elif isinstance(image, np.ndarray):
                    pil_image = Image.fromarray(image)
                else:
                    raise TypeError(f"Unsupported image type {type(image)}")
                pil_image.save(output_bytes, "PNG")
                bytes_data = output_bytes.getvalue()
            return base64.b64encode(bytes_data).decode("utf-8")
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")

    def scale_image(self, image):
        width, height = image.size
        resized_height, resized_width = smart_resize(
            height, width, factor=28, min_pixels=28 * 28, max_pixels=28 * 28 * 2000
        )
        return image.resize((resized_width, resized_height))

    def _as_messages(
        self, image_paths: List[str], prompt: Union[str, dict, list], detail: str = "high"
    ):
        if isinstance(image_paths, str):
            image_paths = [image_paths]

        if isinstance(prompt, str):
            message = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
            for img_path in image_paths:
                # image = self.scale_image(Image.open(img_path))
                image = Image.open(img_path)
                base64_image = self.encode_image(image=image)
                message[0]["content"].append(
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}", "detail": detail}}
                )
            return message

        if isinstance(prompt, list):
            message = []
            for role, msg in prompt:
                if isinstance(msg, dict) and "img_index" in msg:
                    img_index = msg.pop("img_index")
                    # image = self.scale_image(Image.open(image_paths[img_index]))
                    image = Image.open(image_paths[img_index])
                    base64_image = self.encode_image(image=image)
                    message.append(
                        {
                            "role": role,
                            "content": [
                                {"type": "text", "text": msg["text"]},
                                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}", "detail": detail}},
                            ],
                        }
                    )
                else:
                    content = msg if isinstance(msg, dict) else {"text": msg}
                    message.append({"role": role, "content": [{"type": "text", "text": content["text"]}]})
            return message

        if isinstance(prompt, dict):
            message = []
            for role in ["system", "user", "assistant"]:
                if role not in prompt:
                    continue
                entry = {"role": role, "content": []}
                payload = prompt[role]
                if isinstance(payload, str):
                    entry["content"] = payload
                else:
                    entry["content"].append({"type": "text", "text": payload["text"]})
                    if "img_index" in payload:
                        img_index = payload.pop("img_index")
                        # image = self.scale_image(Image.open(image_paths[img_index]))
                        image = Image.open(image_paths[img_index])
                        base64_image = self.encode_image(image=image)
                        entry["content"].append(
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}", "detail": detail}}
                        )
                message.append(entry)
            return message

        raise TypeError(f"Unsupported prompt type: {type(prompt)}")

    def format_query(self, image_paths: Union[str, List[str]], prompt: Union[str, dict, list], detail: str = "high"):
        return self._as_messages(image_paths, prompt, detail=detail)

    def get_response(
        self, image_path: str, prompt: Union[str, dict], temperature: float = 0.5, max_new_tokens: int = 4096, **kwargs: Any
    ):
        response_str, response_state, _ = self.chat(
            image_path=image_path,
            prompt=prompt,
            history=[],
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            **kwargs,
        )
        return response_str, response_state

    def chat(
        self,
        image_path: str,
        prompt: Union[str, dict],
        history=None,
        temperature: float = 0.5,
        max_new_tokens: int = 4096,
        **kwargs: Any,
    ):
        history = history or []
        message = history + self._as_messages(image_path, prompt)
        client = self.clients[random.choice(self.api_url_candidates)]
        response = client.chat.completions.create(
            model=self.model,
            messages=message,
            max_tokens=2000,
            temperature=1,
            timeout=60 * 10,
        )

        response_str = response.choices[0].message.content

        history += message
        history += [{"role": "assistant", "content": response_str}]

        return response_str, 200, history
