from __future__ import annotations

import base64
import io
from typing import Any, Dict, List, Union

import numpy as np
from openai import OpenAI
from PIL import Image

from .base import BaseModel


class GeminiModel(BaseModel):
    """Gemini model proxy that reuses the OpenAI compatible endpoint."""

    def __init__(self, configs: Dict[str, str], seed: int = 2024) -> None:
        self.api_key = configs["OPENAI_API_KEY"]
        self.api_url = configs["OPENAI_API_URL"]
        self.model_name = configs.get("GEMINI_MODEL", "gemini-2.5-pro")
        self.seed = seed
        print(
            f"[GeminiModel] Initializing client with api_url={self.api_url}, "
            f"api_key={self.api_key}, model_name={self.model_name}"
        )
        self.client = OpenAI(api_key=self.api_key, base_url=self.api_url)

    def encode_image(self, image=None, image_path=None) -> str:
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

    def _as_messages(
        self, image_paths: List[str], prompt: Union[str, dict, list], detail: str = "high"
    ):
        if isinstance(image_paths, str):
            image_paths = [image_paths]

        if isinstance(prompt, str):
            message = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
            for img_path in image_paths:
                base64_image = self.encode_image(image_path=img_path)
                message[0]["content"].append(
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}", "detail": detail}}
                )
            return message

        if isinstance(prompt, list):
            message = []
            for role, msg in prompt:
                if isinstance(msg, dict) and "img_index" in msg:
                    img_index = msg.pop("img_index")
                    base64_image = self.encode_image(image_path=image_paths[img_index])
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
                        base64_image = self.encode_image(image_path=image_paths[img_index])
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
        **_: Any,
    ):
        history = history or []
        message = history + self._as_messages(image_path, prompt)
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=message,
            max_tokens=max_new_tokens,
            seed=self.seed,
            temperature=temperature,
            timeout=60 * 10,
        ).choices[0].message

        response_str = response.content
        history += message
        history += [{"role": "assistant", "content": response_str}]

        return response_str, 200, history
