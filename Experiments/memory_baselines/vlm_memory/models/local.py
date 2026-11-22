from __future__ import annotations

import base64
import hashlib
import io
import json
import os
import time
import urllib.request
from pathlib import Path

from openai import OpenAI
from PIL import Image


def image_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def image_content(path: Path, max_side: int) -> tuple[dict, dict]:
    with Image.open(path) as source:
        original_size = source.size
        image = source.convert("RGB")
        image.thumbnail((max_side, max_side))
        stream = io.BytesIO()
        image.save(stream, format="PNG")
        data = base64.b64encode(stream.getvalue()).decode()
        metadata = {"path": str(path), "sha256": image_hash(path),
                    "original_size": list(original_size), "sent_size": list(image.size)}
    return {"type": "image_url", "image_url": {"url": "data:image/png;base64," + data}}, metadata


class LocalModel:
    def __init__(self, config: dict):
        self.config = config
        self.client = OpenAI(base_url=config["base_url"],
                             api_key=os.getenv(config["api_key_env"], "EMPTY"),
                             timeout=config["timeout"], max_retries=0)

    def generate(self, system: str, prompt: str, images: list[tuple[str, Path]],
                 max_tokens: int | None = None, json_mode: bool = False,
                 json_schema: dict | None = None,
                 progress_path: Path | None = None) -> dict:
        content = [{"type": "text", "text": prompt}]
        image_metadata = []
        for label, path in images:
            part, metadata = image_content(path, self.config["image_max_side"])
            content += [{"type": "text", "text": label}, part]
            image_metadata.append({"label": label, **metadata})
        started = time.monotonic()
        messages = [{"role": "system", "content": system}, {"role": "user", "content": content}]
        requested_max_tokens = max_tokens or self.config["max_tokens"]
        effective_max_tokens = requested_max_tokens
        input_token_count = None
        # Large output budgets must leave room for the actual multimodal input.
        # vLLM's tokenizer uses the same chat template and image processor as serving.
        if requested_max_tokens >= 32768 or self.config.get('check_context_budget', False):
            endpoint = self.config["base_url"].removesuffix("/v1").rstrip("/") + "/tokenize"
            request = urllib.request.Request(endpoint,
                data=json.dumps({"model": self.config["name"], "messages": messages}).encode(),
                headers={"Content-Type": "application/json",
                         "Authorization": "Bearer " + os.getenv(self.config["api_key_env"], "EMPTY")})
            with urllib.request.urlopen(request, timeout=self.config["timeout"]) as handle:
                tokenization = json.load(handle)
            input_token_count = tokenization["count"]
            input_limit = self.config.get('max_input_tokens')
            if input_limit is not None and input_token_count > input_limit:
                raise ValueError(f'Input token budget exceeded: {input_token_count} > {input_limit}; split the request')
            effective_max_tokens = min(requested_max_tokens,
                                       tokenization["max_model_len"] - input_token_count - self.config.get('context_reserve_tokens', 0))
            if effective_max_tokens <= 0:
                raise ValueError("Input exhausts the model context window")
            if effective_max_tokens < self.config.get('min_output_tokens', 1):
                raise ValueError('Insufficient remaining context for the minimum output budget; split the request')
        fragments, reasoning = [], []
        usage = {}
        finish_reason = None
        chunks = 0
        request_id = None
        last_report = started

        def record(status, error=None):
            if progress_path is None:
                return
            progress_path.parent.mkdir(parents=True, exist_ok=True)
            payload = {"status": status, "request_id": request_id,
                       "seconds": time.monotonic() - started,
                       "text": "".join(fragments), "reasoning": "".join(reasoning),
                       "stream_chunks": chunks, "usage": usage,
                       "finish_reason": finish_reason}
            if error is not None:
                payload["error"] = f"{type(error).__name__}: {error}"
            temporary = progress_path.with_suffix(progress_path.suffix + ".tmp")
            temporary.write_text(json.dumps(payload, ensure_ascii=False) + "\n")
            temporary.replace(progress_path)

        record("waiting")
        stream = None
        try:
            stream = self.client.chat.completions.create(
                model=self.config["name"], temperature=self.config["temperature"],
                max_tokens=effective_max_tokens,
                messages=messages, stream=True, stream_options={"include_usage": True},
                **({"response_format": {"type": "json_schema", "json_schema": {
                    "name": "prediction", "strict": True, "schema": json_schema}}} if json_schema else
                   ({"response_format": {"type": "json_object"}} if json_mode else {})),
            )
            for event in stream:
                request_id = event.id or request_id
                if event.usage is not None:
                    usage = event.usage.model_dump()
                for choice in event.choices:
                    if choice.index != 0:
                        continue
                    delta = choice.delta
                    if delta.content:
                        fragments.append(delta.content)
                    if getattr(delta, "reasoning_content", None):
                        reasoning.append(delta.reasoning_content)
                    finish_reason = choice.finish_reason or finish_reason
                    chunks += 1
                now = time.monotonic()
                if now - last_report >= 30:
                    print(f"GENERATION {request_id}: elapsed={now-started:.0f}s, "
                          f"chunks={chunks}, text_chars={sum(map(len, fragments))}, "
                          f"reasoning_chars={sum(map(len, reasoning))}", flush=True)
                    record("generating")
                    last_report = now
            if finish_reason is None:
                raise RuntimeError("Generation stream ended without a finish_reason; partial output is not a completed response")
            record("completed")
        except BaseException as exc:
            record("interrupted" if isinstance(exc, KeyboardInterrupt) else "error", exc)
            raise
        finally:
            if stream is not None:
                stream.close()
        return {"text": "".join(fragments), "finish_reason": finish_reason,
                "reasoning": "".join(reasoning), "request_id": request_id,
                "stream_chunks": chunks, "usage": usage,
                "transport": "stream", "transport_timeout_seconds": self.config.get("timeout"),
                "sdk_max_retries": 0,
                "seconds": time.monotonic() - started, "model": self.config["name"],
                "system": system, "prompt": prompt, "images": image_metadata,
                "requested_max_tokens": requested_max_tokens,
                "effective_max_tokens": effective_max_tokens,
                "tokenized_input_tokens": input_token_count,
                "context_limit": tokenization['max_model_len'] if input_token_count is not None else None,
                "json_mode": json_mode, "json_schema": json_schema}


def parse_object(text: str) -> dict:
    text = text.strip()
    decoder = json.JSONDecoder()
    for index, char in enumerate(text):
        if char == "{":
            try:
                value, _ = decoder.raw_decode(text[index:])
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict):
                return value
    raise ValueError("Model did not return a JSON object")
