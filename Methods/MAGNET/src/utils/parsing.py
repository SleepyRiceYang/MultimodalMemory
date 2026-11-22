from __future__ import annotations

import json


def extract_json_block(markdown: str):
    """Best-effort extraction of a JSON object from an LLM markdown response."""
    content = markdown.strip()
    if content.startswith("```json"):
        content = content[7:]
    if content.startswith("```JSON"):
        content = content[7:]
    if content.endswith("```"):
        content = content[:-3]

    if "```json" in content:
        content = content.split("```json", 1)[1]
    if "```JSON" in content:
        content = content.split("```JSON", 1)[1]
    if "```" in content:
        content = content.split("```", 1)[0]

    return json.loads(content)
