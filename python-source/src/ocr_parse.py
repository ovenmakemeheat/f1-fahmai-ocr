from __future__ import annotations

import json
import re
from typing import Any

FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)


def extract_json_text(text: str) -> str:
    stripped = text.strip()
    fence_match = FENCE_RE.search(stripped)
    if fence_match:
        stripped = fence_match.group(1).strip()
    starts = [index for index in (stripped.find("{"), stripped.find("[")) if index != -1]
    if not starts:
        raise ValueError("No JSON object found in model response")
    return stripped[min(starts) :]


def parse_json_object(text: str) -> dict[str, Any]:
    json_text = extract_json_text(text)
    decoder = json.JSONDecoder()
    value, _ = decoder.raw_decode(json_text)
    if isinstance(value, list):
        return {"items": value}
    if not isinstance(value, dict):
        raise ValueError(f"Model response JSON must be an object, got {type(value).__name__}")
    return value
