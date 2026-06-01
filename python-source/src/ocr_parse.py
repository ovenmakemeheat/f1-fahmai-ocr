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
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError("No JSON object found in model response")
    return stripped[start : end + 1]


def parse_json_object(text: str) -> dict[str, Any]:
    json_text = extract_json_text(text)
    value = json.loads(json_text)
    if not isinstance(value, dict):
        raise ValueError("Model response JSON must be an object")
    return value

