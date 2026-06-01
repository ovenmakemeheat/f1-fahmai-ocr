from __future__ import annotations

import json
import os
from collections.abc import Mapping
from typing import Any

SECRET_ENV_HINTS = ("KEY", "TOKEN", "SECRET", "PASSWORD", "CREDENTIAL")


def _secret_values() -> list[str]:
    values: list[str] = []
    for name, value in os.environ.items():
        if any(hint in name.upper() for hint in SECRET_ENV_HINTS) and len(value) >= 8:
            values.append(value)
    return values


def validate_prediction(prediction: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    if not isinstance(prediction, dict):
        return ["prediction must be a JSON object"]
    serialized = json.dumps(prediction, ensure_ascii=False)
    if "```" in serialized:
        errors.append("prediction contains markdown fence")
    for secret in _secret_values():
        if secret and secret in serialized:
            errors.append("prediction appears to contain an environment secret")
            break
    for key in prediction:
        if not isinstance(key, str) or not key.strip():
            errors.append("all keys must be non-empty strings")
            break
    return errors

