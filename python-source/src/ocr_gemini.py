from __future__ import annotations

import base64
import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SUPPORTED_MIME_TYPES = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
}


@dataclass(frozen=True)
class GeminiConfig:
    model: str
    api_key: str | None
    project: str | None
    location: str
    access_token: str | None
    timeout_seconds: int = 120

    @classmethod
    def from_env(cls) -> GeminiConfig:
        return cls(
            model=os.getenv("GEMINI_MODEL", "gemini-3.5-flash"),
            api_key=os.getenv("GEMINI_API_KEY"),
            project=os.getenv("GOOGLE_CLOUD_PROJECT"),
            location=os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1"),
            access_token=os.getenv("GOOGLE_OAUTH_ACCESS_TOKEN"),
            timeout_seconds=int(os.getenv("GEMINI_TIMEOUT_SECONDS", "120")),
        )


class GeminiClient:
    def __init__(self, config: GeminiConfig | None = None) -> None:
        self.config = config or GeminiConfig.from_env()
        if not self.config.api_key and not (self.config.project and self.config.access_token):
            raise RuntimeError(
                "Gemini auth is not configured. Set GEMINI_API_KEY, or set "
                "GOOGLE_CLOUD_PROJECT and GOOGLE_OAUTH_ACCESS_TOKEN for Vertex REST."
            )

    def _endpoint_and_headers(self) -> tuple[str, dict[str, str]]:
        model = self.config.model
        if self.config.api_key:
            endpoint = (
                "https://generativelanguage.googleapis.com/v1beta/models/"
                f"{model}:generateContent?key={self.config.api_key}"
            )
            return endpoint, {"Content-Type": "application/json"}

        endpoint = (
            f"https://{self.config.location}-aiplatform.googleapis.com/v1/"
            f"projects/{self.config.project}/locations/{self.config.location}/"
            f"publishers/google/models/{model}:generateContent"
        )
        return endpoint, {
            "Authorization": f"Bearer {self.config.access_token}",
            "Content-Type": "application/json",
        }

    def generate_json_from_image(self, prompt: str, image_path: Path) -> str:
        suffix = image_path.suffix.lower()
        mime_type = SUPPORTED_MIME_TYPES.get(suffix)
        if not mime_type:
            raise ValueError(f"Unsupported image type for Gemini request: {image_path}")
        image_b64 = base64.b64encode(image_path.read_bytes()).decode("ascii")
        payload = {
            "contents": [
                {
                    "role": "user",
                    "parts": [
                        {"text": prompt},
                        {"inlineData": {"mimeType": mime_type, "data": image_b64}},
                    ],
                }
            ],
            "generationConfig": {
                "temperature": 0,
                "responseMimeType": "application/json",
            },
        }
        return self._post_generate_content(payload)

    def _post_generate_content(self, payload: dict[str, Any]) -> str:
        endpoint, headers = self._endpoint_and_headers()
        body = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(endpoint, data=body, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(request, timeout=self.config.timeout_seconds) as response:
                response_body = response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Gemini HTTP {exc.code}: {detail[:1000]}") from exc
        value = json.loads(response_body)
        try:
            parts = value["candidates"][0]["content"]["parts"]
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError(f"Unexpected Gemini response shape: {response_body[:1000]}") from exc
        texts = [part.get("text", "") for part in parts if isinstance(part, dict)]
        return "\n".join(texts).strip()


def call_with_retries(
    client: GeminiClient,
    prompt: str,
    image_path: Path,
    max_retries: int = 3,
    base_sleep_seconds: float = 2.0,
) -> str:
    last_error: Exception | None = None
    for attempt in range(max_retries + 1):
        try:
            return client.generate_json_from_image(prompt, image_path)
        except Exception as exc:  # noqa: BLE001 - preserve retry context for CLI
            last_error = exc
            if attempt >= max_retries:
                break
            time.sleep(base_sleep_seconds * (2**attempt))
    raise RuntimeError(f"Gemini request failed after {max_retries + 1} attempts") from last_error
