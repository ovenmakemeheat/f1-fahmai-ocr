from __future__ import annotations

import os
import re
import time
from dataclasses import dataclass
from pathlib import Path

from tqdm import tqdm

from .ocr_config import load_dotenv

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
    auth_mode: str
    timeout_seconds: int = 120

    @classmethod
    def from_env(cls) -> GeminiConfig:
        load_dotenv()
        api_key = os.getenv("GEMINI_API_KEY")
        access_token = os.getenv("GOOGLE_OAUTH_ACCESS_TOKEN")
        if api_key and api_key.startswith("ya29."):
            access_token = access_token or api_key
            api_key = None
        return cls(
            model=os.getenv("GEMINI_MODEL", "gemini-3.5-flash"),
            api_key=api_key,
            project=os.getenv("GOOGLE_CLOUD_PROJECT"),
            location=os.getenv("GOOGLE_CLOUD_LOCATION", "global"),
            access_token=access_token,
            auth_mode=os.getenv("GOOGLE_GENAI_AUTH", "adc").lower(),
            timeout_seconds=int(os.getenv("GEMINI_TIMEOUT_SECONDS", "120")),
        )


class GeminiClient:
    def __init__(self, config: GeminiConfig | None = None) -> None:
        self.config = config or GeminiConfig.from_env()
        if not self.config.api_key and not self.config.project:
            raise RuntimeError(
                "Gemini auth is not configured. Set GOOGLE_CLOUD_PROJECT for Vertex AI "
                "with Application Default Credentials, or set GEMINI_API_KEY."
            )
        try:
            import google.auth
            from google import genai
            from google.genai import types
            from google.oauth2.credentials import Credentials
        except ImportError as exc:
            raise RuntimeError("google-genai is not installed. Run `uv sync`.") from exc
        self._genai = genai
        self._types = types
        http_options = types.HttpOptions(timeout=self.config.timeout_seconds * 1000)
        if self.config.project:
            credentials = None
            if self.config.auth_mode == "token":
                if not self.config.access_token:
                    raise RuntimeError(
                        "GOOGLE_GENAI_AUTH=token requires GOOGLE_OAUTH_ACCESS_TOKEN."
                    )
                credentials = Credentials(token=self.config.access_token)
            else:
                try:
                    credentials, _ = google.auth.default(
                        scopes=["https://www.googleapis.com/auth/cloud-platform"]
                    )
                except Exception as exc:
                    raise RuntimeError(
                        "Google Application Default Credentials were not found. Run "
                        "`gcloud auth application-default login` or set "
                        "GOOGLE_APPLICATION_CREDENTIALS to a service-account JSON file. "
                        "For explicit bearer-token mode, set GOOGLE_GENAI_AUTH=token, "
                        "but Vertex may reject unsupported token types."
                    ) from exc
            self._client = genai.Client(
                enterprise=True,
                credentials=credentials,
                project=self.config.project,
                location=self.config.location,
                http_options=http_options,
            )
        else:
            self._client = genai.Client(api_key=self.config.api_key, http_options=http_options)

    def _endpoint_and_headers(self) -> tuple[str, dict[str, str]]:
        """Kept for lightweight diagnostics; generation uses google-genai."""
        model = self.config.model
        if self.config.project:
            host = "aiplatform.googleapis.com"
            if self.config.location != "global":
                host = f"{self.config.location}-aiplatform.googleapis.com"
            endpoint = (
                f"https://{host}/v1/projects/{self.config.project}/"
                f"locations/{self.config.location}/publishers/google/models/{model}:generateContent"
            )
            return endpoint, {"Content-Type": "application/json"}

        if self.config.api_key:
            endpoint = (
                "https://generativelanguage.googleapis.com/v1beta/models/"
                f"{model}:generateContent?key={self.config.api_key}"
            )
            return endpoint, {"Content-Type": "application/json"}

        raise RuntimeError("Gemini auth is not configured.")

    def generate_json_from_image(self, prompt: str, image_path: Path) -> str:
        suffix = image_path.suffix.lower()
        mime_type = SUPPORTED_MIME_TYPES.get(suffix)
        if not mime_type:
            raise ValueError(f"Unsupported image type for Gemini request: {image_path}")
        response = self._client.models.generate_content(
            model=self.config.model,
            contents=[
                prompt,
                self._types.Part.from_bytes(data=image_path.read_bytes(), mime_type=mime_type),
            ],
            config=self._types.GenerateContentConfig(
                temperature=0,
                response_mime_type="application/json",
            ),
        )
        return (response.text or "").strip()


def _redact_secret_like_values(message: str) -> str:
    message = re.sub(r"ya29\.[A-Za-z0-9._-]+", "[REDACTED_OAUTH_TOKEN]", message)
    message = re.sub(r"AIza[0-9A-Za-z_-]+", "[REDACTED_API_KEY]", message)
    return message


def describe_error(exc: Exception) -> str:
    message = _redact_secret_like_values(str(exc))
    if not message:
        message = repr(exc)
    return f"{type(exc).__name__}: {message}"


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
            error_summary = describe_error(exc)
            if attempt >= max_retries:
                break
            sleep_seconds = base_sleep_seconds * (2**attempt)
            tqdm.write(
                f"Gemini request failed for {image_path.name} "
                f"(attempt {attempt + 1}/{max_retries + 1}); retrying in "
                f"{sleep_seconds:.1f}s: {error_summary}"
            )
            time.sleep(sleep_seconds)
    final_summary = describe_error(last_error) if last_error else "unknown error"
    raise RuntimeError(
        f"Gemini request failed for {image_path} after {max_retries + 1} attempts: "
        f"{final_summary}"
    ) from last_error
