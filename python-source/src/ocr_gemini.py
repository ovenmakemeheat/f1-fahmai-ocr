from __future__ import annotations

import base64
import os
import random
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
    provider: str
    model: str
    api_key: str | None
    project: str | None
    location: str
    access_token: str | None
    auth_mode: str
    openrouter_api_key: str | None
    openrouter_model: str
    openrouter_base_url: str
    openrouter_site_url: str | None
    openrouter_app_name: str | None
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
            provider=os.getenv("OCR_PROVIDER", "vertex").lower(),
            model=os.getenv("GEMINI_MODEL", "gemini-3.5-flash"),
            api_key=api_key,
            project=os.getenv("GOOGLE_CLOUD_PROJECT"),
            location=os.getenv("GOOGLE_CLOUD_LOCATION", "global"),
            access_token=access_token,
            auth_mode=os.getenv("GOOGLE_GENAI_AUTH", "adc").lower(),
            openrouter_api_key=os.getenv("OPENROUTER_API_KEY"),
            openrouter_model=os.getenv("OPENROUTER_MODEL", "google/gemini-2.5-flash"),
            openrouter_base_url=os.getenv(
                "OPENROUTER_BASE_URL",
                "https://openrouter.ai/api/v1",
            ),
            openrouter_site_url=os.getenv("OPENROUTER_SITE_URL"),
            openrouter_app_name=os.getenv("OPENROUTER_APP_NAME", "fahmai-ocr-pipeline"),
            timeout_seconds=int(os.getenv("GEMINI_TIMEOUT_SECONDS", "120")),
        )


class GeminiClient:
    def __init__(self, config: GeminiConfig | None = None) -> None:
        self.config = config or GeminiConfig.from_env()
        self._provider = self.config.provider
        if self._provider == "openrouter":
            self._init_openrouter()
            return
        if self._provider not in {"vertex", "gemini"}:
            raise RuntimeError(
                f"Unsupported OCR_PROVIDER={self._provider!r}. Use 'vertex' or 'openrouter'."
            )
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

    def _init_openrouter(self) -> None:
        if not self.config.openrouter_api_key:
            raise RuntimeError("OpenRouter is not configured. Set OPENROUTER_API_KEY.")
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError("openai is not installed. Run `uv sync`.") from exc
        default_headers: dict[str, str] = {}
        if self.config.openrouter_site_url:
            default_headers["HTTP-Referer"] = self.config.openrouter_site_url
        if self.config.openrouter_app_name:
            default_headers["X-Title"] = self.config.openrouter_app_name
        self._openrouter_client = OpenAI(
            api_key=self.config.openrouter_api_key,
            base_url=self.config.openrouter_base_url,
            default_headers=default_headers or None,
            timeout=self.config.timeout_seconds,
        )

    def _endpoint_and_headers(self) -> tuple[str, dict[str, str]]:
        """Kept for lightweight diagnostics; generation uses google-genai."""
        if self._provider == "openrouter":
            return self.config.openrouter_base_url, {"Content-Type": "application/json"}
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
        if self._provider == "openrouter":
            return self._generate_json_from_image_openrouter(prompt, image_path)
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

    def _generate_json_from_image_openrouter(self, prompt: str, image_path: Path) -> str:
        suffix = image_path.suffix.lower()
        mime_type = SUPPORTED_MIME_TYPES.get(suffix)
        if not mime_type:
            raise ValueError(f"Unsupported image type for OpenRouter request: {image_path}")
        image_b64 = base64.b64encode(image_path.read_bytes()).decode("ascii")
        data_url = f"data:{mime_type};base64,{image_b64}"
        response = self._openrouter_client.chat.completions.create(
            model=self.config.openrouter_model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": data_url}},
                    ],
                }
            ],
            temperature=0,
            response_format={"type": "json_object"},
        )
        return (response.choices[0].message.content or "").strip()


def _redact_secret_like_values(message: str) -> str:
    message = re.sub(r"ya29\.[A-Za-z0-9._-]+", "[REDACTED_OAUTH_TOKEN]", message)
    message = re.sub(r"AIza[0-9A-Za-z_-]+", "[REDACTED_API_KEY]", message)
    message = re.sub(r"sk-or-v1-[A-Za-z0-9._-]+", "[REDACTED_OPENROUTER_KEY]", message)
    return message


def describe_error(exc: Exception) -> str:
    message = _redact_secret_like_values(str(exc))
    if not message:
        message = repr(exc)
    return f"{type(exc).__name__}: {message}"


def retry_sleep_seconds(exc: Exception, attempt: int, base_sleep_seconds: float) -> float:
    summary = describe_error(exc)
    if "429" in summary or "RESOURCE_EXHAUSTED" in summary:
        return min(60.0, 10.0 * (2**attempt)) + random.uniform(0.0, 3.0)
    return base_sleep_seconds * (2**attempt) + random.uniform(0.0, 0.5)


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
            sleep_seconds = retry_sleep_seconds(exc, attempt, base_sleep_seconds)
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
