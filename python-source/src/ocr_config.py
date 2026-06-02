from __future__ import annotations

import os
from pathlib import Path


def load_dotenv(path: Path = Path(".env")) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, value = line.split("=", 1)
        name = name.strip()
        value = value.strip()
        if value[:1] not in {"'", '"'} and " #" in value:
            value = value.split(" #", 1)[0].rstrip()
        value = value.strip('"').strip("'")
        if name and name not in os.environ:
            os.environ[name] = value


load_dotenv()

OCR_ARTIFACT_TYPES = (
    "bank_statement",
    "e7_banner",
    "receipt",
    "t2_doc",
    "t3_doc",
    "vendor_invoice",
    "warranty_form",
)

DEFAULT_OCR_DATA_DIR = Path(
    "data/super-ai-engineer-season-6-fah-mai-the-finale-ocr/"
    "fahmai_renders_with_json/fahmai_renders_with_json"
)
DEFAULT_SAMPLE_PATH = Path(
    "data/super-ai-engineer-season-6-fah-mai-the-finale-ocr/sample__submission.csv"
)
DEFAULT_WORK_DIR = Path(os.getenv("OCR_OUTPUT_DIR", "work/ocr"))
