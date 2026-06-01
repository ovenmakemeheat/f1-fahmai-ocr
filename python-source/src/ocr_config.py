from __future__ import annotations

import os
from pathlib import Path

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

