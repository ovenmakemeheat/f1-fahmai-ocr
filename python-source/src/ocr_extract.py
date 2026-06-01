from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from .ocr_gemini import GeminiClient, call_with_retries
from .ocr_jsonl import read_jsonl
from .ocr_parse import parse_json_object
from .ocr_prompts import build_prompt
from .ocr_records import ImagePageRecord
from .ocr_validate import validate_prediction


def load_image_manifest(path: Path) -> list[ImagePageRecord]:
    return [ImagePageRecord.from_json(row) for row in read_jsonl(path)]


def group_images_by_artifact(
    records: list[ImagePageRecord],
) -> dict[str, list[ImagePageRecord]]:
    grouped: dict[str, list[ImagePageRecord]] = defaultdict(list)
    for record in records:
        grouped[record.artifact_id].append(record)
    for artifact_records in grouped.values():
        artifact_records.sort(key=lambda row: (row.page_index, row.image_index))
    return dict(grouped)


def merge_page_predictions(page_predictions: list[dict[str, Any]]) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    for index, prediction in enumerate(page_predictions):
        for key, value in prediction.items():
            if key not in merged:
                merged[key] = value
            else:
                merged[f"P{index + 1}_{key}"] = value
    return merged


class OcrExtractor:
    def __init__(
        self,
        client: GeminiClient,
        work_dir: Path,
        max_retries: int,
        force: bool = False,
    ) -> None:
        self.client = client
        self.work_dir = work_dir
        self.max_retries = max_retries
        self.force = force
        self.raw_dir = work_dir / "raw_responses"
        self.parsed_dir = work_dir / "parsed"
        self.validation_dir = work_dir / "validation"
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.parsed_dir.mkdir(parents=True, exist_ok=True)
        self.validation_dir.mkdir(parents=True, exist_ok=True)

    def should_skip(self, artifact_id: str) -> bool:
        return (self.parsed_dir / f"{artifact_id}.json").exists() and not self.force

    def extract_artifact(self, artifact_id: str, records: list[ImagePageRecord]) -> list[str]:
        page_predictions: list[dict[str, Any]] = []
        raw_payload: list[dict[str, Any]] = []
        for record in records:
            prompt = build_prompt(record.to_json())
            raw_text = call_with_retries(
                self.client,
                prompt,
                record.image_path,
                max_retries=self.max_retries,
            )
            page_predictions.append(parse_json_object(raw_text))
            raw_payload.append(
                {
                    "page_index": record.page_index,
                    "image_index": record.image_index,
                    "image_path": str(record.image_path),
                    "response": raw_text,
                }
            )

        merged = merge_page_predictions(page_predictions)
        errors = validate_prediction(merged)
        self._write_json(self.raw_dir / f"{artifact_id}.json", raw_payload)
        self._write_json(self.parsed_dir / f"{artifact_id}.json", merged, sort_keys=True)
        self._write_json(
            self.validation_dir / f"{artifact_id}.json",
            {"artifact_id": artifact_id, "errors": errors},
        )
        return errors

    @staticmethod
    def _write_json(path: Path, payload: Any, sort_keys: bool = False) -> None:
        with path.open("w", encoding="utf-8") as file:
            json.dump(payload, file, ensure_ascii=False, indent=2, sort_keys=sort_keys)


def extract_from_manifest(
    image_manifest: Path,
    work_dir: Path,
    artifact_type: str | None,
    limit: int | None,
    max_retries: int,
    force: bool,
) -> int:
    image_records = load_image_manifest(image_manifest)
    if artifact_type:
        image_records = [row for row in image_records if row.artifact_type == artifact_type]
    grouped = group_images_by_artifact(image_records)
    extractor = OcrExtractor(
        client=GeminiClient(),
        work_dir=work_dir,
        max_retries=max_retries,
        force=force,
    )

    processed = 0
    for artifact_id, records in grouped.items():
        if extractor.should_skip(artifact_id):
            continue
        if limit is not None and processed >= limit:
            break
        errors = extractor.extract_artifact(artifact_id, records)
        processed += 1
        print(f"{artifact_id}: {'valid' if not errors else 'errors=' + str(errors)}")
    return processed


def validate_parsed_dir(work_dir: Path) -> int:
    validation_dir = work_dir / "validation"
    validation_dir.mkdir(parents=True, exist_ok=True)
    failures = 0
    for parsed_path in sorted((work_dir / "parsed").glob("*.json")):
        with parsed_path.open("r", encoding="utf-8") as file:
            value = json.load(file)
        errors = validate_prediction(value)
        if errors:
            failures += 1
        with (validation_dir / parsed_path.name).open("w", encoding="utf-8") as file:
            json.dump(
                {"artifact_id": parsed_path.stem, "errors": errors},
                file,
                ensure_ascii=False,
                indent=2,
            )
    return failures

