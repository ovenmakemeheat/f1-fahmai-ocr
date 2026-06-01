from __future__ import annotations

import asyncio
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any

from tqdm import tqdm

from .ocr_gemini import GeminiClient, call_with_retries, call_with_retries_async
from .ocr_jsonl import read_jsonl
from .ocr_parse import parse_json_object
from .ocr_prompts import build_prompt
from .ocr_records import ImagePageRecord, portable_path
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


def default_worker_count() -> int:
    env_value = os.getenv("OCR_WORKERS")
    if env_value:
        return max(1, int(env_value))
    cpu_count = os.cpu_count() or 1
    return max(1, min(8, cpu_count))


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
        self.finished_dir = work_dir / "finished"
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.parsed_dir.mkdir(parents=True, exist_ok=True)
        self.validation_dir.mkdir(parents=True, exist_ok=True)
        self.finished_dir.mkdir(parents=True, exist_ok=True)

    def should_skip(self, artifact_id: str) -> bool:
        return is_artifact_finished(self.work_dir, artifact_id) and not self.force

    def extract_artifact(self, artifact_id: str, records: list[ImagePageRecord]) -> list[str]:
        page_predictions: list[dict[str, Any]] = []
        raw_payload: list[dict[str, Any]] = []
        for record in records:
            prompt = build_prompt(record.to_json())
            try:
                raw_text = call_with_retries(
                    self.client,
                    prompt,
                    record.image_path,
                    max_retries=self.max_retries,
                )
            except RuntimeError as exc:
                raise RuntimeError(
                    f"OCR failed for artifact={artifact_id}, page={record.page_index}, "
                    f"image={record.image_index}, path={portable_path(record.image_path)}: {exc}"
                ) from exc
            raw_payload.append(
                {
                    "page_index": record.page_index,
                    "image_index": record.image_index,
                    "image_path": portable_path(record.image_path),
                    "response": raw_text,
                }
            )
            self._write_json(self.raw_dir / f"{artifact_id}.json", raw_payload)
            try:
                page_predictions.append(parse_json_object(raw_text))
            except ValueError as exc:
                raise ValueError(
                    f"Could not parse JSON for artifact={artifact_id}, page={record.page_index}, "
                    f"image={record.image_index}. Raw response saved to "
                    f"{self.raw_dir / f'{artifact_id}.json'}"
                ) from exc

        merged = merge_page_predictions(page_predictions)
        errors = validate_prediction(merged)
        self._write_json(self.raw_dir / f"{artifact_id}.json", raw_payload)
        self._write_json(self.parsed_dir / f"{artifact_id}.json", merged, sort_keys=True)
        self._write_json(
            self.validation_dir / f"{artifact_id}.json",
            {"artifact_id": artifact_id, "errors": errors},
        )
        self.write_finished_marker(artifact_id, records, errors)
        return errors

    async def extract_artifact_async(
        self,
        artifact_id: str,
        records: list[ImagePageRecord],
    ) -> list[str]:
        page_predictions: list[dict[str, Any]] = []
        raw_payload: list[dict[str, Any]] = []
        for record in records:
            prompt = build_prompt(record.to_json())
            try:
                raw_text = await call_with_retries_async(
                    self.client,
                    prompt,
                    record.image_path,
                    max_retries=self.max_retries,
                )
            except RuntimeError as exc:
                raise RuntimeError(
                    f"OCR failed for artifact={artifact_id}, page={record.page_index}, "
                    f"image={record.image_index}, path={portable_path(record.image_path)}: {exc}"
                ) from exc
            raw_payload.append(
                {
                    "page_index": record.page_index,
                    "image_index": record.image_index,
                    "image_path": portable_path(record.image_path),
                    "response": raw_text,
                }
            )
            self._write_json(self.raw_dir / f"{artifact_id}.json", raw_payload)
            try:
                page_predictions.append(parse_json_object(raw_text))
            except ValueError as exc:
                raise ValueError(
                    f"Could not parse JSON for artifact={artifact_id}, page={record.page_index}, "
                    f"image={record.image_index}. Raw response saved to "
                    f"{self.raw_dir / f'{artifact_id}.json'}"
                ) from exc

        merged = merge_page_predictions(page_predictions)
        errors = validate_prediction(merged)
        self._write_json(self.raw_dir / f"{artifact_id}.json", raw_payload)
        self._write_json(self.parsed_dir / f"{artifact_id}.json", merged, sort_keys=True)
        self._write_json(
            self.validation_dir / f"{artifact_id}.json",
            {"artifact_id": artifact_id, "errors": errors},
        )
        self.write_finished_marker(artifact_id, records, errors)
        return errors

    def write_finished_marker(
        self,
        artifact_id: str,
        records: list[ImagePageRecord],
        errors: list[str],
    ) -> None:
        self._write_json(
            self.finished_dir / f"{artifact_id}.json",
            {
                "artifact_id": artifact_id,
                "status": "valid" if not errors else "completed_with_validation_errors",
                "errors": errors,
                "page_count": len(records),
                "raw_response_path": portable_path(self.raw_dir / f"{artifact_id}.json"),
                "parsed_path": portable_path(self.parsed_dir / f"{artifact_id}.json"),
                "validation_path": portable_path(self.validation_dir / f"{artifact_id}.json"),
            },
        )

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
    workers: int | None = None,
) -> int:
    image_records = load_image_manifest(image_manifest)
    if artifact_type:
        image_records = [row for row in image_records if row.artifact_type == artifact_type]
    grouped = group_images_by_artifact(image_records)
    worker_count = workers or default_worker_count()

    artifact_items: list[tuple[str, list[ImagePageRecord]]] = []
    skipped = 0
    grouped_items = list(grouped.items())
    if limit is not None:
        grouped_items = grouped_items[:limit]
    for artifact_id, records in grouped_items:
        if is_artifact_finished(work_dir, artifact_id) and not force:
            skipped += 1
            continue
        artifact_items.append((artifact_id, records))

    if skipped:
        tqdm.write(f"Skipping {skipped} finished artifacts")
    if not artifact_items:
        return 0

    if worker_count <= 1:
        return _extract_serial(
            artifact_items=artifact_items,
            work_dir=work_dir,
            max_retries=max_retries,
            force=force,
        )

    return asyncio.run(
        _extract_async(
            artifact_items=artifact_items,
            work_dir=work_dir,
            max_retries=max_retries,
            force=force,
            workers=worker_count,
        )
    )


def _extract_serial(
    artifact_items: list[tuple[str, list[ImagePageRecord]]],
    work_dir: Path,
    max_retries: int,
    force: bool,
) -> int:
    extractor = OcrExtractor(
        client=GeminiClient(),
        work_dir=work_dir,
        max_retries=max_retries,
        force=force,
    )

    processed = 0
    progress = tqdm(artifact_items, desc="extract", unit="artifact")
    for artifact_id, records in progress:
        progress.set_postfix_str(artifact_id, refresh=False)
        errors = extractor.extract_artifact(artifact_id, records)
        processed += 1
        status = "valid" if not errors else f"errors={len(errors)}"
        progress.write(f"{artifact_id}: {status}")
    return processed


async def _extract_async(
    artifact_items: list[tuple[str, list[ImagePageRecord]]],
    work_dir: Path,
    max_retries: int,
    force: bool,
    workers: int,
) -> int:
    processed = 0
    semaphore = asyncio.Semaphore(workers)
    extractor = OcrExtractor(
        client=GeminiClient(),
        work_dir=work_dir,
        max_retries=max_retries,
        force=force,
    )
    tasks = [
        asyncio.create_task(_extract_artifact_task(semaphore, extractor, artifact_id, records))
        for artifact_id, records in artifact_items
    ]
    progress = tqdm(
        asyncio.as_completed(tasks),
        total=len(tasks),
        desc=f"extract async x{workers}",
        unit="artifact",
    )
    for future in progress:
        artifact_id, errors = await future
        processed += 1
        progress.set_postfix_str(artifact_id, refresh=False)
        status = "valid" if not errors else f"errors={len(errors)}"
        progress.write(f"{artifact_id}: {status}")
    return processed


async def _extract_artifact_task(
    semaphore: asyncio.Semaphore,
    extractor: OcrExtractor,
    artifact_id: str,
    records: list[ImagePageRecord],
) -> tuple[str, list[str]]:
    async with semaphore:
        errors = await extractor.extract_artifact_async(artifact_id, records)
    return artifact_id, errors


def is_artifact_finished(work_dir: Path, artifact_id: str) -> bool:
    parsed_path = work_dir / "parsed" / f"{artifact_id}.json"
    finished_path = work_dir / "finished" / f"{artifact_id}.json"
    if finished_path.exists() and parsed_path.exists():
        return True
    if parsed_path.exists():
        _write_legacy_finished_marker(work_dir, artifact_id)
        return True
    return False


def _write_legacy_finished_marker(work_dir: Path, artifact_id: str) -> None:
    finished_dir = work_dir / "finished"
    finished_dir.mkdir(parents=True, exist_ok=True)
    marker_path = finished_dir / f"{artifact_id}.json"
    if marker_path.exists():
        return
    with marker_path.open("w", encoding="utf-8") as file:
        json.dump(
            {
                "artifact_id": artifact_id,
                "status": "completed_legacy",
                "errors": [],
                "parsed_path": portable_path(work_dir / "parsed" / f"{artifact_id}.json"),
            },
            file,
            ensure_ascii=False,
            indent=2,
        )


def validate_parsed_dir(work_dir: Path) -> int:
    validation_dir = work_dir / "validation"
    validation_dir.mkdir(parents=True, exist_ok=True)
    failures = 0
    parsed_paths = sorted((work_dir / "parsed").glob("*.json"))
    for parsed_path in tqdm(parsed_paths, desc="validate", unit="file"):
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
