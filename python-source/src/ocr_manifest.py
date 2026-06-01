from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from tqdm import tqdm

from .ocr_config import DEFAULT_OCR_DATA_DIR, DEFAULT_SAMPLE_PATH
from .ocr_jsonl import write_jsonl
from .ocr_records import ArtifactRecord, PageRecord


def load_sample_artifact_ids(sample_path: Path = DEFAULT_SAMPLE_PATH) -> list[str]:
    with sample_path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        if reader.fieldnames != ["artifact_id", "pred_json"]:
            raise ValueError(f"Unexpected sample columns in {sample_path}: {reader.fieldnames}")
        return [row["artifact_id"] for row in reader]


def _sidecar_paths(data_dir: Path) -> dict[str, tuple[str, Path]]:
    sidecars: dict[str, tuple[str, Path]] = {}
    per_artifact = data_dir / "per_artifact"
    for artifact_type_dir in sorted(per_artifact.iterdir()):
        if not artifact_type_dir.is_dir():
            continue
        for path in artifact_type_dir.glob("*.json"):
            sidecars[path.stem] = (artifact_type_dir.name, path)
    return sidecars


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        value = json.load(file)
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return value


def load_artifact_manifest(
    data_dir: Path = DEFAULT_OCR_DATA_DIR,
    sample_path: Path = DEFAULT_SAMPLE_PATH,
    limit: int | None = None,
    artifact_type: str | None = None,
) -> list[ArtifactRecord]:
    artifact_ids = load_sample_artifact_ids(sample_path)
    sidecars = _sidecar_paths(data_dir)
    records: list[ArtifactRecord] = []

    scan_ids = artifact_ids
    if limit is not None and artifact_type is None:
        scan_ids = artifact_ids[:limit]
    for artifact_id in tqdm(scan_ids, desc="manifest", unit="artifact"):
        if artifact_id not in sidecars:
            raise FileNotFoundError(f"No sidecar JSON found for artifact_id={artifact_id}")
        detected_type, sidecar_path = sidecars[artifact_id]
        if artifact_type and detected_type != artifact_type:
            continue

        sidecar = _read_json(sidecar_path)
        pages: list[PageRecord] = []
        visible_field_order: list[str] = []
        for page_index, page in enumerate(sidecar.get("pages", [])):
            output_path = page.get("output_path", "")
            if not output_path:
                raise ValueError(f"Missing output_path in {sidecar_path}")
            visible_fields = tuple(page.get("visible_fields", []))
            for field in visible_fields:
                if field not in visible_field_order:
                    visible_field_order.append(field)
            pages.append(
                PageRecord(
                    artifact_id=artifact_id,
                    artifact_type=detected_type,
                    page_index=page_index,
                    page_kind=str(page.get("page_kind", f"page_{page_index}")),
                    source_path=data_dir / output_path,
                    source_fact_table=str(page.get("source_fact_table", "")),
                    source_row_ids=tuple(str(v) for v in page.get("source_row_ids", [])),
                    visible_fields=visible_fields,
                )
            )

        records.append(
            ArtifactRecord(
                artifact_id=artifact_id,
                artifact_type=detected_type,
                renderer_template_id=str(sidecar.get("renderer_template_id", "")),
                template_version=str(sidecar.get("template_version", "")),
                pages=tuple(pages),
                expected_visible_fields=tuple(visible_field_order),
            )
        )
        if artifact_type is not None and limit is not None and len(records) >= limit:
            break
    return records


def manifest_to_jsonl(records: list[ArtifactRecord], output_path: Path) -> None:
    rows = (
        {
            "artifact_id": artifact.artifact_id,
            "artifact_type": artifact.artifact_type,
            "renderer_template_id": artifact.renderer_template_id,
            "template_version": artifact.template_version,
            "page_index": page.page_index,
            "page_kind": page.page_kind,
            "source_path": str(page.source_path),
            "source_fact_table": page.source_fact_table,
            "source_row_ids": list(page.source_row_ids),
            "visible_fields": list(page.visible_fields),
            "expected_visible_fields": list(artifact.expected_visible_fields),
        }
        for artifact in records
        for page in artifact.pages
    )
    write_jsonl(output_path, rows)
