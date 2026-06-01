from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


def portable_path(path: Path) -> str:
    return path.as_posix()


@dataclass(frozen=True)
class PageRecord:
    artifact_id: str
    artifact_type: str
    page_index: int
    page_kind: str
    source_path: Path
    source_fact_table: str
    source_row_ids: tuple[str, ...]
    visible_fields: tuple[str, ...]


@dataclass(frozen=True)
class ArtifactRecord:
    artifact_id: str
    artifact_type: str
    renderer_template_id: str
    template_version: str
    pages: tuple[PageRecord, ...]
    expected_visible_fields: tuple[str, ...]


@dataclass(frozen=True)
class ImagePageRecord:
    artifact_id: str
    artifact_type: str
    renderer_template_id: str
    template_version: str
    page_index: int
    image_index: int
    page_kind: str
    source_path: Path
    image_path: Path
    width: int | None
    height: int | None
    sha256: str
    source_fact_table: str
    source_row_ids: tuple[str, ...]
    visible_fields: tuple[str, ...]
    expected_visible_fields: tuple[str, ...]

    @classmethod
    def from_json(cls, payload: dict[str, Any]) -> ImagePageRecord:
        return cls(
            artifact_id=str(payload["artifact_id"]),
            artifact_type=str(payload["artifact_type"]),
            renderer_template_id=str(payload.get("renderer_template_id", "")),
            template_version=str(payload.get("template_version", "")),
            page_index=int(payload["page_index"]),
            image_index=int(payload["image_index"]),
            page_kind=str(payload["page_kind"]),
            source_path=Path(str(payload["source_path"])),
            image_path=Path(str(payload["image_path"])),
            width=payload.get("width"),
            height=payload.get("height"),
            sha256=str(payload["sha256"]),
            source_fact_table=str(payload.get("source_fact_table", "")),
            source_row_ids=tuple(str(v) for v in payload.get("source_row_ids", [])),
            visible_fields=tuple(str(v) for v in payload.get("visible_fields", [])),
            expected_visible_fields=tuple(
                str(v) for v in payload.get("expected_visible_fields", [])
            ),
        )

    def to_json(self) -> dict[str, Any]:
        return {
            "artifact_id": self.artifact_id,
            "artifact_type": self.artifact_type,
            "renderer_template_id": self.renderer_template_id,
            "template_version": self.template_version,
            "page_index": self.page_index,
            "image_index": self.image_index,
            "page_kind": self.page_kind,
            "source_path": portable_path(self.source_path),
            "image_path": portable_path(self.image_path),
            "width": self.width,
            "height": self.height,
            "sha256": self.sha256,
            "source_fact_table": self.source_fact_table,
            "source_row_ids": list(self.source_row_ids),
            "visible_fields": list(self.visible_fields),
            "expected_visible_fields": list(self.expected_visible_fields),
        }
