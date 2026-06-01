from __future__ import annotations

import hashlib
import shutil
import struct
from pathlib import Path

from .ocr_jsonl import write_jsonl
from .ocr_records import ArtifactRecord, ImagePageRecord, PageRecord


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def png_size(path: Path) -> tuple[int | None, int | None]:
    with path.open("rb") as file:
        header = file.read(24)
    if len(header) >= 24 and header[:8] == b"\x89PNG\r\n\x1a\n" and header[12:16] == b"IHDR":
        return struct.unpack(">II", header[16:24])
    return None, None


def image_path_for_page(
    output_dir: Path,
    page: PageRecord,
    pdf_page_index: int | None = None,
) -> Path:
    safe_kind = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in page.page_kind)
    suffix = f"{page.page_index:03d}__{safe_kind}"
    if pdf_page_index is not None:
        suffix += f"__pdf_{pdf_page_index + 1:03d}"
    return output_dir / page.artifact_type / page.artifact_id / f"{suffix}.png"


def copy_png(page: PageRecord, output_dir: Path, overwrite: bool = False) -> list[Path]:
    destination = image_path_for_page(output_dir, page)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if overwrite or not destination.exists():
        shutil.copy2(page.source_path, destination)
    return [destination]


def render_pdf(
    page: PageRecord,
    output_dir: Path,
    dpi: int = 250,
    overwrite: bool = False,
) -> list[Path]:
    try:
        import fitz  # type: ignore[import-not-found]
    except ImportError as exc:
        raise RuntimeError(
            "PDF conversion requires PyMuPDF. Install it with `uv add pymupdf` "
            "or convert PDFs before running prepare-images."
        ) from exc

    output_paths: list[Path] = []
    zoom = dpi / 72
    matrix = fitz.Matrix(zoom, zoom)
    with fitz.open(page.source_path) as document:
        for pdf_page_index in range(document.page_count):
            destination = image_path_for_page(output_dir, page, pdf_page_index)
            destination.parent.mkdir(parents=True, exist_ok=True)
            if overwrite or not destination.exists():
                pixmap = document[pdf_page_index].get_pixmap(matrix=matrix, alpha=False)
                pixmap.save(destination)
            output_paths.append(destination)
    return output_paths


def prepare_page_images(
    page: PageRecord,
    output_dir: Path,
    dpi: int = 250,
    overwrite: bool = False,
) -> list[Path]:
    suffix = page.source_path.suffix.lower()
    if suffix == ".png":
        return copy_png(page, output_dir, overwrite=overwrite)
    if suffix == ".pdf":
        return render_pdf(page, output_dir, dpi=dpi, overwrite=overwrite)
    raise ValueError(f"Unsupported render type for {page.source_path}")


def prepare_images(
    artifacts: list[ArtifactRecord],
    output_dir: Path,
    manifest_path: Path,
    dpi: int = 250,
    overwrite: bool = False,
) -> None:
    write_jsonl(
        manifest_path,
        (
            image_record.to_json()
            for artifact in artifacts
            for page in artifact.pages
            for image_record in _prepare_image_records(
                artifact,
                page,
                output_dir,
                dpi=dpi,
                overwrite=overwrite,
            )
        ),
    )


def _prepare_image_records(
    artifact: ArtifactRecord,
    page: PageRecord,
    output_dir: Path,
    dpi: int,
    overwrite: bool,
) -> list[ImagePageRecord]:
    image_paths = prepare_page_images(page, output_dir, dpi=dpi, overwrite=overwrite)
    records: list[ImagePageRecord] = []
    for image_index, image_path in enumerate(image_paths):
        width, height = png_size(image_path)
        records.append(
            ImagePageRecord(
                artifact_id=artifact.artifact_id,
                artifact_type=artifact.artifact_type,
                renderer_template_id=artifact.renderer_template_id,
                template_version=artifact.template_version,
                page_index=page.page_index,
                image_index=image_index,
                page_kind=page.page_kind,
                source_path=page.source_path,
                image_path=image_path,
                width=width,
                height=height,
                sha256=sha256_file(image_path),
                source_fact_table=page.source_fact_table,
                source_row_ids=page.source_row_ids,
                visible_fields=page.visible_fields,
                expected_visible_fields=artifact.expected_visible_fields,
            )
        )
    return records
