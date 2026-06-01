from __future__ import annotations

import argparse
import os
from pathlib import Path

from .ocr_config import (
    DEFAULT_OCR_DATA_DIR,
    DEFAULT_SAMPLE_PATH,
    DEFAULT_WORK_DIR,
    OCR_ARTIFACT_TYPES,
)
from .ocr_extract import extract_from_manifest, validate_parsed_dir
from .ocr_images import prepare_images
from .ocr_manifest import load_artifact_manifest, manifest_to_jsonl
from .ocr_submit import write_submission


def cmd_manifest(args: argparse.Namespace) -> None:
    records = load_artifact_manifest(
        data_dir=args.data_dir,
        sample_path=args.sample_path,
        limit=args.limit,
        artifact_type=args.type,
    )
    manifest_to_jsonl(records, args.output)
    print(f"Wrote {args.output} for {sum(len(r.pages) for r in records)} pages")


def cmd_prepare_images(args: argparse.Namespace) -> None:
    records = load_artifact_manifest(
        data_dir=args.data_dir,
        sample_path=args.sample_path,
        limit=args.limit,
        artifact_type=args.type,
    )
    prepare_images(
        records,
        output_dir=args.image_dir,
        manifest_path=args.output,
        dpi=args.dpi,
        overwrite=args.force,
    )
    print(f"Wrote image manifest {args.output}")


def cmd_extract(args: argparse.Namespace) -> None:
    processed = extract_from_manifest(
        image_manifest=args.image_manifest,
        work_dir=args.work_dir,
        artifact_type=args.type,
        limit=args.limit,
        max_retries=args.max_retries,
        force=args.force,
    )
    print(f"Extracted {processed} artifacts")


def cmd_validate(args: argparse.Namespace) -> None:
    failures = validate_parsed_dir(args.work_dir)
    print(f"Validation complete: {failures} failed")


def cmd_submit(args: argparse.Namespace) -> None:
    write_submission(
        parsed_dir=args.work_dir / "parsed",
        output_path=args.output,
        sample_path=args.sample_path,
        allow_missing=args.allow_missing,
    )
    print(f"Wrote {args.output}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="FahMai OCR Gemini pipeline")
    parser.set_defaults(func=None)
    common_data = argparse.ArgumentParser(add_help=False)
    common_data.add_argument("--data-dir", type=Path, default=DEFAULT_OCR_DATA_DIR)
    common_data.add_argument("--sample-path", type=Path, default=DEFAULT_SAMPLE_PATH)
    common_data.add_argument("--type", choices=OCR_ARTIFACT_TYPES)
    common_data.add_argument("--limit", type=int)

    subparsers = parser.add_subparsers(required=True)

    manifest = subparsers.add_parser("manifest", parents=[common_data])
    manifest.add_argument("--output", type=Path, default=DEFAULT_WORK_DIR / "manifest.jsonl")
    manifest.set_defaults(func=cmd_manifest)

    prepare = subparsers.add_parser("prepare-images", parents=[common_data])
    prepare.add_argument("--image-dir", type=Path, default=DEFAULT_WORK_DIR / "images")
    prepare.add_argument("--output", type=Path, default=DEFAULT_WORK_DIR / "image_manifest.jsonl")
    prepare.add_argument("--dpi", type=int, default=250)
    prepare.add_argument("--force", action="store_true")
    prepare.set_defaults(func=cmd_prepare_images)

    extract = subparsers.add_parser("extract")
    extract.add_argument(
        "--image-manifest",
        type=Path,
        default=DEFAULT_WORK_DIR / "image_manifest.jsonl",
    )
    extract.add_argument("--work-dir", type=Path, default=DEFAULT_WORK_DIR)
    extract.add_argument("--type", choices=OCR_ARTIFACT_TYPES)
    extract.add_argument("--limit", type=int)
    extract.add_argument("--max-retries", type=int, default=int(os.getenv("OCR_MAX_RETRIES", "3")))
    extract.add_argument("--force", action="store_true")
    extract.set_defaults(func=cmd_extract)

    validate = subparsers.add_parser("validate")
    validate.add_argument("--work-dir", type=Path, default=DEFAULT_WORK_DIR)
    validate.set_defaults(func=cmd_validate)

    submit = subparsers.add_parser("submit")
    submit.add_argument("--work-dir", type=Path, default=DEFAULT_WORK_DIR)
    submit.add_argument("--sample-path", type=Path, default=DEFAULT_SAMPLE_PATH)
    submit.add_argument("--output", type=Path, default=Path("submissions/ocr_submission.csv"))
    submit.add_argument("--allow-missing", action="store_true")
    submit.set_defaults(func=cmd_submit)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
