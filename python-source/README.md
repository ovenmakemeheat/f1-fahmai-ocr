# FahMai OCR Pipeline

Python OCR/extraction pipeline for the FahMai Kaggle competition data. The code builds an artifact manifest, renders pages to images, calls Gemini on Vertex AI, validates parsed JSON, and writes a Kaggle submission CSV.

## Setup

Install dependencies with `uv`:

```bat
uv sync
```

The project expects the competition OCR data under:

```text
data/super-ai-engineer-season-6-fah-mai-the-finale-ocr/
```

Default input paths are configured in `src/ocr_config.py`.

## Environment

Copy `.env.example` to `.env` and fill in the Vertex settings:

```env
GOOGLE_CLOUD_PROJECT=your-gcp-project-id
GOOGLE_CLOUD_LOCATION=global
GEMINI_MODEL=gemini-3.5-flash
GOOGLE_GENAI_AUTH=adc
GEMINI_API_KEY=
GOOGLE_OAUTH_ACCESS_TOKEN=
GOOGLE_APPLICATION_CREDENTIALS=
OCR_OUTPUT_DIR=work/ocr
OCR_MAX_RETRIES=3
OCR_WORKERS=16
GEMINI_TIMEOUT_SECONDS=120
```

For Vertex auth with `google-genai`, use Application Default Credentials:

```bat
gcloud auth application-default login
gcloud auth application-default set-quota-project YOUR_PROJECT_ID
```

If using a service account JSON file, set `GOOGLE_APPLICATION_CREDENTIALS` to its path. `.env` is ignored by git.

For a short-lived bearer token instead, set `GOOGLE_GENAI_AUTH=token` and `GOOGLE_OAUTH_ACCESS_TOKEN` from:

```bat
gcloud auth application-default print-access-token
```

Do not put the token in `.env.example` or commit it.
`ya29...` values are OAuth bearer tokens, not API keys. If one is accidentally placed in `GEMINI_API_KEY`, the client treats it as a Vertex OAuth token.

## Batch Scripts

Windows wrappers live in `scripts/`.

```bat
scripts\ocr_smoke.bat
```

Creates a 3-artifact manifest and image set only. This is the safest offline check because it does not call Gemini.

```bat
scripts\ocr_test.bat
```

Runs a 3-artifact end-to-end test into `work\ocr_test` and writes `submissions\ocr_test_submission.csv`.

```bat
scripts\ocr_full.bat
```

Runs the default full pipeline:

1. `manifest`
2. `prepare-images`
3. `extract`
4. `validate`
5. `submit`

```bat
scripts\ocr_force_full.bat
```

Same as full, but forces image regeneration and OCR extraction.

```bat
scripts\ocr_receipt_test.bat
```

Runs a 3-artifact receipt-only test into `work\ocr_receipt_test`.

Direct command wrappers pass extra arguments through:

```bat
scripts\ocr_manifest.bat --limit 10
scripts\ocr_prepare_images.bat --limit 10 --force
scripts\ocr_extract.bat --limit 10 --max-retries 5
scripts\ocr_extract.bat --workers 16
scripts\ocr_validate.bat --work-dir work\ocr_test
scripts\ocr_submit.bat --allow-missing
```

## Python CLI

All batch files call the same module:

```bat
uv run python -m src.run_ocr_pipeline manifest
uv run python -m src.run_ocr_pipeline prepare-images
uv run python -m src.run_ocr_pipeline extract
uv run python -m src.run_ocr_pipeline validate
uv run python -m src.run_ocr_pipeline submit
```

Useful options:

```bat
--type receipt
--limit 20
--force
--workers 16
--work-dir work\ocr_test
--image-manifest work\ocr_test\image_manifest.jsonl
--allow-missing
```

`extract` runs artifacts in parallel. Set `OCR_WORKERS` in `.env` or pass `--workers`; higher values are faster until Vertex quota, local network, or rate limits become the bottleneck.

Valid artifact types are defined in `src/ocr_config.py`.

## Outputs

Default outputs:

```text
work/ocr/manifest.jsonl
work/ocr/image_manifest.jsonl
work/ocr/images/
work/ocr/parsed/
submissions/ocr_submission.csv
```

Generated work files and submissions should not be committed unless explicitly needed.
