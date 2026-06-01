@echo off
setlocal
cd /d "%~dp0.."

if "%OCR_WORKERS%"=="" set "OCR_WORKERS=2"

if "%~1"=="" (
  uv run python -m src.run_ocr_pipeline extract --workers %OCR_WORKERS%
  exit /b %ERRORLEVEL%
)

uv run python -m src.run_ocr_pipeline extract %*
