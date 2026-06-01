@echo off
setlocal
cd /d "%~dp0.."

set "WORK_DIR=work\ocr_smoke"

uv run python -m src.run_ocr_pipeline manifest --limit 3 --output "%WORK_DIR%\manifest.jsonl"
if errorlevel 1 exit /b %ERRORLEVEL%

uv run python -m src.run_ocr_pipeline prepare-images --limit 3 --image-dir "%WORK_DIR%\images" --output "%WORK_DIR%\image_manifest.jsonl"
if errorlevel 1 exit /b %ERRORLEVEL%

echo Done.
