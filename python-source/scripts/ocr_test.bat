@echo off
setlocal
cd /d "%~dp0.."

set "WORK_DIR=work\ocr_test"
if "%OCR_WORKERS%"=="" set "OCR_WORKERS=3"

uv run python -m src.run_ocr_pipeline manifest --limit 3 --output "%WORK_DIR%\manifest.jsonl"
if errorlevel 1 exit /b %ERRORLEVEL%

uv run python -m src.run_ocr_pipeline prepare-images --limit 3 --image-dir "%WORK_DIR%\images" --output "%WORK_DIR%\image_manifest.jsonl"
if errorlevel 1 exit /b %ERRORLEVEL%

uv run python -m src.run_ocr_pipeline extract --limit 3 --work-dir "%WORK_DIR%" --image-manifest "%WORK_DIR%\image_manifest.jsonl" --workers %OCR_WORKERS%
if errorlevel 1 exit /b %ERRORLEVEL%

uv run python -m src.run_ocr_pipeline validate --work-dir "%WORK_DIR%"
if errorlevel 1 exit /b %ERRORLEVEL%

uv run python -m src.run_ocr_pipeline submit --work-dir "%WORK_DIR%" --output "submissions\ocr_test_submission.csv" --allow-missing
if errorlevel 1 exit /b %ERRORLEVEL%

echo Done.
