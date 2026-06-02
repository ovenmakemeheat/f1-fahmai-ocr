@echo off
setlocal
cd /d "%~dp0.."

if "%OCR_WORKERS%"=="" set "OCR_WORKERS=2"

uv run python -m src.run_ocr_pipeline manifest
if errorlevel 1 exit /b %ERRORLEVEL%

uv run python -m src.run_ocr_pipeline prepare-images --force
if errorlevel 1 exit /b %ERRORLEVEL%

uv run python -m src.run_ocr_pipeline extract --force --workers %OCR_WORKERS%
if errorlevel 1 exit /b %ERRORLEVEL%

uv run python -m src.run_ocr_pipeline validate
if errorlevel 1 exit /b %ERRORLEVEL%

uv run python -m src.run_ocr_pipeline submit
if errorlevel 1 exit /b %ERRORLEVEL%

echo Done.
