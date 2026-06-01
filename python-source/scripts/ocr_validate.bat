@echo off
setlocal
cd /d "%~dp0.."

uv run python -m src.run_ocr_pipeline validate %*
