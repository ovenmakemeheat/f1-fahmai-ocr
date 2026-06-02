@echo off
setlocal
cd /d "%~dp0.."

set "WORK_DIR=work\ocr"
set "COUNT=0"

for /f %%C in ('dir /b "%WORK_DIR%\parsed\*.json" 2^>nul ^| find /c /v ""') do set "COUNT=%%C"

set "OUTPUT=submissions\ocr_partial_%COUNT%_submission.csv"

uv run python -m src.run_ocr_pipeline submit --work-dir "%WORK_DIR%" --output "%OUTPUT%" --allow-missing %*
if errorlevel 1 exit /b %ERRORLEVEL%

echo Wrote %OUTPUT% from %COUNT% parsed JSON files with missing artifacts allowed.
