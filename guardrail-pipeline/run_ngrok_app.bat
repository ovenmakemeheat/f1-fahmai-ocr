@echo off
setlocal

cd /d "%~dp0"

echo Starting Guardrail Pipeline API with ngrok...
echo Local API: http://127.0.0.1:8000
echo Fixed domain: jolly-polite-gannet.ngrok-free.app
echo Public docs: https://jolly-polite-gannet.ngrok-free.app/docs

uv run python -m src.ngrok_server

endlocal
