@echo off
rem ============================================================
rem  KnowHub backend one-click launcher
rem  1) checks local models under models\ (auto-downloads ~1.1GB
rem     on first run via tools\fetch_models.py, resume-safe)
rem  2) starts uvicorn on port 8000
rem  Models are plain local folders loaded BY PATH from .env, so
rem  no huggingface_hub / HF_ENDPOINT is involved at runtime.
rem ============================================================
cd /d %~dp0

if exist "models\BAAI\bge-small-zh-v1.5\model.safetensors" if exist "models\BAAI\bge-reranker-base\model.safetensors" goto start_backend

echo [1/2] Local models missing under models\BAAI\. Downloading (~1.1GB, first run only) ...
echo       Resume-safe: finished parts are kept, re-run if it fails midway.
echo       hf-mirror sometimes gets SNI-reset in CN; the script alternates
echo       direct / local proxy (127.0.0.1:7890) until it gets through.
uv run python tools\fetch_models.py
if errorlevel 1 (
    echo.
    echo [ERROR] Model download failed. Check your network, then re-run this script.
    pause
    exit /b 1
)

:start_backend
echo.
echo [2/2] Starting backend on http://localhost:8000  (Ctrl+C to stop)
echo       Web UI is at http://localhost:3000 - start frontend separately:
echo       cd ..\frontend ^&^& npm run dev
uv run uvicorn app.main:app --port 8000
pause
