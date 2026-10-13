@echo off
setlocal enabledelayedexpansion

:: ============================================================
::  KnowHub one-click launcher - double-click to run on Windows
::  - Backend: FastAPI/uvicorn -> http://127.0.0.1:8000
::  - Frontend: Next.js dev    -> http://localhost:3000
::  Browser opens automatically when frontend is ready.
::  Two service windows stay open; close a window to stop that service.
::  If the browser does not open, visit http://localhost:3000 manually.
:: ============================================================

:: ---------- config ----------
:: 项目根目录自动取本脚本所在位置, 任意机器上双击即用。
set "ROOT=%~dp0"
:: 去掉 %~dp0 自带的尾部反斜杠, 方便下面拼接
if "%ROOT:~-1%"=="\" set "ROOT=%ROOT:~0,-1%"

:: Python / Node 默认走 PATH; 如用托管运行时, 改成绝对路径即可
set "PYVENV=python"
set "NODEDIR="

:: prepend managed Node to PATH so npm is available (留空则跳过)
if not "%NODEDIR%"=="" set "PATH=%NODEDIR%;%PATH%"

echo [KnowHub] checking runtimes...
where %PYVENV% >nul 2>&1
if errorlevel 1 (
  echo [ERROR] python not found on PATH. Please install Python 3.10+ or edit PYVENV at the top.
  pause
  goto :eof
)
where npm >nul 2>&1
if errorlevel 1 (
  echo [ERROR] npm not found on PATH. Please install Node.js 20+ or set NODEDIR at the top.
  pause
  goto :eof
)

:: ---------- port pre-check - avoid silent EADDRINUSE ----------
echo [KnowHub] pre-checking ports 8000 and 3000 ...
curl -sf http://127.0.0.1:8000/health >nul 2>&1
if not errorlevel 1 (
  echo [ERROR] port 8000 is occupied - backend may already be running.
  echo         Close the previous backend window or free port 8000, then re-run.
  pause
  goto :eof
)
curl -sf http://localhost:3000 >nul 2>&1
if not errorlevel 1 (
  echo [ERROR] port 3000 is occupied - frontend may already be running.
  echo         Close the previous frontend window or free port 3000, then re-run.
  pause
  goto :eof
)

:: ---------- start backend ----------
echo [KnowHub] starting backend - uvicorn on :8000
start "KnowHub-Backend" cmd /k "cd /d %ROOT%\backend && %PYVENV% -m uvicorn app.main:app --host 127.0.0.1 --port 8000"

:: ---------- wait for backend health ----------
set "ATT=0"
:wait_backend
curl -sf http://127.0.0.1:8000/health >nul 2>&1
if not errorlevel 1 goto backend_ok
set /a ATT+=1
if %ATT% geq 40 (
  echo [KnowHub] backend start timed out. Check the KnowHub-Backend window for errors.
  pause
  goto :eof
)
timeout /t 1 >nul
goto wait_backend
:backend_ok
echo [KnowHub] backend ready: http://127.0.0.1:8000

:: ---------- start frontend ----------
echo [KnowHub] starting frontend - Next.js on :3000
start "KnowHub-Frontend" cmd /k "cd /d %ROOT%\frontend && set NEXT_PUBLIC_API_BASE=http://localhost:8000 && npm run dev -- -p 3000"

:: ---------- wait for frontend ----------
set "ATT=0"
:wait_frontend
curl -sf http://localhost:3000 >nul 2>&1
if not errorlevel 1 goto frontend_ok
set /a ATT+=1
if %ATT% geq 90 (
  echo [KnowHub] frontend start timed out. Check the KnowHub-Frontend window for errors.
  pause
  goto :eof
)
timeout /t 1 >nul
goto wait_frontend
:frontend_ok
echo [KnowHub] frontend ready, opening browser...
start "" http://localhost:3000

:end
echo.
echo [KnowHub] done. Keep the two service windows open. Close a window to stop that service.
echo If the browser did not open, visit http://localhost:3000 manually.
echo.
pause
endlocal
