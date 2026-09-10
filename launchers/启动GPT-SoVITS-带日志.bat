@echo off
chcp 65001 >nul
setlocal enabledelayedexpansion
title GPT-SoVITS - ZLUDA - AMD (with log)

rem ============================================================
rem  配置区：自动探测失败时才需要改这三行
rem ============================================================
set "GPT_ROOT="
set "VENV_DIR="
set "ZLUDA_DIR=C:\ZLUDA"
rem ============================================================

for %%I in ("%~dp0..") do set "REPO=%%~fI"

if not defined GPT_ROOT if defined GPT_SOVITS_ROOT set "GPT_ROOT=%GPT_SOVITS_ROOT%"
if not defined GPT_ROOT for %%D in (
    "%REPO%\..\GPT-SoVITS\GPT-SoVITS"
    "%REPO%\..\GPT-SoVITS"
    "%REPO%\GPT-SoVITS"
    "%REPO%"
) do (
    if not defined GPT_ROOT if exist "%%~fD\webui.py" if exist "%%~fD\GPT_SoVITS" set "GPT_ROOT=%%~fD"
)

if not defined VENV_DIR if defined GPT_SOVITS_VENV set "VENV_DIR=%GPT_SOVITS_VENV%"
if not defined VENV_DIR if defined GPT_ROOT for %%V in (
    "%GPT_ROOT%\venv_torch25"
    "%GPT_ROOT%\..\venv_torch25"
    "%REPO%\..\venv_torch25"
    "%GPT_ROOT%\venv"
    "%GPT_ROOT%\.venv"
) do (
    if not defined VENV_DIR if exist "%%~fV\Scripts\python.exe" set "VENV_DIR=%%~fV"
)

if not defined GPT_ROOT (
    echo [错误] 找不到 GPT-SoVITS 项目，请编辑本文件的"配置区"设置 GPT_ROOT。
    pause
    exit /b 1
)
if not defined VENV_DIR (
    echo [错误] 找不到 Python 虚拟环境，请编辑本文件的"配置区"设置 VENV_DIR。
    pause
    exit /b 1
)
if not exist "%ZLUDA_DIR%\bin\zluda.exe" (
    echo [错误] 找不到 ZLUDA: %ZLUDA_DIR%\bin\zluda.exe
    pause
    exit /b 1
)

set "LOG=%GPT_ROOT%\webui_console.log"

rem 兼容层自己会设这些，这里只是保证 stdio 是 UTF-8，避免进度条写
rem 特殊字符时抛 UnicodeEncodeError 让收尾阶段崩掉。
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
set PYTHONUNBUFFERED=1

cd /d "%GPT_ROOT%"

echo ==========================================
echo    GPT-SoVITS ZLUDA 启动器 (带日志)
echo ==========================================
echo.
echo 控制台输出会同时写入:
echo   %LOG%
echo.
echo 正在启动 GPT-SoVITS...
echo.

rem Tee-Object 在 PowerShell 5.1 里写的是 UTF-16LE，跑完转成 UTF-8 方便用别的
rem 编辑器/grep 看（下面的转换即使失败也不影响启动结果）。
powershell -NoProfile -ExecutionPolicy Bypass -Command "& '%ZLUDA_DIR%\bin\zluda.exe' '%VENV_DIR%\Scripts\python.exe' webui.py 2>&1 | Tee-Object -FilePath '%LOG%'"

if exist "%LOG%" powershell -NoProfile -ExecutionPolicy Bypass -Command "try { $c = Get-Content -Raw -LiteralPath '%LOG%'; [IO.File]::WriteAllText('%LOG%', $c, (New-Object Text.UTF8Encoding $false)) } catch { }"

echo.
echo ==========================================
echo GPT-SoVITS 已退出  (日志: %LOG%)
echo ==========================================
pause
