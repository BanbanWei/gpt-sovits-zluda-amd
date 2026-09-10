@echo off
chcp 65001 >nul
title GPT-SoVITS - ZLUDA - RX 6600 (with log)

cd /d "D:\saves\GPT-SoVITS\GPT-SoVITS"

set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
set PYTHONUNBUFFERED=1
set LOG=D:\saves\GPT-SoVITS\GPT-SoVITS\webui_console.log

echo ==========================================
echo    GPT-SoVITS ZLUDA 启动器 (带日志)
echo ==========================================
echo.
echo 控制台输出会同时写入:
echo   %LOG%
echo.

if not exist "webui.py" (
    echo [错误] 找不到 GPT-SoVITS 项目
    pause
    exit /b 1
)

if not exist "D:\saves\GPT-SoVITS\venv_torch25\Scripts\python.exe" (
    echo [错误] 找不到 Python 虚拟环境
    pause
    exit /b 1
)

if not exist "C:\ZLUDA\bin\zluda.exe" (
    echo [错误] 找不到 ZLUDA
    pause
    exit /b 1
)

echo 正在启动 GPT-SoVITS...
echo.

powershell -NoProfile -ExecutionPolicy Bypass -Command "& 'C:\ZLUDA\bin\zluda.exe' 'D:\saves\GPT-SoVITS\venv_torch25\Scripts\python.exe' webui.py 2>&1 | Tee-Object -FilePath '%LOG%'"

echo.
echo ==========================================
echo GPT-SoVITS 已退出
echo ==========================================
pause
