@echo off
chcp 65001 >nul
title GPT-SoVITS - ZLUDA - RX 6600

echo ==========================================
echo        GPT-SoVITS ZLUDA 启动器
echo        AMD RX 6600
echo ==========================================
echo.

cd /d "D:\saves\GPT-SoVITS\GPT-SoVITS"

if not exist "webui.py" (
    echo [错误] 找不到 GPT-SoVITS 项目！
    pause
    exit /b 1
)

if not exist "D:\saves\GPT-SoVITS\venv_torch25\Scripts\python.exe" (
    echo [错误] 找不到 Python 虚拟环境！
    pause
    exit /b 1
)

if not exist "C:\ZLUDA\bin\zluda.exe" (
    echo [错误] 找不到 ZLUDA！
    pause
    exit /b 1
)

rem UTF-8 stdio, otherwise the trainer's progress bar aborts teardown with
rem UnicodeEncodeError when it tries to write a bullet character.
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8

rem The ZLUDA/ROCm compatibility shims (cuBLASLt, cuDNN, TorchScript/NVRTC,
rem exit-time CUDA sync) install themselves automatically through
rem venv_torch25\Lib\site-packages\zluda_compat.pth - nothing to do here.

echo [OK] GPT-SoVITS 项目
echo [OK] Python 3.11
echo [OK] ZLUDA
echo.
echo 正在启动 GPT-SoVITS...
echo.

"C:\ZLUDA\bin\zluda.exe" "D:\saves\GPT-SoVITS\venv_torch25\Scripts\python.exe" webui.py

echo.
echo ==========================================
echo GPT-SoVITS 已退出
echo ==========================================
pause
