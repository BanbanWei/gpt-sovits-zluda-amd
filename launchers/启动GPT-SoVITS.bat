@echo off
chcp 65001 >nul
setlocal enabledelayedexpansion
title GPT-SoVITS - ZLUDA - AMD

rem ============================================================
rem  配置区：自动探测失败时才需要改这三行
rem  （也可以改成先设同名环境变量，脚本会优先用它）
rem ============================================================
set "GPT_ROOT="
set "VENV_DIR="
set "ZLUDA_DIR=C:\ZLUDA"
rem ============================================================

for %%I in ("%~dp0..") do set "REPO=%%~fI"

rem ---- 探测 GPT-SoVITS 项目根目录（含 webui.py 与 GPT_SoVITS\）----
if not defined GPT_ROOT if defined GPT_SOVITS_ROOT set "GPT_ROOT=%GPT_SOVITS_ROOT%"
if not defined GPT_ROOT for %%D in (
    "%REPO%\..\GPT-SoVITS\GPT-SoVITS"
    "%REPO%\..\GPT-SoVITS"
    "%REPO%\GPT-SoVITS"
    "%REPO%"
) do (
    if not defined GPT_ROOT if exist "%%~fD\webui.py" if exist "%%~fD\GPT_SoVITS" set "GPT_ROOT=%%~fD"
)

rem ---- 探测 venv ----
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

echo ==========================================
echo        GPT-SoVITS ZLUDA 启动器
echo ==========================================
echo.

if not defined GPT_ROOT (
    echo [错误] 找不到 GPT-SoVITS 项目。
    echo        请编辑本文件的"配置区"，设置 GPT_ROOT。
    pause
    exit /b 1
)
if not exist "%GPT_ROOT%\webui.py" (
    echo [错误] GPT_ROOT 里没有 webui.py: %GPT_ROOT%
    pause
    exit /b 1
)
if not defined VENV_DIR (
    echo [错误] 找不到 Python 虚拟环境。
    echo        请编辑本文件的"配置区"，设置 VENV_DIR。
    pause
    exit /b 1
)
if not exist "%ZLUDA_DIR%\bin\zluda.exe" (
    echo [错误] 找不到 ZLUDA: %ZLUDA_DIR%\bin\zluda.exe
    echo        请编辑本文件的"配置区"，设置 ZLUDA_DIR。
    pause
    exit /b 1
)

rem 兼容层必须靠 .pth 自动加载；没装的话下面这行会警告（不致命）
set "PTH=%VENV_DIR%\Lib\site-packages\zluda_compat.pth"
if not exist "%PTH%" (
    echo [警告] 未检测到兼容层，建议先运行:
    echo        %VENV_DIR%\Scripts\python.exe compat\install.py
    echo.
)

rem 兼容层自己会设这些，这里只是保证 stdio 是 UTF-8，避免进度条写
rem 特殊字符时抛 UnicodeEncodeError 让收尾阶段崩掉。
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8

cd /d "%GPT_ROOT%"

echo [OK] 项目   %GPT_ROOT%
echo [OK] Python %VENV_DIR%
echo [OK] ZLUDA  %ZLUDA_DIR%
echo.
echo 正在启动 GPT-SoVITS...
echo.

"%ZLUDA_DIR%\bin\zluda.exe" "%VENV_DIR%\Scripts\python.exe" webui.py

echo.
echo ==========================================
echo GPT-SoVITS 已退出
echo ==========================================
pause
