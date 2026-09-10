"""Generic isolated-probe runner.

usage: python run_modes.py <one_shot_script> <timeout_seconds> <mode> [<mode> ...]

Runs each mode in its own process with a hard timeout so a ZLUDA hang is
reported as HANG instead of stalling the whole sweep.  Must itself run under
zluda.exe so that injection propagates into the children.
"""
import os
import subprocess
import sys
import time

import _env

# 本进程自己的 stdout 也要能容错：下面会把子进程日志回显出来，一旦里面有
# 当前控制台编码表示不了的字符（比如 GBK 控制台遇到替换字符 U+FFFD），
# print 会抛 UnicodeEncodeError 并**打断整轮扫描**。
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

PY = _env.gpt_sovits_python()
CWD = _env.gpt_sovits_root()
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "probe_out")

script = os.path.abspath(sys.argv[1])   # 子进程 cwd 不同，必须先转绝对路径
timeout = int(sys.argv[2])
modes = sys.argv[3:]

if not os.path.isfile(PY):
    print("找不到 Python 解释器：%s" % PY)
    print("请设置 GPT_SOVITS_PYTHON 指向 venv 里的 python.exe，例如：")
    print(r'  set GPT_SOVITS_PYTHON=D:\path\to\venv\Scripts\python.exe')
    sys.exit(2)

os.makedirs(OUT, exist_ok=True)
tag = os.path.splitext(os.path.basename(script))[0]

print("RUN %s | %d modes | timeout %ds" % (script, len(modes), timeout), flush=True)
print("=" * 74, flush=True)

summary = []
for i, m in enumerate(modes, 1):
    of = os.path.join(OUT, "%s__%s.txt" % (tag, m))
    env = dict(os.environ)
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    # 必须显式指定：stdout 被重定向到文件时，子进程会退回系统区域编码
    # （中文 Windows 上是 cp936），写的字节和下面按 UTF-8 读回来的不一致，
    # 日志会变成乱码；更糟的是乱码解码出的替换字符会让上面的 print 抛
    # UnicodeEncodeError，把整轮扫描打断。
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"

    t0 = time.time()
    with open(of, "w", encoding="utf-8", errors="replace") as fh:
        p = subprocess.Popen([PY, script, m], stdout=fh, stderr=subprocess.STDOUT, cwd=CWD, env=env)
        try:
            p.wait(timeout=timeout)
            status = "exit=%s" % p.returncode
        except subprocess.TimeoutExpired:
            status = "TIMEOUT"
            subprocess.run(["taskkill", "/F", "/T", "/PID", str(p.pid)], capture_output=True)
    dt = time.time() - t0

    last = "(no output)"
    try:
        with open(of, encoding="utf-8", errors="replace") as fh:
            lines = [ln for ln in fh.read().splitlines() if ln.strip()]
        if lines:
            last = lines[-1]
    except Exception as e:
        last = "unreadable %s" % e

    verdict = "HANG" if status == "TIMEOUT" else ("OK" if "RESULT OK" in last else "FAIL")
    summary.append((m, verdict, dt, last))
    print("[%2d/%2d] %6.1fs  %-24s %-4s  %s" % (i, len(modes), dt, m, verdict, last[:130]), flush=True)

print("=" * 74, flush=True)
bad = [s for s in summary if s[1] != "OK"]
print("OK: %d  BAD: %d" % (len(summary) - len(bad), len(bad)), flush=True)
for m, v, dt, last in bad:
    print("  BAD %-24s %-4s %6.1fs  %s" % (m, v, dt, last[:110]), flush=True)
print("ALL DONE", flush=True)
