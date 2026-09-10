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

PY = r"D:\saves\GPT-SoVITS\venv_torch25\Scripts\python.exe"
CWD = r"D:\saves\GPT-SoVITS\GPT-SoVITS"
OUT = r"D:\saves\deepseek-harness\probe_out"

script = sys.argv[1]
timeout = int(sys.argv[2])
modes = sys.argv[3:]

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
