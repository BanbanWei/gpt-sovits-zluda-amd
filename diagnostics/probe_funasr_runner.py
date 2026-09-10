"""Run funasr_probe_one.py on cpu and cuda, reporting exit codes.

My first attempt died silently on BOTH devices right after the checkpoint loaded,
with no Python traceback - i.e. a native abort, not a Python exception.  Two
suspicions:

  1. webui.py sets KMP_DUPLICATE_LIB_OK=TRUE ("OMP: Error #15" aborts the process
     with no traceback, and FunASR pulls in MKL/OpenMP).  The first probe did not
     set it.
  2. Resource pressure: the machine only had 9.3 GB free with the user's ASR
     running.  It is now free (19.8 GB).

This run sets the same env webui uses and prints the exit code, so a failure can
be told apart from a crash.
"""
import os
import subprocess
import time

PY = r"D:\saves\GPT-SoVITS\venv_torch25\Scripts\python.exe"
ONE = r"D:\saves\deepseek-harness\funasr_probe_one.py"
OUT = r"D:\saves\deepseek-harness\probe_out"
CWD = r"D:\saves\GPT-SoVITS\GPT-SoVITS"
TIMEOUT = 420

os.makedirs(OUT, exist_ok=True)

print("=" * 74, flush=True)
for dev in ("cpu", "cuda"):
    env = dict(os.environ)
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUNBUFFERED"] = "1"
    env["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # what webui.py sets

    of = os.path.join(OUT, "funasr_%s.txt" % dev)
    t0 = time.time()
    with open(of, "w", encoding="utf-8", errors="replace") as fh:
        p = subprocess.Popen([PY, ONE, dev], stdout=fh, stderr=subprocess.STDOUT, cwd=CWD, env=env)
        try:
            rc = p.wait(timeout=TIMEOUT)
            status = "exit=%s" % rc
        except subprocess.TimeoutExpired:
            status = "TIMEOUT(HANG)"
            subprocess.run(["taskkill", "/F", "/T", "/PID", str(p.pid)], capture_output=True)
    dt = time.time() - t0

    print("[%s] %-14s %6.1fs" % (dev, status, dt), flush=True)
    try:
        with open(of, encoding="utf-8", errors="replace") as fh:
            lines = [ln for ln in fh.read().splitlines() if ln.strip()]
        print("    --- last 6 lines ---", flush=True)
        for ln in lines[-6:]:
            print("    " + ln[:150], flush=True)
    except Exception as e:
        print("    (unreadable: %s)" % e, flush=True)
    print("-" * 74, flush=True)

print("ALL DONE", flush=True)
