"""Can FunASR (Fun-ASR-Nano) run on the GPU under ZLUDA?

argv[1] is the device to request: "cpu" (baseline) or "cuda" (the question).
Mirrors tools/asr/funasr_asr.py::create_fun_asr_nano_model exactly, minus the
hardcoded device, and is run in its own process with a hard timeout because the
rocFFT failure mode on this GPU is a HANG, not an exception.
"""
import os
import sys
import time
import traceback

DEVICE = sys.argv[1]
PROJ = r"D:\saves\GPT-SoVITS\GPT-SoVITS"
WAV = os.path.join(PROJ, "logs", "GBC", "5-wav32k", "GBC.m4a_0000123520_0000258880.wav")
MODEL_ID = "FunAudioLLM/Fun-ASR-Nano-2512"


def log(*a):
    print(*a, flush=True)


log("START device=%s" % DEVICE)
log("  cwd            :", os.getcwd())
log("  wav exists     :", os.path.exists(WAV), WAV)

try:
    import torch

    log("  torch          :", torch.__version__)
    log("  cuda available :", torch.cuda.is_available())
    if torch.cuda.is_available():
        log("  device name    :", torch.cuda.get_device_name(0))

    from funasr import AutoModel

    t0 = time.time()
    model = AutoModel(
        model=MODEL_ID,
        hub="hf",
        trust_remote_code=True,
        vad_model="fsmn-vad",
        device=DEVICE,
        disable_update=True,
    )
    load_s = time.time() - t0
    log("  model loaded   : %.1fs" % load_s)

    t0 = time.time()
    res = model.generate(input=WAV)
    gen_s = time.time() - t0
    text = res[0]["text"] if res else "<empty result>"
    log("  generate       : %.1fs" % gen_s)

    if torch.cuda.is_available() and DEVICE == "cuda":
        torch.cuda.synchronize()
        log("  cuda sync      : ok")

    log("RESULT OK device=%s load=%.1fs gen=%.1fs text=%r" % (DEVICE, load_s, gen_s, str(text)[:110]))

except Exception as e:
    log("RESULT FAIL", type(e).__name__, str(e)[:250])
    log(traceback.format_exc()[-1200:])
