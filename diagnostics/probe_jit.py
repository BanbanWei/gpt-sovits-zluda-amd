"""Find a JIT setting that stops TorchScript from NVRTC-compiling fused kernels.

The failure is at module/modules.py:197 -> commons.fused_add_tanh_sigmoid_multiply,
a @torch.jit.script function, where the fuser emits a fused CUDA kernel that NVRTC
cannot build for this ZLUDA device ("invalid value for --gpu-architecture").
argv[1] selects the mitigation.
"""
import os
import sys
import traceback

MODE = sys.argv[1]
import _env
PROJ = _env.gpt_sovits_root()
sys.path.insert(0, PROJ)
sys.path.insert(0, os.path.join(PROJ, "GPT_SoVITS"))

import torch


def log(*a):
    print(*a, flush=True)


log("START", MODE)

try:
    if MODE == "default":
        pass
    elif MODE == "texpr_off":
        torch._C._jit_set_texpr_fuser_enabled(False)
    elif MODE == "profiling_off":
        torch._C._jit_set_profiling_executor(False)
        torch._C._jit_set_profiling_mode(False)
    elif MODE == "fusers_off":
        torch._C._jit_set_texpr_fuser_enabled(False)
        torch._C._jit_set_profiling_executor(False)
        torch._C._jit_set_profiling_mode(False)
        torch._C._jit_override_can_fuse_on_gpu(False)
        torch._C._jit_override_can_fuse_on_cpu(False)
    elif MODE == "jit_state_disable":
        torch.jit._state.disable()
    elif MODE == "eager_manual":
        # skip the scripted function entirely, do the maths by hand
        pass
    else:
        log("RESULT FAIL unknown mode")
        raise SystemExit(0)

    try:
        log("  jit state enabled =", torch.jit._state._enabled)
    except Exception as e:
        log("  jit state unreadable:", e)
    try:
        log("  texpr fuser enabled =", torch._C._jit_texpr_fuser_enabled())
    except Exception as e:
        log("  texpr fuser unreadable:", e)

    from module.commons import fused_add_tanh_sigmoid_multiply as fam

    log("  imported fam, type =", type(fam).__name__)

    a = torch.randn(2, 384, 50, device="cuda").half()
    b = torch.randn(2, 384, 50, device="cuda").half()
    n = torch.tensor([192], device="cuda")

    with torch.autocast("cuda", dtype=torch.float16):
        if MODE == "eager_manual":
            in_act = a + b
            out = torch.tanh(in_act[:, :192, :]) * torch.sigmoid(in_act[:, 192:, :])
        else:
            out = fam(a, b, n)

    torch.cuda.synchronize()
    log("RESULT OK", tuple(out.shape), out.dtype)

except Exception as e:
    log("RESULT FAIL", type(e).__name__, str(e)[:200])
    log("---- traceback ----")
    log(traceback.format_exc()[-1200:])
