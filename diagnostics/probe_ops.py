"""Granular op probes to localise "GET was unable to find an engine".

Run with DISABLE_ADDMM_CUDA_LT=1 so cuBLASLt is out of the picture and any
remaining failure is attributable to the op itself.  Prints full tracebacks.
"""
import os
import sys
import traceback

MODE = sys.argv[1]
import torch


def log(*a):
    print(*a, flush=True)


log("START", MODE)
log("  env DISABLE_ADDMM_CUDA_LT =", os.environ.get("DISABLE_ADDMM_CUDA_LT", "<unset>"))
log("  torch.backends.cudnn.enabled =", torch.backends.cudnn.enabled)
log("  zluda_compat loaded =", "zluda_compat" in sys.modules)

if os.environ.get("PROBE_NO_CUDNN", "").strip().lower() in ("1", "true", "yes", "on"):
    torch.backends.cudnn.enabled = False
    torch.backends.cuda.matmul.allow_tf32 = False
    log("  (forced) cudnn disabled =", not torch.backends.cudnn.enabled)

try:
    if MODE == "gn_fwd":
        m = torch.nn.GroupNorm(4, 192).cuda()
        with torch.no_grad():
            o = m(torch.randn(2, 192, 500, device="cuda"))
        torch.cuda.synchronize()
        log("RESULT OK", tuple(o.shape))

    elif MODE == "gn_fwd_bwd":
        m = torch.nn.GroupNorm(4, 192).cuda()
        o = m(torch.randn(2, 192, 500, device="cuda"))
        o.float().pow(2).mean().backward()
        torch.cuda.synchronize()
        log("RESULT OK", tuple(o.shape))

    elif MODE == "gn_fwd_bwd_fp16":
        m = torch.nn.GroupNorm(4, 192).cuda().half()
        o = m(torch.randn(2, 192, 500, device="cuda").half())
        o.float().pow(2).mean().backward()
        torch.cuda.synchronize()
        log("RESULT OK", tuple(o.shape))

    elif MODE == "ln_fwd_bwd":
        m = torch.nn.LayerNorm(192).cuda()
        o = m(torch.randn(2, 500, 192, device="cuda"))
        o.pow(2).mean().backward()
        torch.cuda.synchronize()
        log("RESULT OK", tuple(o.shape))

    elif MODE == "conv1d_bwd":
        m = torch.nn.Conv1d(192, 192, 3, padding=1).cuda()
        o = m(torch.randn(2, 192, 500, device="cuda"))
        o.pow(2).mean().backward()
        torch.cuda.synchronize()
        log("RESULT OK", tuple(o.shape))

    elif MODE == "conv1d_bwd_fp16":
        m = torch.nn.Conv1d(192, 192, 3, padding=1).cuda().half()
        o = m(torch.randn(2, 192, 500, device="cuda").half())
        o.float().pow(2).mean().backward()
        torch.cuda.synchronize()
        log("RESULT OK", tuple(o.shape))

    elif MODE == "lrelu_bwd":
        x = torch.randn(2, 192, 500, device="cuda", requires_grad=True)
        torch.nn.functional.leaky_relu(x, 0.1).pow(2).mean().backward()
        torch.cuda.synchronize()
        log("RESULT OK grad", tuple(x.grad.shape))

    elif MODE == "linear_bwd_fp16":
        m = torch.nn.Linear(192, 256).cuda().half()
        o = m(torch.randn(2, 500, 192, device="cuda").half())
        o.float().pow(2).mean().backward()
        torch.cuda.synchronize()
        log("RESULT OK", tuple(o.shape))

    elif MODE == "net_conv_linear_bwd":
        # no GroupNorm at all
        c = torch.nn.Conv1d(192, 192, 3, padding=1).cuda().half()
        l = torch.nn.Linear(192, 256).cuda().half()
        x = torch.randn(2, 192, 500, device="cuda").half()
        o = l(c(x).transpose(1, 2))
        o.float().pow(2).mean().backward()
        torch.cuda.synchronize()
        log("RESULT OK", tuple(o.shape))

    elif MODE == "net_conv_gn_bwd":
        c = torch.nn.Conv1d(192, 192, 3, padding=1).cuda().half()
        g = torch.nn.GroupNorm(4, 192).cuda().half()
        x = torch.randn(2, 192, 500, device="cuda").half()
        o = g(c(x))
        o.float().pow(2).mean().backward()
        torch.cuda.synchronize()
        log("RESULT OK", tuple(o.shape))

    elif MODE == "net_full_bwd":
        c = torch.nn.Conv1d(192, 192, 3, padding=1).cuda().half()
        g = torch.nn.GroupNorm(4, 192).cuda().half()
        l = torch.nn.Linear(192, 256).cuda().half()
        x = torch.randn(2, 192, 500, device="cuda").half()
        o = l(torch.nn.functional.leaky_relu(g(c(x)), 0.1).transpose(1, 2))
        o.float().pow(2).mean().backward()
        torch.cuda.synchronize()
        log("RESULT OK", tuple(o.shape))

    elif MODE == "convtranspose_bwd_fp16":
        m = torch.nn.ConvTranspose1d(192, 192, 4, stride=2, padding=1).cuda().half()
        o = m(torch.randn(2, 192, 250, device="cuda").half())
        o.float().pow(2).mean().backward()
        torch.cuda.synchronize()
        log("RESULT OK", tuple(o.shape))

    elif MODE == "interpolate_fp16":
        x = torch.randn(2, 192, 100, device="cuda").half()
        o = torch.nn.functional.interpolate(x, scale_factor=2, mode="nearest")
        torch.cuda.synchronize()
        log("RESULT OK", tuple(o.shape))

    elif MODE == "instance_norm_fwd":
        m = torch.nn.InstanceNorm1d(192).cuda()
        o = m(torch.randn(2, 192, 500, device="cuda"))
        torch.cuda.synchronize()
        log("RESULT OK", tuple(o.shape))

    elif MODE == "batchnorm_fwd":
        m = torch.nn.BatchNorm1d(192).cuda()
        o = m(torch.randn(2, 192, 500, device="cuda"))
        torch.cuda.synchronize()
        log("RESULT OK", tuple(o.shape))

    else:
        log("RESULT FAIL unknown mode")

except Exception as e:
    log("RESULT FAIL", type(e).__name__, str(e)[:200])
    log("---- traceback ----")
    log(traceback.format_exc()[-2000:])
