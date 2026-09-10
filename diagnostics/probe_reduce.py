"""One reduction / scalar-readback probe per process. argv[1] selects it.

These isolate which CUDA op hangs under ZLUDA.  mel_spectrogram_torch begins with
`if torch.min(y) < -1.2:` which forces a device->host scalar read, so a hang
there would block s2 training independently of the FFT issue.
"""
import sys

MODE = sys.argv[1]
import torch


def log(*a):
    print(*a, flush=True)


log("START", MODE)
try:
    if MODE == "randn_alloc":
        x = torch.randn(32000, device="cuda")
        log("RESULT OK", tuple(x.shape))

    elif MODE == "sync_only":
        torch.randn(32000, device="cuda")
        torch.cuda.synchronize()
        log("RESULT OK synced")

    elif MODE == "min_item":
        x = torch.randn(32000, device="cuda")
        v = torch.min(x).item()
        log("RESULT OK min=", v)

    elif MODE == "min_float":
        x = torch.randn(32000, device="cuda")
        v = float(torch.min(x))
        log("RESULT OK min=", v)

    elif MODE == "min_str":
        x = torch.randn(32000, device="cuda")
        s = str(torch.min(x))
        log("RESULT OK repr=", s)

    elif MODE == "max_float":
        x = torch.randn(32000, device="cuda")
        log("RESULT OK max=", float(torch.max(x)))

    elif MODE == "sum_float":
        x = torch.randn(32000, device="cuda")
        log("RESULT OK sum=", float(x.sum()))

    elif MODE == "mean_float":
        x = torch.randn(32000, device="cuda")
        log("RESULT OK mean=", float(x.mean()))

    elif MODE == "argmin_int":
        x = torch.randn(32000, device="cuda")
        log("RESULT OK argmin=", int(torch.argmin(x)))

    elif MODE == "bool_compare":
        # exactly what mel_spectrogram_torch does
        x = torch.randn(32000, device="cuda") * 4
        if float(torch.min(x)) < -1.2:
            log("RESULT OK branch-taken")
        else:
            log("RESULT OK branch-skipped")

    elif MODE == "cpu_copy":
        x = torch.randn(32000, device="cuda")
        y = x.cpu()
        log("RESULT OK", tuple(y.shape), str(y.device))

    elif MODE == "scalar_cpu_copy":
        x = torch.randn(32000, device="cuda")
        y = x.sum().cpu()
        log("RESULT OK", tuple(y.shape), str(y.device))

    elif MODE == "min_cpu_after_cuda":
        x = torch.randn(32000, device="cuda")
        y = torch.min(x).cpu()
        log("RESULT OK", float(y))

    elif MODE == "layernorm_sum":
        x = torch.randn(32000, device="cuda")
        y = torch.nn.functional.layer_norm(x, (32000,))
        log("RESULT OK", float(y.sum()))

    elif MODE == "softmax_sum":
        x = torch.randn(8, 100, device="cuda")
        log("RESULT OK", float(torch.softmax(x, -1).sum()))

    elif MODE == "sum_float_large":
        x = torch.randn(1024, 1024, device="cuda")
        log("RESULT OK", float(x.sum()))

    elif MODE == "min_float_half":
        x = torch.randn(32000, device="cuda").half()
        log("RESULT OK", float(torch.min(x)))

    else:
        log("RESULT FAIL unknown mode")

except Exception as e:
    import traceback

    log("RESULT FAIL", type(e).__name__, str(e)[:300])
    log(traceback.format_exc()[-1200:])
