"""Probe cuBLASLt support: F.linear / conv under ZLUDA. argv[1] selects mode."""
import sys

MODE = sys.argv[1]
import torch
import torch.nn.functional as F


def log(*a):
    print(*a, flush=True)


log("START", MODE)
log("  DISABLE_ADDMM_CUDA_LT =", __import__("os").environ.get("DISABLE_ADDMM_CUDA_LT", "<unset>"))

try:
    if MODE == "linear_fp32":
        m = torch.nn.Linear(512, 512).cuda()
        o = m(torch.randn(8, 100, 512, device="cuda"))
        torch.cuda.synchronize()
        log("RESULT OK", tuple(o.shape))

    elif MODE == "linear_fp16":
        m = torch.nn.Linear(512, 512).cuda().half()
        o = m(torch.randn(8, 100, 512, device="cuda").half())
        torch.cuda.synchronize()
        log("RESULT OK", tuple(o.shape))

    elif MODE == "linear_refenc_fp32":
        # closest analogue of the layer that failed: ref_enc -> spectral(conv) -> fc
        m = torch.nn.Linear(768, 1024).cuda()
        y = torch.randn(4, 704, 768, device="cuda")
        o = m(y)
        torch.cuda.synchronize()
        log("RESULT OK", tuple(o.shape))

    elif MODE == "linear_bigk_fp32":
        m = torch.nn.Linear(2048, 1024).cuda()
        o = m(torch.randn(2, 500, 2048, device="cuda"))
        torch.cuda.synchronize()
        log("RESULT OK", tuple(o.shape))

    elif MODE == "linear_noncontig_fp32":
        m = torch.nn.Linear(512, 512).cuda()
        y = torch.randn(64, 512, device="cuda")[:, :200].t().contiguous().t()
        o = m(y)
        torch.cuda.synchronize()
        log("RESULT OK", tuple(o.shape))

    elif MODE == "matmul_bt_fp32":
        a = torch.randn(1000, 512, device="cuda")
        b = torch.randn(512, 1000, device="cuda")
        o = a @ b
        torch.cuda.synchronize()
        log("RESULT OK", tuple(o.shape))

    elif MODE == "addmm_fp32":
        o = torch.addmm(torch.randn(512, device="cuda"), torch.randn(1000, 512, device="cuda"), torch.randn(512, 512, device="cuda"))
        torch.cuda.synchronize()
        log("RESULT OK", tuple(o.shape))

    elif MODE == "conv1d_fp32":
        m = torch.nn.Conv1d(512, 512, 5, padding=2).cuda()
        o = m(torch.randn(8, 512, 1000, device="cuda"))
        torch.cuda.synchronize()
        log("RESULT OK", tuple(o.shape))

    elif MODE == "conv1d_fp16":
        m = torch.nn.Conv1d(512, 512, 5, padding=2).cuda().half()
        o = m(torch.randn(8, 512, 1000, device="cuda").half())
        torch.cuda.synchronize()
        log("RESULT OK", tuple(o.shape))

    elif MODE == "conv2d_fp16":
        m = torch.nn.Conv2d(1, 32, 3, padding=1).cuda().half()
        o = m(torch.randn(2, 1, 64, 64, device="cuda").half())
        torch.cuda.synchronize()
        log("RESULT OK", tuple(o.shape))

    elif MODE == "linear_3d_fp16":
        m = torch.nn.Linear(1024, 1024).cuda().half()
        o = m(torch.randn(4, 200, 1024, device="cuda").half())
        torch.cuda.synchronize()
        log("RESULT OK", tuple(o.shape))

    elif MODE == "embedding_fp32":
        e = torch.nn.Embedding(1025, 512).cuda()
        o = e(torch.randint(0, 1024, (4, 100), device="cuda"))
        torch.cuda.synchronize()
        log("RESULT OK", tuple(o.shape))

    elif MODE == "full_small_net_fp16":
        # generator-ish: conv + linear + groupnorm, half, with backward
        net = torch.nn.Sequential(
            torch.nn.Conv1d(192, 192, 3, padding=1),
            torch.nn.GroupNorm(4, 192),
            torch.nn.LeakyReLU(0.1),
        ).cuda().half()
        lin = torch.nn.Linear(192, 256).cuda().half()
        x = torch.randn(2, 192, 500, device="cuda").half()
        o = lin(net(x).transpose(1, 2))
        loss = o.float().pow(2).mean()
        loss.backward()
        torch.cuda.synchronize()
        log("RESULT OK", tuple(o.shape))

    else:
        log("RESULT FAIL unknown mode")

except Exception as e:
    log("RESULT FAIL", type(e).__name__, str(e)[:200])
