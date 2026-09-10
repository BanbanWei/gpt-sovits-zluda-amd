"""Run exactly ONE probe then exit. argv[1] selects the probe.

Kept deliberately tiny: each probe runs in its own process so that a ZLUDA
hang can be killed and reported as TIMEOUT instead of stalling everything.
"""
import os
import sys

MODE = sys.argv[1]
import _env
NOW = _env.gpt_sovits_root()
sys.path.insert(0, NOW)
sys.path.insert(0, os.path.join(NOW, "GPT_SoVITS", "eres2net"))


def log(*a):
    print(*a, flush=True)


def cuda():
    import torch

    return torch


log("START", MODE)

try:
    if MODE == "cuda_basic":
        t = cuda()
        log("RESULT OK", t.cuda.get_device_name(0), t.cuda.get_device_capability(0))

    elif MODE.startswith("rfft_"):
        t = cuda()
        dtype = {"fp32": t.float32, "fp16": t.float16}[MODE.split("_")[1]]
        x = torch.randn(1, 16000, device="cuda").to(dtype) if False else t.randn(1, 16000, device="cuda").to(dtype)
        y = t.fft.rfft(x, n=512, dim=-1)
        t.cuda.synchronize()
        log("RESULT OK", y.dtype, tuple(y.shape))

    elif MODE.startswith("fbank_"):
        tag = MODE.split("_")[1]
        t = cuda()
        import kaldi as Kaldi

        x = t.randn(1, 16000)
        if tag == "fp32":
            x = x.cuda().float()
        elif tag == "fp16":
            x = x.cuda().half()
        else:
            x = x.float()  # cpu
        f = Kaldi.fbank(x, num_mel_bins=80, sample_frequency=16000, dither=0)
        if str(x.device).startswith("cuda"):
            t.cuda.synchronize()
        log("RESULT OK", f.dtype, tuple(f.shape), str(f.device))

    elif MODE.startswith("sv_"):
        # exercise the REAL sv.py code path
        import torch

        tag = MODE.split("_")[1]
        import sv as svmod

        if tag == "fixed":
            # candidate fix: fbank in fp32, then cast features to the model dtype
            import kaldi as Kaldi
            from ERes2NetV2 import ERes2NetV2

            state = torch.load(svmod.sv_path, map_location="cpu", weights_only=False)
            m = ERes2NetV2(baseWidth=24, scale=4, expansion=4)
            m.load_state_dict(state)
            m.eval().half().cuda()
            w = torch.randn(1, 16000).cuda()
            feat = torch.stack([Kaldi.fbank(w0.float().unsqueeze(0), num_mel_bins=80, sample_frequency=16000, dither=0) for w0 in w])
            feat = feat.half()
            emb = m.forward3(feat)
            torch.cuda.synchronize()
            log("RESULT OK", "feat", tuple(feat.shape), "emb", tuple(emb.shape))
        else:
            is_half = tag == "fp16"
            device = "cpu" if tag == "cpu" else "cuda"
            s = svmod.SV(device, is_half)
            w = torch.randn(1, 32000)
            emb = s.compute_embedding3(w.to(device))
            if device == "cuda":
                torch.cuda.synchronize()
            log("RESULT OK", tuple(emb.shape), str(emb.device), str(emb.dtype))

    elif MODE.startswith("matmul_"):
        t = cuda()
        dt = {"fp32": t.float32, "fp16": t.float16, "bf16": t.bfloat16}[MODE.split("_")[1]]
        a = t.randn(512, 512, device="cuda").to(dt)
        b = t.randn(512, 512, device="cuda").to(dt)
        c = a @ b
        t.cuda.synchronize()
        log("RESULT OK", c.dtype, tuple(c.shape))

    elif MODE == "conv_fp16":
        t = cuda()
        y1 = t.nn.Conv1d(192, 192, 3, padding=1).cuda().half()(t.randn(2, 192, 1000, device="cuda").half())
        y2 = t.nn.ConvTranspose1d(192, 192, 4, stride=2, padding=1).cuda().half()(t.randn(2, 192, 500, device="cuda").half())
        y3 = t.nn.GroupNorm(4, 192).cuda().half()(t.randn(2, 192, 100, device="cuda").half())
        y4 = t.nn.LayerNorm(512).cuda().half()(t.randn(4, 100, 512, device="cuda").half())
        t.cuda.synchronize()
        log("RESULT OK", tuple(y1.shape), tuple(y2.shape), tuple(y3.shape), tuple(y4.shape))

    elif MODE == "autocast_fp16":
        t = cuda()
        with t.autocast("cuda", dtype=t.float16):
            r = t.randn(256, 256, device="cuda") @ t.randn(256, 256, device="cuda")
        t.cuda.synchronize()
        log("RESULT OK", r.dtype)

    elif MODE == "gradscaler":
        t = cuda()
        s = t.cuda.amp.GradScaler(enabled=True)
        m = t.nn.Linear(64, 64).cuda()
        loss = m(t.randn(8, 64, device="cuda")).sum()
        s.scale(loss).backward()
        s.step(t.optim.SGD(m.parameters(), lr=0.1))
        s.update()
        t.cuda.synchronize()
        log("RESULT OK gradscaler")

    elif MODE == "amp_train_step":
        # closest cheap analogue of an s2_train iteration
        t = cuda()
        net = t.nn.Sequential(
            t.nn.Conv1d(192, 192, 3, padding=1), t.nn.GroupNorm(4, 192), t.nn.LeakyReLU(0.1),
            t.nn.ConvTranspose1d(192, 192, 4, stride=2, padding=1),
        ).cuda().half()
        opt = t.optim.AdamW(net.parameters(), lr=1e-4)
        scaler = t.cuda.amp.GradScaler(enabled=True)
        with t.autocast("cuda", dtype=t.float16):
            out = net(t.randn(2, 192, 200, device="cuda").half())
            loss = out.float().pow(2).mean()
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        t.cuda.synchronize()
        log("RESULT OK amp_train_step loss=", float(loss))

    elif MODE == "stft":
        t = cuda()
        w = t.hann_window(2048, device="cuda")
        s = t.stft(t.randn(16000, device="cuda"), 2048, 640, 2048, window=w, return_complex=True)
        y = t.istft(s, 2048, 640, 2048, window=w)
        t.cuda.synchronize()
        log("RESULT OK", tuple(s.shape), tuple(y.shape))

    elif MODE == "pin_memory":
        t = cuda()
        p = t.empty(8).pin_memory()
        log("RESULT OK pinned=", p.is_pinned())

    elif MODE == "dataloader_workers":
        t = cuda()
        from torch.utils.data import DataLoader, TensorDataset

        ds = TensorDataset(t.arange(64).float().unsqueeze(1))
        dl = DataLoader(ds, batch_size=8, num_workers=4, persistent_workers=True, prefetch_factor=2)
        n = sum(1 for _ in dl)
        log("RESULT OK batches=", n)

    else:
        log("RESULT FAIL unknown mode")

except Exception as e:
    import traceback

    log("RESULT FAIL", type(e).__name__, str(e)[:300])
    log(traceback.format_exc()[-1500:])
