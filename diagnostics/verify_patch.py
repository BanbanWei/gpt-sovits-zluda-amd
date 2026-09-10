"""Verify the FFT CPU-fallback patches + the print fix, avoiding known ZLUDA landmines.

Known-bad under ZLUDA (avoided here): str()/repr() of a CUDA tensor, argmin/argmax,
and exiting the process with unsynchronised GPU work still in flight.
"""
import os
import sys
import time

import torch

NOW = r"D:\saves\GPT-SoVITS\GPT-SoVITS"
sys.path.insert(0, NOW)
sys.path.insert(0, os.path.join(NOW, "GPT_SoVITS", "eres2net"))
sys.path.insert(0, os.path.join(NOW, "GPT_SoVITS"))


def log(*a):
    print(*a, flush=True)


log("cuda      :", torch.cuda.is_available(), torch.cuda.get_device_name(0))

import kaldi as Kaldi
from module.mel_processing import mel_spectrogram_torch, spectrogram_torch

TOL = 2e-3
ARG = (2048, 128, 32000, 640, 2048, 0.0, None)

# ---------------------------------------------------- mel: in-range audio
# real training audio is normalised to [-1, 1] -> the min/max print is skipped
audio = torch.rand(1, 32000) * 2 - 1
ref = mel_spectrogram_torch(audio, *ARG)

for dt in (torch.float32, torch.float16):
    t0 = time.time()
    got = mel_spectrogram_torch(audio.cuda().to(dt), *ARG)
    torch.cuda.synchronize()
    d = float((ref - got.float().cpu()).abs().max())
    log("mel in-range  cuda %-8s : %s maxdiff=%.2e %s  %.2fs"
        % (str(dt).replace("torch.", ""), tuple(got.shape), d,
           "OK" if d < TOL else "*** MISMATCH ***", time.time() - t0))

# ------------------------------- mel: out-of-range, exercises the fixed print
noisy = torch.randn(1, 32000) * 3
log("-- expecting a 'min value is <float>' line below (this used to hang) --")
t0 = time.time()
got = mel_spectrogram_torch(noisy.cuda(), *ARG)
torch.cuda.synchronize()
log("mel out-of-range cuda   : %s  %.2fs  OK" % (tuple(got.shape), time.time() - t0))

# ------------------------------------------------------------ spectrogram
sp_ref = spectrogram_torch(audio, 2048, 32000, 640, 2048)
sp_got = spectrogram_torch(audio.cuda(), 2048, 32000, 640, 2048)
torch.cuda.synchronize()
d = float((sp_ref - sp_got.cpu()).abs().max())
log("spectrogram_torch       : %s maxdiff=%.2e %s"
    % (tuple(sp_got.shape), d, "OK" if d < TOL else "*** MISMATCH ***"))

# ------------------------------------------------------------------ fbank
x = torch.randn(1, 16000) * 0.3
fb_ref = Kaldi.fbank(x.clone(), num_mel_bins=80, sample_frequency=16000, dither=0)
fb32 = Kaldi.fbank(x.clone().cuda(), num_mel_bins=80, sample_frequency=16000, dither=0)
fb16 = Kaldi.fbank(x.clone().cuda().half(), num_mel_bins=80, sample_frequency=16000, dither=0)
torch.cuda.synchronize()
d32 = float((fb_ref - fb32.float().cpu()).abs().max())
d16 = float((fb_ref - fb16.float().cpu()).abs().max())
log("fbank fp32 cuda         : %s maxdiff=%.2e %s" % (tuple(fb32.shape), d32, "OK" if d32 < TOL else "*** MISMATCH ***"))
log("fbank fp16 cuda         : %s maxdiff=%.2e %s" % (tuple(fb16.shape), d16, "OK" if d16 < TOL else "*** MISMATCH ***"))

# ------------------------------------------- real SV end-to-end (was hanging)
import sv as svmod

t0 = time.time()
svm = svmod.SV("cuda", True)
log("SV model loaded (is_half=True) in %.1fs" % (time.time() - t0))

t0 = time.time()
emb = svm.compute_embedding3(torch.randn(1, 32000, device="cuda") * 0.3)
torch.cuda.synchronize()
log("SV compute_embedding3   : %s %s  %.2fs  <-- used to hang forever"
    % (tuple(emb.shape), emb.dtype, time.time() - t0))

t0 = time.time()
for _ in range(5):
    svm.compute_embedding3(torch.randn(1, 32000, device="cuda") * 0.3)
torch.cuda.synchronize()
log("SV x5 more              : %.2fs total (%.3fs each)" % (time.time() - t0, (time.time() - t0) / 5))

torch.cuda.synchronize()
log("VERIFY DONE")
