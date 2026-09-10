"""Isolate the "work done but process never exits" hang under ZLUDA.

argv[1] selects the probe.  Every probe does its work, prints RESULT OK, and then
either returns normally, synchronises first, or hard-exits with os._exit(0).
"""
import os
import sys

MODE = sys.argv[1]
NOW = r"D:\saves\GPT-SoVITS\GPT-SoVITS"
sys.path.insert(0, NOW)
sys.path.insert(0, os.path.join(NOW, "GPT_SoVITS", "eres2net"))
sys.path.insert(0, os.path.join(NOW, "GPT_SoVITS"))


def log(*a):
    print(*a, flush=True)


log("START", MODE)
import torch

if MODE == "torch_import":
    log("RESULT OK cuda=", torch.cuda.is_available())

elif MODE == "pth_loaded":
    hook = sys.modules.get("zluda_exit_fix")
    log("RESULT OK zluda_exit_fix loaded=", hook is not None)

elif MODE == "alloc_exit":
    torch.randn(1000, device="cuda")
    log("RESULT OK allocated, exiting with unsynced work")

elif MODE == "alloc_sync_exit":
    torch.randn(1000, device="cuda")
    torch.cuda.synchronize()
    log("RESULT OK allocated + synced")

elif MODE == "alloc_os_exit":
    torch.randn(1000, device="cuda")
    log("RESULT OK allocated, hard exit")
    os._exit(0)

elif MODE == "svmodel_exit":
    import sv as svmod

    svm = svmod.SV("cuda", True)
    log("RESULT OK model loaded, no compute")

elif MODE == "svmodel_os_exit":
    import sv as svmod

    svm = svmod.SV("cuda", True)
    log("RESULT OK model loaded, hard exit")
    os._exit(0)

elif MODE == "sv_full_exit":
    import sv as svmod

    svm = svmod.SV("cuda", True)
    emb = svm.compute_embedding3(torch.randn(1, 32000, device="cuda") * 0.3)
    log("RESULT OK embedding", tuple(emb.shape))

elif MODE == "sv_full_sync_exit":
    import sv as svmod

    svm = svmod.SV("cuda", True)
    emb = svm.compute_embedding3(torch.randn(1, 32000, device="cuda") * 0.3)
    emb.cpu()
    torch.cuda.synchronize()
    log("RESULT OK embedding + sync", tuple(emb.shape))

elif MODE == "sv_full_os_exit":
    import sv as svmod

    svm = svmod.SV("cuda", True)
    emb = svm.compute_embedding3(torch.randn(1, 32000, device="cuda") * 0.3)
    log("RESULT OK embedding, hard exit", tuple(emb.shape))
    os._exit(0)

elif MODE == "torch_save_exit":
    torch.randn(1000, device="cuda")
    torch.save({"a": torch.zeros(4)}, os.path.join(NOW, "TEMP", "_probe_tmp.pth"))
    log("RESULT OK saved, exiting")

else:
    log("RESULT FAIL unknown mode")
