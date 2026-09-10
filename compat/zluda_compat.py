"""Compatibility shims for running PyTorch on ZLUDA (AMD/ROCm on Windows).

Activated for every interpreter that processes site-packages, via the companion
zluda_compat.pth.  Disable the whole thing with ZLUDA_COMPAT=0.

Why each shim exists (all verified on ROCm 6.2.4 + ZLUDA 3 + RX 6600 / gfx1032,
torch 2.5.1+cu118):

1. DISABLE_ADDMM_CUDA_LT=1
   ZLUDA's cublasLt.dll is a stub, so every Linear/addmm that goes through the
   cuBLASLt heuristic API dies with:
       CUDA error: CUBLAS_STATUS_NOT_SUPPORTED when calling
       `cublasLtMatmulAlgoGetHeuristic(...)`
   Forcing the plain cuBLAS path fixes it.

2. torch.backends.cudnn.enabled = False
   ZLUDA's cudnn.dll cannot serve convolution *backward*: fp16 conv backward
   raises "GET was unable to find an engine to execute this computation",
   ConvTranspose1d backward hangs outright, and BatchNorm1d fails with
   CUDNN_STATUS_EXECUTION_FAILED.  PyTorch's native conv kernels work for both
   directions, so cuDNN is switched off.

3. synchronize before exit
   With unsynchronised GPU work still in flight, the process deadlocks inside
   ExitProcess -> DLL_PROCESS_DETACH and never terminates (os._exit(0) does not
   help either).  GPT-SoVITS hits this at the end of every preprocessing step,
   so webui's Popen(...).wait() blocks forever and the UI looks stuck even
   though the work finished.
"""
import atexit
import builtins
import os
import sys

VERBOSE = os.environ.get("ZLUDA_COMPAT_VERBOSE", "").strip().lower() in ("1", "true", "yes", "on")


def _say(*a):
    if VERBOSE:
        print("[zluda_compat]", *a, flush=True)


# ---------------------------------------------------------------- 1. env flags
os.environ.setdefault("DISABLE_ADDMM_CUDA_LT", "1")
os.environ.setdefault("TORCH_BLAS_PREFER_HIPBLASLT", "0")
# ZLUDA's nvrtc cannot compile the fused kernels the TorchScript fuser emits for
# this device ("nvrtc: error: invalid value for --gpu-architecture (-arch)"),
# which aborts s2 training inside commons.fused_add_tanh_sigmoid_multiply.
# Running eagerly removes TorchScript and therefore the NVRTC path entirely.
os.environ.setdefault("PYTORCH_JIT", "0")

_FLAGS_APPLIED = False
_APPLYING = False


def _apply_torch_flags(torch_mod):
    """Turn off the backends that ZLUDA cannot serve.

    Retried after every torch-related import until it sticks: while torch is
    still initialising, sys.modules["torch"] exists but is only partially built
    and torch.backends is not reachable yet, so an early attempt must not be
    treated as final.
    """
    global _FLAGS_APPLIED, _APPLYING
    if _FLAGS_APPLIED or _APPLYING:
        return
    _APPLYING = True
    try:
        backends = getattr(torch_mod, "backends", None)
        cudnn = getattr(backends, "cudnn", None) if backends is not None else None
        if cudnn is None:
            return  # torch still initialising; retry on the next import
        cudnn.enabled = False
        try:
            backends.cuda.matmul.allow_tf32 = False
        except Exception as e:
            _say("could not disable tf32:", e)
        # ZLUDA cannot launch the CUTLASS/fmha attention kernels, so PyTorch keeps
        # probing and rejecting them, printing thousands of
        # "FATAL: kernel `fmha_cutlassF_...` is for sm80-sm100" lines to stderr
        # before falling back anyway.  Turn those backends off and let the math
        # backend handle attention.
        for flag in ("enable_cudnn_sdp", "enable_flash_sdp", "enable_mem_efficient_sdp"):
            setter = getattr(backends.cuda, flag, None)
            if setter is None:
                continue
            try:
                setter(False)
                _say(flag, "-> False")
            except Exception as e:
                _say("could not call", flag, ":", e)
        _FLAGS_APPLIED = True
        _say("cudnn.enabled = False, matmul.allow_tf32 = False")
    except Exception as e:
        _say("could not apply torch flags:", e)
    finally:
        _APPLYING = False


# --------------------------------------------- 2. torch flags right after import
_orig_import = builtins.__import__


def _import_hook(name, globals=None, locals=None, fromlist=(), level=0):
    mod = _orig_import(name, globals, locals, fromlist, level)
    if not _FLAGS_APPLIED and (name == "torch" or name.startswith("torch.")):
        t = sys.modules.get("torch")
        if t is not None:
            _apply_torch_flags(t)
    return mod


# --------------------------------------------------------- 3. sync before exit
def _sync_before_exit():
    mod = sys.modules.get("torch")
    if mod is None:
        return
    try:
        if mod.cuda.is_available():
            mod.cuda.synchronize()
            _say("synchronised before exit")
    except Exception:
        pass


def install():
    builtins.__import__ = _import_hook
    atexit.register(_sync_before_exit)
    # If torch was somehow already imported before this module, fix it up now.
    t = sys.modules.get("torch")
    if t is not None:
        _apply_torch_flags(t)
    _say("installed")


if os.environ.get("ZLUDA_COMPAT", "1").strip().lower() not in ("0", "false", "no", "off"):
    install()
