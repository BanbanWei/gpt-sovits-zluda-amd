"""Check whether zluda_compat's SDPA/cudnn flags actually applied."""
import os
import sys

import _env
sys.path.insert(0, _env.gpt_sovits_site_packages())
import torch

print("zluda_compat loaded      :", "zluda_compat" in sys.modules)
print("DISABLE_ADDMM_CUDA_LT    :", os.environ.get("DISABLE_ADDMM_CUDA_LT", "<unset>"))
print("PYTORCH_JIT              :", os.environ.get("PYTORCH_JIT", "<unset>"))
print("cudnn.enabled            :", torch.backends.cudnn.enabled)
print("matmul.allow_tf32        :", torch.backends.cuda.matmul.allow_tf32)

for name in ("cudnn_sdp_enabled", "flash_sdp_enabled", "mem_efficient_sdp_enabled", "math_sdp_enabled"):
    fn = getattr(torch.backends.cuda, name, None)
    try:
        print("%-24s : %s" % (name, fn() if fn else "<missing>"))
    except Exception as e:
        print("%-24s : ERR %s" % (name, e))

for name in ("enable_cudnn_sdp", "enable_flash_sdp", "enable_mem_efficient_sdp", "enable_math_sdp"):
    print("%-24s : %s" % (name, "present" if getattr(torch.backends.cuda, name, None) else "MISSING"))
