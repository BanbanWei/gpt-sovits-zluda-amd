"""Is slicing with a 0-dim CUDA tensor as the bound broken under ZLUDA?

t2s_model.forward_old does:
    x_len = x_lens.max()                      # 0-dim CUDA tensor
    logits = self.ar_predict_layer(xy_dec[:, x_len-1:])
which produced a 0-length slice on a (4, 167, 512) tensor with x_len == 62.
"""
import os
import torch

print("device:", torch.cuda.get_device_name(0))
print("=" * 60)

x = torch.zeros(4, 167, 512, device="cuda")
xl = torch.tensor([28, 22, 62, 42], device="cuda")
x_len = xl.max()

print("x_lens            :", xl.tolist())


def show(label, fn):
    try:
        r = fn()
        torch.cuda.synchronize()
        print("%-34s -> %s" % (label, r))
    except Exception as e:
        print("%-34s -> EXC %s: %s" % (label, type(e).__name__, str(e)[:120]))


show("x_len tensor repr", lambda: (str(x_len.dtype), tuple(x_len.shape)))
show("x_len.item()", lambda: x_len.item())
show("int(x_len)", lambda: int(x_len))
show("(x_len-1).item()", lambda: (x_len - 1).item())
show("float(x_len)", lambda: float(x_len))
show("__index__ via list index", lambda: list(range(1000))[x_len - 1])

show("x[:, 61:].shape            (int bound)", lambda: tuple(x[:, 61:].shape))
show("x[:, x_len-1:].shape       (tensor bound)", lambda: tuple(x[:, x_len - 1:].shape))
show("x[:, int(x_len)-1:].shape  (int() bound)", lambda: tuple(x[:, int(x_len) - 1:].shape))
show("x[:, (x_len-1).item():].shape (.item bound)", lambda: tuple(x[:, (x_len - 1).item():].shape))
show("x[:, x_len.item()-1:].shape", lambda: tuple(x[:, x_len.item() - 1:].shape))
show("x[:, :x_len].shape         (tensor end)", lambda: tuple(x[:, :x_len].shape))
show("x[:, :62].shape            (int end)", lambda: tuple(x[:, :62].shape))
show("x[:, x_len:].shape         (tensor start)", lambda: tuple(x[:, x_len:].shape))
show("x[:, 62:].shape            (int start)", lambda: tuple(x[:, 62:].shape))

# the exact expression from forward_old
show("exact: xy_dec[:, x_len-1:]", lambda: tuple(x[:, x_len - 1:].shape))

# and on CPU for comparison
xc = torch.zeros(4, 167, 512)
xlc = torch.tensor([28, 22, 62, 42])
show("CPU x[:, x_len-1:].shape", lambda: tuple(xc[:, xlc.max() - 1:].shape))

torch.cuda.synchronize()
print("DONE")
