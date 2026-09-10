# 8 个兼容坑：报错原文、根因、验证

这些问题的共同特点是：**报错信息具有误导性**，而且**只修其中几个仍然跑不起来**。本文按"症状 → 报错原文 → 根因 → 如何验证 → 解法"逐条记录。

> 排查方法统一为：**把可疑操作放进独立进程、带硬超时跑一次**（见 `diagnostics/`）。
> 原因见坑 1 和坑 3 —— ZLUDA 下**失败经常表现为挂死而不是报错**，同进程排查会卡在第一个问题上。

---

## 坑 1：GPU FFT 不可用，且失败后进程空转

### 症状

`2-get-sv.py`（v2Pro 的 SV 特征提取）跑很久不结束，`7-sv_cn` 目录**始终 0 个文件**；控制台刷满同一个异常。

### 报错原文

```
File "GPT_SoVITS/eres2net/kaldi.py", line 524, in fbank
    spectrum = torch.fft.rfft(strided_input).abs()
RuntimeError: cuFFT error: CUFFT_INTERNAL_ERROR
```

同时 PyTorch 给出关键提示：

```
kaldi.py:524: UserWarning: ComplexHalf support is experimental and many operators
don't support it yet.
```

### 但"改成 fp32"没用

第一反应是"fp16 的问题，改成 float32"。**实测无效**：

| 探针 | baseline | `HSA_OVERRIDE_GFX_VERSION=10.3.0` |
|---|---|---|
| `rfft` (fp32) | **HANG** cuFFT error | **HANG** cuFFT error |
| `stft` | **HANG** cuFFT error | **HANG** cuFFT error |
| `matmul` (fp32) | **OK** 3.3s | **OK** 5.2s |

**结论：整个 GPU FFT 路径都废，与 dtype 无关；`HSA_OVERRIDE_GFX_VERSION` 也救不了。** 矩阵乘是好的（说明 gfx1032 补丁把 rocBLAS 修通了），FFT 不是。

### 根因

ZLUDA 的 `cufft.dll` 是个**转发壳**，它引用的是 ROCm 的 `hipfft`：

```powershell
$b=[System.IO.File]::ReadAllBytes('C:\ZLUDA\bin\cufft.dll')
[System.Text.Encoding]::ASCII.GetString($b).Contains('hipfft')   # → True
```

调用链是 `torch → ZLUDA\cufft.dll → ROCm\hipfft.dll → rocfft.dll`，而 **gfx1032 补丁只覆盖了 rocBLAS 与 Tensile 内核，没有覆盖 rocFFT**，所以 rocFFT 在该架构上不可用。

### 第二个更隐蔽的问题：报错后进程不退出

在独立探针里观察到：**异常已经打印出来了，但进程继续空转**，直到 120 秒硬超时才被 `taskkill` 杀掉。这解释了为什么 `2-get-sv.py` "看起来在跑"却什么都不产出——它其实早就失败了。

> **教训：在 ZLUDA 上"进程还活着"完全不能说明它在正常工作。**

### 解法

把 FFT 挪到 CPU。共 **3 处调用点**：

| 文件 | 位置 | 内容 |
|---|---|---|
| `GPT_SoVITS/module/mel_processing.py` | `spectrogram_torch` | `torch.stft(...)` |
| `GPT_SoVITS/module/mel_processing.py` | `mel_spectrogram_torch` | `torch.stft(...)` |
| `GPT_SoVITS/eres2net/kaldi.py` | `fbank` | `torch.fft.rfft(...)` |

做法：检测到 ZLUDA 设备时，把张量搬到 CPU 做 FFT，再把结果搬回原设备。**注意 CPU 的 FFT 不支持 fp16**，所以 CPU 路径固定用 float32 计算、最后转回原 dtype。

```python
def _fft_needs_cpu() -> bool:
    forced = os.environ.get("GPT_SOVITS_FFT_CPU")
    if forced is not None:
        return forced.strip().lower() in ("1", "true", "yes", "on")
    if not torch.cuda.is_available():
        return False
    try:
        return "ZLUDA" in torch.cuda.get_device_name(0)
    except Exception:
        return False
```

用环境变量 `GPT_SOVITS_FFT_CPU=1` 可强制开启。

### 验证

| 检查 | 结果 |
|---|---|
| `mel_spectrogram_torch`（真实音频，±1 内） | cuda fp32 maxdiff **3.58e-07** ✅ |
| `mel_spectrogram_torch`（cuda fp16） | maxdiff **1.87e-03** ✅ |
| `spectrogram_torch` | maxdiff **0.00e+00** ✅ |
| `Kaldi.fbank`（cuda fp32） | maxdiff **1.08e-04** ✅ |
| `SV.compute_embedding3`（真实类，`is_half=True`） | **从"永久挂死"变成 0.090 秒/次** ✅ |
| `logs/GBC/7-sv_cn` | **0 个文件 → 317/317（约 90 秒跑完）** ✅ |

---

## 坑 2：打印 CUDA 张量会挂死

### 症状

处理某些音频时，`mel_spectrogram_torch` 一开始就卡住，且**卡在打印语句上**。

### 报错原文

没有报错——**它挂死**。输出停在：

```
min value is
```

### 根因

`mel_processing.py` 里有这样两行诊断代码：

```python
if torch.min(y) < -1.2:
    print("min value is ", torch.min(y))
if torch.max(y) > 1.2:
    print("max value is ", torch.max(y))
```

`print(a, b)` 会**先求值所有参数**，所以能打出 `min value is` 说明 `torch.min(y)` 已经返回了。真正挂住的是 `print` 内部的 `str(0维CUDA张量)` —— 它会触发设备到主机的同步拷贝。

探针实测（这是关键对比）：

| 探针 | 结果 |
|---|---|
| `float(torch.min(x))` | **OK**，4.2s |
| `torch.min(x).item()` | **OK**，4.2s |
| **`str(torch.min(x))`** | **HANG**（45 秒超时被杀） |
| `int(torch.argmin(x))` | **HANG** |

**`float()` / `.item()` 都正常，只有 `str()` 会挂。**

### 为什么这在训练里更致命

`s2_train` 会调用 `mel_spectrogram_torch(y_hat, ...)`，而 `y_hat` 是**生成器的输出**。**训练早期权重随机，`y_hat` 极容易超出 ±1.2** → 触发打印 → **整个训练挂死**。这是一个必然会在某个时刻踩到的坑。

### 解法

把 4 处 `print(..., torch.min/max(y))` 改成 `float(...)`（`spectrogram_torch` 与 `mel_spectrogram_torch` 各 2 处）。

### 验证

| 输入 | 修复前 | 修复后 |
|---|---|---|
| 真实音频（±1 内，不触发分支） | OK | OK |
| 超范围输入（min ≈ -11，**触发分支**） | **HANG** | **打印 `min value is -11.058...` 并完成，0.00s** ✅ |

---

## 坑 3：干完活进程不退出

### 症状

`2-get-sv.py` 把 317 个文件**全部写完**了，但进程**永不退出**；webui 的 `Popen(...).wait()` 永远等不到，界面一直显示"运行中"。

监控日志是铁证——文件数从 20:55 到 22:16（**80 多分钟**）一直是 `317/317`，进程却还在：

```
[20:53:48] 7-sv_cn = 317/317
...
[22:16:08] 7-sv_cn = 317/317
FINISHED  exit=  7-sv_cn = 317/317
```

### 根因

探针实测矩阵（每个独立进程、40 秒超时）：

| 探针 | 结果 |
|---|---|
| `torch_import`（只导入 torch） | OK 2.5s |
| `alloc_exit`（分配 CUDA 张量后直接退出） | **HANG** |
| `alloc_sync_exit`（**退出前 synchronize**） | **OK 3.3s** ✅ |
| `alloc_os_exit`（分配后 `os._exit(0)`） | **HANG** |
| `svmodel_exit`（只加载模型，不计算） | OK |
| `sv_full_exit`（算完 embedding 直接退出） | **HANG** |
| `sv_full_sync_exit`（**算完 + synchronize**） | **OK 5.0s** ✅ |
| `torch_save_exit`（存盘后退出） | **HANG** |

**规律非常干净：退出时只要还有未同步的 GPU 任务，就死锁在 `ExitProcess` 的 DLL 卸载阶段。**

注意 `os._exit(0)` **也救不了** —— 因为死锁发生在更底层（DLL_PROCESS_DETACH / 驱动卸载），Python 层无法干预。

### 解法

用 `site-packages` 的 `.pth` 注册一个 `atexit` 钩子，在进程退出前 `torch.cuda.synchronize()`。放在 `.pth` 而不是改某个入口文件，是因为**训练/前处理都是 `Popen` 拉起的独立进程**，必须对任意解释器生效。

```python
def _sync_before_exit():
    mod = sys.modules.get("torch")        # 只有真导入了 torch 才做，避免拖慢所有进程
    if mod is None:
        return
    try:
        if mod.cuda.is_available():
            mod.cuda.synchronize()
    except Exception:
        pass

atexit.register(_sync_before_exit)
```

### 验证

| 探针 | 修复前 | 修复后 |
|---|---|---|
| `alloc_exit` | HANG | **OK 3.9s** ✅ |
| `sv_full_exit` | HANG | **OK 5.7s** ✅ |
| `torch_save_exit` | HANG | **OK 3.9s** ✅ |

实测 `2-get-sv.py` 从"写完文件后永久卡住"变成 `rc=0` 干净退出；训练结束后 webui 也能正常回到可点击状态。

---

## 坑 4：cuBLASLt 是空壳，所有 Linear 报错

### 症状

s2 训练能启动、能加载预训练权重、能开始迭代，然后在第一个 `Linear` 上崩。

### 报错原文

```
File "GPT_SoVITS/module/models.py", line 939, in forward
    ge = self.ref_enc(y[:, :704] * y_mask, y_mask)
File "GPT_SoVITS/module/modules.py", line 527, in forward
    out = self.fc(input)
File "torch/nn/modules/linear.py", line 125, in forward
    return F.linear(input, self.weight, self.bias)
RuntimeError: CUDA error: CUBLAS_STATUS_NOT_SUPPORTED when calling
  `cublasLtMatmulAlgoGetHeuristic( ltHandle, computeDesc.descriptor(), ...)`
```

### 根因

`cublasLt`（cuBLAS 的 Lite API）在 ZLUDA 里**没有实现**。判别方法：

```powershell
# cublasLt.dll 不引用任何 ROCm 库 → 空壳
$b=[System.IO.File]::ReadAllBytes('C:\ZLUDA\bin\cublasLt.dll')   # 仅 185,856 字节
# 对比 cublas.dll：引用 rocblas → 真转发
```

### 实测矩阵

| 探针 | baseline | `DISABLE_ADDMM_CUDA_LT=1` |
|---|---|---|
| `linear_fp32` | **FAIL** | **OK** ✅ |
| `linear_3d_fp16` | **FAIL** | **OK** ✅ |
| `linear_refenc_fp32` | **FAIL** | **OK** ✅ |
| `linear_bigk_fp32` | **FAIL** | **OK** ✅ |
| `addmm_fp32` | **FAIL** | **OK** ✅ |
| `matmul_bt_fp32` | OK | OK |
| `conv1d_fp16` | OK | OK |

**一个环境变量解决全部 Linear/addmm 问题。**

### 解法

在兼容层里设：

```python
os.environ.setdefault("DISABLE_ADDMM_CUDA_LT", "1")
os.environ.setdefault("TORCH_BLAS_PREFER_HIPBLASLT", "0")
```

**必须在导入 torch 之前设置**（PyTorch 会缓存这个标志），所以放在 `.pth` 里、torch 导入之前执行。

> 有趣的是：前一轮排查者其实**已经摸到了这个开关** —— 它躺在一个从未被任何地方 import 的 `zluda_patch.py` 里。**光找到问题不够，必须接到实际执行路径上。**

---

## 坑 5：cuDNN 无法做卷积反向

### 症状

s2 训练穿过坑 4 之后，在下一个位置崩。

### 报错原文

```
File "GPT_SoVITS/module/modules.py", line 197, in forward
    acts = commons.fused_add_tanh_sigmoid_multiply(x_in, g_l, n_channels_tensor)
RuntimeError: The following operation failed in the TorchScript interpreter.
RuntimeError: nvrtc: error: invalid value for --gpu-architecture (-arch)
```

—— 注意这是**坑 6**。修掉坑 6 之后才露出坑 5：

```
File "GPT_SoVITS/module/data_utils.py", line ...
RuntimeError: GET was unable to find an engine to execute this computation
```

### 根因实测定性

逐层隔离（带上 `DISABLE_ADDMM_CUDA_LT=1` 排除干扰）：

| 探针 | 结果 |
|---|---|
| `gn_fwd_bwd_fp16`（GroupNorm） | OK |
| `ln_fwd_bwd`（LayerNorm） | OK |
| `linear_bwd_fp16` | OK |
| `interpolate_fp16` | OK |
| `instance_norm_fwd` | OK |
| **`conv1d_bwd_fp16`** | **FAIL** `GET was unable to find an engine` |
| **`convtranspose_bwd_fp16`** | **HANG** |
| **`batchnorm_fwd`** | **FAIL** `cuDNN error: CUDNN_STATUS_EXECUTION_FAILED` |
| `net_conv_linear_bwd` / `net_conv_gn_bwd` / `net_full_bwd` | 全 FAIL |

**结论：conv 前向正常，反向挂；BatchNorm 也挂。** 而且 `dev` 那一步（`get_device_name`）就指向了 `cudnn.dll`（260 KB 的转发壳，拿不出可用实现）。

### 解法

关掉 cuDNN，让 PyTorch 走原生卷积实现：

```python
torch_mod.backends.cudnn.enabled = False
```

### 验证

| 探针 | 修复前 | 修复后 |
|---|---|---|
| `conv1d_bwd` | FAIL | **OK** ✅ |
| `conv1d_bwd_fp16` | FAIL | **OK** ✅ |
| `convtranspose_bwd_fp16` | **HANG** | **OK** ✅ |
| `net_conv_linear_bwd` | FAIL | **OK** ✅ |
| `net_conv_gn_bwd` | FAIL | **OK** ✅ |
| `net_full_bwd` | FAIL | **OK** ✅ |
| `batchnorm_fwd` | FAIL | **OK** ✅ |
| `gn_fwd_bwd_fp16` | OK | OK |

**8/8 通过。**

---

## 坑 6：TorchScript 融合内核无法编译

### 报错原文

```
File "GPT_SoVITS/module/modules.py", line 197, in forward
    acts = commons.fused_add_tanh_sigmoid_multiply(x_in, g_l, n_channels_tensor)
RuntimeError: The following operation failed in the TorchScript interpreter.
RuntimeError: nvrtc: error: invalid value for --gpu-architecture (-arch)
nvrtc compilation failed:
#define NAN __int_as_float(0x7fffffff)
...
extern "C" __global__
void fused__autocast_to_f_9475641431357273780(half* tinput_b_1, half* tinput_a_1, float* aten_add) {
```

### 根因

`commons.py` 里 `fused_add_tanh_sigmoid_multiply` 带 `@torch.jit.script`。TorchScript 的**张量融合器**把 `input_a + input_b` 和 autocast 融合成一个 CUDA 内核，然后**用 NVRTC 现场编译**。

问题在于架构名：ZLUDA 把设备 capability 报成 `(8, 8)`，PyTorch 于是传 `--gpu-architecture=compute_88`，而 ZLUDA 的 `nvrtc.dll` 转发给 AMD 的 HIPRTC，**HIPRTC 只认 `gfx1032` 这类名字** → 编译失败。

### 一个失败的假设

一开始怀疑是"关掉 JIT 导致的形状问题"（因为 `t2s_model.py` 里也有 `@torch.jit.script` 的类）。**做了对照实验**：`PYTORCH_JIT=1` 重跑 s1，**失败完全一样** → 证明该形状问题与 JIT 无关。（这个结论后来导向了坑 7。）

### 解法

让 TorchScript 整体退回 eager：

```python
os.environ.setdefault("PYTORCH_JIT", "0")
```

必须在导入 torch 前设置。

### 验证

| 探针（无 env） | `PYTORCH_JIT=1` | `PYTORCH_JIT=0` |
|---|---|---|
| `default` 模式返回 dtype | float32（走了 TorchScript，autocast 未生效） | **float16（eager，autocast 正常）** |
| s2 训练 | 崩在 `modules.py:197` | **穿过该点继续训练** ✅ |

dtype 的变化本身就证明 `@torch.jit.script` 已退化成普通函数。

---

## 坑 7：小张量的 CUDA 归约返回错误值

### 症状

s1 训练报"形状对不上"，而且**每次报的数字还不一样**。

### 报错原文（第一次）

```
File "GPT_SoVITS/AR/models/t2s_model.py", line 508, in forward_old
    loss = F.cross_entropy(logits, targets, reduction="sum")
RuntimeError: Expected target size [4, 0], got [4, 120]
```

### 排查过程（这几步很关键）

**第一步：怀疑数据。** 扫了全部 79 个 batch，确认**数据集完全一致**（`semantic_ids` 的填充宽度 == `semantic_ids_len.max()`）：

```
num_workers=0 to_cuda=False -> 79 batches, mismatches=0
num_workers=2 to_cuda=False -> 79 batches, mismatches=0
num_workers=4 to_cuda=False -> 79 batches, mismatches=0
num_workers=0 to_cuda=True  -> 79 batches, mismatches=0
num_workers=4 to_cuda=True  -> 79 batches, mismatches=0
```

数据、多进程传输、H2D 拷贝**全都干净**。

**第二步：给真实前向插桩**，打印每一步的形状。结果：

```
phoneme_ids : (4, 62)   lens [28, 22, 62, 42]
semantic_ids: (4, 105)  lens [105, 102, 96, 90]
x_lens.max()  : 62
y_lens.max()  : 105
----------------------------------------------------------
  pad_y_eos -> y (4, 105)  targets (4, 106)
  transformer: in (4, 167, 512) -> out (4, 167, 512)
  ar_predict_layer: in (4, 0, 512)      ← 切片切出了 0 长度！
  cross_entropy: logits (4, 1025, 0)  targets (4, 106)
```

transformer 输入 `167 = 62 + 105` 完全正常，但 `xy_dec[:, x_len-1:]` 在 `x_len=62` 时应得 `167-61=106`，却得到 **0**。

**第三步：验证"用张量做切片下标"是否坏掉** —— 结果**完全正常**：

```
x[:, 61:].shape            (int bound)    -> (4, 106, 512)
x[:, x_len-1:].shape       (tensor bound) -> (4, 106, 512)
```

**第四步：把 `forward_old` 的函数体逐步复制出来带打印**，终于抓到：

```
x_len = 62  y_len = 105
...
(x_len-1).item() = 167          ← 同一个变量，从 62 变成了 168！
xy_dec[:, x_len-1:]        (4, 0, 512)
xy_dec[:, int(x_len)-1:]   (4, 1, 512)
```

`x_len` 是 `x_lens.max()` 返回的 **0 维 CUDA 张量**。刚打出来是 62，经过中间几十个张量分配之后**再读它变成了 167/168** —— 被后续分配覆盖/复用了。

### 根因

**ZLUDA 下小张量的 CUDA 归约结果不可靠**：

- 0 维归约结果的生命周期有问题，**延迟读取会拿到脏值**；
- 更糟的是，`y_lens` 这种**输入**小张量在分配压力下值也会变（后面又观察到 mask 宽度变成 119 / 61，而真值是 134 —— 那些数字像是**别的 batch 的长度值**）。

这解释了为什么报错里的数字每次都不一样。

### 解法（两层）

**第一层：mask 宽度改为从数据张量形状推导**，彻底不信任 `*_lens` 的数值：

```python
# 原来是：x_mask = make_pad_mask_left(x_lens)      # 宽度取自可能变脏的 x_lens.max()
x_len = x.shape[1]                                 # 填充后的真实宽度
x_mask = make_pad_mask_left(x_lens, x_len)         # 显式传入，不再让库自己去 max()
```

同时让 `make_pad_mask` / `make_pad_mask_left` 中**显式传入的 `max_len` 具有权威性**（上游语义是 `max(max_len, lengths.max())`，会被脏值顶大）。

**第二层：长度归约移 CPU**（这些张量只有几个元素，CPU 计算精确且零成本）：

```python
def seq_max_len(lengths: torch.Tensor) -> int:
    if lengths.is_cuda:
        return int(lengths.detach().to("cpu").max())
    return int(lengths.max())
```

### 验证

失败点从 `t2s_model.py:508`（epoch 0/79）一路推进 → `1/79` → 直到**完成 79/79 全部步数**：

```
Epoch 0/0  ---------------- 79/79
`Trainer.fit` stopped: `max_epochs=1` reached.
GPT_weights_v2Pro after : 1
```

---

## 坑 8：单卡套 DDP 会原生崩溃

### 报错原文

不是 Python 异常，而是**原生崩溃 + 调用栈**：

```
Exception Code: 0xC0000005
 #0 0x00007ffd866c14a1 c10d::Reducer::mark_variable_ready_dense(unsigned __int64)
 #1 0x00007ffd866c135a c10d::Reducer::mark_variable_ready(unsigned __int64)
 #2 0x00007ffd866ba759 c10d::Reducer::autograd_hook(unsigned __int64)
 ...
 #7 0x00007ffd85e094b1 torch::autograd::Engine::evaluate_function(...)
 #8 0x00007ffd85e0ee75 torch::autograd::Engine::thread_main(...)
```

### 根因

`s1_train.py` 只要检测到 CUDA 就无条件套 DDP：

```python
strategy=DDPStrategy(process_group_backend="nccl" if platform.system() != "Windows" else "gloo")
if torch.cuda.is_available()
else "auto",
```

崩溃发生在 **DDP 梯度归约器的反向钩子**里（`mark_variable_ready_dense` 会把梯度拷进 bucket）。单卡场景下 DDP 本来就没意义，却引入了额外的梯度分桶与拷贝路径。

### 解法

只有多卡时才套 DDP：

```python
strategy=DDPStrategy(...)
if torch.cuda.is_available() and torch.cuda.device_count() > 1
else "auto",
```

### 连带修复：又一处"无条件 dist 调用"

去掉 DDP 之后，立刻暴露出第二个同类问题：

```
File "GPT_SoVITS/AR/data/bucket_sampler.py", line 42, in __init__
    num_replicas = dist.get_world_size() if torch.cuda.is_available() else 1
ValueError: Default process group has not been initialized
```

作者的判断条件是"CUDAS 是否可用"，而不是"进程组是否已初始化"。单卡无进程组时必崩。修法：

```python
num_replicas = dist.get_world_size() if (torch.cuda.is_available() and dist.is_initialized()) else 1
rank = dist.get_rank() if (torch.cuda.is_available() and dist.is_initialized()) else 0
```

**同一个毛病在 `GPT_SoVITS/module/core_vq.py` 里还有一处**，而且它**只在首次推理时**触发（VQ 码本的惰性 kmeans 初始化）：

```
File "GPT_SoVITS/module/core_vq.py", line 165, in init_embed_
    if dist.get_rank() == 0:
ValueError: Default process group has not been initialized
```

这里作者**判断了** `SyncFunction` 那一步，却**漏判了下面三行**：

```python
if dist.is_available() and dist.is_initialized():   # ← 判断了
    data = SyncFunction.apply(data)

if dist.get_rank() == 0:              # ← 没判断
    embed, cluster_size = kmeans(...)
dist.broadcast(embed, src=0)          # ← 没判断
dist.broadcast(cluster_size, src=0)   # ← 没判断
```

单进程下显然应该是"本地跑 kmeans，不广播"。`expire_codes_` 里同样的问题一并修掉。

> **排查建议**：在整个项目里搜一遍，把所有无条件调用都找出来加判断：
> ```bash
> grep -rn 'dist\.\(get_rank\|get_world_size\|broadcast\|all_reduce\|all_gather\|barrier\)' --include=*.py
> ```

---

## 附：几个"看着像错其实无害"的东西

| 输出 | 真相 |
|---|---|
| `KeyError: ''` + 紧跟 `Warning: ... Skipped.` + 计数继续 | 坏数据被 `except` 捕获并跳过。作者把 `traceback.print_exc()` 写在 `except` 里，所以"被完美处理的异常"长得跟崩溃一样。**判据是后面有没有继续输出** |
| `FutureWarning: You are using torch.load with weights_only=False` | PyTorch 预告未来默认值变更。加载本地可信文件无风险 |
| `FATAL: kernel 'fmha_cutlassF_f16_aligned_64x64_rf_sm80' is for sm80-sm100, but was built for sm37` | ZLUDA 拒绝某个 CUTLASS 注意力内核并回退。**刷几千行纯属噪音**，可用关闭 SDPA 后端消除（见 VERIFICATION 第 9 节） |
| `Precision 16-mixed is not supported by the model summary` | Lightning 说它没法准确估算模型体积。无影响 |
| `Disabling CTC timestamps because the checkpoint did not initialize 86 of 86 required CTC tensors` | FunASR 的模型没带 CTC 头权重，关掉时间戳功能。**下一句就写着 `Text transcription remains available`** —— 转写照常 |
| `UnicodeEncodeError: 'gbk' codec can't encode character '\u2022'` | rich 进度条想往 GBK 控制台写 `•`。启动器里加 `chcp 65001` + `PYTHONUTF8=1` 即可 |

---

## 附二：会**静默毁掉数据**的坑 —— 别开 FunASR 的 batch

这个不属于"8 个兼容问题"，但后果比它们都严重，所以单独列出来。

**诱因**：Fun-ASR-Nano 里确实有一个正经的批实现 `_inference_llm_batch`（左填充、修正 `position_ids`、一次 `llm.generate` 跑完一批），而且因为官方 checkpoint 不含 CTC 权重、FunASR 会自动禁用 CTC 解码器，这个分支**不改一行代码就可达**。看到这些的人很容易顺手把 `batch_size_s` 调大来加速。

**实际后果**：输出变成退化的重复文本，**但进程不报错、退出码为 0、进度条一切正常**。

| 方式 | 耗时 | 输出 |
|---|---|---|
| 串行 | 1.12 s/条 | ✅ `よろしくね。ええ、ケチ。あ。` |
| 变长批 ×8 | 5.63 s/条 | ❌ `よろろろしくしくくくくくくねくねね…` |
| **等长批 ×8（零填充）** | 0.41 s/条 | ❌ 仍然退化 |

**决定性证据是"等长批"**：8 条**完全相同**的输入、零填充、`position_ids` 就是平凡的 `0..T-1` —— 输出依然是垃圾，**而且 8 条结果互不相同**（长度 43/44/45/46/47 混杂）。

确定性贪心解码对相同输入给出 8 个不同答案，说明**批路径的数值本身就是错的**，与填充逻辑无关。这与坑 7（ZLUDA 小张量归约返回错误值）同源。

**为什么危险**：ASR 是训练数据的前置步骤。批量输出的垃圾会直接写进 `*.list` 标注文件，然后被 SoVITS/GPT 训练当成正确答案学进去。你从控制台上**看不出任何异常**。

**结论**：
- 不要用 `batch_size_s`、`merge_vad` 之类的开关给 ASR 加速。
- 另外注意在 CPU 上 FunASR 本来就会强行关掉组批（`auto_model.py` 里 `if kwargs["device"] == "cpu": batch_size = 0`），所以这个坑只在 GPU 上才会踩到。
- 顺带一条测量陷阱：**不加 `zluda.exe` 时 `torch.cuda.is_available()` 是 False，而 `AutoModel(device="cuda")` 不报错、直接退回 CPU**。症状是"GPU"和 CPU 的耗时逐条几乎完全一样。做任何 GPU 对比前先确认 `torch.cuda.is_available()`。

完整实测数据见 [`VERIFICATION.md` 第 9 节](VERIFICATION.md)。

