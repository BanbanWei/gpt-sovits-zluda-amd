# 算子可用性实测矩阵

本文是**所有结论的原始证据**。每个探针都在**独立进程 + 硬超时**下运行，判定为：

- **OK** — 正常返回
- **FAIL** — 抛出异常
- **HANG** — 超时被杀（**ZLUDA 特有的失败模式**）

复现方式：

```powershell
C:\ZLUDA\bin\zluda.exe <venv>\Scripts\python.exe diagnostics\run_modes.py diagnostics\probe_ops.py 60 <模式名> [<模式名> ...]
```

---

## 1. 设备与基础

| 探针 | 结果 |
|---|---|
| `cuda_basic` | **OK** — `AMD Radeon RX 6600 [ZLUDA]`，capability **`(8, 8)`** |
| `torch_import` | OK 2.5s |
| 矩阵乘 fp32 | **OK** 3.3s |
| 矩阵乘 fp16 / bf16 | OK |

**capability 报 `(8,8)` 的连锁影响**：GPT-SoVITS 的 `config.py` 据此判定 `is_half=True` → SV 特征提取把 wav 转 fp16 → 触发 fp16 的 `torch.fft.rfft` → 正是坑 1 的入口。

**注入是否传播到子进程**（决定性验证，因为 webui 用 `Popen` 拉起训练）：

```
PARENT  cuda : True    device: AMD Radeon RX 6600 [ZLUDA]
CHILD   cuda : True    device: AMD Radeon RX 6600 [ZLUDA]
```

✅ **注入会传播**。所以 `zluda.exe` 包住 webui 之后，它拉起的训练/前处理进程同样拿到 GPU。

---

## 2. BLAS / 线性层（坑 4）

| 探针 | baseline | `DISABLE_ADDMM_CUDA_LT=1` |
|---|---|---|
| `linear_fp32` | **FAIL** `CUBLAS_STATUS_NOT_SUPPORTED` | **OK** |
| `linear_3d_fp16` | **FAIL** 同上 | **OK** |
| `linear_refenc_fp32`（复现真实失败点） | **FAIL** 同上 | **OK** |
| `linear_bigk_fp32` | **FAIL** 同上 | **OK** |
| `addmm_fp32` | **FAIL** 同上 | **OK** |
| `matmul_bt_fp32` | OK | OK |
| `conv1d_fp16` | OK | OK |

报错原文：

```
RuntimeError: CUDA error: CUBLAS_STATUS_NOT_SUPPORTED when calling
  `cublasLtMatmulAlgoGetHeuristic(ltHandle, computeDesc.descriptor(), ...)`
```

**所有普通矩阵乘走 `cublasLt`，全部失败；只有裸 `@` 运算绕过了它。** 一个环境变量全解决。

---

## 3. 卷积 / 归一化（坑 5）

`DISABLE_ADDMM_CUDA_LT=1` + `PROBE_NO_CUDNN=1`：

| 探针 | baseline | cudnn 关闭后 |
|---|---|---|
| `conv1d_bwd` | FAIL | **OK** ✅ |
| `conv1d_bwd_fp16` | FAIL `GET was unable to find an engine` | **OK** ✅ |
| `convtranspose_bwd_fp16` | **HANG** | **OK** ✅ |
| `net_conv_linear_bwd` | FAIL | **OK** ✅ |
| `net_conv_gn_bwd` | FAIL | **OK** ✅ |
| `net_full_bwd` | FAIL | **OK** ✅ |
| `batchnorm_fwd` | FAIL `cuDNN error: CUDNN_STATUS_EXECUTION_FAILED` | **OK** ✅ |
| `gn_fwd_bwd_fp16` | OK | OK |
| `ln_fwd_bwd` | OK | OK |
| `linear_bwd_fp16` | OK | OK |
| `interpolate_fp16` | OK | OK |
| `instance_norm_fwd` | OK | OK |

**结论：conv 前向正常、反向挂；`ConvTranspose1d` 反向直接挂死；BatchNorm 前向就挂。关掉 cuDNN 后 8/8 通过。**

---

## 4. FFT（坑 1）

| 探针 | baseline | `HSA_OVERRIDE_GFX_VERSION=10.3.0` |
|---|---|---|
| `rfft` fp32 | **HANG** | **HANG** |
| `rfft` fp16 | **HANG** | — |
| `stft` fp32 | **HANG** | **HANG** |
| `matmul` fp32 | OK 3.3s | OK 5.2s |

报错原文（fp32 也一样）：

```
RuntimeError: cuFFT error: CUFFT_INTERNAL_ERROR
```

**注意所有 HANG 的实测耗时都是 60–120 秒**：异常其实**已经打印出来了**，但进程继续空转直到硬超时。这就是"看起来在跑其实早失败了"的机制。

**`HSA_OVERRIDE_GFX_VERSION=10.3.0` 无效** —— 所以这不是"内核库选错架构"，而是 FFT 这条路本身不通。

### 项目级修复后的数值正确性

| 检查 | 结果 |
|---|---|
| `mel_spectrogram_torch` cuda fp32 vs CPU 参考 | maxdiff **3.58e-07** ✅ |
| `mel_spectrogram_torch` cuda fp16 vs CPU 参考 | maxdiff **1.87e-03** ✅ |
| `spectrogram_torch` cuda vs CPU | maxdiff **0.00e+00** ✅ |
| `Kaldi.fbank` cuda fp32 vs CPU | maxdiff **1.08e-04** ✅ |
| `Kaldi.fbank` cuda fp16 vs CPU | maxdiff **5.51e-01**（fp16 表示精度所致，与上游 half 路径同量级） |
| `SV.compute_embedding3`（`is_half=True`） | **从永久挂死 → 0.090 秒/次**，输出 `(1, 20480) float16`，与代码注释一致 ✅ |

`7-sv_cn` 产出：**0 个文件 → 317/317**。

---

## 5. 归约与标量回读（坑 2 / 坑 7）

| 探针 | 结果 |
|---|---|
| `sync_only` | **OK** 4.1s |
| `min_item`（`torch.min(x).item()`） | **OK** 4.2s |
| `min_float`（`float(torch.min(x))`） | **OK** 4.2s |
| `max_float` | **OK** 3.9s |
| `sum_float` | **OK** 3.9s |
| `mean_float` | **OK** 3.9s |
| `bool_compare`（`if float(torch.min(x)) < -1.2`） | **OK** 4.1s |
| `cpu_copy` / `scalar_cpu_copy` / `min_cpu_after_cuda` | **OK** |
| `layernorm_sum` / `softmax_sum` | **OK** |
| `min_float_half` | **OK** 4.2s |
| **`min_str`（`str(torch.min(x))`）** | **HANG** 46.2s |
| **`argmin_int`（`int(torch.argmin(x))`）** | **HANG** 46.2s |
| **`randn_alloc`（分配后直接退出）** | **HANG** 45.2s（见第 6 节） |

**规律**：

- `float()`、`.item()`、`int()` 对**归约标量**都正常；
- **`str()` 张量挂死**（坑 2）；
- **`argmin` 挂死**；
- 归约结果**延迟读取**会变成脏值（坑 7 的根因，见下）。

### 坑 7 的原始证据

给真实前向插桩后抓到的决定性输出：

```
x_len = 62  y_len = 105
...
(x_len-1).item() = 167          ← 同一个 0 维张量变量，从 62 变成了 168
xy_dec[:, x_len-1:]        (4, 0, 512)      ← 切片出 0 长度
xy_dec[:, int(x_len)-1:]   (4, 1, 512)
```

`x_len` 来自 `x_lens.max()`。刚打印是 62，经过中间几十个张量分配后再读变成 167/168 —— **被后续分配覆盖**。

后续还观察到 mask 宽度出现 **119** 和 **61**（真值 **134**），那些数字像是**别的 batch 的长度值**。

### 数据通路是无辜的（排除性验证）

为排除"数据集/多进程/H2D 拷贝"的可能，扫了全部 79 个 batch：

```
num_workers=0 to_cuda=False -> 79 batches, mismatches=0
num_workers=2 to_cuda=False -> 79 batches, mismatches=0
num_workers=4 to_cuda=False -> 79 batches, mismatches=0
num_workers=0 to_cuda=True  -> 79 batches, mismatches=0
num_workers=4 to_cuda=True  -> 79 batches, mismatches=0
```

判定条件是 `semantic_ids` 的填充宽度 == `semantic_ids_len.max()`（音素同理）。**全 0 不匹配 → 问题在模型侧，不在数据侧。**

---

## 6. 进程退出行为（坑 3）

独立进程、40 秒超时：

| 探针 | 结果 |
|---|---|
| `torch_import` | OK 2.5s |
| `alloc_exit`（分配 CUDA 张量后正常退出） | **HANG** 41.1s |
| **`alloc_sync_exit`（退出前 `synchronize`）** | **OK 3.3s** ✅ |
| `alloc_os_exit`（`os._exit(0)`） | **HANG** 41.2s |
| `svmodel_exit`（只加载模型） | OK 3.6s |
| `svmodel_os_exit` | OK 4.3s |
| `sv_full_exit`（算完 embedding 直接退） | **HANG** 41.2s |
| **`sv_full_sync_exit`（算完 + `synchronize`）** | **OK 5.0s** ✅ |
| `torch_save_exit`（存盘后退出） | **HANG** 40.3s |

**关键点：`os._exit(0)` 也救不了** —— 死锁在 `ExitProcess` 的 DLL 卸载阶段，比 Python 层更深。

### 加装 atexit 钩子后的对照

| 探针 | 修复前 | 修复后 |
|---|---|---|
| `pth_loaded` | — | `zluda_exit_fix loaded = True` |
| `alloc_exit` | HANG | **OK 3.9s** ✅ |
| `sv_full_exit` | HANG | **OK 5.7s** ✅ |
| `torch_save_exit` | HANG | **OK 3.9s** ✅ |
| `alloc_sync_exit` | OK | OK |

端到端效果：`2-get-sv.py` 写完 317 个文件后**干净退出（rc=0）**，webui 的 `p.wait()` 能正常返回。

---

## 7. 兼容层生效自检

```
zluda_compat loaded      : True
DISABLE_ADDMM_CUDA_LT    : 1
PYTORCH_JIT              : 0
cudnn.enabled            : False
matmul.allow_tf32        : False
cudnn_sdp_enabled        : False
flash_sdp_enabled        : False
mem_efficient_sdp_enabled: False
math_sdp_enabled         : True
```

**注意 `enable_cudnn_sdp` 单独关掉是不够的。** 实测发现刷屏的内核名是 `fmha_cutlassF_*`，而它来自 **flash attention** 后端（不是 cuDNN）。三个一起关才彻底消除：

```
# 只关 cudnn_sdp 时，刷屏依旧：
cudnn_sdp_enabled : False
flash_sdp_enabled : True     ← 元凶
mem_efficient_sdp_enabled : True
```

关掉后 SDPA 走 MATH 后端（永远可用）。这三个后端本来在 ZLUDA 上就跑不了 CUTLASS 内核、一直在回退，所以关掉只是**省掉无效重试和数万行日志**，不损失能力。

---

## 8. JIT 编译代价（性能特征）

### s2 训练

```
1/81 [11:20<15:07:23, 680.54s/it]     ← 首轮第一个 batch 680 秒
2/81 [11:23<6:11:18,  282.00s/it]
7/81 [11:40<36:27,     29.57s/it]
...
100%|██████████| 81/81 [19:34<00:00, 14.50s/it]    ← 首轮总耗时 19 分 34 秒
```

**缓存命中后的第二轮**：

```
1/81 [00:51<1:08:39, 51.49s/it]       ← 首步从 680 秒降到 51 秒
100%|██████████| 81/81 [06:36<00:00,  4.89s/it]    ← 总耗时 6 分 36 秒（快 3 倍）
```

### 推理 AR 循环

```
12503:   0%|  | 1/1500 [03:03<76:12:01, 183.00s/it]   ← 第 1 步 183 秒
12510:   1%|  | 17/1500 [03:04<53:30,     2.16s/it]
12523:   3%|  | 50/1500 [03:06<01:33,    15.55it/s]   ← 已恢复
12800:  41%|  | 613/1500 [03:39<00:50,   17.47it/s]   ← 稳态 17.5 步/秒
```

**前 50 步花了 3 分钟，后面 900 多步只要 50 秒。** 也就是说"慢"的 95% 是首次编译。

缓存位置：`%LOCALAPPDATA%\zluda\ComputeCache\zluda.db`（实测 159.9 MB）。

> **重要**：它在 `LOCALAPPDATA`，**不在 TEMP**里 —— 所以 GPT-SoVITS 启动时清空 `TEMP` 目录的行为**不会**破坏内核缓存。但**不要手动删它**。

---

## 9. ASR：CPU vs GPU

用同一个音频、同一模型，两种设备各跑一次（独立进程）：

| | CPU | GPU (`device="cuda"`) |
|---|---|---|
| 退出码 | **0** ✅ | **0** ✅ |
| 模型加载 | 52.8s | 52.6s |
| **forward** | **2.346s** | **2.167s** |
| generate 总耗时 | 2.7s | 2.4s |
| RTF | 0.555 | 0.512 |
| 识别文本 | `よろしくね。ええ、ケチー。あっ。` | **完全相同** |

对照数据集里的标准答案（`2-name2text.txt` 第 1 行）——**逐字一致**，说明两条路径都正确。

**结论：GPU 可用，但只快约 8%。** 因为 Fun-ASR-Nano 是逐 token 自回归解码的小模型，batch=1，GPU 大部分时间在等 CPU 派发。**因此项目把它固定在 CPU。**

### 附带发现：裸跑 FunASR 会静默崩溃

前两次探针 **CPU 和 GPU 都静默退出、无 traceback**，输出停在 checkpoint 加载完成后。原因有两个，都排掉后一次通过：

1. **缺 `KMP_DUPLICATE_LIB_OK=TRUE`** —— 这是 `webui.py` 会设的变量。FunASR 会拉进 MKL/OpenMP，重复加载 OpenMP 运行时会 **abort 进程且不产生 Python traceback**。
2. **内存压力** —— 当时机器只剩 9.3 GB 可用（用户自己的 ASR 还占着 4.5 GB）。清空其他进程后可用内存回到 19.8 GB。

**所以：别在 webui 外面裸跑 FunASR**，要么走 webui，要么自己设 `KMP_DUPLICATE_LIB_OK=TRUE`。

---

## 10. 端到端结果

| 项目 | 结果 |
|---|---|
| SoVITS (s2) 训练 | 81 步 × 8 轮，~3.66 s/步，`saving ckpt 安和昴_e8:Success.`，`rc=0` |
| GPT (s1) 训练 | 79 步/轮 × 15 轮，~88 秒/轮，`epoch=4-step=95.ckpt`(888.3 MB) 正常落盘 |
| 小权重产出 | `SoVITS_weights_v2Pro\安和昴_e8_s648.pth`(128.7 MB)、`GPT_weights_v2Pro\安和昴-e*.ckpt`(148 MB) |
| 前处理 | `4-cnhubert` / `5-wav32k` / `7-sv_cn` 各 317，全通 |
| 推理 | 可选新权重合成；首次调用慢（JIT），之后正常 |

### 一个值得记录的教训

早期用 **1 epoch + batch 4** 做验证时产出的权重，推理结果是**一分钟的 "a a a a a"**。诊断依据：

```
AR 循环最终状态:  100%|█████████▉| 1499/1500 [04:27...]   ← 跑满硬上限
EOS 相关行数: 0                                          ← 从头到尾没输出过 EOS
```

**模型根本没学会在句尾停止**，于是一路生成到 1500 步上限、反复吐同一个 token。

**这不是 ZLUDA 的问题**（当时整条链路零报错），纯粹是训练量不足。1 epoch 只能用于"验证流程是否跑通"，**不能用于产出可用模型**。GPT-SoVITS 微调一般需要 SoVITS 8~25 轮、GPT 15~25 轮。
