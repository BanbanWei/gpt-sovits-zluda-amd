# 诊断探针

这套探针是**定位到全部 8 个根因的工具**，也是本仓库区别于"抄一份配置"的地方。

## 0. 两条使用前提

### 必须用 `zluda.exe` 包住运行器

```powershell
C:\ZLUDA\bin\zluda.exe <venv>\Scripts\python.exe run_modes.py <探针脚本> <超时秒数> <模式1> [<模式2> ...]
```

这样注入才会传播到它拉起的每个子进程。（实测：`zluda.exe` 的注入**会**传播到子进程。）

### 装了 `zluda_compat` 之后，探针默认显示"已修复"的结果

兼容层是全局生效的，所以它会顺手把探针的 baseline 也改掉。想复现**原始失败**，先关掉它：

```powershell
$env:ZLUDA_COMPAT='0'      # 关掉兼容层，看原始行为
# ... 跑探针 ...
Remove-Item env:ZLUDA_COMPAT
```

| 想看的 | 设置 |
|---|---|
| 原始失败（验证"坑"存在） | `ZLUDA_COMPAT=0` |
| 修复后的行为（验证"解法有效"） | 默认（不设） |

## 1. 路径配置

所有探针通过 `_env.py` 解析路径，**不需要改源码**。它会按"环境变量 → 自动探测 → 示例兜底"的顺序决定：

```powershell
$env:GPT_SOVITS_ROOT   = 'D:\path\to\GPT-SoVITS'          # 仓库根目录
$env:GPT_SOVITS_PYTHON = 'D:\path\to\venv\Scripts\python.exe'
$env:GPT_SOVITS_TEST_WAV = 'D:\some\audio.wav'            # 探针用的测试音频（可选）
```

不设置时从当前目录逐级向上找含 `webui.py` + `GPT_SoVITS/` 的目录，并在仓库同级/内部找 `venv_torch25` / `venv` / `.venv`。

自检：

```powershell
<venv>\Scripts\python.exe _env.py
```

输出示例：

```
GPT_SOVITS_ROOT          : D:\saves\GPT-SoVITS\GPT-SoVITS
GPT_SOVITS_PYTHON        : D:\saves\GPT-SoVITS\venv_torch25\Scripts\python.exe True
GPT_SOVITS_SITE_PACKAGES : D:\saves\GPT-SoVITS\venv_torch25\Lib\site-packages True
TEST_WAV                 : D:\...\logs\GBC\5-wav32k\GBC.m4a_0000123520_0000258880.wav True
```

## 2. 为什么要"一个探针一个进程 + 硬超时"

ZLUDA 下有两种会毁掉排查的失败模式：

1. **失败后进程挂死空转，而不是抛异常退出。** 如果所有探针跑在同一个进程里，**你会在第一个问题上永久卡住**。
2. **退出时死锁。** 只要退出时还有未同步的 GPU 任务，进程就卡在 `ExitProcess` 的 DLL 卸载阶段。

`run_modes.py` 因此给每个模式起**独立子进程 + 硬超时**，超时就 `taskkill /F /T`，结果归为：

| 判定 | 含义 |
|---|---|
| `OK` | 输出里有 `RESULT OK` |
| `FAIL` | 进程退出了但没成功（通常抛了异常） |
| `HANG` | **超时被杀** —— ZLUDA 特有的失败模式 |

每个模式的完整日志落在 `diagnostics\probe_out\`。

---

## 3. 各探针支持的模式（**按脚本区分，别搞混**）

> 下面是从源码里逐条取出的**实际模式名**。模式名用错会得到 `RESULT FAIL unknown mode` —— 那不是环境问题。

### `probe_fft.py` —— 基础设备、FFT、以及各类模型/训练相关算子

这个脚本覆盖面最广，**基础设备查询和矩阵乘也在它这里**。

| 模式 | 用途 |
|---|---|
| `cuda_basic` | **设备枚举**（任何排查的第一步） |
| `rfft_fp32` / `rfft_fp16` | 一维 FFT |
| `stft` | STFT / ISTFT |
| `fbank_fp32` / `fbank_fp16` / `fbank_cpu` | Kaldi fbank（SV 特征用） |
| `sv_fp16` / `sv_fp32` / `sv_cpu` / `sv_fixed` | 真实 `sv.py` 路径（`sv_fixed` 是候选修法） |
| `matmul_fp32` / `matmul_fp16` / `matmul_bf16` | 裸矩阵乘 |
| `conv_fp16` | conv1d + convtranspose1d + groupnorm + layernorm |
| `autocast_fp16` / `gradscaler` / `amp_train_step` | AMP 训练相关 |
| `pin_memory` / `dataloader_workers` | DataLoader 相关 |

### `probe_blas.py` —— cuBLAS / cuBLASLt

| 模式 | 用途 |
|---|---|
| `linear_fp32` / `linear_fp16` | `nn.Linear`（走 cuBLASLt） |
| `linear_refenc_fp32` | 复刻真实失败点（`ref_enc` → `spectral` → `fc`） |
| `linear_bigk_fp32` / `linear_noncontig_fp32` | 大 K / 非连续输入 |
| `matmul_bt_fp32` | 裸 `a @ b`（**唯一不经过 cuBLASLt 的对照**） |
| `addmm_fp32` | `torch.addmm` |
| `conv1d_fp32` / `conv1d_fp16` / `conv2d_fp16` | 卷积前向 |
| `linear_3d_fp16` / `embedding_fp32` / `full_small_net_fp16` | 其它常见组合 |

**用法**：`ZLUDA_COMPAT=0` 时全部 `linear_*` 会 FAIL；默认（装了兼容层）全部 OK。这个对比就是坑 4 的证据。

### `probe_ops.py` —— 卷积反向与归一化

| 模式 | 用途 |
|---|---|
| `gn_fwd` / `gn_fwd_bwd` / `gn_fwd_bwd_fp16` | GroupNorm |
| `ln_fwd_bwd` | LayerNorm |
| `conv1d_bwd` / `conv1d_bwd_fp16` | **卷积反向**（坑 5 的核心） |
| `convtranspose_bwd_fp16` | 转置卷积反向（未修时**挂死**） |
| `lrelu_bwd` / `linear_bwd_fp16` | 激活 / 线性层反向 |
| `net_conv_linear_bwd` / `net_conv_gn_bwd` / `net_full_bwd` | 组合网络反向（最接近真实训练） |
| `interpolate_fp16` / `instance_norm_fwd` / `batchnorm_fwd` | 其它 |

**注意**：这个脚本才是坑 5 的主角。`probe_fft.py` 里的 `conv_fp16` 只测前向，**前向是好的**，所以别用前向来判断坑 5。

### `probe_reduce.py` —— 归约与标量回读

| 模式 | 用途 |
|---|---|
| `randn_alloc` | 分配张量后退出（测退出行为，见坑 3） |
| `sync_only` | 只做 `cuda.synchronize()` 的对照 |
| `min_item` / `min_float` | `torch.min(x).item()` / `float(torch.min(x))` |
| **`min_str`** | **`str(0维CUDA张量)`** —— 坑 2 的核心，预期 `HANG` |
| `max_float` / `sum_float` / `mean_float` | 其它归约 |
| **`argmin_int`** | `int(torch.argmin(x))` —— 预期 `HANG` |
| `bool_compare` | 复刻 `if torch.min(y) < -1.2:` 的写法 |
| `cpu_copy` / `scalar_cpu_copy` / `min_cpu_after_cuda` | D2H 拷贝 |
| `layernorm_sum` / `softmax_sum` | 归约类算子 |
| `sum_float_large` / `min_float_half` | 大张量 / fp16 |

### `probe_exit.py` —— 进程退出行为（坑 3）

| 模式 | 含义 |
|---|---|
| `torch_import` / `pth_loaded` | 基线 / 兼容层是否加载 |
| `alloc_exit` | 分配张量后直接退出 —— **未修时挂死** |
| `alloc_sync_exit` | 退出前 `synchronize` —— **修好的样子** |
| `alloc_os_exit` | 用 `os._exit(0)` —— **同样挂死**（证明死锁在 Python 层之下） |
| `svmodel_exit` / `svmodel_os_exit` | 只加载模型不计算 |
| `sv_full_exit` / `sv_full_sync_exit` / `sv_full_os_exit` | 算完 embedding 后的退出行为 |
| `torch_save_exit` | 存盘后退出（最接近真实预处理脚本） |

### `probe_jit.py` —— TorchScript / NVRTC（坑 6）

| 模式 | 含义 |
|---|---|
| `default` | 默认设置（预期报 `nvrtc: invalid value for --gpu-architecture`） |
| `texpr_off` | 关 TensorExpr 融合器 |
| `profiling_off` | 关 profiling executor |
| `fusers_off` | 关全部融合器 |
| `jit_state_disable` | `torch.jit._state.disable()` |
| `eager_manual` | 绕过 scripted 函数，手写等价 eager 运算（对照） |

判定技巧：**看返回 dtype**。走 TorchScript 时 autocast 不生效（返回 float32），退回 eager 后生效（返回 float16）。

### `probe_funasr_device.py` + `probe_funasr_runner.py` —— ASR 设备对比

`probe_funasr_device.py` 接收**设备名**而不是模式名（`cpu` / `cuda`），由 `probe_funasr_runner.py` 驱动两种设备各跑一次并对比耗时。

```powershell
C:\ZLUDA\bin\zluda.exe <venv>\Scripts\python.exe probe_funasr_runner.py
```

> **注意**：`probe_funasr_runner.py` 会设 `KMP_DUPLICATE_LIB_OK=TRUE`。这是必须的 —— 见下方"附带发现"。

---

## 4. 几个不能直接跑的脚本

以下脚本有额外前提，用前请先看代码头部注释。

| 文件 | 前提 |
|---|---|
| `diag_s1_replicate.py` | 需要一个 s1 训练配置。默认读 `<root>\TEMP\tmp_s1.yaml`（**webui 启动一次 GPT 训练就会生成**）；也可用 `GPT_SOVITS_S1_CONFIG` 指定 |
| `diag_s1_forward.py` | 同上 |
| `diag_s1_batches.py` | 同上。**会加载数据集并扫描全部 batch**，需要已完成的训练集前处理 |
| `diag_s1_dataset.py` | 同上 |
| `diag_slice.py` | 独立可跑（不需要项目数据） |
| `check_sdp_flags.py` | 独立可跑（读兼容层开关状态） |
| `verify_patch.py` | **需要项目已打好 FFT 补丁**，用于验证 CPU 回退的数值正确性 |

---

## 5. 附带的两个重要发现

### 裸跑 FunASR 会静默崩溃

初次测试时 **CPU 和 GPU 都静默退出、无 traceback**，输出停在 checkpoint 加载完成后。原因有两个，都排掉后一次通过：

1. **缺 `KMP_DUPLICATE_LIB_OK=TRUE`** —— 这是 `webui.py` 自己会设的变量。FunASR 会拉进 MKL/OpenMP，重复加载 OpenMP 运行时会 **abort 进程且不产生 Python traceback**。
2. **内存压力** —— 当时只剩 9.3 GB 可用内存（用户自己的 ASR 还占着 4.5 GB）。清空其他进程后回到 19.8 GB，问题消失。

**所以：别在 webui 外面裸跑 FunASR。**

### "进程静默退出、无 traceback" 的通用含义

只要看到这个组合，就是**原生代码崩溃**，不是 Python 异常。常见原因：内存不足、OpenMP 重复加载、GPU 驱动层问题。排查方向是"看 Windows 事件查看器 / 减小内存占用"，而不是去读 Python 栈。

---

## 6. 一套可复用的排查流程

1. **先确认设备枚举**（`probe_fft.py` 的 `cuda_basic`）。失败就别往下走，问题在 ROCm/ZLUDA 层。
2. **拿到报错的最后一行**，定位对应算子。
3. **把该算子单独丢进探针**，`ZLUDA_COMPAT=0` 下跑一次 —— 确认能复现。
4. **加一个候选开关**再跑一次，对比。
5. **回归验证**：修完把整组探针重跑一遍，确认没把别的东西弄坏。

第 3 步是精髓：**"3 秒内能复现的最小用例"比在完整训练里猜快一个数量级。**

### 一个反面教材（值得记）

排查坑 6 时，曾怀疑"关掉 JIT 导致了形状错误"，做了对照实验：**`PYTORCH_JIT=1` 重跑，失败完全一样** → 假设被证伪。

**这次证伪的价值不在于猜对，而在于用对照实验快速排除了错误方向** —— 正是它把思路推向了真正的根因（坑 7）。
