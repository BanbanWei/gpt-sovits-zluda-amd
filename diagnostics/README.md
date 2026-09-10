# 诊断探针

这套探针是**定位到全部 8 个根因的工具**，也是本仓库区别于"抄一份配置"的地方。

## 为什么要"一个探针一个进程 + 硬超时"

ZLUDA 下有两种会毁掉排查的失败模式：

1. **失败后进程挂死空转，而不是抛异常退出。**
   实测：`rfft` 的异常打印出来了，但进程继续空转，直到被硬超时杀掉。如果所有探针跑在同一个进程里，**你会在第一个问题上永久卡住**，后面什么都测不到。

2. **退出时死锁。**
   只要退出时还有未同步的 GPU 任务，进程就卡在 `ExitProcess` 的 DLL 卸载阶段，永不退出。

所以 `run_modes.py` 的设计是：**每个模式起一个独立子进程，配一个硬超时，超时就 `taskkill /F /T`**，然后把结果归类为：

| 判定 | 含义 |
|---|---|
| `OK` | 输出里有 `RESULT OK` |
| `FAIL` | 进程退出了但没成功（通常是抛了异常） |
| `HANG` | **超时被杀** —— 这就是 ZLUDA 特有的失败模式 |

## 用法

```powershell
C:\ZLUDA\bin\zluda.exe <venv>\Scripts\python.exe run_modes.py <探针脚本> <超时秒数> <模式1> [模式2] ...
```

例：

```powershell
# 只测基础设备与矩阵乘
... run_modes.py probe_ops.py 60 matmul_fp32 conv1d_fwd linear_fp32

# 测 FFT（预期 HANG —— 这是已知且已被项目补丁绕过的）
... run_modes.py probe_fft.py 60 rfft_fp32 stft

# 测退出行为（预期未修时 HANG，装了 zluda_compat 后 OK）
... run_modes.py probe_exit.py 45 alloc_exit sv_full_exit
```

**必须用 `zluda.exe` 包住 `run_modes.py`** —— 这样注入才会传播到它拉起的每个子进程。

输出文件（每个模式的完整日志）落在 `probe_out\` 下，方便逐个查看原始输出。

## 各探针测什么

### `probe_ops.py` —— 算子覆盖（最常用）

GPT-SoVITS 训练/推理实际会用到的算子逐个验证。

| 模式 | 测什么 |
|---|---|
| `cuda_basic` | 设备枚举、capability、显存 |
| `matmul_fp32` / `matmul_fp16` / `matmul_bf16` | 裸矩阵乘 |
| `linear_fp32` / `linear_fp16` | `nn.Linear`（走 cuBLASLt） |
| `conv1d_bwd` / `conv1d_bwd_fp16` | **卷积反向**（cuDNN 问题） |
| `convtranspose_bwd_fp16` | 转置卷积反向（未修时**挂死**） |
| `gn_fwd_bwd_fp16` / `ln_fwd_bwd` | GroupNorm / LayerNorm 前后向 |
| `batchnorm_fwd` | BatchNorm（cuDNN 问题） |
| `net_full_bwd` | 组合网络（conv+gn+lrelu+linear）的完整反向 |
| `interpolate_fp16` / `instance_norm_fwd` | 其它常见算子 |

### `probe_blas.py` —— cuBLAS / cuBLASLt

区分"裸矩阵乘 OK"和"`F.linear` 失败"的关键。默认情况下所有 `linear_*` 都 FAIL，加 `DISABLE_ADDMM_CUDA_LT=1` 后全部 OK。

### `probe_fft.py` —— FFT / STFT

验证 GPU FFT 是否可用。**在 gfx1032 上预期全部 HANG**（rocFFT 无该架构支持）。

> 项目补丁已把 3 处 FFT 调用挪到 CPU，所以**项目本身不依赖 GPU FFT** —— 这里 HANG 不代表项目跑不起来。

### `probe_exit.py` —— 进程退出行为

| 模式 | 含义 |
|---|---|
| `alloc_exit` | 分配张量后直接退出（**未修时挂死**） |
| `alloc_sync_exit` | 退出前 `synchronize`（**修好的样子**） |
| `alloc_os_exit` | 用 `os._exit(0)` —— **同样挂死**，证明死锁在 Python 层之下 |
| `svmodel_exit` | 只加载模型不计算（正常退出） |
| `sv_full_exit` / `sv_full_sync_exit` | 算完 embedding 后退出 / 加同步后退出 |
| `torch_save_exit` | 存盘后退出（最接近真实预处理脚本） |

### 其它可用的探针脚本

仓库之外的排查过程中还写过这些，模式名与上面一致，可按需取用：

| 脚本 | 用途 |
|---|---|
| `jit_probe_one.py` | 定位 TorchScript/NVRTC 问题（坑 6） |
| `red_probe_one.py` | 归约与标量回读（`str(tensor)` 挂死、`argmin` 挂死） |
| `diag_s1_replicate.py` | 把 `forward_old` 逐步复制出来插桩，抓到坑 7 的脏值 |
| `diag_s1_batches.py` | 排除"数据/多进程/H2D 拷贝"嫌疑 |
| `diag_slice.py` | 验证"用张量做切片下标"是否正常 |
| `check_sdp_flags.py` | 确认兼容层各开关是否真的生效 |

## 排查方法论

这套探针背后是一个可复用的流程：

1. **先确认设备枚举**（`cuda_basic`）。失败就别往下走，问题在 ROCm/ZLUDA 层。
2. **拿到报错的最后一行**，找到它对应的算子。
3. **把这个算子单独丢进探针**，在 baseline 下跑一次 —— 确认能复现。
4. **加一个候选开关**再跑一次，对比。
5. **回归验证**：修完要把整组探针重跑一遍，确认没有把别的东西弄坏。

第 3 步是精髓：**"能在 3 秒内复现的最小用例"比在完整训练里猜快一个数量级。**

## 一个反例（值得记）

排查坑 6 时，曾怀疑"关掉 JIT 导致了形状错误"，于是做了对照实验：**`PYTORCH_JIT=1` 重跑，失败完全一样** → 该假设被证伪。

**结论价值不在于"猜对了"，而在于用对照实验快速排除了错误方向。** 正是这次证伪把思路推向了真正的根因（坑 7）。
