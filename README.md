# GPT-SoVITS 在 AMD 显卡上本地训练（Windows）

**实测跑通的完整方案。** 目标环境：Windows + AMD Radeon RX 6600 (gfx1032) + ROCm HIP SDK + ZLUDA。

本站的起点是一台"装好了 ROCm、装好了 ZLUDA，但训练一跑就挂死"的机器。经过逐项隔离排查，最终定位到 **8 个相互独立的兼容问题**，全部修好后 GPT-SoVITS 的 SoVITS 与 GPT 训练均可完整跑完并产出可用权重。

本文给出：可复现的版本矩阵、每个坑的**根因**（不是"试试这个"）、补丁与脚本、以及一套**只读诊断探针**，方便你判断自己卡在哪一步。

---

## 一、实测结果

| 阶段 | 配置 | 结果 |
|---|---|---|
| SoVITS 训练 (s2, v2Pro) | 8 epoch, batch 4 | 81 步/轮 × 8 轮，约 **3.66 s/步**，约 5 分钟/轮，权重正常存出 |
| GPT 训练 (s1, v2Pro) | 15 epoch, batch 4 | 79 步/轮，约 **88 秒/轮**，`epoch=4-step=95.ckpt` 正常落盘 |
| 数据集 | 317 条切片 / 约 25 分钟音频 | 预处理全通（`4-cnhubert` / `5-wav32k` / `7-sv_cn` 各 317） |
| 推理 | v2Pro | 可正常合成；首次调用慢（见下文 JIT），之后正常 |

**性能特征（重要，别误判为卡死）**

ZLUDA 会为每个 GPU 内核做一次 JIT 编译，**首次调用极慢，之后走缓存**：

| 场景 | 首次 | 缓存命中后 |
|---|---|---|
| s2 训练第一个 batch | **680 秒** | 51 秒 |
| 推理 AR 循环第 1 步 | **183 秒** | 稳态 17.5 步/秒 |

缓存落在 `%LOCALAPPDATA%\zluda\ComputeCache\zluda.db`（实测已增长到 160 MB）。**它不在 TEMP 里**，所以不会被程序清理 TEMP 的行为破坏——但也**不要手动删它**，否则每次都要重新编译几分钟。

---

## 二、实测环境版本矩阵

全部为**实测值**，不是推荐值。完整采集命令见 [`docs/VERSIONS.md`](docs/VERSIONS.md)。

| 组件 | 版本 | 备注 |
|---|---|---|
| 操作系统 | Windows 11 专业版 build **26200** | |
| GPU | **AMD Radeon RX 6600** (gfx1032, Navi 23) | `hipInfo` 报 `major:10 minor:3`，7.98 GB 显存 |
| 显卡驱动 | **32.0.21045.9003**（Adrenalin 25.10.45.09） | |
| ROCm / HIP SDK | **6.2.4** | `hip_version.h`: MAJOR 6 / MINOR 2 / PATCH 41512 / githash `db3292736` |
| HIP 安装路径 | `C:\Program Files\AMD\ROCm\6.2\` | `HIP_PATH` 指向它 |
| gfx1032 补丁 | `rocm.gfx1032.for.hip.sdk.6.2.4.navi21.logic` | **144 个文件 / 223.6 MB**，替换 `rocblas.dll`(80.8 MB) + 预编译 gfx1032 Tensile 内核 |
| ZLUDA | **3.9.5** | `zluda.exe --version` |
| Python | **3.11.9** | |
| PyTorch | **2.5.1+cu118** | 用 CUDA 11.8 构建的 wheel 跑在 ZLUDA 上（这是 ZLUDA 的标准用法） |
| torchaudio | 2.5.1+cu118 | |
| GPT-SoVITS | commit **`48b1a01`** (`Fix Fun-ASR-Nano Transformers requirement`) | 上游 `RVC-Boss/GPT-SoVITS`，分支 `main` |

关键 Python 依赖（完整清单见 [`docs/DEPENDENCIES.md`](docs/DEPENDENCIES.md)）：

```
numpy 1.26.4          transformers 4.56.2    gradio 4.44.1
funasr 1.4.15         modelscope 1.40.0      ctranslate2 4.8.2
pytorch-lightning 2.6.5    tokenizers 0.22.2    librosa 0.10.2
```

> **注意**：`numpy` 必须 `<2`，`gradio` 必须 `<5`，`pydantic` 需 `<=2.10.6`。

---

## 三、为什么"装了 ROCm + ZLUDA 还是跑不起来"

这是本文最有价值的部分。**这 8 个问题相互独立**，只修其中几个仍然跑不起来，所以很容易误判为"ZLUDA 根本不行"。

| # | 现象 | 根因 | 解法 |
|---|---|---|---|
| 1 | `torch.fft.rfft` / `torch.stft` 报 `CUFFT error: CUFFT_INTERNAL_ERROR`，**随后进程永久空转** | ZLUDA 的 `cufft.dll` 转发到 ROCm `hipfft`→`rocfft`，而 rocFFT 在 gfx1032 上不可用（fp32/fp16 都崩）。`HSA_OVERRIDE_GFX_VERSION=10.3.0` **无效**。rocBLAS 被上面的补丁救活了，**rocFFT 没有** | 把 FFT 挪到 CPU（3 处调用点） |
| 2 | 音频数值超出 ±1.2 时挂死 | `print("min value is ", torch.min(y))` —— **`str(CUDA 张量)` 会挂死**（而 `float()` 正常） | 改成 `float(...)` |
| 3 | 干完活进程不退出，webui 永远显示"运行中" | 退出时若有未同步的 GPU 任务，死锁在 `ExitProcess` 的 DLL 卸载阶段。**`os._exit(0)` 也救不了**（卡在更底层） | 退出前 `torch.cuda.synchronize()`（atexit 钩子） |
| 4 | 所有 `Linear`/`addmm` 报 `CUBLAS_STATUS_NOT_SUPPORTED`（`cublasLtMatmulAlgoGetHeuristic`） | ZLUDA 的 `cublasLt.dll` 只有 185 KB、不引用任何 ROCm 库，**是个空壳** | `DISABLE_ADDMM_CUDA_LT=1` |
| 5 | conv **反向**报 `GET was unable to find an engine`；`ConvTranspose1d` 反向挂死；`BatchNorm1d` 报 `CUDNN_STATUS_EXECUTION_FAILED` | `cudnn.dll` 无法为卷积反向提供实现 | `torch.backends.cudnn.enabled = False`（走 PyTorch 原生卷积） |
| 6 | s2 在 `commons.fused_add_tanh_sigmoid_multiply` 崩溃 | TorchScript 融合内核走 NVRTC，设备报 `compute_88`，而 HIPRTC 只认 `gfx1032` → `nvrtc: error: invalid value for --gpu-architecture` | `PYTORCH_JIT=0`，整体退回 eager |
| 7 | s1 报 `Expected target size [4, 0]`，或 mask 宽度与数据宽度不符 | **ZLUDA 上小张量的 CUDA 归约会返回错误值**（真值 134 却返回 119 / 61）；且 `*_lens` 这类小张量在分配压力下值会变脏 | mask 宽度改为**从数据张量形状推导**，不信任 `*_lens` 的数值 |
| 8 | s1 反向传播时原生崩溃 `0xC0000005`（访问违例） | 崩在 `c10d::Reducer::mark_variable_ready_dense`（DDP 梯度分桶） | 单卡时不再套 DDP |

附带修掉的同类问题：`bucket_sampler.py`、`core_vq.py` 里**无条件调用 `dist.get_rank()/broadcast()`**（单进程无进程组即崩）。

每个坑的**完整报错原文 + 复现方法 + 验证过程**见 [`docs/TROUBLESHOOTING.md`](docs/TROUBLESHOOTING.md)，实测的算子可用性矩阵见 [`docs/VERIFICATION.md`](docs/VERIFICATION.md)。

---

## 四、安装步骤

### 0. 前置说明

- **只支持 AMD 独显**（本方案针对 gfx1032 验证；gfx1030/1100 等 ROCm 官方支持的型号更简单，见文末）。
- 全程需要约 **250 GB 磁盘**（ROCm + gfx1032 补丁 + 模型 + 权重）。
- 建议 ≥ 32 GB 内存。**内存不足会导致 FunASR 等原生扩展静默崩溃（无 traceback）**，实测踩过。

### 1. 安装 ROCm / HIP SDK 6.2.4

从 AMD 官方下载 **HIP SDK 6.2.4 for Windows** 并安装，默认路径 `C:\Program Files\AMD\ROCm\6.2\`。

安装后确认：

```powershell
$env:HIP_PATH                      # 应指向 ...\AMD\ROCm\6.2\
& 'C:\Program Files\AMD\ROCm\6.2\bin\hipInfo.exe'
# 关注最后几行：major: 10  minor: 3   → gfx1032
```

`hipInfo` 能否正确列出你的卡，是"ROCm 层面是否就绪"的唯一判据。

### 2. 打 gfx1032 补丁（关键）

HIP SDK 6.2.4 **官方不支持 gfx1032**，必须替换 rocBLAS 与 Tensile 内核库，否则矩阵乘等基础算子会失败。

补丁包含 144 个文件 / 223.6 MB，结构：

```
rocm gfx1032 for hip skd 6.2.4(navi21 logic)\
├── rocblas.dll                                              (80.8 MB，替换用)
└── library\
    ├── Kernels.so-000-gfx1032.hsaco                         (预编译 gfx1032 内核)
    ├── TensileLibrary_lazy_gfx1032.dat
    └── TensileLibrary_Type_*.dat / *.hsaco                  (各类矩阵乘逻辑)
```

把 `rocblas.dll` 覆盖到 `C:\Program Files\AMD\ROCm\6.2\bin\`，`library\` 里的文件覆盖到 `C:\Program Files\AMD\ROCm\6.2\bin\rocblas\library\`（**先备份原文件**）。

> 本仓库不附带该补丁（体积大且非我原创）。它在中文社区以 `rocm.gfx1032.for.hip.sdk.6.2.4.navi21.logic` 之名流传，按此名搜索即可。若你的卡是 gfx1030 / gfx1100 等 ROCm 官方支持的型号，**跳过这一步**。

**验证**：用本仓库的 `diagnostics/` 探针跑 `matmul_fp32`（在 `probe_fft.py` 里），应 `OK`：

```powershell
C:\ZLUDA\bin\zluda.exe <venv>\Scripts\python.exe diagnostics\run_modes.py diagnostics\probe_fft.py 60 matmul_fp32
```

### 3. 安装 ZLUDA 3.9.5

解压 ZLUDA 到 `C:\ZLUDA\`，确认存在 `C:\ZLUDA\bin\zluda.exe` 与 `nvcuda.dll`。

ZLUDA 的用法是**用 `zluda.exe` 包裹 python 启动**，它把 CUDA 库注入目标进程：

```powershell
C:\ZLUDA\bin\zluda.exe C:\path\to\python.exe your_script.py
```

**注意**：`zluda.exe` 会把自己的选项和参数混在一起解析，所以 `python -c "..."` 这种写法会报 `Unrecognized argument: "-c"`。**启动脚本必须写成 .py 文件**。

已验证：**注入会传播到子进程**（webui 用 `Popen` 拉起的训练进程同样拿到 GPU）。

### 4. 建 Python 环境并装依赖

```powershell
py -3.11 -m venv venv
venv\Scripts\python.exe -m pip install --upgrade pip
```

安装 **CUDA 11.8 版本**的 torch（ZLUDA 需要 cu118 wheel）：

```powershell
venv\Scripts\python.exe -m pip install torch==2.5.1+cu118 torchaudio==2.5.1+cu118 ^
  --index-url https://download.pytorch.org/whl/cu118
```

再装 GPT-SoVITS 依赖。仓库根目录的 `requirements_zluda.txt` 已经锁好了版本约束（`numpy<2`、`gradio<5`、`pydantic<=2.10.6` 等）：

```powershell
venv\Scripts\python.exe -m pip install -r requirements_zluda.txt
```

**缺失的包（本环境实测需要额外处理）**

| 包 | 情况 | 处理 |
|---|---|---|
| `jieba_fast` | Python 3.11 无 wheel，需 C 编译器 | **不要装**。本项目补丁已让 3 处 import 回退到 `jieba` |
| `eunjeon` | 同样编译失败 | 仅韩文需要。不用韩文可忽略 |
| nltk 语料 | `g2p_en` 需要，且**下载会被部分网络环境拦** | 见下一步 |
| `fast_langdetect` 模型 | 需要一个**目录**先存在，库不会自动创建 | 见下一步 |

### 5. 补齐数据文件

**5.1 nltk 语料**（英文前端需要；缺了会导致"目标文本里含拉丁字母"时崩）

下载到 `C:\Users\<你>\nltk_data\`：

| 文件 | 大小 | 放置位置 |
|---|---|---|
| `cmudict.zip` | 875 KB | `corpora\` |
| `averaged_perceptron_tagger.zip` | 2.5 MB | `taggers\` |
| `averaged_perceptron_tagger_eng.zip` | 1.5 MB | `taggers\` |
| `punkt.zip` | 13.6 MB | `tokenizers\` |

来源为 nltk_data 仓库的 `packages/` 目录。**如果 `nltk.download()` 报 `SSRF attempt to restricted IP`**（某些 DNS/代理环境会把 `raw.githubusercontent.com` 解析到保留网段，并被 Python 层的安全组件拦截），用 **`curl.exe`** 下载即可绕过——该拦截只钩在 Python 的 urllib 上：

```powershell
curl.exe -sSL -o "$env:USERPROFILE\nltk_data\corpora\cmudict.zip" `
  "https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages/corpora/cmudict.zip"
```

**5.2 fast_langdetect 模型**（语言检测，必需）

该库对**用户指定的缓存目录不会自动创建**，父目录不存在就直接抛 `FileNotFoundError: Cache directory not found`。先建目录，库就会自己下载：

```powershell
mkdir "<GPT-SoVITS根目录>\GPT_SoVITS\pretrained_models\fast_langdetect"
```

之后首次运行会自动下载 `lid.176.bin`（125.2 MB）。也可以手动下载放进该目录。

**5.3 ASR 模型**（仅前处理需要）

使用 Fun-ASR-Nano 后端时，首次运行会自动从 HuggingFace 拉取：

| 模型 | 大小 |
|---|---|
| `FunAudioLLM/Fun-ASR-Nano-2512` | 1.9 GB |
| `funasr/fsmn-vad` | 4 MB |

缓存位置 `C:\Users\<你>\.cache\huggingface\hub\`。

### 6. 打本项目的兼容补丁

**6.1 核心：`zluda_compat` 兼容 shim**

`compat/zluda_compat.py` 用一个 `site-packages` 的 `.pth` 自动加载，一次性解决第 **3、4、5、6** 号坑，并且对 webui 拉起的**所有子进程**同样生效（这点很关键——训练是子进程）。

```powershell
venv\Scripts\python.exe compat\install.py
```

它会：
1. 把 `zluda_compat.py` 与 `zluda_compat.pth` 复制进 `venv\Lib\site-packages\`
2. 打印自检结果

它做的事：

| 动作 | 时机 | 解决 |
|---|---|---|
| `DISABLE_ADDMM_CUDA_LT=1`<br>`TORCH_BLAS_PREFER_HIPBLASLT=0`<br>`PYTORCH_JIT=0` | 解释器启动时（torch 导入前） | 坑 4、6 |
| `torch.backends.cudnn.enabled = False`<br>`matmul.allow_tf32 = False`<br>关闭 cuDNN / flash / mem_efficient 三个 SDPA 后端 | torch 导入后（通过 import hook） | 坑 5，并消除数万行 `FATAL: kernel ... sm80 ... built for sm37` 刷屏 |
| `atexit` 里 `torch.cuda.synchronize()` | 进程退出前 | 坑 3 |

可用 `ZLUDA_COMPAT=0` 整体关闭，`ZLUDA_COMPAT_VERBOSE=1` 打印它做了什么。

> **为什么用 `.pth` 而不是改 `webui.py`？** 因为训练/前处理是 `Popen` 拉起的独立进程，改入口文件覆盖不全；`.pth` 对任何解释器都生效。

**6.2 项目代码补丁**

```powershell
cd <GPT-SoVITS根目录>
git apply <本仓库>\patches\gpt-sovits-zluda.patch
```

补丁针对 commit **`48b1a01`**，共 **11 个文件 / 204 行新增**：

| 文件 | 解决 |
|---|---|
| `GPT_SoVITS/module/mel_processing.py` | 坑 1（`stft` CPU 回退）+ 坑 2（print 修复） |
| `GPT_SoVITS/eres2net/kaldi.py` | 坑 1（`rfft` CPU 回退，SV 特征用） |
| `GPT_SoVITS/AR/models/t2s_model.py`、`AR/models/utils.py` | 坑 7（mask 宽度取自形状 + 长度归约移 CPU） |
| `GPT_SoVITS/s1_train.py`、`AR/data/bucket_sampler.py` | 坑 8（单卡不套 DDP） |
| `GPT_SoVITS/module/core_vq.py` | 修 `dist.get_rank()` 无进程组即崩（首次推理必踩） |
| `GPT_SoVITS/text/chinese.py`、`chinese2.py`、`tone_sandhi.py` | `jieba_fast` 回退到 `jieba` |
| `tools/asr/funasr_asr.py` | FunASR 默认走 GPU + 加 `FUNASR_DEVICE` 覆盖（见下文"已知限制"） |

验证补丁是否与你的代码兼容：

```powershell
git apply --check <本仓库>\patches\gpt-sovits-zluda.patch
```

**如果 `--check` 失败**（你的 GPT-SoVITS 比 `48b1a01` 新/旧，上游改动撞车了），两个选择：

```powershell
# 1) 三方合并：能自动合的自动合，剩下的标冲突让你手动解决
git apply --3way <本仓库>\patches\gpt-sovits-zluda.patch

# 2) 直接切到与补丁一致的版本再打
git stash && git checkout 48b1a01 && git apply <本仓库>\patches\gpt-sovits-zluda.patch
```

> **`--3way` 的前提**：补丁里带了 11 个前置 blob 哈希（`index abcd123..`），三方合并要求这些 blob 在**你的**仓库对象库里存在。你的 clone 只要包含 `48b1a01`（哪怕 HEAD 更新）就满足；如果报 `repository lacks the necessary blob`，说明缺历史，改用方案 2，或者手工对照文档逐个坑重做。

补丁改的都是**小范围、语义明确**的地方（拿形状替代归约、把 FFT 挪 CPU、包一层 `is_initialized`），所以手工重做完全可行 —— [`docs/TROUBLESHOOTING.md`](docs/TROUBLESHOOTING.md) 按"根因 → 解法 → 验证"组织，就是为了让你不仅会打补丁，也能自己改。

**6.3 启动器**

`launchers/` 下两个 `.bat`（**必须 CRLF 换行**，否则 cmd 解析多行 `if ( )` 块会出错）：

| 文件 | 特点 |
|---|---|
| `启动GPT-SoVITS.bat` | 输出直接到控制台 → **GPT 训练的实时进度条可见** |
| `启动GPT-SoVITS-带日志.bat` | 输出同时写文件 → 便于事后排查，**但 GPT 进度条不显示** |

> 进度条的差异不是 bug：s1 用的是 PyTorch Lightning 的 **rich** 进度条（原地重绘，只发 `\r` 不发换行），一旦输出被重定向/管道接管，就被按行缓冲吞掉；s2 用的 **tqdm** 会自动退化成逐行打印，所以照样可见。**想要 GPT 进度条就不要重定向。**

**路径不用改** —— 两个 bat 会自己找：从 `launchers\` 的上级目录出发，按 `..\GPT-SoVITS\GPT-SoVITS`、`..\GPT-SoVITS`、`GPT-SoVITS`、当前目录的顺序找含 `webui.py` + `GPT_SoVITS\` 的目录，再在它附近找 `venv_torch25` / `venv` / `.venv`。找不到会明确告诉你去改哪个变量。

只有布局特殊时才需要动顶部的「配置区」三行，或者先设环境变量（优先级更高）：

```bat
set "GPT_ROOT=D:\path\to\GPT-SoVITS"
set "VENV_DIR=D:\path\to\venv_torch25"
set "ZLUDA_DIR=C:\ZLUDA"
```

也支持 `GPT_SOVITS_ROOT` / `GPT_SOVITS_VENV` 环境变量。

### 7. 首次运行验证

```powershell
cd <GPT-SoVITS根目录>
C:\ZLUDA\bin\zluda.exe <venv>\Scripts\python.exe -c "..."   # ✗ 会报 Unrecognized argument
```

正确做法是用仓库里的验证脚本：

```powershell
C:\ZLUDA\bin\zluda.exe <venv>\Scripts\python.exe diagnostics\run_modes.py diagnostics\probe_fft.py 60 cuda_basic matmul_fp32 conv_fp16 rfft_fp32
```

期望：`cuda_basic` / `matmul_fp32` / `conv_fp16` 为 `OK`，`rfft_fp32` 为 `FAIL` 或 `HANG`（**这是预期的**——rocFFT 在 gfx1032 上不可用，项目补丁已把 FFT 挪到 CPU，所以项目本身不依赖 GPU FFT）。

> 模式名必须和脚本对上：`cuda_basic` / `matmul_fp32` / `rfft_fp32` 在 `probe_fft.py`，`linear_*` 在 `probe_blas.py`，卷积反向在 `probe_ops.py`。用错脚本会得到 `unknown mode`，那不是环境问题。完整对照表见 `diagnostics/README.md`。

---

## 五、排障速查

| 症状 | 先查什么 |
|---|---|
| 控制台满屏红字 | **看有没有 `Traceback`。** 只有 `Warning` / `FutureWarning` / `FATAL: kernel ... sm37` 都是噪音 |
| `Traceback` 后面跟着 `Warning: ... Skipped.` 且计数继续 | **被捕获的坏数据，无害**。项目把 `traceback.print_exc()` 写在 `except` 里，所以"被完美处理的异常"长得像崩溃 |
| 进程卡住、CPU 满载但 GPU 空闲 | 典型的 ZLUDA 挂死。原语：**任何 GPU 操作失败后进程可能空转而不退出** |
| 训练"没有进度条" | 若用了带日志的启动器，是 rich 被管道吞了；换成普通启动器 |
| 首次跑极慢（几分钟没动静） | JIT 编译。**看 GPU 是否满载**来区分"在编译"和"卡死" |
| `CUDA error: CUBLAS_STATUS_NOT_SUPPORTED` | `zluda_compat` 没生效（检查 `.pth` 是否在 site-packages 里） |
| `GET was unable to find an engine to execute this computation` | cuDNN 没关掉（卷积反向） |
| `nvrtc: error: invalid value for --gpu-architecture` | `PYTORCH_JIT=0` 没生效（TorchScript 融合） |
| `Default process group has not been initialized` | 又一处无条件 `dist.*` 调用。用 `grep -rn "dist\.\(get_rank\|get_world_size\|broadcast\|all_reduce\)"` 找出来加 `is_initialized()` 判断 |
| 进程**静默退出、无 traceback** | 原生扩展崩溃。常见原因：内存不足，或缺 `KMP_DUPLICATE_LIB_OK=TRUE`（OpenMP 重复加载会 abort 进程） |

**黄金判据：红字之后还有没有继续输出。** 有 = 无害；停在那就 = 真崩。

---

## 六、已知限制

1. **GPU FFT 不可用。** 本项目已把 3 处 FFT 调用挪到 CPU，所以不影响训练/推理；但**其他依赖 GPU FFT 的库在 gfx1032 上仍会挂死**，注意别引入。

2. **FunASR (ASR) 默认走 GPU，但别开 batch。**
   ASR 上 GPU 是**同时赢速度和内存**的（实测同环境）：

   | 配置 | 吞吐 | 峰值内存 | 317 条 |
   |---|---|---|---|
   | CPU 单进程 | 0.47 条/秒 | 6.85 GB | ~11.4 min |
   | CPU 双进程（各 1 线程） | 0.748 条/秒 | 13.7 GB | ~7.1 min |
   | **GPU 单进程** | **1.034 条/秒** | **6.85 GB** | **~5.1 min** |

   GPU 比 CPU 单进程快约 **2.2×**、比双进程快约 **1.38×**，内存只要一半。代价是 **3.33 GB 显存**和一次性 **55–100 秒 JIT 预热**。
   **训练同时跑 ASR 时用 `set FUNASR_DEVICE=cpu` 强制回 CPU**（显存留给训练）。

   另外两条重要事实：
   - **"ASR 上 GPU 会踩 rocFFT" 是错的。** FunASR 的 fbank 永远在 CPU 上算（调用点在 `to_device` 之前），实测 CPU/GPU 输出逐条一致。
   - **不要开 batch。** 批路径在 ZLUDA 下输出退化的重复文本（连"8 条完全相同的输入"都会给出 8 个不同的垃圾答案），会把标注悄悄污染。**别用 `batch_size_s` / `merge_vad` 之类的开关加速。**

3. **不要裸跑 FunASR。** 走 webui（它会设 `KMP_DUPLICATE_LIB_OK`），或在你的脚本里自己设，否则可能静默 abort。

4. **单卡不套 DDP。** `s1_train.py` 已改；若你后来插了第二张卡、想用 DDP，需自行验证是否会再触发坑 8。

5. **`HSA_OVERRIDE_GFX_VERSION=10.3.0` 对本方案的 FFT 问题无效**（实测过，仍然崩）。它可能对别的场景有用，但别指望它解决 rocFFT。

6. **仅验证了 v2Pro。** v1/v2/v3/v4 的算子路径不同，未必全部适用；但 `zluda_compat` 那层是通用的。

---

## 七、目录说明

```
├── README.md                          ← 本文
├── docs/
│   ├── VERSIONS.md                    ← 实测版本矩阵 + 采集命令
│   ├── TROUBLESHOOTING.md             ← 8 个坑的完整报错原文与验证过程
│   ├── VERIFICATION.md                ← 算子可用性实测矩阵
│   └── DEPENDENCIES.md                ← 依赖清单与版本约束
├── compat/
│   ├── zluda_compat.py                ← 核心兼容 shim
│   ├── zluda_compat.pth               ← 自动加载入口
│   └── install.py                     ← 装进 venv 的脚本
├── patches/
│   └── gpt-sovits-zluda.patch         ← 项目代码补丁（11 文件）
├── launchers/
│   ├── 启动GPT-SoVITS.bat
│   └── 启动GPT-SoVITS-带日志.bat
└── diagnostics/
    ├── run_modes.py                   ← 通用隔离探针运行器
    ├── probe_ops.py                   ← 算子探针（conv/gn/ln/attention…）
    ├── probe_blas.py                  ← cuBLAS / cuBLASLt 探针
    ├── probe_fft.py                   ← FFT / STFT 探针
    └── probe_exit.py                  ← 进程退出行为探针
```

### 关于 `diagnostics/`

这套探针是本方案能定位到根因的关键工具，也是它区别于"抄一份配置"的地方。

**每个探针都在独立进程里跑，并带硬超时**——因为 ZLUDA 的失败模式经常是**挂死而不是报错**，放在同一个进程里会让整个排查卡在第一个问题上。

```powershell
# 每个模式一个独立子进程，60 秒超时；同一脚本内可一次给多个模式
C:\ZLUDA\bin\zluda.exe <venv>\Scripts\python.exe diagnostics\run_modes.py diagnostics\probe_blas.py 60 linear_fp32 addmm_fp32
C:\ZLUDA\bin\zluda.exe <venv>\Scripts\python.exe diagnostics\run_modes.py diagnostics\probe_ops.py  60 conv1d_bwd convtranspose_bwd_fp16 net_full_bwd
```

输出判定：`OK` / `FAIL`（有异常）/ `HANG`（超时被杀）。

**怀疑某个算子时，第一步就是把它丢进探针跑一次**，这比在完整训练里猜快一个数量级。

---

## 八、原理小结

为什么这套组合能跑起来，一句话版本：

1. **ROCm 侧**：gfx1032 不被官方支持，用社区补丁把 rocBLAS + Tensile 内核换成 gfx1032 版本 → **矩阵乘可用**。
2. **ZLUDA 侧**：它把 CUDA 调用翻译到 HIP。BLAS 通了，但 **FFT / cuBLASLt / cuDNN 卷积反向 / NVRTC 这四块要么是空壳、要么不支持该架构** → 所以针对性关掉（cuBLASLt、cuDNN、TorchScript）或绕开（FFT 挪 CPU）。
3. **稳定化**：ZLUDA 下"失败"经常表现为挂死，且小张量归约值可能变脏 → 需要退出同步钩子、长度归约移 CPU、mask 宽度取自数据形状。

**核心思路不是"让 ZLUDA 支持一切"，而是"识别出哪四块不可用，然后绕开它们"。**

---

## 九、授权与致谢

- 本仓库中的脚本、补丁、文档：**MIT**。
- GPT-SoVITS 本体版权归 [RVC-Boss/GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS)（MIT），补丁中的片段来自该项目。
- [ZLUDA](https://github.com/vosen/ZLUDA) by vosen。
- gfx1032 的 rocBLAS/Tensile 补丁来自中文社区（非本仓库作品）。
- 感谢 FunASR / ModelScope / HuggingFace 提供的 ASR 模型。

**本方案在真实项目上端到端验证过**：317 条日语数据、SoVITS 8 轮 + GPT 15 轮训练完整跑完、权重可加载并用于推理。

如果你在别的 AMD 卡上试通了（或卡住了），欢迎开 issue 补充你的 `hipInfo` 输出与探针结果——那是最有用的信息。
