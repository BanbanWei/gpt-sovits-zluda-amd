# 实测版本矩阵

全部为**实测值**，并附上采集命令，方便你逐项比对自己的环境。

> 环境不同未必不能跑，但**差异越大、踩到未记录问题的概率越高**。遇到问题请附上本文所有命令的输出。

---

## 1. 硬件与操作系统

| 项目 | 实测值 |
|---|---|
| 操作系统 | Microsoft Windows 11 专业版 |
| 版本 / Build | 10.0.**26200** |
| 架构 | 64 位 |
| 内存 | 31.82 GB |
| GPU | **AMD Radeon RX 6600** |
| GPU 计算单元 | 14 CU（`multiProcessorCount: 14`） |
| 显存 | 7.98 GB（`totalGlobalMem`） |
| GPU 架构 | **gfx1032**（`hipInfo` 报 `major: 10` / `minor: 3`） |
| 显卡驱动 | **32.0.21045.9003** |
| Adrenalin 版本 | **25.10.45.09** |
| 驱动日期 | 2026-08-27 |

采集命令：

```powershell
Get-CimInstance Win32_OperatingSystem | Select Caption,Version,BuildNumber,OSArchitecture

# 若 Win32_VideoController 被拒（某些环境的 WMI 权限），用注册表兜底：
Get-CimInstance Win32_VideoController | Select Name,DriverVersion,DriverDate
Get-ItemProperty 'HKLM:\SOFTWARE\AMD\CN' | Select CNVersion,DriverVersion,InstallDir
```

---

## 2. ROCm / HIP SDK

| 项目 | 实测值 |
|---|---|
| 版本 | **HIP SDK 6.2.4** |
| 安装路径 | `C:\Program Files\AMD\ROCm\6.2\` |
| `HIP_PATH` | `C:\Program Files\AMD\ROCm\6.2\` |
| `HIP_VERSION_MAJOR` | 6 |
| `HIP_VERSION_MINOR` | 2 |
| `HIP_VERSION_PATCH` | 41512 |
| `HIP_VERSION_GITHASH` | `db3292736` |
| 关键运行库 | `amdhip64_6.dll`(23.7 MB)、`rocblas.dll`(84.8 MB)、`rocfft.dll`(13.2 MB)、`rocsparse.dll`(686 MB)、`rocsolver.dll`(749 MB)、`amd_comgr0602.dll`(112 MB) |

采集命令：

```powershell
$env:HIP_PATH
Get-Content 'C:\Program Files\AMD\ROCm\6.2\include\hip\hip_version.h' | Select-String VERSION

# 最有价值的一条：确认运行时能否正确识别你的 GPU
& 'C:\Program Files\AMD\ROCm\6.2\bin\hipInfo.exe'
#   Name: AMD Radeon RX 6600
#   totalGlobalMem: 7.98 GB
#   major: 10
#   minor: 3          ← gfx1032
```

### gfx1032 补丁

| 项目 | 实测值 |
|---|---|
| 名称 | `rocm.gfx1032.for.hip.sdk.6.2.4.navi21.logic` |
| 文件数 | **144** |
| 总大小 | **223.6 MB** |
| 结构 | 根目录 1 个 `rocblas.dll`（80.8 MB）+ `library\` 下 143 个 Tensile 逻辑/内核文件 |
| 典型 library 文件 | `Kernels.so-000-gfx1032.hsaco`(230 KB)、`TensileLibrary_lazy_gfx1032.dat`(22 KB)、`TensileLibrary_Type_*_gfx1032.hsaco`(数十~数百 KB) |

**它的作用**：HIP SDK 6.2.4 官方不支持 gfx1032，替换后 rocBLAS 才有可用的 gfx1032 内核。**没有它，矩阵乘直接失败**，训练无从谈起。

> 该补丁非本仓库作品，也不便附带（体积大）。按上述名称在中文社区搜索即可。
> 若你的卡是 gfx1030 / gfx1100 等 ROCm 官方支持的型号，**不需要这一步**。

---

## 3. ZLUDA

| 项目 | 实测值 |
|---|---|
| 版本 | **ZLUDA 3.9.5**（`zluda.exe --version`） |
| 安装路径 | `C:\ZLUDA\bin\` |
| 原始压缩包 | `C:\ZLUDA\zluda.zip`（5.92 MB） |
| 内核缓存 | `C:\Users\<你>\AppData\Local\zluda\ComputeCache\zluda.db`（实测增长到 **159.9 MB**） |

`C:\ZLUDA\bin\` 内的文件与大小（可作为"是否为完整发行版"的校验）：

| 文件 | 大小 | 说明 |
|---|---|---|
| `nvcuda.dll` | 10,225,152 | 主实现 |
| `zluda_dump.dll` | 8,621,568 | 调试用 |
| `cudnn.dll` | 260,608 | **转发壳，卷积反向不可用** |
| `cublas.dll` | 247,296 | 转发到 rocBLAS，**可用** |
| `cublasLt.dll` | 185,856 | **空壳**（不引用任何 ROCm 库） |
| `cufft.dll` | 190,976 | 转发到 `hipfft`，**gfx1032 上不可用** |
| `cufftw.dll` | 142,336 | |
| `cusparse.dll` | 210,944 | |
| `nvml.dll` | 157,696 | |
| `nvrtc.dll` | 142,336 | 架构名不匹配，**编译 gfx1032 会失败** |
| `nccl.dll` | 121,344 | |
| `zluda_redirect.dll` | 190,976 | 注入用 |
| `zluda.exe` | 300,032 | 启动器 |

**判断一个 DLL 是"真实现"还是"转发壳"的方法**：在二进制里搜它引用了哪个 ROCm 库——

```powershell
$b=[System.IO.File]::ReadAllBytes('C:\ZLUDA\bin\cufft.dll')
$t=[System.Text.Encoding]::ASCII.GetString($b)
foreach($k in @('rocfft','rocblas','amdhip64','hipfft','hipsolver','rocsparse')){ if($t.Contains($k)){ "cufft.dll 引用: $k" } }
```

实测结果：

```
cufft.dll   → 引用 hipfft        （转发到 ROCm 的 hipfft → rocfft）
cublas.dll  → 引用 rocblas       （转发到 rocBLAS，被 gfx1032 补丁救活）
nvcuda.dll  → 引用 amdhip64      （真实现）
cublasLt.dll / cufftw.dll / zluda_redirect.dll → 不引用任何 ROCm 库（壳）
```

---

## 4. Python 与 PyTorch

| 项目 | 实测值 |
|---|---|
| Python | **3.11.9** (`tags/v3.11.9:de54cf5, Apr 2 2024, 10:12:12`) [MSC v.1938 64 bit] |
| venv 路径 | 任意（本文示例为 `venv_torch25\`） |
| PyTorch | **2.5.1+cu118** |
| torchaudio | **2.5.1+cu118** |
| ZLUDA 报告的 capability | `(8, 8)` ← 影响 `is_half` 判定，进而决定用不用 fp16 |
| ZLUDA 报告的设备名 | `AMD Radeon RX 6600 [ZLUDA]` |

```powershell
& <venv>\Scripts\python.exe -c "import torch,sys;print(sys.version);print(torch.__version__,torch.version.cuda)"
```

> **为什么用 cu118 wheel？** ZLUDA 的定位就是"用 AMD 卡假冒 NVIDIA"，它拦截 CUDA 的库调用，因此需要**CUDA 构建**的 PyTorch，而不是 ROCm 构建的。这是 ZLUDA 的标准用法。

> **capability `(8,8)` 的连锁影响**：GPT-SoVITS 的 `config.py` 会据此判定 `is_half=True`，于是 SV 特征提取把 wav 转成 fp16，进而触发 fp16 的 `torch.fft.rfft` —— 正是坑 1 的入口。

---

## 5. GPT-SoVITS

| 项目 | 实测值 |
|---|---|
| commit | **`48b1a0169a28582a8984402f82cf438d3bfa6aca`** |
| commit subject | `Fix Fun-ASR-Nano Transformers requirement (#2824)` |
| commit date | Tue Aug 18 17:16:25 2026 +0800 |
| 远程 | `https://github.com/RVC-Boss/GPT-SoVITS.git` |
| 分支 | `main` |
| 硬编码版本 | `webui.py` 第 4 行：`os.environ["version"] = version = "v2Pro"` |

```powershell
cd <GPT-SoVITS根目录>
git log -1 --format='%H%n%s%n%cd'
git remote -v
```

**补丁基线**：`patches/gpt-sovits-zluda.patch` 是针对上述 commit 生成的。若你的代码更新了，先 `git apply --check` 试一下；失败则需手工合并（改动集中在 11 个文件，逐处对照 `docs/TROUBLESHOOTING.md` 即可）。

---

## 6. 需要额外下载的数据（不在 pip 里）

| 项 | 大小 | 位置 | 用途 |
|---|---|---|---|
| `fast_langdetect/lid.176.bin` | 125.2 MB | `<GPT-SoVITS>\GPT_SoVITS\pretrained_models\fast_langdetect\` | 语言检测（**必需**） |
| `nltk_data/corpora/cmudict.zip` | 875 KB | `C:\Users\<你>\nltk_data\` | 英文 G2P（**必需**） |
| `nltk_data/taggers/averaged_perceptron_tagger.zip` | 2.5 MB | 同上 | 同上 |
| `nltk_data/taggers/averaged_perceptron_tagger_eng.zip` | 1.5 MB | 同上 | 同上 |
| `nltk_data/tokenizers/punkt.zip` | 13.6 MB | 同上 | 同上 |
| `models--FunAudioLLM--Fun-ASR-Nano-2512` | 1.9 GB | `C:\Users\<你>\.cache\huggingface\hub\` | ASR（仅前处理） |
| `models--funasr--fsmn-vad` | 4 MB | 同上 | ASR VAD（仅前处理） |

**注意**：`fast_langdetect` 的**目录必须先存在**，该库对用户指定的缓存目录不会自动创建（详见 README 5.2）。

---

## 7. 磁盘占用参考

| 项 | 实测 |
|---|---|
| ROCm 安装 | 约 10 GB（含 rocsparse/rocsolver 等大库） |
| gfx1032 补丁 | 223.6 MB |
| ZLUDA | 约 21 MB（+ 缓存 160 MB） |
| venv（含 torch） | 约 8–10 GB |
| 预训练模型（v2Pro + hubert + bert + sv + s1v3） | 约 1.2 GB |
| 每条数据的前处理产物 | 317 条 ≈ 25 分钟音频 → 约 200 MB（`4-cnhubert` + `5-wav32k` + `7-sv_cn`） |
| 训练 checkpoint（s2 每轮） | `G_*.pth` 951 MB + `D_*.pth` 758 MB |
| 训练小权重（可推理） | SoVITS ≈ 129 MB / GPT ≈ 148 MB |
| **建议预留** | **≥ 250 GB** |
