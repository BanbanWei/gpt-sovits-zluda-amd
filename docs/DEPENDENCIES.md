# 依赖清单与版本约束

## 一、核心（必须严格匹配）

| 包 | 实测版本 | 约束原因 |
|---|---|---|
| `torch` | **2.5.1+cu118** | 必须是 **CUDA 构建**的 wheel（ZLUDA 拦截 CUDA 调用）。`+cu118` 是实测可用的组合 |
| `torchaudio` | **2.5.1+cu118** | 必须与 torch 同版本同构建 |
| `numpy` | **1.26.4** | **必须 `<2`**。numpy 2.x 会导致大量二进制扩展 ABI 不兼容 |
| `gradio` | **4.44.1** | **必须 `<5`**。gradio 5 改了 API（`{"__type__":"update"}` 等写法失效） |
| `pydantic` | **2.10.6** | 需 `<=2.10.6`（gradio 4.44 与更高版本 pydantic 冲突） |
| `transformers` | **4.56.2** | 需 `>=4.51,<5` |

---

## 二、GPT-SoVITS 训练/推理

| 包 | 实测版本 |
|---|---|
| `pytorch-lightning` | 2.6.5 |
| `torchmetrics` | 1.5.0 |
| `tokenizers` | 0.22.2 |
| `x-transformers` | 2.28.4 |
| `rotary-embedding-torch` | 0.8.8 |
| `torch-complex` | 0.4.4 |
| `torch-einops-utils` | 0.1.25 |
| `einops` | 0.8.2 |
| `peft` | 0.17.1 |
| `sentencepiece` | 0.2.2 |
| `scipy` | 1.17.1 |
| `librosa` | 0.10.2 |
| `soundfile` | 0.14.0 |
| `numba` | 0.67.0 |
| `onnxruntime` | 1.29.0 |
| `onnxruntime-gpu` | 1.29.0 |
| `ctranslate2` | 4.8.2 |
| `av` | 18.1.0 |

## 三、文本前端

| 包 | 实测版本 | 说明 |
|---|---|---|
| `jieba` | 0.42.1 | **`jieba_fast` 不要装**（Python 3.11 无 wheel），本项目补丁已回退到它 |
| `pypinyin` | 0.55.0 | 中文 |
| `cn2an` | 0.5.24 | 中文数字 |
| `g2p-en` | 2.1.0 | 英文；**依赖 nltk 的 `cmudict` / `averaged_perceptron_tagger` 语料**（见 README 5.1） |
| `ToJyutping` | 3.2.0 | 粤语 |
| `wordsegment` | 1.3.1 | 英文分词 |
| `fast-langdetect` | 1.0.1 | 语言检测；**需要 `lid.176.bin` 且缓存目录须先存在** |
| `split-lang` | 2.1.1 | 多语种切分 |
| `chardet` | 7.6.0 | |

**缺失/不可安装的包（实测）**

| 包 | 状态 | 影响 |
|---|---|---|
| `jieba_fast` | ❌ 无 Python 3.11 wheel，需 MSVC 编译，实测失败 | 本项目补丁回退到 `jieba` |
| `eunjeon` | ❌ 同样编译失败 | **仅韩文需要**。不用韩文可忽略 |

---

## 四、ASR（仅训练集前处理需要）

| 包 | 实测版本 |
|---|---|
| `funasr` | 1.4.15 |
| `modelscope` | 1.40.0 |
| `faster-whisper` | 1.2.1 |

**同时装了 `onnxruntime` 与 `onnxruntime-gpu`**，实测无冲突。

---

## 五、工具

| 包 | 实测版本 |
|---|---|
| `fastapi` | 0.115.2 |
| `PyYAML` | 6.0.3 |
| `psutil` | 7.2.2 |
| `tqdm` | 4.70.0 |

---

## 六、可直接使用的依赖文件

GPT-SoVITS 仓库根目录的 **`requirements_zluda.txt`** 已包含上述版本约束（`numpy<2`、`gradio<5`、`pydantic<=2.10.6`、`torchmetrics<=1.5` 等），是从上游 `requirements.txt` 调整而来。

安装顺序建议：

```powershell
# 1) 先装 cu118 版 torch（必须单独指定 index-url）
<venv>\Scripts\python.exe -m pip install torch==2.5.1+cu118 torchaudio==2.5.1+cu118 `
  --index-url https://download.pytorch.org/whl/cu118

# 2) 再装其余依赖
<venv>\Scripts\python.exe -m pip install -r requirements_zluda.txt
```

**顺序很重要**：先装 `requirements` 再装 torch，pip 可能把 torch 换成 CPU 版或 CUDA 12 版，ZLUDA 就不能用了。

---

## 七、完整体检命令

装完后一次性核对关键版本：

```powershell
<venv>\Scripts\python.exe -m pip list | Select-String -Pattern `
  '^(torch|torchaudio|numpy|transformers|tokenizers|gradio|pydantic|funasr|modelscope|onnxruntime|onnxruntime-gpu|ctranslate2|librosa|numba|pytorch-lightning|torchmetrics|x-transformers|peft|sentencepiece|jieba|pypinyin|g2p-en|fast-langdetect|split-lang)\s'
```

再确认 torch 能"看见"GPU：

```powershell
# 注意：必须写成脚本文件用 zluda.exe 启动，python -c 会被 zluda.exe 吃掉参数
C:\ZLUDA\bin\zluda.exe <venv>\Scripts\python.exe diagnostics\run_modes.py diagnostics\probe_fft.py 30 cuda_basic
```

期望输出包含：

```
RESULT OK AMD Radeon RX 6600 [ZLUDA] (8, 8)
```

若这里就失败，说明问题在 ROCm/ZLUDA 层，**不必往下走**——先解决设备枚举。

---

## 八、环境变量总表

| 变量 | 值 | 由谁设置 | 作用 |
|---|---|---|---|
| `HIP_PATH` | `C:\Program Files\AMD\ROCm\6.2\` | ROCm 安装器 | ROCm 定位 |
| `DISABLE_ADDMM_CUDA_LT` | `1` | `zluda_compat` | 绕开空壳 `cublasLt`（坑 4） |
| `TORCH_BLAS_PREFER_HIPBLASLT` | `0` | `zluda_compat` | 同上 |
| `PYTORCH_JIT` | `0` | `zluda_compat` | 禁用 TorchScript，绕开 NVRTC（坑 6） |
| `PYTHONUTF8` | `1` | 启动器 bat | 避免 rich 进度条写 `•` 到 GBK 控制台时 `UnicodeEncodeError` |
| `PYTHONIOENCODING` | `utf-8` | 启动器 bat | 同上 |
| `KMP_DUPLICATE_LIB_OK` | `TRUE` | **webui.py 自己设** | OpenMP 重复加载会 **abort 进程且无 traceback**。裸跑 FunASR 时必须自己设 |
| `FUNASR_DEVICE` | `cpu` 可强制回 CPU | 可选 | 不给则 ASR 默认走 GPU。训练同时跑 ASR 时设 `cpu`（GPU 路径占约 3.33 GB 显存） |
| `ZLUDA_COMPAT` | `0` 可关 | 可选 | 整体关闭兼容层 |
| `ZLUDA_COMPAT_VERBOSE` | `1` 开日志 | 可选 | 打印兼容层做了什么 |

`zluda_compat` 设置的四个变量实测有效：

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
