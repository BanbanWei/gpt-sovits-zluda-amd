# 启动器使用说明

两个 `.bat` 功能相同，**唯一区别是输出走不走管道**。

| 文件 | GPT 训练进度条 | 写日志文件 | 用途 |
|---|---|---|---|
| `启动GPT-SoVITS.bat` | ✅ **可见** | ❌ | 日常使用 |
| `启动GPT-SoVITS-带日志.bat` | ❌ 不可见 | ✅ `webui_console.log` | 出问题要排查时 |

## 为什么进度条会不一样

不是 bug，是两套进度条对"输出被重定向"的反应不同：

| | SoVITS (s2) | GPT (s1) |
|---|---|---|
| 库 | **tqdm** | **rich**（PyTorch Lightning 的 RichProgressBar） |
| 检测到不是终端时 | 自动退化成"每次打一整行 + 换行" | 仍是**原地重绘**模式，只发 `\r`、从不发 `\n` |
| 结果 | 满屏可见 | 被 PowerShell / 重定向的**按行缓冲**吞掉，一个字都不显示 |

**任何重定向（写文件、管道）都会让 rich 判定"不是终端"从而不画进度条。** 想看到 GPT 的实时进度，就必须用普通那个。

## 必须改的三处路径

两个 bat 里都有这几个硬编码路径，按你自己的环境替换：

```bat
cd /d "<GPT-SoVITS 根目录>"

if not exist "<venv>\Scripts\python.exe" ( ... )

"C:\ZLUDA\bin\zluda.exe" "<venv>\Scripts\python.exe" webui.py
```

## 必须保持 CRLF 换行

这两个文件**必须是 CRLF 换行**。如果被某个编辑器/工具转成 LF，cmd 解析多行 `if not exist (...)` 块会出错，表现为"双击没反应"。

检查与修复：

```powershell
$f = '启动GPT-SoVITS.bat'
$b = [System.IO.File]::ReadAllBytes($f)
$crlf=0; $lf=0
for($i=0;$i -lt $b.Length;$i++){ if($b[$i] -eq 10){ if($b[$i-1] -eq 13){$crlf++}else{$lf++} } }
"CRLF=$crlf  bare-LF=$lf"      # bare-LF 必须为 0

# 修复：
$enc = New-Object System.Text.UTF8Encoding($false)
$t = [System.IO.File]::ReadAllText($f, $enc)
[System.IO.File]::WriteAllText($f, $t.Replace("`r`n","`n").Replace("`n","`r`n"), $enc)
```

## 编码说明

bat 采用 **UTF-8（无 BOM）+ 文件第二行 `chcp 65001`**。因为第一二行是纯 ASCII，cmd 能正确解析，之后中文 echo 在 UTF-8 代码页下正常显示。

`PYTHONUTF8=1` / `PYTHONIOENCODING=utf-8` 是为了避免 rich 往 GBK 控制台写 `•` 时报：

```
UnicodeEncodeError: 'gbk' codec can't encode character '\u2022'
```

## 关于 `zluda.exe` 的参数解析

`zluda.exe` 会把 `-` 开头的参数当成**自己的**选项，所以下面这种写法会失败：

```
zluda.exe python.exe -c "print(1)"
# → Unrecognized argument: "-c". Did you mean "--nccl"?
```

**给 zluda.exe 启动的 python 传脚本，必须写成 `.py` 文件**，不能用 `-c`。

在 bat 里 `zluda.exe ... webui.py` 这种不带 `-` 的写法没有问题。

## 端口

`config.py` 里定义的端口：

| 用途 | 端口 |
|---|---|
| 主界面 | 9874 |
| TTS 推理子界面 | 9872 |
| UVR5 | 9873 |
| API | 9880 |

**主界面同一时间只能开一个。** 上一个实例没关干净时会占住 9874，新启动的会绑不上端口。检查：

```powershell
Get-NetTCPConnection -LocalPort 9874 -State Listen | Select LocalPort,OwningProcess
```
