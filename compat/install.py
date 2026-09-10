"""把 zluda_compat 兼容层装进指定的 Python 环境。

用法
----
    python compat/install.py                    # 自动探测常见 venv 名字
    python compat/install.py <venv目录>          # 指定 venv 根目录
    python compat/install.py <site-packages>     # 也可以直接给 site-packages
    python compat/install.py --uninstall <venv>
    python compat/install.py --check <venv>      # 只做自检，不改文件

它做两件事
----------
1. 复制 zluda_compat.py 到 <venv>\\Lib\\site-packages\\
2. 写一个 zluda_compat.pth，让**每个**解释器启动时自动 import 它

为什么用 .pth 而不是改 webui.py
-------------------------------
GPT-SoVITS 的训练与前处理步骤都是 webui.py 用 Popen 拉起的**独立进程**。
只改入口文件覆盖不全，而 .pth 对任何解释器都生效 —— 这一点是这套方案能
跑通训练的关键之一。
"""

import argparse
import os
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
MODULE = "zluda_compat.py"
PTH = "zluda_compat.pth"

CANDIDATE_VENVS = ("venv_torch25", "venv", ".venv", "env")


def log(*a):
    print(*a, flush=True)


def resolve_site_packages(target):
    """把用户给的路径归一化成 site-packages 目录。"""
    target = os.path.abspath(target)

    # 直接指向 site-packages
    if os.path.basename(target).lower() == "site-packages":
        return target

    # 指向 venv 根目录
    for sub in (
        os.path.join("Lib", "site-packages"),          # Windows
        os.path.join("lib", "site-packages"),          # 少数情况
    ):
        p = os.path.join(target, sub)
        if os.path.isdir(p):
            return p

    # 指向 Scripts / bin 的父目录
    if os.path.basename(target).lower() in ("scripts", "bin"):
        return resolve_site_packages(os.path.dirname(target))

    return None


def autodetect():
    """在常见位置找 venv。"""
    roots = [
        os.path.dirname(HERE),                       # 仓库同级
        os.path.dirname(os.path.dirname(HERE)),
        os.getcwd(),
        os.path.dirname(os.getcwd()),
    ]
    tried = []
    for root in roots:
        for name in CANDIDATE_VENVS:
            cand = os.path.join(root, name)
            tried.append(cand)
            if os.path.isdir(cand):
                sp = resolve_site_packages(cand)
                if sp:
                    return cand, sp
    log("自动探测失败，已尝试：")
    for t in tried:
        log("  " + t)
    return None, None


def do_install(sp_dir):
    src_mod = os.path.join(HERE, MODULE)
    src_pth = os.path.join(HERE, PTH)
    for f in (src_mod, src_pth):
        if not os.path.isfile(f):
            log("[错误] 缺少文件：%s" % f)
            return False

    dst_mod = os.path.join(sp_dir, MODULE)
    dst_pth = os.path.join(sp_dir, PTH)

    # 备份已有版本
    for dst in (dst_mod, dst_pth):
        if os.path.isfile(dst):
            bak = dst + ".bak"
            shutil.copy2(dst, bak)
            log("  已备份旧文件 -> %s" % os.path.basename(bak))

    shutil.copy2(src_mod, dst_mod)
    shutil.copy2(src_pth, dst_pth)
    log("  已写入 %s" % dst_mod)
    log("  已写入 %s" % dst_pth)
    return True


def do_uninstall(sp_dir):
    removed = []
    for name in (MODULE, PTH):
        p = os.path.join(sp_dir, name)
        if os.path.isfile(p):
            os.remove(p)
            removed.append(name)
    if removed:
        log("  已删除：%s" % ", ".join(removed))
    else:
        log("  没有找到已安装的文件（无需卸载）")
    return True


def do_check(venv_root):
    """用目标环境的 python 跑一次自检。"""
    py = None
    for sub in (os.path.join("Scripts", "python.exe"), os.path.join("bin", "python")):
        p = os.path.join(venv_root, sub)
        if os.path.isfile(p):
            py = p
            break
    if py is None:
        log("[跳过自检] 在 %s 里找不到 python" % venv_root)
        return

    code = (
        "import sys;"
        "print('zluda_compat loaded :', 'zluda_compat' in sys.modules);"
        "import os;"
        "print('DISABLE_ADDMM_CUDA_LT:', os.environ.get('DISABLE_ADDMM_CUDA_LT','<unset>'));"
        "print('PYTORCH_JIT          :', os.environ.get('PYTORCH_JIT','<unset>'));"
        "import torch;"
        "print('cudnn.enabled        :', torch.backends.cudnn.enabled);"
        "print('matmul.allow_tf32    :', torch.backends.cuda.matmul.allow_tf32);"
        "f=getattr(torch.backends.cuda,'enable_cudnn_sdp',None);"
        "print('cudnn_sdp_enabled    :', torch.backends.cuda.cudnn_sdp_enabled() if hasattr(torch.backends.cuda,'cudnn_sdp_enabled') else 'n/a');"
        "print('flash_sdp_enabled    :', torch.backends.cuda.flash_sdp_enabled() if hasattr(torch.backends.cuda,'flash_sdp_enabled') else 'n/a');"
    )
    log("")
    log("--- 自检（用 %s）---" % py)
    try:
        subprocess.run([py, "-c", code], check=False)
    except Exception as e:
        log("  自检失败：%s" % e)

    log("")
    log("期望看到的输出：")
    log("  zluda_compat loaded  : True")
    log("  DISABLE_ADDMM_CUDA_LT: 1")
    log("  PYTORCH_JIT          : 0")
    log("  cudnn.enabled        : False")


def main():
    ap = argparse.ArgumentParser(description="安装 zluda_compat 兼容层")
    ap.add_argument("target", nargs="?", help="venv 根目录或 site-packages 目录")
    ap.add_argument("--uninstall", action="store_true", help="卸载")
    ap.add_argument("--check", action="store_true", help="只做自检，不改文件")
    ap.add_argument("--no-check", action="store_true", help="装完不自检")
    args = ap.parse_args()

    venv_root = None
    sp_dir = None

    if args.target:
        sp_dir = resolve_site_packages(args.target)
        if sp_dir is None:
            log("[错误] 无法从 %r 定位 site-packages" % args.target)
            log("       请给出 venv 根目录，或形如 <venv>\\Lib\\site-packages 的路径")
            return 1
        # 反推 venv 根目录（用于自检）
        guess = os.path.dirname(os.path.dirname(sp_dir))   # ...\<venv>\Lib\site-packages -> <venv>
        if os.path.basename(os.path.dirname(sp_dir)).lower() == "lib":
            venv_root = guess
    else:
        venv_root, sp_dir = autodetect()
        if sp_dir is None:
            log("")
            log("请显式指定 venv：")
            log("  python %s <你的venv目录>" % os.path.join("compat", "install.py"))
            return 1

    log("venv          : %s" % (venv_root or "<未知>"))
    log("site-packages : %s" % sp_dir)
    log("")

    if args.check:
        if venv_root:
            do_check(venv_root)
        else:
            log("给的是 site-packages，无法定位 python 做自检；请改为传 venv 根目录")
        return 0

    if args.uninstall:
        log("卸载中...")
        do_uninstall(sp_dir)
        return 0

    log("安装中...")
    if not do_install(sp_dir):
        return 1

    if not args.no_check and venv_root:
        do_check(venv_root)

    log("")
    log("完成。这个兼容层对以下情况自动生效：")
    log("  * webui.py 拉起的训练/前处理子进程")
    log("  * api.py / 任意直接运行的脚本")
    log("可用 ZLUDA_COMPAT=0 关闭，ZLUDA_COMPAT_VERBOSE=1 查看它做了什么。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
