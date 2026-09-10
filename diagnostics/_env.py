"""共用路径解析：环境变量优先，其次自动探测，最后回退到示例路径。

这个模块让 `diagnostics/` 里的探针可以脱离原作者目录结构使用。

配置方式（二选一即可）
--------------------
    set GPT_SOVITS_ROOT=D:\\path\\to\\GPT-SoVITS
    set GPT_SOVITS_PYTHON=D:\\path\\to\\venv\\Scripts\\python.exe

不设置时会自动探测：从当前工作目录和本文件所在目录**逐级向上**寻找同时含有
`webui.py` 与 `GPT_SoVITS/` 的目录；python 解释器则在仓库同级/内部的常见 venv
名字（`venv_torch25` / `venv` / `.venv`）里找。
"""

import glob
import os

# 最后兜底用的示例路径（在别人的机器上会被自动探测覆盖）
_EXAMPLE_ROOT = r"D:\GPT-SoVITS"
_EXAMPLE_VENV = "venv_torch25"
_VENV_NAMES = ("venv_torch25", "venv", ".venv", "env")


def _looks_like_root(p):
    return bool(p) and os.path.isfile(os.path.join(p, "webui.py")) and os.path.isdir(os.path.join(p, "GPT_SoVITS"))


def gpt_sovits_root():
    """GPT-SoVITS 仓库根目录。"""
    env = os.environ.get("GPT_SOVITS_ROOT")
    if env and _looks_like_root(env):
        return os.path.abspath(env)

    here = os.path.dirname(os.path.abspath(__file__))
    for start in (os.getcwd(), here):
        cur = os.path.abspath(start)
        for _ in range(8):
            if _looks_like_root(cur):
                return cur
            parent = os.path.dirname(cur)
            if parent == cur:
                break
            cur = parent

    if env:  # 用户明确指定了，即使不像也照用
        return os.path.abspath(env)
    return _EXAMPLE_ROOT


def gpt_sovits_python():
    """venv 里的 python.exe。"""
    env = os.environ.get("GPT_SOVITS_PYTHON")
    if env and os.path.isfile(env):
        return os.path.abspath(env)

    root = gpt_sovits_root()
    bases = [os.path.dirname(root), root]
    for base in bases:
        for name in _VENV_NAMES:
            for sub in (os.path.join("Scripts", "python.exe"), os.path.join("bin", "python")):
                p = os.path.join(base, name, sub)
                if os.path.isfile(p):
                    return p

    return os.path.join(os.path.dirname(root), _EXAMPLE_VENV, "Scripts", "python.exe")


def gpt_sovits_site_packages():
    """venv 的 site-packages 目录（由 python 路径反推）。"""
    py = gpt_sovits_python()
    venv = os.path.dirname(os.path.dirname(py))          # <venv>\Scripts\python.exe -> <venv>
    for sub in (os.path.join("Lib", "site-packages"), os.path.join("lib", "site-packages")):
        p = os.path.join(venv, sub)
        if os.path.isdir(p):
            return p
    return os.path.join(venv, "Lib", "site-packages")


def find_test_wav():
    """随便找一个可用于测试的音频（探针只需要一段真实 32k wav）。

    优先环境变量 GPT_SOVITS_TEST_WAV；否则在 <root>/logs/*/5-wav32k/ 下找第一个。
    **找不到时返回 None**（调用方应给出明确提示，而不是拿一个假路径去报
    FileNotFoundError）。
    """
    env = os.environ.get("GPT_SOVITS_TEST_WAV")
    if env and os.path.isfile(env):
        return env

    root = gpt_sovits_root()
    for pat in (
        os.path.join(root, "logs", "*", "5-wav32k", "*.wav"),
        os.path.join(root, "logs", "*", "*.wav"),
        os.path.join(root, "output", "slicer_opt", "*.wav"),
    ):
        hits = sorted(glob.glob(pat))
        if hits:
            return hits[0]
    return None


if __name__ == "__main__":
    _wav = find_test_wav()
    print("GPT_SOVITS_ROOT          :", gpt_sovits_root())
    print("GPT_SOVITS_PYTHON        :", gpt_sovits_python(), os.path.isfile(gpt_sovits_python()))
    print("GPT_SOVITS_SITE_PACKAGES :", gpt_sovits_site_packages(), os.path.isdir(gpt_sovits_site_packages()))
    if _wav:
        print("TEST_WAV                 :", _wav, os.path.isfile(_wav))
    else:
        print("TEST_WAV                 : (未找到) 需要真实音频的探针请设 GPT_SOVITS_TEST_WAV")
