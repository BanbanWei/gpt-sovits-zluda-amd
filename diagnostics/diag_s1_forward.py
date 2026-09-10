"""Instrument forward_old with the real GBC batch to find where the length goes to 0."""
import os
import sys
import traceback

import torch
import torch.nn.functional as F
import yaml

import _env
PROJ = _env.gpt_sovits_root()
sys.path.insert(0, PROJ)
sys.path.insert(0, os.path.join(PROJ, "GPT_SoVITS"))

cfg_path = os.environ.get("GPT_SOVITS_S1_CONFIG") or os.path.join(PROJ, "TEMP", "tmp_s1.yaml")
if not os.path.isfile(cfg_path):
    raise SystemExit(
        "找不到 s1 训练配置: %s\n"
        "请先通过 webui 启动一次 GPT 训练（webui 会写出 TEMP/tmp_s1.yaml），\n"
        "或用环境变量 GPT_SOVITS_S1_CONFIG 指定你的配置路径。" % cfg_path
    )
cfg = yaml.safe_load(open(cfg_path, encoding="utf-8"))

from AR.data.dataset import Text2SemanticDataset

ds = Text2SemanticDataset(
    phoneme_path=cfg["train_phoneme_path"],
    semantic_path=cfg["train_semantic_path"],
    max_sec=cfg["data"]["max_sec"],
    pad_val=cfg["data"]["pad_val"],
)
print("dataset len:", len(ds))

batch = ds.collate([ds[i] for i in range(4)])
x = batch["phoneme_ids"].cuda()
xl = batch["phoneme_ids_len"].cuda()
y = batch["semantic_ids"].cuda()
yl = batch["semantic_ids_len"].cuda()
bf = batch["bert_feature"].cuda()

print("phoneme_ids    :", tuple(x.shape), "lens", xl.tolist())
print("semantic_ids   :", tuple(y.shape), "lens", yl.tolist())
print("bert_feature   :", tuple(bf.shape))
print("x_lens.max()   :", int(xl.max()))
print("y_lens.max()   :", int(yl.max()))

from AR.models.t2s_model import Text2SemanticDecoder

m = Text2SemanticDecoder(config=cfg, top_k=3).cuda()
m.eval()

_orig_pad = m.pad_y_eos


def pad_instrumented(*a, **k):
    yy, tt = _orig_pad(*a, **k)
    print("  pad_y_eos -> y", tuple(yy.shape), " targets", tuple(tt.shape))
    return yy, tt


m.pad_y_eos = pad_instrumented

_orig_ce = F.cross_entropy


def ce_instrumented(inp, tgt, *a, **k):
    print("  cross_entropy: logits", tuple(inp.shape), " targets", tuple(tgt.shape))
    return _orig_ce(inp, tgt, *a, **k)


F.cross_entropy = ce_instrumented

_orig_h = m.h.forward


def h_instrumented(*a, **k):
    inp = a[0][0] if isinstance(a[0], tuple) else a[0]
    out = _orig_h(*a, **k)
    o = out[0] if isinstance(out, tuple) else out
    print("  transformer: in", tuple(inp.shape), "-> out", tuple(o.shape))
    return out


m.h.forward = h_instrumented

_orig_pred = m.ar_predict_layer.forward


def pred_instrumented(inp):
    out = _orig_pred(inp)
    print("  ar_predict_layer: in", tuple(inp.shape), "-> out", tuple(out.shape))
    return out


m.ar_predict_layer.forward = pred_instrumented

print("-" * 60)
try:
    loss, acc = m.forward_old(x, xl, y, yl, bf)
    print("forward_old OK  loss=", float(loss), " acc=", acc)
except Exception:
    traceback.print_exc()

torch.cuda.synchronize()
