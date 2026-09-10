"""Replicate t2s_model.forward_old step by step with shape prints.

Slicing with a tensor bound is fine, so the 0-length slice means x_len is not 62
at line 505.  Prime suspect: self.ar_text_position(x), which may pad the sequence.
"""
import os
import sys

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
from AR.models.t2s_model import Text2SemanticDecoder
from AR.models.utils import make_pad_mask, make_pad_mask_left

ds = Text2SemanticDataset(
    phoneme_path=cfg["train_phoneme_path"],
    semantic_path=cfg["train_semantic_path"],
    max_sec=cfg["data"]["max_sec"],
    pad_val=cfg["data"]["pad_val"],
)
batch = ds.collate([ds[i] for i in range(4)])

x0 = batch["phoneme_ids"].cuda()
xl = batch["phoneme_ids_len"].cuda()
y0 = batch["semantic_ids"].cuda()
yl = batch["semantic_ids_len"].cuda()
bf = batch["bert_feature"].cuda()


def p(label, t):
    print("%-26s %s" % (label, tuple(t.shape) if hasattr(t, "shape") else t))


p("phoneme_ids in", x0)
p("semantic_ids in", y0)
print("xl", xl.tolist(), "yl", yl.tolist())

m = Text2SemanticDecoder(config=cfg, top_k=3).cuda()
m.eval()
print("-" * 60)

with torch.no_grad():
    x = m.ar_text_embedding(x0)
    p("ar_text_embedding", x)
    x = x + m.bert_proj(bf.transpose(1, 2))
    p("+ bert_proj", x)
    x = m.ar_text_position(x)
    p("ar_text_position", x)          # <-- does this change the length?
    x_mask = make_pad_mask_left(xl)
    p("x_mask", x_mask)

    y_mask = make_pad_mask(yl)
    p("y_mask", y_mask)
    y_mask_int = y_mask.type(torch.int64)
    codes = y0.type(torch.int64) * (1 - y_mask_int)
    yy, targets = m.pad_y_eos(codes, y_mask_int, eos_id=m.EOS)
    p("pad_y_eos y", yy)
    p("pad_y_eos targets", targets)

    x_len = xl.max()
    y_len = yl.max()
    print("x_len =", x_len.item(), " y_len =", y_len.item())

    y_emb = m.ar_audio_embedding(yy)
    p("y_emb", y_emb)
    y_pos = m.ar_audio_position(y_emb)
    p("y_pos", y_pos)

    xy_padding_mask = torch.concat([x_mask, y_mask], dim=1)
    p("xy_padding_mask", xy_padding_mask)

    x_attn_mask = F.pad(
        torch.zeros((x_len, x_len), dtype=torch.bool, device=x.device),
        (0, y_len),
        value=True,
    )
    p("x_attn_mask", x_attn_mask)
    y_attn_mask = F.pad(
        torch.triu(torch.ones(y_len, y_len, dtype=torch.bool, device=x.device), diagonal=1),
        (x_len, 0),
        value=False,
    )
    p("y_attn_mask", y_attn_mask)
    xy_attn_mask = torch.concat([x_attn_mask, y_attn_mask], dim=0)
    p("xy_attn_mask", xy_attn_mask)

    bsz, src_len = x.shape[0], x_len + y_len
    print("bsz =", bsz, " src_len =", src_len.item() if hasattr(src_len, "item") else src_len)
    _xy_padding_mask = (
        xy_padding_mask.view(bsz, 1, 1, src_len)
        .expand(-1, m.num_head, -1, -1)
        .reshape(bsz * m.num_head, 1, src_len)
    )
    p("_xy_padding_mask", _xy_padding_mask)
    xy_attn_mask = xy_attn_mask.logical_or(_xy_padding_mask)
    new_attn_mask = torch.zeros_like(xy_attn_mask, dtype=x.dtype)
    new_attn_mask.masked_fill_(xy_attn_mask, float("-inf"))
    xy_attn_mask = new_attn_mask
    p("final xy_attn_mask", xy_attn_mask)

    xy_pos = torch.concat([x, y_pos], dim=1)
    p("xy_pos", xy_pos)

    xy_dec, _ = m.h((xy_pos, None), mask=xy_attn_mask)
    p("xy_dec", xy_dec)

    print("(x_len-1).item() =", (x_len - 1).item())
    sliced = xy_dec[:, x_len - 1 :]
    p("xy_dec[:, x_len-1:]", sliced)
    sliced2 = xy_dec[:, int(x_len) - 1 :]
    p("xy_dec[:, int(x_len)-1:]", sliced2)

    logits = m.ar_predict_layer(sliced).permute(0, 2, 1)
    p("logits", logits)
    print("targets len =", targets.shape[-1])
    try:
        loss = F.cross_entropy(logits, targets, reduction="sum")
        print("cross_entropy OK loss =", float(loss))
    except Exception as e:
        print("cross_entropy FAILED:", type(e).__name__, e)

torch.cuda.synchronize()
print("DONE")
