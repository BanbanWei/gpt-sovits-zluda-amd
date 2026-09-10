"""Scan every s1 training batch looking for the degenerate shape that breaks
F.cross_entropy in t2s_model.forward_old (logits length 0 vs targets length 120).
"""
import os
import sys

import torch
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
cfg["data"]["num_workers"] = 0  # keep it in-process

from AR.data.data_module import Text2SemanticDataModule

print("phoneme path:", cfg["train_phoneme_path"])
print("semantic path:", cfg["train_semantic_path"])

dm = Text2SemanticDataModule(
    cfg,
    train_semantic_path=cfg["train_semantic_path"],
    train_phoneme_path=cfg["train_phoneme_path"],
)
dm.setup()
ds = dm._train_dataset
print("dataset len:", len(ds))

plens = [len(p) for s, p in ds.semantic_phoneme]
slens = [len(s) for s, p in ds.semantic_phoneme]
print("phoneme len  min/max:", min(plens), max(plens), " zeros:", sum(1 for x in plens if x == 0))
print("semantic len min/max:", min(slens), max(slens), " zeros:", sum(1 for x in slens if x == 0))

for i, n in enumerate(ds.item_names):
    if "0023521600" in n:
        s, p = ds.semantic_phoneme[i]
        print("SUSPECT item still present at idx", i, n, "sem_len", len(s), "ph_len", len(p))

print("-" * 60)
dl = dm.train_dataloader()
bad = 0
n = 0
for bi, batch in enumerate(dl):
    n += 1
    pshape = tuple(batch["phoneme_ids"].shape)
    sshape = tuple(batch["semantic_ids"].shape)
    pmax = int(batch["phoneme_ids_len"].max())
    smax = int(batch["semantic_ids_len"].max())
    if pshape[1] == 0 or sshape[1] == 0 or pmax == 0 or smax == 0:
        bad += 1
        print("BAD BATCH #%d  phoneme_ids%s semantic_ids%s  pmax=%d smax=%d"
              % (bi, pshape, sshape, pmax, smax))
        print("   phoneme_lens:", batch["phoneme_ids_len"].tolist())
        print("   semantic_lens:", batch["semantic_ids_len"].tolist())
        print("   ids:", batch["ids"])
        for j in batch["ids"]:
            print("      idx %d name=%s p=%d s=%d"
                  % (j, ds.item_names[j], plens[j], slens[j]))
        if bad >= 5:
            break

print("-" * 60)
print("scanned batches:", n, " bad:", bad)

# What forward_old would compute for the first batch
b0 = ds.collate([ds[i] for i in range(4)])
x_len = int(b0["phoneme_ids_len"].max())
y_len = int(b0["semantic_ids_len"].max())
print("first batch: x_len=%d y_len=%d" % (x_len, y_len))
print("forward_old logits length would be:", (x_len + y_len) - (x_len - 1))
