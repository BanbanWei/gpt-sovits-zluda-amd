"""Is the s1 batch internally consistent, especially across DataLoader workers?

For every batch: does the padded width of semantic_ids match semantic_ids_len.max()?
Also repeats the check with the tensors moved to CUDA, to cover the H2D step.
Runs the multi-worker cases under a __main__ guard (Windows spawn).
"""
import os
import sys

import torch
import yaml

PROJ = r"D:\saves\GPT-SoVITS\GPT-SoVITS"
sys.path.insert(0, PROJ)
sys.path.insert(0, os.path.join(PROJ, "GPT_SoVITS"))

cfg = yaml.safe_load(open(r"D:\saves\deepseek-harness\tmp_s1.yaml", encoding="utf-8"))

from torch.utils.data import DataLoader

from AR.data.bucket_sampler import DistributedBucketSampler
from AR.data.dataset import Text2SemanticDataset


def build():
    return Text2SemanticDataset(
        phoneme_path=cfg["train_phoneme_path"],
        semantic_path=cfg["train_semantic_path"],
        max_sec=cfg["data"]["max_sec"],
        pad_val=cfg["data"]["pad_val"],
    )


def check(nw, to_cuda=False, bs=4, verbose=False):
    ds = build()
    sampler = DistributedBucketSampler(ds, batch_size=bs)
    kw = {}
    if nw > 0:
        kw = dict(num_workers=nw, persistent_workers=True, prefetch_factor=16)
    dl = DataLoader(ds, batch_size=bs, sampler=sampler, collate_fn=ds.collate, **kw)

    n = 0
    bad = 0
    for b in dl:
        n += 1
        if to_cuda:
            b = {k: (v.cuda() if torch.is_tensor(v) else v) for k, v in b.items()}
        P = b["phoneme_ids"].shape[1]
        pl = int(b["phoneme_ids_len"].max().cpu())
        S = b["semantic_ids"].shape[1]
        sl = int(b["semantic_ids_len"].max().cpu())
        if P != pl or S != sl:
            bad += 1
            if bad <= 4:
                print("  MISMATCH batch %d: phoneme pad=%d max_len=%d | semantic pad=%d max_len=%d"
                      % (n, P, pl, S, sl), flush=True)
                print("    phoneme_lens :", b["phoneme_ids_len"].tolist(), flush=True)
                print("    semantic_lens:", b["semantic_ids_len"].tolist(), flush=True)
        elif verbose:
            print("  batch %d ok: P=%d pl=%d S=%d sl=%d" % (n, P, pl, S, sl), flush=True)
    print("num_workers=%d to_cuda=%s -> scanned %d batches, mismatches=%d"
          % (nw, to_cuda, n, bad), flush=True)
    print("-" * 70, flush=True)


if __name__ == "__main__":
    print("cuda available:", torch.cuda.is_available(), flush=True)

    # DistributedBucketSampler asks dist.get_world_size() whenever CUDA is
    # available, so a process group must exist even for this single-process check.
    import torch.distributed as dist

    if torch.cuda.is_available() and not dist.is_initialized():
        os.environ.setdefault("MASTER_ADDR", "localhost")
        os.environ.setdefault("MASTER_PORT", "29511")
        os.environ["USE_LIBUV"] = "0"  # this torch build has no libuv support
        dist.init_process_group(backend="gloo", rank=0, world_size=1)

    check(0)
    check(2)
    check(4)
    check(0, to_cuda=True)
    check(4, to_cuda=True)
    print("DONE", flush=True)
