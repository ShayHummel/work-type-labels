"""Repair fine-tuning outputs written by the first finetune.py version (no "predict_order" in done.json).

That version used Trainer.predict() with train_sampling_strategy="group_by_length", and transformers 5.x then
also length-groups the prediction batches: row j of val_proba / holdout_proba belongs to dataset index order[j],
not j. The order is deterministic (LengthGroupedSampler, eval batch size, seed 42), so it is rebuilt here from
the same tokenisation and inverted. The repaired files are checked by the validation score (a wrong order scores
at chance, ≈ 0.18 accuracy).

Usage: uv run python src/fix_ft_order.py data/ft/<key>/fold0 [more fold dirs …]
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score
from transformers import AutoTokenizer
from transformers.trainer_pt_utils import LengthGroupedSampler

from finetune import LABELS, MODELS, TextDataset, encode

ROOT = Path(__file__).resolve().parents[1]


def grouped_order(ds, batch_size: int, seed: int = 42) -> np.ndarray:
    sampler = LengthGroupedSampler(batch_size, dataset=ds, model_input_name="input_ids",
                                   generator=torch.Generator().manual_seed(seed))
    return np.array(list(iter(sampler)))


def repair(fdir: Path) -> None:
    done = json.loads((fdir / "done.json").read_text())
    if done.get("predict_order"):
        print(f"{fdir}: already in dataset order ({done['predict_order']}) — nothing to do")
        return
    cfg = MODELS[done["key"]]
    tok = AutoTokenizer.from_pretrained(cfg["hf"])
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    train = pd.read_json(ROOT / "train.jsonl", lines=True)
    hold = pd.read_json(ROOT / "holdout.jsonl", lines=True)
    y = train.label.map({lab: i for i, lab in enumerate(LABELS)}).values
    va_idx = np.load(fdir / "val_idx.npy")
    eval_bs = done["batch"] // cfg["accum"] * 2  # per_device_eval_batch_size = 2 × per-device train batch
    fixed = {}
    for name, texts in (("val_proba", train.text.values[va_idx]), ("holdout_proba", hold.text.values)):
        ds = TextDataset(encode(tok, list(texts), done["max_len"]), None)
        order = grouped_order(ds, eval_bs)
        assert sorted(order) == list(range(len(ds)))
        saved = np.load(fdir / f"{name}.npy")
        out = np.empty_like(saved)
        out[order] = saved
        fixed[name] = out
    f_old = f1_score(y[va_idx], np.load(fdir / "val_proba.npy").argmax(1), labels=range(10), average=None)
    f_new = f1_score(y[va_idx], fixed["val_proba"].argmax(1), labels=range(10), average=None)
    acc_new = (fixed["val_proba"].argmax(1) == y[va_idx]).mean()
    print(f"{fdir}: stored order macro-F1 {f_old.mean():.3f} → repaired {f_new.mean():.3f} "
          f"(min-F1 {f_new.min():.3f}, acc {acc_new:.3f})")
    if acc_new < 0.4:
        print("  repaired order still scores near chance — NOT writing; the order could not be reproduced")
        return
    for name, arr in fixed.items():
        np.save(fdir / f"{name}.npy", arr)
    done.update({"predict_order": "repaired-by-fix_ft_order", "val_macro_F1": float(f_new.mean()),
                 "val_min_F1": float(f_new.min()), "val_accuracy": float(acc_new),
                 "val_F1": dict(zip(LABELS, map(float, f_new)))})
    (fdir / "done.json").write_text(json.dumps(done, indent=2))


if __name__ == "__main__":
    for d in sys.argv[1:]:
        repair(Path(d))
