"""Import result zips downloaded from Colab (My Drive/zuzai-hw/exports/) into data/ and validate them.

Handles embedding zips (emb_<key>.zip → data/emb/<key>/) and fine-tuning zips (ft_<key>.zip → data/ft/<key>/fold*/).

Usage: uv run python src/import_colab.py ~/Downloads/emb_qwen3-8b.zip [more.zip …]
"""

import hashlib
import json
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def local_ids_sha1() -> str:
    ids = []
    for name in ("train.jsonl", "holdout.jsonl"):
        with open(ROOT / name) as f:
            ids += [json.loads(line)["id"] for line in f]
    return hashlib.sha1("\n".join(ids).encode()).hexdigest()


def main(paths: list[str]) -> None:
    sha = local_ids_sha1()
    folds = pd.read_csv(ROOT / "data" / "folds.csv").fold.values
    n_hold = sum(1 for _ in open(ROOT / "holdout.jsonl"))
    for p in paths:
        with zipfile.ZipFile(p) as z:
            z.extractall(ROOT / "data")
            manifests = [n for n in z.namelist() if n.endswith("manifest.json")]
            dones = [n for n in z.namelist() if n.endswith("done.json")]
        for m in dones:  # fine-tuning folds
            d = (ROOT / "data" / m).parent
            done = json.loads((d / "done.json").read_text())
            vi, vp, hp = (np.load(d / f) for f in ("val_idx.npy", "val_proba.npy", "holdout_proba.npy"))
            problems = []
            if not done.get("limit") and not np.array_equal(vi, np.flatnonzero(folds == done["fold"])):
                problems.append("val rows differ from data/folds.csv")
            if vp.shape != (len(vi), 10) or hp.shape[1] != 10 or (not done.get("limit") and hp.shape[0] != n_hold):
                problems.append(f"shape mismatch {vp.shape} {hp.shape}")
            if not (np.isfinite(vp).all() and np.isfinite(hp).all()):
                problems.append("non-finite values")
            if done.get("limit") or done.get("max_steps"):
                problems.append("SMOKE TEST run (limit/max_steps set) — not a real result")
            status = "OK" if not problems else "PROBLEM: " + "; ".join(problems)
            print(f"{p} → {d.relative_to(ROOT)}  [{done['hf']}, fold {done['fold']}, val macro-F1 "
                  f"{done['val_macro_F1']:.3f}, min-F1 {done['val_min_F1']:.3f}, {done['device']}]  {status}")
        for m in manifests:
            d = (ROOT / "data" / m).parent
            man = json.loads((d / "manifest.json").read_text())
            problems = []
            if man["ids_sha1"] != sha:
                problems.append("ids hash differs from local jsonl")
            if (d / "train.npy").exists():  # embeddings
                tr, ho = np.load(d / "train.npy"), np.load(d / "holdout.npy")
                if tr.shape != (man["n_train"], man["dim"]) or ho.shape != (man["n_holdout"], man["dim"]):
                    problems.append(f"shape mismatch {tr.shape} {ho.shape}")
                if not (np.isfinite(tr.astype(np.float32)).all() and np.isfinite(ho.astype(np.float32)).all()):
                    problems.append("non-finite values")
            status = "OK" if not problems else "PROBLEM: " + "; ".join(problems)
            print(f"{p} → {d.relative_to(ROOT)}  [{man.get('model_id', '?')}, dim {man.get('dim', '?')}]  {status}")


if __name__ == "__main__":
    main(sys.argv[1:])
