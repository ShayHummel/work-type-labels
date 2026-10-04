"""Collect fine-tuning folds (data/ft/<key>/fold0..4) into the standard experiment format.

OOF = each fold's validation probabilities placed at its rows; holdout = mean of the 5 fold models' holdout
probabilities. Then evaluated and registered like every other experiment.
Run: uv run python src/08_collect_ft.py <key> [<key> …]
"""

import json
import sys

import numpy as np

from experiments import DATA, load_data, record

train, _ = load_data()
for key in sys.argv[1:]:
    oof = np.full((len(train), 10), np.nan, dtype=np.float32)
    hold, dones = [], []
    for k in range(5):
        d = DATA / "ft" / key / f"fold{k}"
        done = json.loads((d / "done.json").read_text())
        assert done.get("predict_order"), f"{d}: prediction order not verified — run src/fix_ft_order.py"
        vi = np.load(d / "val_idx.npy")
        assert np.array_equal(vi, np.flatnonzero(train.fold.values == k))
        oof[vi] = np.load(d / "val_proba.npy")
        hold.append(np.load(d / "holdout_proba.npy"))
        dones.append(done)
    assert not np.isnan(oof).any()
    exp = f"ft-{key}"
    np.save(DATA / "oof" / f"{exp}.npy", oof)
    np.save(DATA / "holdout_scores" / f"{exp}.npy", np.mean(hold, axis=0).astype(np.float32))
    d0 = dones[0]
    record(exp, train, oof, {"block": 6, "features": "fine-tuned (Colab)", "model": f"{d0['hf']} LoRA={d0['lora']}",
                             "params": json.dumps({k: d0[k] for k in ("max_len", "epochs", "lr", "batch",
                                                                      "class_weight", "truncation")}),
                             "cv_seconds": round(sum(d["total_seconds"] for d in dones), 1),
                             "selection": "fixed hyper-parameters (no tuning on folds)"})
