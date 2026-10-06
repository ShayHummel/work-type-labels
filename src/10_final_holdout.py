"""Final holdout scores for stack-H, computed the way the meta-model was trained.

The meta-model (LogReg on standardised base scores) is trained on OUT-OF-FOLD scores, i.e. on scores of models that
each saw 4 of the 5 folds. For the holdout, feeding it the *average* of the 5 fine-tuned fold models (smoother
probabilities) and a *full-train* refit of the embedding LR (sharper probabilities) shifts its inputs: holdout
agreement with the fine-tune drops from 0.939 (OOF) to 0.922. Here, for each fold k, the holdout is scored by the
fine-tuned fold-k model and by an embedding LR trained on the same 4 folds; the meta-model is applied to that pair,
and the 5 meta outputs are averaged. Same inputs as in training, then a 5-model ensemble.

Writes data/holdout_scores/stack-H-perfold.npy. Run: uv run python src/10_final_holdout.py
"""

import json

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from experiments import DATA, load_data

FT, EMB = "ft-qwen3-8b-lora", "emb-qwen3-8b-instr_lr"
train, hold = load_data()
y, folds = train.y.values, train.fold.values


def feats(ft_proba, emb_proba):
    """Same transform as src/06_stack.py, including float64: the meta-LR's solver stops at a tolerance, so float32
    inputs give a slightly different solution (12 OOF labels differ) than the meta-model that was evaluated."""
    return np.hstack([np.log(np.clip(np.asarray(ft_proba, np.float64), 1e-6, 1)),
                      np.log(np.clip(np.asarray(emb_proba, np.float64), 1e-6, 1))])


# meta-model on all OOF rows (identical to the one behind stack-H's holdout scores)
X = feats(np.load(DATA / "oof" / f"{FT}.npy"), np.load(DATA / "oof" / f"{EMB}.npy"))
sc = StandardScaler().fit(X)
meta = LogisticRegression(C=1.0, max_iter=3000).fit(sc.transform(X), y)

emb_params = json.loads(__import__("pandas").read_csv(DATA.parent / "reports" / "results.csv")
                        .set_index("exp_id").loc[EMB, "params"])
E_tr = np.load(DATA / "emb" / "qwen3-8b-instr" / "train.npy").astype(np.float32)
E_ho = np.load(DATA / "emb" / "qwen3-8b-instr" / "holdout.npy").astype(np.float32)
outs = []
for k in range(5):
    ft_k = np.load(DATA / "ft" / "qwen3-8b-lora" / f"fold{k}" / "holdout_proba.npy")
    lr_k = LogisticRegression(C=emb_params["C"], class_weight=emb_params["cw"], max_iter=3000).fit(
        E_tr[folds != k], y[folds != k])
    outs.append(meta.predict_proba(sc.transform(feats(ft_k, lr_k.predict_proba(E_ho)))))
final = np.mean(outs, axis=0).astype(np.float32)
np.save(DATA / "holdout_scores" / "stack-H-perfold.npy", final)
np.save(DATA / "oof" / "stack-H-perfold.npy", np.load(DATA / "oof" / "stack-H.npy"))  # same OOF estimate

old = np.load(DATA / "holdout_scores" / "stack-H.npy").argmax(1)
new = final.argmax(1)
ft_avg = np.load(DATA / "holdout_scores" / f"{FT}.npy").argmax(1)
print(f"holdout labels changed vs stack-H (averaged inputs): {(old != new).sum()} of {len(new)} "
      f"({(old != new).mean():.1%})")
print(f"agreement with the fine-tune: per-fold {(new == ft_avg).mean():.3f}, averaged-inputs {(old == ft_avg).mean():.3f}, "
      f"OOF reference 0.939")
print("fold-to-fold agreement of meta outputs:",
      np.round([(outs[i].argmax(1) == outs[j].argmax(1)).mean() for i in range(5) for j in range(i + 1, 5)], 3))
