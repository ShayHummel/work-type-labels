"""Block 7 — Late fusion (stacking) of base models + cross-fitted per-class decision bias.

Stacking: the OOF class scores of the chosen base models are concatenated (log-probabilities for models with
probabilities, raw decision values for SVM/Ridge, all standardised) and a multinomial LogReg meta-model is
cross-fitted on the same 5 folds: for fold k it is trained on the OOF rows of the other folds and scores fold k.
For the holdout, the meta-model is trained on all OOF rows and applied to the base models' holdout scores.

Decision bias (PLAN §6): an additive bias b (10 values) on the meta log-probabilities, prediction =
argmax(log p + b), chosen by coordinate ascent to maximise min-F1, then macro-F1. Cross-fitted: b is learned on
4 folds' meta-OOF and applied to the 5th; the reported score is the cross-fitted one. The final b (all folds) is
saved for the holdout.

Run: uv run python src/06_stack.py
"""

import json

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler

from experiments import DATA, load_data, record

K = 10
BIAS_GRID = np.round(np.arange(-3, 3.01, 0.25), 2)


def base_features(exp_ids: list[str], split: str) -> np.ndarray:
    folder = "oof" if split == "oof" else "holdout_scores"
    feats = []
    for e in exp_ids:
        s = np.load(DATA / folder / f"{e}.npy").astype(np.float64)
        is_proba = np.allclose(np.load(DATA / "oof" / f"{e}.npy")[:100].sum(1), 1, atol=1e-3) and s.min() >= 0
        feats.append(np.log(np.clip(s, 1e-6, 1)) if is_proba else s)
    return np.hstack(feats)


def f1s(y, pred):
    return f1_score(y, pred, labels=range(K), average=None, zero_division=0)


def fit_bias(logp: np.ndarray, y: np.ndarray, rounds: int = 3) -> np.ndarray:
    b = np.zeros(K)

    def score(bb):
        f = f1s(y, (logp + bb).argmax(1))
        return f.min(), f.mean()

    best = score(b)
    for _ in range(rounds):
        for c in range(K):
            for v in BIAS_GRID:
                cand = b.copy()
                cand[c] = v
                sc = score(cand)
                if sc > best:
                    best, b = sc, cand
    return b


def stack(name: str, exp_ids: list[str], train, C: float = 1.0, cw=None) -> None:
    y, folds = train.y.values, train.fold.values
    X, Xh = base_features(exp_ids, "oof"), base_features(exp_ids, "holdout")
    meta_oof = np.zeros((len(y), K))
    for k in range(5):
        tr, va = folds != k, folds == k
        sc = StandardScaler().fit(X[tr])
        m = LogisticRegression(C=C, class_weight=cw, max_iter=3000).fit(sc.transform(X[tr]), y[tr])
        meta_oof[va] = m.predict_proba(sc.transform(X[va]))
    sc = StandardScaler().fit(X)
    m = LogisticRegression(C=C, class_weight=cw, max_iter=3000).fit(sc.transform(X), y)
    meta_hold = m.predict_proba(sc.transform(Xh))
    meta = {"block": 7, "features": "stack of " + ", ".join(exp_ids), "model": "meta-LR",
            "params": json.dumps({"C": C, "cw": cw})}
    np.save(DATA / "oof" / f"{name}.npy", meta_oof.astype(np.float32))
    np.save(DATA / "holdout_scores" / f"{name}.npy", meta_hold.astype(np.float32))
    record(name, train, meta_oof, meta)

    # cross-fitted decision bias
    logp, logh = np.log(np.clip(meta_oof, 1e-9, 1)), np.log(np.clip(meta_hold, 1e-9, 1))
    biased = np.zeros_like(logp)
    per_fold_b = []
    for k in range(5):
        tr, va = folds != k, folds == k
        b = fit_bias(logp[tr], y[tr])
        per_fold_b.append(b.tolist())
        biased[va] = logp[va] + b
    b_all = fit_bias(logp, y)
    np.save(DATA / "oof" / f"{name}+bias.npy", np.exp(biased - biased.max(1, keepdims=True)).astype(np.float32))
    np.save(DATA / "holdout_scores" / f"{name}+bias.npy",
            np.exp(logh + b_all - (logh + b_all).max(1, keepdims=True)).astype(np.float32))
    record(f"{name}+bias", train, biased, {**meta, "model": "meta-LR + cross-fitted class bias",
                                           "params": json.dumps({"C": C, "cw": cw, "bias_all": b_all.tolist(),
                                                                 "bias_per_fold": per_fold_b})})


if __name__ == "__main__":
    import sys
    train, _ = load_data()
    sets = json.loads(open(sys.argv[1]).read()) if len(sys.argv) > 1 else {
        "stack-A": ["tfidf-WC_lr", "emb-qwen3-0.6b-instr_lr"],
    }
    for name, ids in sets.items():
        stack(name, ids, train)
