"""Block 7 — Late fusion (stacking) of base models + cross-fitted per-class decision bias.

Stacking: the OOF class scores of the chosen base models are concatenated (log-probabilities for models with
probabilities, raw decision values for SVM/Ridge, all standardised) and a multinomial LogReg meta-model is
cross-fitted on the same 5 folds: for fold k it is trained on the OOF rows of the other folds and scores fold k.
For the holdout, the meta-model is trained on all OOF rows and applied to the base models' holdout scores.

Decision bias (PLAN §6): an additive bias b (10 values) on the meta log-probabilities, prediction =
argmax(log p + b), chosen by coordinate ascent to maximise min-F1, then macro-F1. Cross-fitted: b is learned on
4 folds' meta-OOF and applied to the 5th; the reported score is the cross-fitted one. The final b (all folds) is
saved for the holdout.

Meta-model: multinomial LogReg (default) or XGBoost (--xgb; depth 3, eta 0.05, early stopping on an inner 10%
split of each training part; the holdout model uses the mean of the per-fold best iteration counts).

Run: uv run python src/06_stack.py [sets.json] [--xgb]
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


XGB_PARAMS = dict(objective="multi:softprob", num_class=K, max_depth=3, learning_rate=0.05, subsample=0.8,
                  colsample_bytree=0.8, min_child_weight=5, reg_lambda=1.0, tree_method="hist", n_jobs=8,
                  eval_metric="mlogloss")


def fit_xgb(X, y, seed=0):
    """XGBoost meta-model with early stopping on an inner stratified 10% split of the given training rows."""
    import xgboost as xgb
    from sklearn.model_selection import train_test_split
    Xa, Xb, ya, yb = train_test_split(X, y, test_size=0.1, stratify=y, random_state=seed)
    m = xgb.XGBClassifier(n_estimators=3000, early_stopping_rounds=100, random_state=seed, **XGB_PARAMS)
    m.fit(Xa, ya, eval_set=[(Xb, yb)], verbose=False)
    return m


def stack(name: str, exp_ids: list[str], train, C: float = 1.0, cw=None, meta_model: str = "lr") -> None:
    y, folds = train.y.values, train.fold.values
    X, Xh = base_features(exp_ids, "oof"), base_features(exp_ids, "holdout")
    meta_oof = np.zeros((len(y), K))
    best_iters = []
    for k in range(5):
        tr, va = folds != k, folds == k
        if meta_model == "xgb":
            m = fit_xgb(X[tr], y[tr], seed=k)
            best_iters.append(int(m.best_iteration))
            meta_oof[va] = m.predict_proba(X[va])
        else:
            sc = StandardScaler().fit(X[tr])
            m = LogisticRegression(C=C, class_weight=cw, max_iter=3000).fit(sc.transform(X[tr]), y[tr])
            meta_oof[va] = m.predict_proba(sc.transform(X[va]))
    if meta_model == "xgb":
        import xgboost as xgb
        n_est = int(np.mean(best_iters)) + 1  # all OOF rows, no early-stopping split: mean of the fold optima
        m = xgb.XGBClassifier(n_estimators=n_est, random_state=0, **XGB_PARAMS).fit(X, y)
        meta_hold = m.predict_proba(Xh)
        params = {**{k: v for k, v in XGB_PARAMS.items() if k not in ("objective", "num_class")},
                  "best_iterations_per_fold": best_iters, "n_estimators_final": n_est}
    else:
        sc = StandardScaler().fit(X)
        m = LogisticRegression(C=C, class_weight=cw, max_iter=3000).fit(sc.transform(X), y)
        meta_hold = m.predict_proba(sc.transform(Xh))
        params = {"C": C, "cw": cw}
    meta = {"block": 7, "features": "stack of " + ", ".join(exp_ids),
            "model": "meta-XGBoost" if meta_model == "xgb" else "meta-LR", "params": json.dumps(params)}
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
    record(f"{name}+bias", train, biased, {**meta, "model": meta["model"] + " + cross-fitted class bias",
                                           "params": json.dumps({**json.loads(meta["params"]),
                                                                 "bias_all": b_all.tolist(),
                                                                 "bias_per_fold": per_fold_b})})


if __name__ == "__main__":
    import sys
    train, _ = load_data()
    files = [a for a in sys.argv[1:] if not a.startswith("--")]
    sets = json.loads(open(files[0]).read()) if files else {
        "stack-A": ["tfidf-WC_lr", "emb-qwen3-0.6b-instr_lr"],
    }
    meta_model = "xgb" if "--xgb" in sys.argv else "lr"
    for name, ids in sets.items():
        stack(name + ("-xgb" if meta_model == "xgb" else ""), ids, train, meta_model=meta_model)
