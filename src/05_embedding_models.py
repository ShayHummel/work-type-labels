"""Block 2 — Fixed embeddings × {Ridge, LogReg, LinearSVC, kNN}, and early fusion with TF-IDF words+chars.

Embeddings come from data/emb/<key>/ (src/embed.py, locally or imported from Colab). Same protocol as block 1:
hyper-parameters chosen on fold 0 (min-F1, then macro-F1), then 5-fold OOF + a full-train fit for the holdout.

Run: uv run python src/05_embedding_models.py [emb keys …] [--no-fusion]   (default: every key in data/emb/)
"""

import sys

import numpy as np
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC

from experiments import DATA, load_data, run_cv, select_on_fold0

TFIDF_WC = [dict(analyzer="word", ngram_range=(1, 3), min_df=2, max_df=0.9, sublinear_tf=True,
                 token_pattern=r"(?u)\b\w+\b", max_features=500_000, dtype="float32"),
            dict(analyzer="char_wb", ngram_range=(3, 5), min_df=3, sublinear_tf=True, max_features=500_000,
                 dtype="float32")]

MODELS = {
    "ridge": (lambda p: RidgeClassifier(alpha=p["alpha"], class_weight=p["cw"]),
              [{"alpha": a, "cw": cw} for a in (0.1, 1.0, 10.0) for cw in (None, "balanced")]),
    "lr": (lambda p: LogisticRegression(C=p["C"], class_weight=p["cw"], max_iter=3000),
           [{"C": c, "cw": cw} for c in (1.0, 10.0, 100.0) for cw in (None, "balanced")]),
    "svm": (lambda p: LinearSVC(C=p["C"], class_weight=p["cw"]),
            [{"C": c, "cw": cw} for c in (0.1, 1.0, 10.0) for cw in (None, "balanced")]),
    "knn": (lambda p: KNeighborsClassifier(n_neighbors=p["k"], weights="distance", metric="cosine"),
            [{"k": k} for k in (5, 15, 50)]),
}
FUSION_GRID = [{"C": c, "alpha_emb": a, "cw": cw} for c in (3.0, 10.0) for a in (1.0, 3.0) for cw in (None, "balanced")]


def scores(clf, X):
    return clf.predict_proba(X) if hasattr(clf, "predict_proba") else clf.decision_function(X)


def emb_fit_predict(E_train, E_hold, model_key):
    def fp(df_fit, df_score, params):
        Xf = E_train[df_fit.index.values]
        Xs = E_hold if df_score.attrs.get("holdout") else E_train[df_score.index.values]
        return scores(MODELS[model_key][0](params).fit(Xf, df_fit.y.values), Xs)
    return fp


def fusion_fit_predict(E_train, E_hold):
    def fp(df_fit, df_score, params):
        vecs = [TfidfVectorizer(**cfg) for cfg in TFIDF_WC]
        Ef = E_train[df_fit.index.values]
        Es = E_hold if df_score.attrs.get("holdout") else E_train[df_score.index.values]
        Xf = sp.hstack([v.fit_transform(df_fit.text) for v in vecs] + [sp.csr_matrix(Ef * params["alpha_emb"])])
        Xs = sp.hstack([v.transform(df_score.text) for v in vecs] + [sp.csr_matrix(Es * params["alpha_emb"])])
        clf = LogisticRegression(C=params["C"], class_weight=params["cw"], max_iter=3000)
        return clf.fit(Xf.tocsr(), df_fit.y.values).predict_proba(Xs.tocsr())
    return fp


if __name__ == "__main__":
    train, hold = load_data()
    hold.attrs["holdout"] = True
    no_fusion = "--no-fusion" in sys.argv  # early fusion is slow (block 2: up to 90 min); late fusion in block 7
    keys = [a for a in sys.argv[1:] if not a.startswith("--")] or sorted(p.name for p in (DATA / "emb").iterdir() if (p / "manifest.json").exists())
    for key in keys:
        E_train = np.load(DATA / "emb" / key / "train.npy").astype(np.float32)
        E_hold = np.load(DATA / "emb" / key / "holdout.npy").astype(np.float32)
        # each worker holds a copy of the matrix: 4096-d × 24.8k rows ≈ 0.4 GB → fewer workers for big models
        n_jobs = 6 if E_train.shape[1] <= 1024 else 2
        for model in ("lr", "svm", "ridge", "knn"):
            exp = f"emb-{key}_{model}"
            if (DATA / "oof" / f"{exp}.npy").exists() and "--redo" not in sys.argv:
                print(exp, "already done — skipping", flush=True)
                continue
            fp = emb_fit_predict(E_train, E_hold, model)
            # Ridge/kNN on >1024-d float32 segfault inside loky workers (fine in-process); they take seconds anyway
            jobs = 1 if (model in ("ridge", "knn") and E_train.shape[1] > 1024) else n_jobs
            best, grid = select_on_fold0(train, fp, MODELS[model][1], n_jobs=jobs)
            print(exp, "fold-0 grid:\n", grid.round(3).to_string(index=False), flush=True)
            run_cv(exp, train, hold, fp, best, {"block": 2 if E_train.shape[1] <= 1024 else 5, "features": f"emb-{key}", "model": model,
                                                 "selection": "fold-0 grid, min-F1 then macro-F1"},
                   n_jobs=min(jobs, 5))
        if no_fusion:
            continue
        exp = f"fusion-tfidfWC+{key}_lr"
        fp = fusion_fit_predict(E_train, E_hold)
        best, grid = select_on_fold0(train, fp, FUSION_GRID, n_jobs=4)
        print(exp, "fold-0 grid:\n", grid.round(3).to_string(index=False), flush=True)
        run_cv(exp, train, hold, fp, best, {"block": 2, "features": f"tfidf-WC + emb-{key} (early fusion)",
                                             "model": "lr", "selection": "fold-0 grid, min-F1 then macro-F1"})
