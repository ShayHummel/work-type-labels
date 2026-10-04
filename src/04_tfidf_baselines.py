"""Block 1 — TF-IDF baselines: {words 1–3, chars 3–5, words+chars} × {Ridge, LogReg, LinearSVC}.

Vectorizers are fitted inside each training split (no IDF from scored rows). Hyper-parameters are chosen on
fold 0 (train folds 1–4 → score fold 0) by min-F1 then macro-F1, then all 5 folds are run with that choice.
Run: uv run python src/04_tfidf_baselines.py
"""

import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.svm import LinearSVC

from experiments import load_data, run_cv, select_on_fold0

FEATURES = {
    "W": [dict(analyzer="word", ngram_range=(1, 3), min_df=2, max_df=0.9, sublinear_tf=True,
               token_pattern=r"(?u)\b\w+\b", max_features=500_000, dtype="float32")],
    "C": [dict(analyzer="char_wb", ngram_range=(3, 5), min_df=3, sublinear_tf=True, max_features=500_000,
               dtype="float32")],
}
FEATURES["WC"] = FEATURES["W"] + FEATURES["C"]

MODELS = {
    "ridge": (lambda p: RidgeClassifier(alpha=p["alpha"], class_weight=p["cw"], solver="sparse_cg"),
              [{"alpha": a, "cw": cw} for a in (0.3, 1.0, 3.0) for cw in (None, "balanced")]),
    "lr": (lambda p: LogisticRegression(C=p["C"], class_weight=p["cw"], max_iter=2000),
           [{"C": c, "cw": cw} for c in (3.0, 10.0, 30.0) for cw in (None, "balanced")]),
    "svm": (lambda p: LinearSVC(C=p["C"], class_weight=p["cw"]),
            [{"C": c, "cw": cw} for c in (0.1, 0.3, 1.0) for cw in (None, "balanced")]),
}


def make_fit_predict(feat_key, model_key):
    def fit_predict(df_fit, df_score, params):
        vecs = [TfidfVectorizer(**cfg) for cfg in FEATURES[feat_key]]
        Xf = sp.hstack([v.fit_transform(df_fit.text) for v in vecs]).tocsr()
        Xs = sp.hstack([v.transform(df_score.text) for v in vecs]).tocsr()
        clf = MODELS[model_key][0](params).fit(Xf, df_fit.y.values)
        return clf.predict_proba(Xs) if hasattr(clf, "predict_proba") else clf.decision_function(Xs)
    return fit_predict


if __name__ == "__main__":
    train, hold = load_data()
    for feat in ("W", "C", "WC"):
        for model in ("ridge", "lr", "svm"):
            exp = f"tfidf-{feat}_{model}"
            fp = make_fit_predict(feat, model)
            best, grid = select_on_fold0(train, fp, MODELS[model][1])
            print(exp, "fold-0 grid:\n", grid.round(3).to_string(index=False), flush=True)
            run_cv(exp, train, hold, fp, best, {"block": 1, "features": f"tfidf-{feat}", "model": model,
                                                 "selection": "fold-0 grid, min-F1 then macro-F1"})
