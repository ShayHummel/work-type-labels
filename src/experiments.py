"""Shared experiment runner: 5-fold out-of-fold (OOF) scores, holdout scores, evaluation, results registry.

Every experiment saves
    data/oof/<exp_id>.npy            (24792, 10) class scores, train.jsonl order (OOF)
    data/holdout_scores/<exp_id>.npy (6199, 10)  scores from a model fit on all of train
and appends one row to reports/results.csv (metrics + runtime), plus a per-experiment markdown section.
"""

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.metrics import f1_score

from evaluation import LABELS, evaluate, plot_confusion, plot_curves, to_ids, to_markdown

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
REPORTS = ROOT / "reports"
for d in (DATA / "oof", DATA / "holdout_scores", REPORTS / "figures" / "exp"):
    d.mkdir(parents=True, exist_ok=True)


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    train = pd.read_json(ROOT / "train.jsonl", lines=True)
    hold = pd.read_json(ROOT / "holdout.jsonl", lines=True)
    folds = pd.read_csv(DATA / "folds.csv")
    assert (folds.id.values == train.id.values).all()
    train["fold"] = folds.fold.values
    train["y"] = to_ids(train.label)
    return train, hold


def select_on_fold0(train: pd.DataFrame, fit_predict, grid: list[dict], n_jobs: int = 6) -> tuple[dict, pd.DataFrame]:
    """Pick hyper-parameters on fold 0 only (train on folds 1–4, score fold 0): min-F1, then macro-F1."""
    tr, va = train.fold.values != 0, train.fold.values == 0

    def one(params):
        s = fit_predict(train[tr], train[va], params)
        f1 = f1_score(train.y.values[va], s.argmax(1), labels=range(len(LABELS)), average=None, zero_division=0)
        return {**params, "min_F1": f1.min(), "macro_F1": f1.mean()}

    res = pd.DataFrame(Parallel(n_jobs=n_jobs)(delayed(one)(p) for p in grid))
    best = res.sort_values(["min_F1", "macro_F1"], ascending=False).iloc[0]
    return {k: best[k] for k in grid[0]}, res


def run_cv(exp_id: str, train: pd.DataFrame, hold: pd.DataFrame, fit_predict, params: dict,
           meta: dict, n_jobs: int = 5) -> dict:
    """fit_predict(df_fit, df_score, params) -> (n_score, 10) scores."""
    t0 = time.time()
    parts = Parallel(n_jobs=n_jobs)(
        delayed(fit_predict)(train[train.fold != k], train[train.fold == k], params) for k in range(5))
    oof = np.zeros((len(train), len(LABELS)), dtype=np.float32)
    for k, s in enumerate(parts):
        oof[train.fold.values == k] = s
    t_cv = time.time() - t0
    t1 = time.time()
    hold_scores = fit_predict(train, hold, params).astype(np.float32)
    t_full = time.time() - t1
    np.save(DATA / "oof" / f"{exp_id}.npy", oof)
    np.save(DATA / "holdout_scores" / f"{exp_id}.npy", hold_scores)
    return record(exp_id, train, oof, {**meta, "params": json.dumps(params, default=str),
                                       "cv_seconds": round(t_cv, 1), "full_fit_seconds": round(t_full, 1)})


def record(exp_id: str, train: pd.DataFrame, oof: np.ndarray, meta: dict) -> dict:
    res = evaluate(train.y.values, oof.argmax(1), oof)
    plot_confusion(res["cm"], REPORTS / "figures" / "exp" / f"{exp_id}_cm.png", exp_id)
    plot_curves(train.y.values, oof, REPORTS / "figures" / "exp" / f"{exp_id}_curves.png", exp_id)
    per, ov = res["per_class"], res["overall"]["value"]
    row = {"exp_id": exp_id, **meta, "accuracy": ov["accuracy"], "macro_F1": ov["macro-F1"],
           "min_F1": ov["min-F1 (bar)"], "n_ge_0.8": int(ov["classes with F1 ≥ 0.8"]),
           "macro_ROC_AUC": ov.get("macro ROC-AUC"), "macro_PR_AUC": ov.get("macro PR-AUC"),
           **{f"F1 {lab}": per.loc[lab, "F1"] for lab in LABELS},
           "time": time.strftime("%Y-%m-%d %H:%M")}
    path = REPORTS / "results.csv"
    df = pd.read_csv(path) if path.exists() else pd.DataFrame()
    df = pd.concat([df[df.get("exp_id", pd.Series(dtype=str)) != exp_id], pd.DataFrame([row])], ignore_index=True)
    df.to_csv(path, index=False)
    md = to_markdown(res, exp_id, f"figures/exp/{exp_id}_cm.png", f"figures/exp/{exp_id}_curves.png")
    (REPORTS / "exp").mkdir(exist_ok=True)
    (REPORTS / "exp" / f"{exp_id}.md").write_text(f"# {exp_id}\n\n```\n{json.dumps(meta, indent=1, default=str)}\n```\n\n"
                                                  + md.replace("figures/", "../figures/"))
    print(f"{exp_id}: macro-F1 {row['macro_F1']:.3f}  min-F1 {row['min_F1']:.3f}  acc {row['accuracy']:.3f}  "
          f"({meta.get('cv_seconds', '?')}s CV)", flush=True)
    return row
