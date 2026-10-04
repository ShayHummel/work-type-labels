"""Evaluation protocol shared by every model.

Inputs are out-of-fold (OOF) predictions over train: y_true, y_pred and, when the model has scores,
an (n, 10) probability matrix in LABELS order.

Reported per class: precision, recall, F1, support, ROC-AUC, PR-AUC, with 95% bootstrap CIs; overall:
accuracy, macro-F1, min-F1 (the TASK.md bar is min-F1 ≥ 0.8); and the confusion matrix.

Confidence intervals are computed at **holdout scale**: each bootstrap replicate has 6,199 rows (the size
of holdout.jsonl) with the *train* class proportions (e.g. 39 Optimize rows), drawn per class from the OOF
rows of that class, with replacement. The interval therefore says how much a score on a holdout-sized
sample can move by sampling alone. Only train.jsonl is used; the support column printed in TASK.md is not
an input (we cannot verify what it counts).

p-values:
- `p(F1 ≤ 0.8)`: one-sided bootstrap test of H0 "class F1 is at or below the bar"; small = bar met.
- `p(AUC = 0.5)`: one-sided Mann–Whitney U test of H0 "scores do not rank the class above the rest".
- `p(macro-F1 = chance)`: permutation test (labels shuffled).
- `compare()`: paired bootstrap p-value per class for "model B's F1 is not higher than model A's", and
  McNemar's exact test on per-row correctness.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import binomtest, mannwhitneyu
from sklearn.metrics import average_precision_score, confusion_matrix, roc_auc_score

LABELS = ["Bug fix", "Feature dev", "Refactoring", "Architecting", "Researching",
          "Testing", "Review", "Optimize", "Setup", "Other"]
K = len(LABELS)
LABEL2ID = {lab: i for i, lab in enumerate(LABELS)}
HOLDOUT_ROWS = 6199  # len(holdout.jsonl)
BAR = 0.8


def to_ids(y) -> np.ndarray:
    y = np.asarray(y)
    return np.array([LABEL2ID[v] for v in y]) if y.dtype.kind in "OUS" else y.astype(int)


def _prf_from_cm(cm: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """cm[..., true, pred] → precision, recall, F1 per class (last axis)."""
    tp = np.diagonal(cm, axis1=-2, axis2=-1)
    pred_n = cm.sum(axis=-2)
    true_n = cm.sum(axis=-1)
    with np.errstate(divide="ignore", invalid="ignore"):
        p = np.where(pred_n > 0, tp / pred_n, 0.0)
        r = np.where(true_n > 0, tp / true_n, 0.0)
        f = np.where(p + r > 0, 2 * p * r / (p + r), 0.0)
    return p, r, f


def boot_class_counts(y: np.ndarray) -> np.ndarray:
    """Rows per class in a holdout-sized replicate: train proportions × HOLDOUT_ROWS (largest remainder)."""
    exact = np.bincount(y, minlength=K) / len(y) * HOLDOUT_ROWS
    counts = np.floor(exact).astype(int)
    counts[np.argsort(-(exact - counts))[:HOLDOUT_ROWS - counts.sum()]] += 1
    return counts


def _stratified_boot_idx(y: np.ndarray, rng: np.random.Generator, n_boot: int) -> np.ndarray:
    """(n_boot, HOLDOUT_ROWS) row indices, class counts fixed by boot_class_counts."""
    parts = []
    for c, n_c in enumerate(boot_class_counts(y)):
        rows = np.flatnonzero(y == c)
        parts.append(rows[rng.integers(0, len(rows), size=(n_boot, n_c))])
    return np.concatenate(parts, axis=1)


def evaluate(y_true, y_pred, proba: np.ndarray | None = None, n_boot: int = 1000,
             n_boot_auc: int = 200, n_perm: int = 200, seed: int = 0) -> dict:
    y, p = to_ids(y_true), to_ids(y_pred)
    rng = np.random.default_rng(seed)
    cm = confusion_matrix(y, p, labels=range(K))
    prec, rec, f1 = _prf_from_cm(cm)

    idx = _stratified_boot_idx(y, rng, n_boot)
    codes = y[idx] * K + p[idx]
    cms = np.stack([np.bincount(row, minlength=K * K).reshape(K, K) for row in codes])
    bp, br, bf = _prf_from_cm(cms)
    b_acc = np.trace(cms, axis1=1, axis2=2) / cms.sum(axis=(1, 2))
    lo, hi = 2.5, 97.5

    per = pd.DataFrame({
        "support (OOF)": cm.sum(axis=1),
        "precision": prec, "P lo": np.percentile(bp, lo, 0), "P hi": np.percentile(bp, hi, 0),
        "recall": rec, "R lo": np.percentile(br, lo, 0), "R hi": np.percentile(br, hi, 0),
        "F1": f1, "F1 lo": np.percentile(bf, lo, 0), "F1 hi": np.percentile(bf, hi, 0),
        "p(F1≤0.8)": (bf <= BAR).mean(axis=0),
    }, index=LABELS)

    if proba is not None:
        roc, pr, roc_ci, pr_ci, p_auc = [], [], [], [], []
        auc_idx = idx[:n_boot_auc]
        for c in range(K):
            yc = (y == c).astype(int)
            s = proba[:, c]
            roc.append(roc_auc_score(yc, s))
            pr.append(average_precision_score(yc, s))
            p_auc.append(mannwhitneyu(s[yc == 1], s[yc == 0], alternative="greater").pvalue)
            br_roc = [roc_auc_score(yc[r], s[r]) for r in auc_idx]
            br_pr = [average_precision_score(yc[r], s[r]) for r in auc_idx]
            roc_ci.append(np.percentile(br_roc, [lo, hi]))
            pr_ci.append(np.percentile(br_pr, [lo, hi]))
        per["ROC-AUC"] = roc
        per["ROC lo"], per["ROC hi"] = np.array(roc_ci).T
        per["p(AUC=0.5)"] = p_auc
        per["PR-AUC"] = pr
        per["PR lo"], per["PR hi"] = np.array(pr_ci).T
        per["PR chance"] = np.bincount(y, minlength=K) / len(y)

    macro = f1.mean()
    perm_macro = []
    for _ in range(n_perm):
        yp = rng.permutation(y)
        perm_macro.append(_prf_from_cm(confusion_matrix(yp, p, labels=range(K)))[2].mean())
    overall = pd.DataFrame({
        "value": [np.trace(cm) / cm.sum(), macro, f1.min(), int((f1 >= BAR).sum())],
        "lo": [np.percentile(b_acc, lo), np.percentile(bf.mean(1), lo), np.percentile(bf.min(1), lo), np.nan],
        "hi": [np.percentile(b_acc, hi), np.percentile(bf.mean(1), hi), np.percentile(bf.min(1), hi), np.nan],
        "p-value": [np.nan, (1 + np.sum(np.array(perm_macro) >= macro)) / (n_perm + 1),
                    (bf.min(1) <= BAR).mean(), np.nan],
    }, index=["accuracy", "macro-F1", "min-F1 (bar)", "classes with F1 ≥ 0.8"])
    return {"per_class": per, "overall": overall, "cm": cm, "boot_f1": bf}


def compare(y_true, pred_a, pred_b, n_boot: int = 1000, seed: int = 0) -> pd.DataFrame:
    """Is model B better than model A? Paired, on the same OOF rows."""
    y, a, b = to_ids(y_true), to_ids(pred_a), to_ids(pred_b)
    rng = np.random.default_rng(seed)
    idx = _stratified_boot_idx(y, rng, n_boot)

    def boot_f1(p):
        codes = y[idx] * K + p[idx]
        return _prf_from_cm(np.stack([np.bincount(r, minlength=K * K).reshape(K, K) for r in codes]))[2]

    fa, fb = boot_f1(a), boot_f1(b)
    d = fb - fa
    res = pd.DataFrame({"F1 A": _prf_from_cm(confusion_matrix(y, a, labels=range(K)))[2],
                        "F1 B": _prf_from_cm(confusion_matrix(y, b, labels=range(K)))[2],
                        "Δ lo": np.percentile(d, 2.5, 0), "Δ hi": np.percentile(d, 97.5, 0),
                        "p(B≤A)": (d <= 0).mean(axis=0)}, index=LABELS)
    res.loc["macro-F1"] = [res["F1 A"].mean(), res["F1 B"].mean(), *np.percentile(d.mean(1), [2.5, 97.5]),
                           (d.mean(1) <= 0).mean()]
    ca, cb = a == y, b == y
    n01, n10 = int((~ca & cb).sum()), int((ca & ~cb).sum())
    res.attrs["mcnemar"] = (n01, n10, binomtest(n01, n01 + n10, 0.5, alternative="greater").pvalue
                            if n01 + n10 else 1.0)
    return res


def plot_confusion(cm: np.ndarray, path: Path, title: str) -> None:
    norm = cm / cm.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    for i in range(K):
        for j in range(K):
            if cm[i, j]:
                ax.text(j, i, f"{norm[i, j]:.2f}\n{cm[i, j]}", ha="center", va="center", fontsize=6.5,
                        color="white" if norm[i, j] > .5 else "black")
    ax.set_xticks(range(K), LABELS, rotation=40, ha="right")
    ax.set_yticks(range(K), LABELS)
    ax.set_xlabel("predicted")
    ax.set_ylabel("teacher label")
    ax.set_title(f"{title}\n(row-normalised recall; count below)")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def to_markdown(res: dict, name: str, fig_rel: str | None = None) -> str:
    per = res["per_class"]
    cols = ["support (OOF)", "precision", "recall", "F1"]
    show = per[cols].copy()
    show["P 95% CI"] = per.apply(lambda r: f"{r['P lo']:.3f}–{r['P hi']:.3f}", axis=1)
    show["R 95% CI"] = per.apply(lambda r: f"{r['R lo']:.3f}–{r['R hi']:.3f}", axis=1)
    show["F1 95% CI"] = per.apply(lambda r: f"{r['F1 lo']:.3f}–{r['F1 hi']:.3f}", axis=1)
    show["p(F1≤0.8)"] = per["p(F1≤0.8)"]
    if "ROC-AUC" in per:
        show["ROC-AUC (CI)"] = per.apply(lambda r: f"{r['ROC-AUC']:.3f} ({r['ROC lo']:.3f}–{r['ROC hi']:.3f})", 1)
        show["p(AUC=0.5)"] = per["p(AUC=0.5)"].map(lambda v: f"{v:.1e}")
        show["PR-AUC (CI)"] = per.apply(lambda r: f"{r['PR-AUC']:.3f} ({r['PR lo']:.3f}–{r['PR hi']:.3f})", 1)
        show["PR chance"] = per["PR chance"]
    lines = [f"### {name}", "", show.to_markdown(floatfmt=".3f"), "", res["overall"].to_markdown(floatfmt=".3f"), ""]
    if fig_rel:
        lines += [f"![confusion]({fig_rel})", ""]
    return "\n".join(lines)
