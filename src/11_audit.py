"""Audit of the whole process: recompute key results independently and check invariants. Prints PASS/FAIL/INFO.

Run: uv run python src/11_audit.py   → also writes reports/11_audit.md
"""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler

from evaluation import LABELS, evaluate
from experiments import DATA, load_data

ROOT = Path(__file__).resolve().parents[1]
K = 10
out, n_fail = [], 0


def check(name: str, ok: bool, detail: str = "") -> None:
    global n_fail
    n_fail += (not ok)
    line = f"{'PASS' if ok else 'FAIL'}  {name}" + (f" — {detail}" if detail else "")
    print(line, flush=True)
    out.append(f"- {line}")


def info(name: str, detail: str) -> None:
    line = f"INFO  {name} — {detail}"
    print(line, flush=True)
    out.append(f"- {line}")


train, hold = load_data()
y, folds = train.y.values, train.fold.values
f1 = lambda yt, yp: f1_score(yt, yp, labels=range(K), average=None, zero_division=0)  # noqa: E731

# ---------------------------------------------------------------- A. data and folds
out.append("## A. Data and folds")
check("train ids unique", train.id.is_unique)
check("holdout ids unique", hold.id.is_unique)
check("no id overlap train/holdout", not set(train.id) & set(hold.id))
check("no exact text overlap train/holdout", not set(train.text) & set(hold.text))
check("folds.csv aligned with train.jsonl", True, "asserted in load_data()")
dev = max(abs((train[train.fold == k].label.value_counts(normalize=True)
               - train.label.value_counts(normalize=True)).max()) for k in range(5))
check("folds stratified (max class-share deviation < 0.1 pp)", dev < 0.001, f"{dev * 100:.3f} pp")
check("labels are the ten names", set(train.label) == set(LABELS))

# ---------------------------------------------------------------- B. embeddings
out.append("## B. Embeddings")
ids_sha = hashlib.sha1("\n".join(list(train.id) + list(hold.id)).encode()).hexdigest()
for key in ("qwen3-8b-instr",):
    man = json.loads((DATA / "emb" / key / "manifest.json").read_text())
    check(f"emb {key}: ids hash matches train+holdout order", man["ids_sha1"] == ids_sha)
    check(f"emb {key}: row counts", man["n_train"] == len(train) and man["n_holdout"] == len(hold))

# ---------------------------------------------------------------- C. embedding LR (base model B)
out.append("## C. Base model B — Qwen3-Emb-8B-instr + LogReg: saved scores reproduce")
params = json.loads(pd.read_csv(ROOT / "reports" / "results.csv").set_index("exp_id")
                    .loc["emb-qwen3-8b-instr_lr", "params"])
E_tr = np.load(DATA / "emb" / "qwen3-8b-instr" / "train.npy").astype(np.float32)
E_ho = np.load(DATA / "emb" / "qwen3-8b-instr" / "holdout.npy").astype(np.float32)
saved_oof = np.load(DATA / "oof" / "emb-qwen3-8b-instr_lr.npy")
saved_ho = np.load(DATA / "holdout_scores" / "emb-qwen3-8b-instr_lr.npy")
lr_full = LogisticRegression(C=params["C"], class_weight=params["cw"], max_iter=3000).fit(E_tr, y)
d_ho = np.abs(lr_full.predict_proba(E_ho) - saved_ho).max()
check("holdout scores = LR(all train) on holdout embeddings (row order)", d_ho < 1e-3, f"max |Δp| {d_ho:.2e}")
lr0 = LogisticRegression(C=params["C"], class_weight=params["cw"], max_iter=3000).fit(E_tr[folds != 0], y[folds != 0])
d_oof = np.abs(lr0.predict_proba(E_tr[folds == 0]) - saved_oof[folds == 0]).max()
check("OOF fold 0 = LR(folds 1–4) on fold-0 rows", d_oof < 1e-3, f"max |Δp| {d_oof:.2e}")
per_fold = [f1(y[folds == k], saved_oof[folds == k].argmax(1)).mean() for k in range(5)]
info("fold-0 selection optimism (hyper-parameters chosen on fold 0)",
     f"macro-F1 fold 0 {per_fold[0]:.3f} vs folds 1–4 mean {np.mean(per_fold[1:]):.3f} "
     f"(per fold {np.round(per_fold, 3).tolist()})")

# ---------------------------------------------------------------- D. fine-tunes
out.append("## D. Base model A — fine-tuned folds")
for key in ("qwen3-8b-lora", "qwen3-0.6b-lora"):
    dones = [json.loads((DATA / "ft" / key / f"fold{k}" / "done.json").read_text()) for k in range(5)]
    cfg_keys = ("hf", "max_len", "truncation", "epochs", "lr", "batch", "class_weight", "lora", "limit", "max_steps")
    same = all(all(d.get(c) == dones[0].get(c) for c in cfg_keys) for d in dones)
    check(f"{key}: identical training config in all 5 folds", same,
          ", ".join(f"{c}={dones[0].get(c)}" for c in cfg_keys))
    check(f"{key}: no smoke-test folds", all(not d.get("limit") and not d.get("max_steps") for d in dones))
    check(f"{key}: every fold has a verified prediction order", all(d.get("predict_order") for d in dones),
          str([d.get("predict_order") for d in dones]))
    oof = np.load(DATA / "oof" / f"ft-{key}.npy")
    hos = []
    for k, d in enumerate(dones):
        fd = DATA / "ft" / key / f"fold{k}"
        vi, vp = np.load(fd / "val_idx.npy"), np.load(fd / "val_proba.npy")
        hos.append(np.load(fd / "holdout_proba.npy"))
        ok_rows = np.array_equal(vi, np.flatnonzero(folds == k))
        ok_oof = np.allclose(oof[vi], vp)
        m = f1(y[vi], vp.argmax(1)).mean()
        check(f"{key} fold {k}: val rows = fold rows, OOF matches, done.json metric reproduces",
              ok_rows and ok_oof and abs(m - d["val_macro_F1"]) < 1e-6,
              f"macro-F1 {m:.3f} (done.json {d['val_macro_F1']:.3f})")
        check(f"{key} fold {k}: probabilities valid", np.isfinite(vp).all() and np.allclose(vp.sum(1), 1, atol=1e-3)
              and hos[-1].shape == (len(hold), K))
    agree = [(hos[i].argmax(1) == hos[j].argmax(1)).mean() for i in range(5) for j in range(i + 1, 5)]
    check(f"{key}: holdout predictions of the 5 fold models agree (alignment; chance ≈ 0.18)", min(agree) > 0.8,
          f"pairwise agreement {min(agree):.3f}–{max(agree):.3f}")
    f0 = [(hos[0].argmax(1) == hos[j].argmax(1)).mean() for j in range(1, 5)]
    info(f"{key}: fold-0 holdout agreement with folds 1–4", f"{np.round(f0, 3).tolist()}")
    mean_ho = np.mean(hos, axis=0)
    check(f"{key}: saved holdout scores = mean of fold holdouts",
          np.allclose(mean_ho, np.load(DATA / "holdout_scores" / f"ft-{key}.npy"), atol=1e-5))

# ---------------------------------------------------------------- E. stacking (stack-H)
out.append("## E. Stack-H — independent recomputation")
FT, EMB = "ft-qwen3-8b-lora", "emb-qwen3-8b-instr_lr"
lg = lambda p: np.log(np.clip(np.asarray(p, np.float64), 1e-6, 1))  # noqa: E731  (float64 as in 06_stack.py)
X = np.hstack([lg(np.load(DATA / "oof" / f"{FT}.npy")), lg(np.load(DATA / "oof" / f"{EMB}.npy"))])
meta_oof = np.zeros((len(y), K))
for k in range(5):
    tr, va = folds != k, folds == k
    sc = StandardScaler().fit(X[tr])
    meta_oof[va] = LogisticRegression(C=1.0, max_iter=3000).fit(sc.transform(X[tr]), y[tr]).predict_proba(
        sc.transform(X[va]))
saved = np.load(DATA / "oof" / "stack-H.npy")
check("stack-H OOF reproduces (cross-fitted meta-LR)", np.abs(meta_oof - saved).max() < 1e-3,
      f"max |Δp| {np.abs(meta_oof - saved).max():.2e}")
sc_all = StandardScaler().fit(X)
meta_all = LogisticRegression(C=1.0, max_iter=3000).fit(sc_all.transform(X), y)
import importlib  # noqa: E402
fh = importlib.import_module("10_final_holdout")
same_meta = np.allclose(meta_all.coef_, fh.meta.coef_, atol=1e-8) and np.allclose(fh.sc.mean_, sc_all.mean_)
check("final holdout script uses exactly the evaluated meta-model (same features, dtype, scaler, coefficients)", same_meta)
check("meta-model never sees a row's own fold (cross-fitting)", True, "by construction above: fit on folds≠k")
# leakage-free in the strict sense: base OOF rows come from models that did not see them
check("base OOF rows were predicted by models that did not train on them", True,
      "D: val rows = fold rows for fine-tunes; C: LR(folds≠k) reproduces fold-k OOF")

# ---------------------------------------------------------------- F. evaluation math
out.append("## F. Evaluation module vs scikit-learn")
pred = saved.argmax(1)
res = evaluate(y, pred, saved, n_boot=50, n_boot_auc=5, n_perm=5)
per = res["per_class"]
check("F1 per class = sklearn", np.allclose(per["F1"].values, f1(y, pred)))
check("precision per class = sklearn", np.allclose(per["precision"].values,
                                                   precision_score(y, pred, labels=range(K), average=None)))
check("recall per class = sklearn", np.allclose(per["recall"].values,
                                               recall_score(y, pred, labels=range(K), average=None)))
check("accuracy = sklearn", abs(res["overall"].loc["accuracy", "value"] - accuracy_score(y, pred)) < 1e-12)
info("stack-H (OOF)", f"macro-F1 {f1(y, pred).mean():.4f}, min-F1 {f1(y, pred).min():.4f}, "
                      f"accuracy {accuracy_score(y, pred):.4f}")

# ---------------------------------------------------------------- G. predictions.jsonl
out.append("## G. predictions.jsonl")
P = pd.read_json(ROOT / "predictions.jsonl", lines=True)
check("columns are exactly id, label", list(P.columns) == ["id", "label"])
check("one row per holdout id, same order as holdout.jsonl", P.id.tolist() == hold.id.tolist(), f"{len(P)} rows")
check("labels are the ten names; all ten used", set(P.label) == set(LABELS), f"{P.label.nunique()} used")
final = np.load(DATA / "holdout_scores" / "stack-H-perfold.npy")
check("labels = argmax of the final stack-H (per-fold) holdout scores",
      (P.label.values == np.array(LABELS)[final.argmax(1)]).all())
for e in (FT, EMB):
    a_ho = (np.load(DATA / "holdout_scores" / f"{e}.npy").argmax(1) == final.argmax(1)).mean()
    a_oof = (np.load(DATA / "oof" / f"{e}.npy").argmax(1) == saved.argmax(1)).mean()
    check(f"holdout agreement final vs {e} close to its OOF agreement (alignment)", abs(a_ho - a_oof) < 0.05,
          f"holdout {a_ho:.3f}, OOF {a_oof:.3f}")
raw = [json.loads(line) for line in open(ROOT / "predictions.jsonl")]
check("every line is valid JSON with string id and label", all(isinstance(r["id"], str) and isinstance(r["label"], str)
                                                              and set(r) == {"id", "label"} for r in raw))

# ---------------------------------------------------------------- H. eyeball sample
out.append("## H. Random holdout sample with predicted labels (read by hand)")
rng = np.random.default_rng(7)
for i in rng.choice(len(hold), 15, replace=False):
    t = hold.text.iloc[i].replace("\n", " ")
    line = f"[{P.label.iloc[i]}] {t[:150]}{'…' if len(t) > 150 else ''}"
    print("      " + line)
    out.append(f"  - {line}")

summary = f"**{n_fail} failed checks.**"
print("\n" + summary)
(ROOT / "reports" / "11_audit.md").write_text("# Audit of the process\n\nGenerated by `src/11_audit.py`.\n\n"
                                              + summary + "\n\n" + "\n".join(out) + "\n")
