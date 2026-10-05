"""Final step — write predictions.jsonl for the holdout from a chosen experiment, with format checks.

Reads data/holdout_scores/<exp_id>.npy (6199 × 10, LABELS order), takes the arg-max label, and checks:
every holdout id exactly once, only the ten label names, all ten labels used, row count. Also writes
reports/09_predictions.md: predicted class mix on the holdout vs the train mix, and the expected per-class F1 of
the chosen model (out-of-fold estimate with 95% CI at holdout scale, from src/evaluation.py).

Run: uv run python src/09_predict.py <exp_id> [--out predictions.jsonl]
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from evaluation import LABELS, evaluate
from experiments import DATA, load_data

ROOT = Path(__file__).resolve().parents[1]

args = [a for a in sys.argv[1:] if not a.startswith("--")]
exp = args[0]
out = Path(sys.argv[sys.argv.index("--out") + 1]) if "--out" in sys.argv else ROOT / "predictions.jsonl"
out = out if out.is_absolute() else ROOT / out

train, hold = load_data()
scores = np.load(DATA / "holdout_scores" / f"{exp}.npy")
oof = np.load(DATA / "oof" / f"{exp}.npy")
assert scores.shape == (len(hold), len(LABELS)), scores.shape
assert np.isfinite(scores).all()
pred = np.array(LABELS)[scores.argmax(1)]

# ---- format checks (TASK.md: every holdout id once, every label one of the ten names)
problems = []
if len(pred) != len(hold) or hold.id.duplicated().any():
    problems.append("row count / duplicate ids")
if not set(pred) <= set(LABELS):
    problems.append("unknown label")
missing = sorted(set(LABELS) - set(pred))
if missing:
    problems.append(f"labels never predicted: {missing}")
if problems:
    sys.exit("NOT WRITTEN — " + "; ".join(problems))

with open(out, "w") as f:
    for i, lab in zip(hold.id, pred):
        f.write(json.dumps({"id": i, "label": lab}) + "\n")
# re-read and verify the file itself
back = pd.read_json(out, lines=True)
assert list(back.columns) == ["id", "label"] and len(back) == len(hold)
assert back.id.tolist() == hold.id.tolist() and back.label.isin(LABELS).all()

# ---- report
mix = pd.DataFrame({"holdout predicted": pd.Series(pred).value_counts(), "train (teacher)": train.label.value_counts()})
mix = mix.reindex(LABELS).fillna(0).astype(int)
mix["holdout predicted %"] = (mix["holdout predicted"] / len(hold) * 100).round(1)
mix["train %"] = (mix["train (teacher)"] / len(train) * 100).round(1)
res = evaluate(train.y.values, oof.argmax(1), oof)
per = res["per_class"]
exp_f1 = pd.DataFrame({"expected F1 (OOF)": per["F1"].round(3),
                       "95% CI (holdout scale)": per.apply(lambda r: f"{r['F1 lo']:.3f}–{r['F1 hi']:.3f}", 1),
                       "p(F1≤0.8)": per["p(F1≤0.8)"].round(3)})
ov = res["overall"]
lines = [f"# Holdout predictions — `{exp}`", "", f"Written to `{out.relative_to(ROOT)}`: {len(back)} rows, "
         f"{back.id.nunique()} unique ids, {back.label.nunique()} of 10 labels used. All format checks passed.", "",
         "## Predicted class mix (holdout) vs teacher mix (train)", "", mix.to_markdown(), "",
         "## Expected per-class F1 (out-of-fold on train, 95% CI for a holdout-sized sample)", "",
         exp_f1.to_markdown(), "", ov.to_markdown(floatfmt=".3f"), ""]
(ROOT / "reports" / "09_predictions.md").write_text("\n".join(lines))
print("\n".join(lines))
