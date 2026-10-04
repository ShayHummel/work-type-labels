# Work log — work-type labels

Time budget: 9 hours, started **2026-10-04 08:28 IDT**.
Each step ends with a review checkpoint where I decide on the next phase.
Scripts live in `src/` (numbered per phase), outputs in `reports/`.

---

## Step 0 — Setup (08:28–08:35)

**Done**
- Read `TASK.md`. Key constraints: 10 fixed labels, **0.8 F1 on every label**, single scoring on holdout, train only on `train.jsonl` (+ self-created text), no holdout rows in training, no dropping labels.
- Prior baselines given by Zuzai: fastText ≈ 0.50 macro-F1; DistilBERT (3 ep, len 256, inverse-freq class weights) 0.603 macro-F1 / 0.695 acc. Weakest: Review 0.42, Optimize 0.46, Architecting 0.47, Refactoring 0.50. Synthetic data for weak classes and keyword regexes did not help. Larger encoder collapsed due to fp16 + weighted loss → keep loss in fp32.
- Forked the repo, cloned, working on branch `solution`.
- Python 3.12 venv (`.venv`), base packages pinned in `requirements.txt`. Hardware: Apple M5 Pro, 24 GB, MPS GPU.

**Observations**
- Holdout class mix (from Zuzai's support counts) is very imbalanced: Other 25%, Feature dev 22%, Researching 20%, Bug fix 14%, … Optimize 0.8% (49 rows). Per-class F1 on the small classes will have high variance (one error on Optimize ≈ 1–2 F1 points).
- Labels come from an LLM teacher, so part of the gap to 0.8 may be label noise/ambiguity rather than model error. EDA must quantify this.

**Next** — Step 1: exploratory data analysis.
