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
- Python 3.12 project managed with **uv** (`pyproject.toml` + `uv.lock`; run scripts with `uv run python src/<script>.py`). Hardware: Apple M5 Pro, 24 GB, MPS GPU.

**Observations**
- Holdout class mix (from Zuzai's support counts) is very imbalanced: Other 25%, Feature dev 22%, Researching 20%, Bug fix 14%, … Optimize 0.8% (49 rows). Per-class F1 on the small classes will have high variance (one error on Optimize ≈ 1–2 F1 points).
- Labels come from an LLM teacher, so part of the gap to 0.8 may be label noise/ambiguity rather than model error. EDA must quantify this.

**Next** — Step 1: exploratory data analysis.

---

## Step 1 — Exploratory data analysis (08:55–09:35)

Script: `src/01_eda.py` → `reports/01_eda.md` + `reports/figures/01_*.png`.

**Findings**
1. **Clean data.** No nulls, empty texts, duplicate ids or exact train/holdout text overlap. One source (`devgpt-*`, shared ChatGPT conversations). Newlines were stripped: every prompt is one line, so code structure is lost.
2. **Train and holdout have the same class mix** (within ±0.4 pp). Imbalance is 41× (Other 25.7% … Optimize 0.6% = 155 rows).
3. **Length.** The median prompt is about 21 words, but the tail is heavy: about 17% exceed 256 tokens (the DistilBERT cut) and about 10% exceed 512. Review and Refactoring are the longest classes (about 30% over 256 tokens), since they usually carry pasted code. Researching and Setup are short. Truncation hurts exactly the weak classes.
4. **Many prompts are conversational follow-ups** ("Still truncated. Try again.", "cant the deal status be merged with the lead status", "Do you see in the tests that I send…"). Their label can only be inferred from the conversation they belong to.
5. **The rows are in conversation order.**
   - P(same label as previous row) is 0.47, against 0.18 if the order were random.
   - Consecutive rows are 14× more similar than random pairs.
   - The holdout keeps the same global order. The position of each holdout row and the position of its nearest train row lie on the diagonal (median gap 1% of the file).
   - So the holdout is a row-level sample out of the same conversations, and each holdout prompt's conversational neighbours are in train.
6. **Teacher-label noise is real.**
   - Among near-duplicate pairs (TF-IDF cosine ≥ 0.9), 28% have different labels.
   - Above cos 0.5, nearest-neighbour agreement levels off at about 65–72%.
   - Typical collisions are near-identical text with different labels: "convert to arabic <strings…>" gets Feature dev, Other or Refactoring, and readme pastes get Researching or Review.
   - The most frequent colliding pair is Bug fix / Feature dev, then Feature dev with Other, Review and Refactoring.
7. **Templates repeat.** "I will provide you with a passage…" appears 631× and is always Other. "Task: implement the following feature…" appears 319× and is 90% Feature dev.
8. **Keywords are weak**, as TASK.md said:
   - Only 61% of rows containing "refactor" are Refactoring, and 13.5% of Refactoring rows contain it.
   - "faster" spreads across Other, Researching and Optimize (about 20% each).
   - Only 19% of rows containing "review" are Review.
   - Optimize vs Refactoring is separated by intent ("more efficient", "optimize this", "slow"), not by vocabulary.
9. 6% of prompts are non-English (mostly Chinese and Japanese), so the encoder must be multilingual or at least robust to it.

**Implications for modelling**
- Context matters: a prompt's neighbours in the conversation carry strong label signal. There are two ways to use it, and they differ in how legitimate they are:
  - (a) **Neighbour text** (the previous and next prompts of the session). Zuzai has the session history in production, so this is valid.
  - (b) **Neighbour train labels** (the teacher labels of adjacent train rows). This exploits the row-level split, and would not exist in production beyond the model's own predictions. If used at all, it must be reported separately.
- Use a long-context encoder (≥ 512 tokens) or head+tail truncation, multilingual if possible.
- Expect a noise ceiling below 0.8 on some pairs (Bug fix/Feature dev, Review/Researching, Refactoring/Feature dev). Error analysis must separate label collisions from model errors.

**Next** — decision pending.
