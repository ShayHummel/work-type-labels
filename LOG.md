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

**Next** — review with Shay.

### Step 1 follow-up — review questions (09:40–10:20)

- **Label count table and top n-grams per label:** `src/01b_label_ngrams.py` → `reports/01b_label_ngrams.md`.
  - Per label, a TF-IDF vectorizer is fitted for unigrams, bigrams and trigrams; the top-10 by mean weight are listed.
  - A second table ranks n-grams by how distinctive they are (class vs rest).
  - Repeated prompt templates dominate several lists: "passage … book" for Other, "shell script creates changes" for Feature dev and Refactoring.
- **"Still truncated. Try again." (rows 210–211) is an anecdote, not evidence.** It was withdrawn as support. Measured instead: 12.9% of prompts have ≤ 6 words, mostly Other (34%) and Researching (28%).
- **Alignment claim, quantified.** For each holdout row, find the nearest train row by TF-IDF cosine, and keep the 3,419 confident matches (cos > 0.3). Then check how far apart the two rows sit in relative file position.
  - Within ±1% / ±2% / ±5% of the file: 46.9% / 61.8% / 65.6% of matches.
  - The same check with the holdout order shuffled: 2.0% / 4.1% / 9.8%.
- **Decision (Shay): do not use neighbour/conversation context.**
  - Rationale: the classifier must label a single prompt, as it will in production on future data. The order effect only tells us how the split was made.
  - The one consequence kept: since the holdout is a random row-level sample, a stratified random CV mirrors it.
- **Teacher labels** = the labels in `train.jsonl`, produced by a strong LLM (TASK.md), not by humans. Holdout scoring uses the same teacher, so the target is agreement with the teacher.

---

## Step 2 — Data preparation & evaluation protocol (10:20–10:50)

- **Folds:** `src/02_folds.py` → `data/folds.csv`. 5-fold StratifiedKFold, shuffled, seed 42. Each class's share per fold is within 0.02 pp of its overall share.
- **Evaluation module** (`src/evaluation.py`), used by every model on out-of-fold predictions:
  - Per class: precision, recall, F1, ROC-AUC and PR-AUC (one-vs-rest).
  - Overall: confusion matrix, accuracy, macro-F1, and **min-F1**, the bar from TASK.md.
  - **95% bootstrap CIs at holdout scale.** Each replicate has 6,199 rows (the holdout size) with the train class proportions (e.g. 39 Optimize rows), so each interval shows how much a score on a holdout-sized sample can move from sampling alone.
- **p-values:**
  - `p(F1≤0.8)`: one-sided bootstrap test that the class misses the bar.
  - `p(AUC=0.5)`: Mann–Whitney U.
  - `p(macro-F1 = chance)`: permutation test.
  - `compare()`: model vs model, with a paired bootstrap per class and McNemar's exact test.
- **Harness check:**
  - A random classifier gives AUC ≈ 0.5, `p(F1≤0.8)` = 1 for every class, and no significant AUCs.
  - Under random predictions, 10 runs of the permutation p-value come out roughly uniform (0.005–0.99), so the test is calibrated.
  - Results: `reports/02_folds.md`.

- **Revision (Shay's review):** the first version scaled the bootstrap to the per-class "Support" column printed in TASK.md. That column presumably counts the holdout's teacher labels, but TASK.md does not say so explicitly, and those labels are noisy anyway. The bootstrap now uses train proportions × 6,199, and the evaluation depends on `train.jsonl` only. The TASK.md column is shown in the EDA as unverified context and is not used anywhere else.
- **Reminder on label quality:** all scores, ours and Zuzai's, measure agreement with the teacher LLM, not correctness. Disagreement between the teacher's labels on near-identical text (EDA §4) caps what any model can reach. The note will separate model errors from these collisions.

- **Revision (Shay's review, 2):**
  - The per-label top-10 n-gram tables moved into the EDA report as §7 (7a: separate vectorizer per label, as requested; 7b: distinctive, label vs rest). The standalone `01b` script was removed.
  - EDA §5 now explains how to read the alignment plot:
    - **Diagonal** (|Δpos| ≤ 0.02): 2,114 rows, median cosine 0.55. The nearest train row is a turn of the same conversation.
    - **Background scatter**: 1,305 rows, median cosine 0.41. Generic short prompts matched to a similar prompt anywhere.
    - **Horizontal bands**: single generic train prompts that are nearest for many holdout rows.
  - ROC-AUC and PR-AUC now have their own table ("Ranking metrics"): value, 95% CI, Mann–Whitney p, and PR chance level. Macro ROC-AUC and macro PR-AUC are in the overall table, and per-class ROC and PR curves are plotted (`plot_curves`).

**Next** — Step 3, model exploration (candidates proposed by Shay: code-tuned embeddings + linear/LogReg/SVM; DeBERTa-v3 fine-tune; Hugging Face search for task-related models).

---

## Step 3 — Experiment planning (10:50–11:15)

- Benchmarked embedding runtime on this Mac (`src/03_embed_benchmark.py` → `reports/03_embed_benchmark.md`). BGE-M3: 3 min for all 30,991 prompts. Qwen3-Embedding-0.6B: 9 min. 4B / 7B / 8B models: ≥ 1–2 h each locally, so they go to Colab.
- Added JEV (typed-decision classifiers: open-jev-deberta-v3-large, simple-jev) to the candidates at Shay's request.
- Wrote `PLAN.md`: features, models, evaluation protocol, decision layer, noise analysis, runtime policy, timed experiment queue and open decisions.

**Next** — review of PLAN.md with Shay.

### Step 3 follow-up — plan review and Colab tooling (11:20–11:35)

- Shay's decisions are recorded in PLAN §11:
  - Colab jobs must survive GPU loss and be one click.
  - The Haiku classifier is in scope, lowest priority, after JEV.
  - 7–8B embedding models are acceptable in production.
- Colab design in PLAN §13: thin notebooks, with all logic in `src/` scripts that run identically on CUDA and MPS.
- `src/embed.py`: resumable embedding job.
  - Writes 2,000-row chunks atomically. A rerun skips finished chunks and models.
  - On OOM it halves the batch size. Models too large for the GPU are skipped. A manifest records the ids hash, dimension and device, and finished models can be zipped for export.
  - Tested by killing a run, deleting one chunk and rerunning: only the missing chunk was recomputed.
- `notebooks/colab_embeddings.ipynb`: Run all → Qwen3-Embedding-8B, 4B, nomic-embed-code, 8B with instruction. Outputs go to `My Drive/zuzai-hw/`.
- `src/import_colab.py`: unzips the exported results into `data/` and validates them (ids hash, shapes, NaNs).
- Started locally in the background: BGE-M3, Qwen3-0.6B, and Qwen3-0.6B with instruction (max_len 1024).

### Fine-tuning tooling (11:40–12:00)

- Decision (Shay): fine-tuning runs on Colab, and the results must be downloadable and kept for future comparison.
- `src/finetune.py`: one job per (model, fold), the same code on CUDA and MPS.
  - Weighted CE computed in fp32 on fp32 logits.
  - Head+tail truncation (last 128 tokens kept), batches grouped by length, bf16 on A100/L4.
  - Checkpoint to Drive every 300 steps; a rerun resumes from it and skips finished folds.
  - Each finished fold writes `val_proba.npy`, `val_idx.npy`, `holdout_proba.npy` and `done.json` (config, device, timings, fold metrics), then re-zips all finished folds of the model to `exports/ft_<model>.zip`.
  - Models: DeBERTa-v3 base and large, ModernBERT-large, mmBERT-base, Qwen3 0.6B and 1.7B + LoRA.
- `notebooks/colab_finetune.ipynb`: Run all. Fold 0 of DeBERTa-v3-base, ModernBERT-large and Qwen3-0.6B-LoRA first, then DeBERTa-v3-base folds 1–4.
- `src/import_colab.py` now also validates fine-tune zips: fold rows match `data/folds.csv`, shapes, finite values, and it flags smoke-test runs.
- Smoke test passed locally (deberta-v3-xsmall, 400 rows, 12 steps, MPS): files written, zip exported, import validated. Resume after a lost GPU relies on HF Trainer's `resume_from_checkpoint` and was not tested end-to-end here.
- Block 1 started in the background (`src/04_tfidf_baselines.py`, shared runner `src/experiments.py`, which writes OOF scores, holdout scores and `reports/results.csv`).
