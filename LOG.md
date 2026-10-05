# Work log — work-type labels

Time budget: 9 hours of Shay's working time (not wall-clock or compute time), started **2026-10-04 08:28 IDT**. Long runs on Colab or in the background do not count against it.
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

---

## Block 1 — TF-IDF baselines (11:45–12:14)

Script `src/04_tfidf_baselines.py`. Results: `reports/results.csv` and `reports/exp/tfidf-*.md`, with confusion matrices and ROC/PR curves in `reports/figures/exp/`.
Hyper-parameters were chosen on fold 0, then all 5 folds were run. The vectorizers are fitted inside each training split.

| Features | Ridge (macro / min F1) | LogReg | LinearSVC |
|---|---|---|---|
| Words 1–3 | 0.516 / 0.234 | 0.531 / 0.207 | 0.520 / 0.222 |
| Chars 3–5 | 0.509 / 0.242 | 0.536 / 0.264 | 0.526 / 0.247 |
| Words + chars | 0.549 / 0.254 | **0.558 / 0.267** | 0.551 / 0.280 |

- **Best: words+chars LogReg.** Accuracy 0.667, macro ROC-AUC 0.923, macro PR-AUC 0.586. Every grid selected `class_weight=balanced`.
- **Words+chars vs words only:** +0.027 macro-F1 (95% CI 0.013–0.040, paired bootstrap p < 0.001; McNemar p ≈ 1e-34).
- **LogReg vs SVM on words+chars:** not significant (Δ macro-F1 CI −0.006 to 0.019, p = 0.13).
- **Per class (best model):** Other 0.78, Bug fix 0.71, Feature dev 0.68, Testing 0.67, Researching 0.65, Setup 0.58, Refactoring 0.45, Architecting 0.44, Review 0.35, Optimize 0.27 (holdout-scale CI 0.11–0.43). `p(F1≤0.8)` is 1.00 for every class.
- **Conclusion:** this matches Zuzai's fastText (about 0.50) and confirms the bag-of-words ceiling of about 0.55 macro-F1. The weak classes (Review, Optimize, Architecting, Refactoring) are the same ones that were weak for DistilBERT.
- **Local embeddings finished** (`data/emb/`): BGE-M3 4 min, Qwen3-0.6B 19 min, Qwen3-0.6B with instruction 21 min. A quick fold-0 check of BGE-M3 + LogReg gives macro 0.567 / min 0.308.
- **Colab (T4):** Qwen3-Embedding-8B is skipped (needs about 17 GB); the 4B runs at about 3.6 min per 2,000-row chunk, about 58 min in total. Fixed both scripts to use fp16 on GPUs without native bf16.

### Checkpoint after block 1 (12:20)

- Shay's decisions:
  - No Drive-queue worker for Colab; the notebooks stay manual.
  - Wait for the Colab results (Qwen3-4B embeddings, then fine-tuning) before continuing.
  - The 9-hour budget is Shay's working time; about 2–3 hours used so far on 2026-10-04.

---

## Block 2 — Embeddings × linear models / kNN, and TF-IDF fusion (12:30–14:50)

Script `src/05_embedding_models.py`, same protocol as block 1. A bug in the grid selection (pandas turned `class_weight=None` into NaN) crashed the first run; it was fixed in `select_on_fold0` and the block rerun. Block 1 was unaffected, since all its grids chose `balanced`.

| Features | LogReg | SVM | Ridge | kNN | Fusion with TF-IDF words+chars (LogReg) |
|---|---|---|---|---|---|
| BGE-M3 | 0.546 / 0.302 | 0.540 / 0.294 | 0.501 / 0.223 | 0.506 / 0.268 | 0.611 / 0.374 |
| Qwen3-Emb-0.6B | 0.576 / 0.323 | 0.596 / 0.288 | 0.545 / 0.278 | 0.496 / 0.247 | 0.639 / 0.391 |
| Qwen3-Emb-0.6B + instruction | 0.587 / 0.361 | 0.602 / 0.368 | 0.548 / 0.262 | 0.505 / 0.269 | **0.640 / 0.410** |

(macro-F1 / min-F1, 5-fold OOF)

- **Best: TF-IDF words+chars ⊕ Qwen3-0.6B-instr, LogReg.** Accuracy 0.743, macro ROC-AUC 0.957, macro PR-AUC 0.701. It beats Zuzai's DistilBERT (0.603 macro).
  - Per class: Other 0.85, Bug fix 0.77, Feature dev 0.74, Researching 0.73, Testing 0.71, Setup 0.66, Refactoring 0.53, Architecting 0.51, Optimize 0.48, Review 0.41.
- **Fusion vs best TF-IDF:** +0.08 macro-F1 (95% CI 0.063–0.105, p < 0.001).
- **The instruction prompt** helps the embedding alone (LogReg min-F1 0.32 → 0.36, SVM 0.29 → 0.37). Inside the fusion it makes no significant macro-F1 difference (Δ CI −0.018 to 0.020, p = 0.42).
- **kNN is the weakest**: neighbours often carry different teacher labels, consistent with the EDA noise finding.
- **Runtime issue:** early fusion (sparse TF-IDF plus a dense block in one LogReg) is slow. CV took 2.5 min for BGE, 22 min for Qwen, and 90 min for Qwen-instr (no class weights, slow lbfgs convergence). For later fusions (e.g. Qwen3-4B), use **late fusion** (stack the OOF probabilities of the separate models) instead. It is much cheaper and needs no refit.

### Fine-tuning job hardened before the Colab run (14:50–15:00)

- **Smoke tests on MPS** (64 rows, 3 steps): ModernBERT-large and Qwen3-0.6B + LoRA both train, predict, write their outputs and export.
  - At their default settings (fp32, batch 16, length 1024) they run out of memory on this Mac. That is a hardware limit, not a code issue.
  - Added a `--batch-size` override.
- **Resume test:** start a run, kill it as soon as a checkpoint exists, rerun.
  - **Bug found:** the kill landed during a checkpoint save, and HF's `get_last_checkpoint` picked the half-written folder, so the run failed with "Can't find a valid checkpoint".
  - **Fix:** `last_complete_checkpoint()` resumes from the newest checkpoint that has trainer state, optimizer, scheduler and RNG files, and deletes partial folders.
  - **Retest:** the rerun resumed from checkpoint-5 and finished the fold.
- **Colab:** Qwen3-Embedding-8B is running on an A100 (40 GB) at about 40 s per 2,000-row chunk, so about 11 min per 8B model.

---

## Block 5 — Large embeddings from Colab × linear models / kNN (15:25–16:10)

- **Colab outputs** (Qwen3-Embedding-8B with and without instruction, 4B, nomic-embed-code; A100) were copied by Shay into `data/emb/`. Validated with `src/import_colab.py --check-emb`: ids hash, shapes, finite values, unit norm, no collapsed rows. All OK.
- **Runs crashed three times.** The causes were memory pressure and a native segfault of Ridge on > 1024-d float32 inside joblib/loky workers; the same fits run fine in-process.
  - Fixes: experiments that are already done are skipped; 2 workers for large embeddings; Ridge and kNN run sequentially on > 1024-d (seconds each).
- Early fusion was skipped (too slow, see block 2). Late fusion is in block 7.

| Embedding (dim) | LogReg | SVM | Ridge | kNN |
|---|---|---|---|---|
| **Qwen3-8B + instruction (4096)** | **0.740 / 0.573** | 0.742 / 0.555 | 0.697 / 0.518 | 0.677 / 0.446 |
| Qwen3-8B (4096) | 0.670 / 0.417 | 0.670 / 0.448 | 0.638 / 0.408 | 0.538 / 0.264 |
| Qwen3-4B (2560) | 0.639 / 0.409 | 0.655 / 0.424 | 0.623 / 0.392 | 0.523 / 0.269 |
| nomic-embed-code (3584) | 0.645 / 0.421 | 0.623 / 0.396 | 0.610 / 0.343 | 0.506 / 0.269 |

(macro-F1 / min-F1, 5-fold OOF)

- **Best single model: Qwen3-8B-instr + LogReg.**
  - Accuracy 0.810, macro ROC-AUC 0.976. CI at holdout scale: macro-F1 0.722–0.759, min-F1 0.505–0.620.
  - 4 classes are ≥ 0.8: Other 0.88, Bug fix 0.85, Feature dev 0.81, Researching 0.81. Testing is at 0.796.
  - Below the bar: Setup 0.75, Refactoring 0.68, Optimize 0.64, Architecting 0.63, Review 0.57.
- **The task instruction is the biggest single lever at 8B:** +0.07 macro-F1 and +0.16 min-F1 over the same model without it. At 0.6B the gain was small.
- **4B was only embedded without the instruction**, so a 4B-instr run is a missing comparison (about 6 min on an A100).
- **nomic-embed-code** (a code-retrieval model) is no better than general-purpose Qwen3-4B, as expected for an intent task.

---

## Block 7 (part 1) — Stacking + cross-fitted class bias (16:15–16:30)

`src/06_stack.py` with sets in `data/stack_sets.json`. The meta-model is a multinomial LogReg on standardised base scores (log-probabilities, or decision values for SVM), cross-fitted on the 5 folds.

| Set | Base models | macro-F1 / min-F1 | + class bias (cross-fitted) |
|---|---|---|---|
| A | TF-IDF WC LR, Qwen3-0.6B-instr LR | 0.640 / 0.413 | 0.621 / 0.424 |
| B | Qwen3-8B-instr LR, TF-IDF WC LR | 0.738 / 0.546 | 0.728 / 0.553 |
| **C** | Qwen3-8B-instr LR + SVM, TF-IDF WC LR, Qwen3-0.6B-instr LR | **0.750 / 0.584** | 0.724 / 0.563 |
| D | C + Qwen3-8B LR, Qwen3-4B SVM, nomic LR, BGE-M3 LR | 0.748 / 0.579 | 0.722 / 0.554 |

- **Best: stack-C.** Accuracy 0.822, macro ROC-AUC 0.980, macro PR-AUC 0.826.
- **Stack-C vs the best single model (Qwen3-8B-instr LR):**
  - Macro-F1 +0.010 (95% CI −0.003 to 0.023, p = 0.065). McNemar p ≈ 8e-13: more rows right (993 vs 702 discordant).
  - Significant per-class gains on the big classes: Bug fix, Feature dev, Researching, Other (p < 0.05).
  - The weak classes do not move significantly.
- **Adding more, weaker embeddings (set D) adds nothing.**
- **The class bias does not generalise once cross-fitted.**
  - The bias vectors learned on different 4-fold subsets disagree in sign (e.g. Setup −1.5 vs +1.0), and the held-out fold gets worse on both macro-F1 and min-F1.
  - Optimising min-F1 directly chases the noisiest class (Optimize, 31 rows per fold).
  - Conclusion: do not use a min-F1-tuned bias. If a decision layer is used, it needs a smoother objective (e.g. macro-F1, or a single shared prior-correction parameter) and must again be cross-fitted.
- **Where it stands:** 4 classes are clearly ≥ 0.8 (Other 0.89, Bug fix 0.86, Feature dev 0.82, Researching 0.82). Testing is at 0.79. Setup 0.76, Refactoring 0.70, Optimize 0.65, Architecting 0.64 and Review 0.58 are still short of 0.8.

### Colab fine-tuning: Google Drive ran out of space (16:30)

- **Cause:** checkpoints were written to Drive. Each one holds model + optimizer state (DeBERTa-v3-base ≈ 2.2 GB, ModernBERT-large ≈ 4.7 GB), and the peak is ×2 while a checkpoint is being replaced. About 2.5 GB of embedding chunks and zips were also still on Drive.
- **Fix:** new `--ckpt-root` option. On Colab, checkpoints now go to the VM disk (`/content/ckpt`), and only the results (≈ 0.5 MB per fold) go to Drive.
  - Trade-off: if the VM itself is replaced, the running fold restarts (≈ 7–15 min on A100) instead of resuming.
- **Notebook:** a cleanup cell removes embedding chunks and old checkpoint folders from Drive (the Drive trash must be emptied by hand).
- **Jobs (Shay: fine-tune in parallel with the error analysis; Qwen + LoRA only, expected to be more efficient than DeBERTa):** Qwen3-0.6B + LoRA folds 0–4.

---

## Block 8 — Error analysis of stack-C (16:00–16:25)

Script `src/07_error_analysis.py` → `reports/07_error_analysis.md`. The first run hung: cleanlab spawned worker processes, and on macOS each one re-ran the script (no `__main__` guard). Fixed with `n_jobs=1`; the analysis itself takes 4 s.

1. **Confident learning (cleanlab):** 1,946 rows (7.8%) are flagged as likely label issues.
   - The weak classes are flagged 2–4× as often as the strong ones: Review 20.2%, Optimize 18.1%, Architecting 17.5%, Refactoring 16.4%, Setup 11.7%, Testing 10.6%, against Other 4.4%, Feature dev 6.2%, Bug fix 6.6%.
2. **Teacher self-consistency on near-duplicates** (Qwen3-8B-instr cosine ≥ 0.95, 16% of rows): the near-duplicate carries the same teacher label in 96.8% of cases for Other, 88% for Bug fix, 85% for Feature dev, but only **64–68% for Refactoring, Architecting and Review**, and 72% for Optimize.
   - So on near-identical prompts the teacher itself disagrees about a third of the time in those classes.
   - Label agreement with the nearest other row rises from 0.57 to 0.94 as similarity rises, but even at 0.95–0.98 cosine it is only 0.84.
3. **Error attribution:**
   - For the weak classes, **68–77% of false positives** (model says Review, Architecting, Optimize or Refactoring; teacher says otherwise) are on rows cleanlab flags as inconsistent.
   - About 45% of false negatives are flagged or near-duplicate collisions; about 55% are unexplained.
4. **F1 on rows with self-consistent labels** (8.2% excluded): every class reaches ≥ 0.80 except Review (0.76).
   - Refactoring 0.70 → 0.85, Architecting 0.64 → 0.81, Optimize 0.65 → 0.80, Setup 0.76 → 0.87, Testing 0.79 → 0.88.
   - This is not a holdout estimate (the holdout keeps its noisy labels). It locates the gap: most of the distance to 0.8 on the weak classes sits on rows where the teacher's labels are inconsistent.
5. **Reading the error samples (§5 of the report) by hand:**
   - Many errors are short follow-ups whose meaning depends on the conversation ("I meant this in the __init__ method" labelled Review; "Is there another, non-blocking way?" labelled Researching).
   - Others are defensible either way ("would it be faster to search all tables at the same time" is Researching for the teacher, Optimize for the model; "Whats the process of updating workflows … give me database models" is Architecting vs Researching).
   - Clear model errors exist too, e.g. "convert this code to fetch the data in parallel across regions" (teacher Optimize, model Feature dev).
- **Conclusion for the note:** for Review, Architecting, Refactoring and Optimize, a large part of the gap to 0.8 is label collision, not model capacity. Reaching 0.8 on the holdout for those classes is unlikely for any model that sees only the prompt.

### Colab: Qwen3-0.6B + LoRA failed at start
- Colab ships torchao 0.10, and peft 0.21 refuses torchao < 0.16 when injecting LoRA layers. We do not use torchao, so the notebook now uninstalls it.
- **Second Colab failure: CUDA out of memory** (A100 40 GB) on the first step of Qwen3-0.6B + LoRA: fp32 base, batch 16, 1,024 tokens, and the length-grouped sampler starts with the longest batch.
  - **Fix for the LoRA configs:**
    - the frozen base is loaded in bf16, while the adapters and the new head are kept in fp32 and the loss is still computed in fp32;
    - gradient checkpointing is on;
    - max length is 512 (about 90% of prompts fit whole; head+tail for the rest);
    - `expandable_segments` is set to reduce fragmentation.
  - **Verified locally:** a bf16 base with fp32 adapters under autocast runs forward and backward with checkpointing; all 393 trainable tensors get fp32 gradients. The full job passes the smoke test.

### Colab Qwen3-0.6B + LoRA, fold 0: chance-level validation score → prediction-order bug (17:50–18:00)

- **What we saw:** fold 0 trained normally (loss 4.4 → about 0.13 over 3 epochs, 33 min on A100), but scored val macro-F1 0.099 and accuracy 0.176. That is chance level: random agreement is Σp² ≈ 0.18 (EDA §5).
- **Cause (my bug):** `train_sampling_strategy="group_by_length"` in transformers 5.18 is also applied by `_get_eval_sampler`, so `Trainer.predict()` returned predictions in length-grouped order, not dataset order. The model was fine; the saved rows were shuffled.
  - The earlier smoke tests could not catch it: 12 steps on 400 rows also gives chance-level scores.
- **Fix:**
  - `finetune.py` predicts with its own loop: batches are length-sorted for speed and the original order is restored. `done.json` now carries `predict_order`.
  - Verified locally: deberta-v3-xsmall, 150 steps on 1,200 rows → val accuracy 0.49 (chance 0.18).
- **Recovery without retraining:** `src/fix_ft_order.py` rebuilds the deterministic sampler order (same tokenisation, eval batch size, seed 42) and inverts it. It only writes if the repaired order scores clearly above chance.
- **Second, smaller bug:** head+tail truncation broke when max_len < 2 × 128 tail tokens (local smoke tests only; Colab used 512). Fixed.
- Shay stopped the Colab run during fold 1 (old code).

### Qwen3-0.6B + LoRA fold 0: repaired and evaluated (19:00)

- `fix_ft_order.py` rebuilt the Colab prediction order locally, so it is reproducible across torch versions. Fold 0 went from macro-F1 0.099 (shuffled) to **0.729, min-F1 0.571, accuracy 0.801**.
- Comparison on the same rows (fold 0, n = 4,959, macro / min F1):
  - stack-C 0.743 / 0.500
  - Qwen3-8B-instr embedding + LR 0.733 / 0.566
  - **Qwen3-0.6B + LoRA 0.729 / 0.571**
  - stack-C + LoRA (meta-LR, inner 5-fold CV within fold 0) 0.763 / 0.586
  - average of stack-C and LoRA probabilities 0.758 / 0.596
- A 0.6B model fine-tuned with LoRA matches the 8B embedding model. It is the strongest single model on Review and Optimize, and adding it to the stack gives about +0.02 macro-F1 and +0.09 min-F1 on fold 0.
  - Single fold, so the CIs are wide.
  - Training loss reached about 0.13–0.2 by epoch 3 (memorisation), so 2 epochs may do as well at two-thirds of the cost.

---

## Block 6/7 — Qwen3-0.6B + LoRA, all 5 folds, and stacking (22:00)

- Colab finished folds 1–4 with the fixed code. Fold 1 resumed from a checkpoint left by the stopped run; training is unchanged between versions.
  - Per fold macro / min F1: 0.729 / 0.571 (fold 0, repaired), 0.714 / 0.594, 0.719 / 0.542, 0.732 / 0.602, 0.735 / 0.594. About 22–33 min per fold on an A100.
- `src/08_collect_ft.py` assembles the folds into OOF scores, with the holdout as the mean of the 5 fold models, and registers the result as `ft-qwen3-0.6b-lora`.
  - It refuses folds whose prediction order is not verified.

| Model | macro-F1 | min-F1 | accuracy | classes ≥ 0.8 |
|---|---|---|---|---|
| ft-qwen3-0.6b-lora (single model) | 0.726 | **0.603** | 0.798 | 3 |
| stack-C (no fine-tune) | 0.750 | 0.584 | 0.822 | 4 |
| **stack-E** = stack-C bases + LoRA | **0.766** | **0.635** | **0.833** | 4 |
| stack-F = Qwen3-8B-instr LR + LoRA | 0.764 | 0.624 | 0.831 | 5 |

- **Stack-E vs stack-C:** macro-F1 +0.016 (95% CI 0.004–0.028, p = 0.004); McNemar p ≈ 3e-13.
  - Significant per-class gains: Review +0.051 (p = 0.02), Architecting +0.030 (p = 0.02), Feature dev, Researching.
  - Optimize and Testing do not move.
- **Stack-E per class, with 95% CI at holdout scale:**
  - Other 0.895, Bug fix 0.866 (0.849–0.881), Feature dev 0.838, Researching 0.830, Testing 0.799 (0.741–0.853), Setup 0.762 (0.725–0.799), Refactoring 0.719 (0.674–0.766), Architecting 0.670 (0.623–0.718), Optimize 0.648 (0.515–0.767), Review 0.635 (0.577–0.693).
  - Overall: macro-F1 0.766 (0.749–0.785), min-F1 0.635 (0.515–0.670).
- **The class bias again lowers both metrics** when cross-fitted (stack-E+bias 0.754 / 0.627), so it is not used.

---

## Day 2 (2026-10-05) — larger fine-tunes and an XGBoost meta-model

- **Shay's requests:**
  1. fine-tune Qwen3-8B with and without LoRA, 3 epochs, 5 folds;
  2. try XGBoost as the stacking meta-model.
- **Full fine-tuning of Qwen3-8B does not fit one A100 40 GB.** bf16 weights 16 GB + gradients 16 GB + fp32 Adam 64 GB + fp32 master 32 GB ≈ 130 GB. Even an 8-bit optimizer needs ≈ 50 GB; CPU offload needs ≈ 96 GB RAM, and Colab has ≈ 83 GB.
  - So the LoRA-vs-full comparison is done on 0.6B and 1.7B (1.7B full ≈ 27 GB + activations). Qwen3-8B is fine-tuned with LoRA.
- **New `finetune.py` configs:**
  - LoRA: `qwen3-4b-lora`, `qwen3-8b-lora`, and `qwen3-emb-8b-lora` (the Qwen3-Embedding-8B model, our best embedding, as a fine-tuning base).
  - Full fine-tuning: `qwen3-0.6b-full`, `qwen3-1.7b-full` (fp32 master weights, bf16 autocast, gradient checkpointing).
- **`--bench` mode:** N real training steps with no checkpoints and no predictions. It writes seconds per step, peak GPU memory and the estimated time per fold to `bench/<model>.json`. Tested locally.
- **`notebooks/colab_ft_benchmark.ipynb`:** benchmarks 8B-LoRA, Emb-8B-LoRA, 4B-LoRA, 1.7B-full and 0.6B-full, 40 steps each, before the GPU is committed to a 5-fold run.

### XGBoost as the stacking meta-model (09:15)

- Installed `libomp` with Homebrew (Shay approved); XGBoost needs the OpenMP runtime on macOS.
- **Setup:** depth-3 trees, eta 0.05, subsample and colsample 0.8, early stopping on an inner stratified 10% split of each training part. The best iteration counts per fold were 162–186, very stable. Cross-fitted on the same folds as the LR meta-model; the holdout model uses the mean (176 trees).

| Stack | LR meta (macro / min F1) | XGBoost meta | Δ macro-F1 (95% CI) | McNemar p |
|---|---|---|---|---|
| C | 0.750 / 0.584 | 0.742 / 0.569 | −0.008 (−0.019 to 0.003) | 0.99 |
| E | 0.766 / 0.635 | 0.766 / 0.633 | 0.000 (−0.011 to 0.012) | 0.14 |
| F | 0.764 / 0.624 | 0.761 / 0.618 | −0.003 (−0.013 to 0.006) | 0.86 |

- **No gain from XGBoost.** Per class the two meta-models are within ±0.01 (Setup +0.012 with XGBoost), and macro PR-AUC is identical (0.844).
- **Why:** the meta-features are already calibrated class scores that combine almost linearly, so there are no interactions left for trees to find.
- **Decision:** keep the logistic-regression meta-model. It is simpler, and its coefficients show which base model is trusted for which class.
- The class bias again hurts once cross-fitted.

### Fine-tuning benchmark on A100 (Colab, 40 steps per model) and choice of GPU (09:30)

| Model | s/step | peak GB | per fold, 3 ep | 5 folds, 3 ep |
|---|---|---|---|---|
| Qwen3-0.6B full | 0.30 | 9.8 | 22 min | 1.8 h |
| Qwen3-1.7B full | 0.63 | 31.5 | 42 min | 3.5 h |
| Qwen3-4B LoRA | 1.33 | 10.7 | 86 min | 7.2 h |
| Qwen3-8B LoRA | 1.35 | 18.9 | 87 min | 7.2 h |
| Qwen3-Emb-8B LoRA | 1.36 | 18.8 | 87 min | 7.2 h |

- 4B and 8B LoRA run at the same speed: the bottleneck is gradient checkpointing and small batches, not compute. 8B therefore costs no more than 4B.
- Shay has access to an H100 (80 GB). Added `--no-gc` (gradient checkpointing off) and `--accum` overrides.
- The benchmark notebook now compares Qwen3-8B LoRA as-is, with no gc, and with no gc + one batch of 16 (same effective batch). Then the 5-fold run uses the fastest setting that fits (`EXTRA` in the fine-tune notebook).
- TPUs were not considered: the code is PyTorch/CUDA.
