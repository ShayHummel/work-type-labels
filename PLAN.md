# Experiment plan — work-type labels

Status: **draft for review** (2026-10-04 11:10 IDT, 2h40m of the 9h budget used; hard stop 17:28).
Companion files: `LOG.md` (what was done), `reports/` (results), `src/` (code).

---

## 1. Goal and success criteria

| | |
|---|---|
| Task | One label per prompt (the `text` field), 10 fixed labels, no label added or dropped. |
| Target | The teacher labels (an LLM) in `train.jsonl`. The holdout is scored against the same teacher. |
| Bar | **F1 ≥ 0.8 on every label** on the holdout, scored once by Zuzai. |
| Reference | DistilBERT (Zuzai): macro-F1 0.603, min-F1 0.423 (Review). fastText: about 0.50. |
| Deliverables | `predictions.jsonl` (6,199 rows, one valid label each) and `note.md` (what was tried, what not to repeat, per-class F1 we believe we earned, where labels collide). |

**Model selection criterion**, in order:
1. **min-F1** across the 10 classes (the bar).
2. Number of classes with F1 ≥ 0.8.
3. macro-F1.
4. Inference cost (the model must be usable behind the chart).

A difference only counts if the paired bootstrap p-value is < 0.05 (§3.4).

## 2. Data and validation design

- **Training data:** `train.jsonl` only (24,792 rows). Holdout rows are never used for fitting, tuning, thresholds, or vocabulary/IDF statistics. The holdout support column in TASK.md is not used.
- **Unit of classification:** a single prompt. No conversation or neighbour features (decision 2026-10-04; see LOG Step 1).
- **Folds:** `data/folds.csv`, 5-fold StratifiedKFold, shuffled, seed 42. Class shares per fold are within 0.02 pp of overall. A row-level random split mirrors how the holdout was drawn (EDA §5).
- **Out-of-fold (OOF) protocol:** every model produces OOF class scores for all 24,792 rows, saved to `data/oof/<exp_id>.npy` (n × 10, LABELS order). All metrics, comparisons, ensembles and threshold tuning use OOF scores only.
- **Holdout predictions:** for every promoted experiment, the same 5 fold models (or one model refit on all train, for the cheap ones) score the holdout, saved to `data/holdout_scores/<exp_id>.npy`. Only the final pipeline writes `predictions.jsonl`.
- **Preprocessing:** none beyond what each model needs. Text is kept as is: no lowercasing for encoders, no stripping of code or non-English text.
  - Truncation variants for encoders: **head 512** (default) vs **head 384 + tail 128**. EDA: 10–12% of prompts exceed 512 tokens, concentrated in Review and Refactoring.

## 3. Evaluation protocol (`src/evaluation.py`)

### 3.1 Metrics, for every experiment
- **Per class:** precision, recall, F1 (argmax and, separately, after threshold tuning, §6), support.
- **Per class, ranking:** ROC-AUC and PR-AUC (average precision, one-vs-rest), with the chance level (prevalence).
- **Overall:** accuracy, macro-F1, **min-F1**, number of classes with F1 ≥ 0.8, macro ROC-AUC, macro PR-AUC.
- **Confusion matrix:** counts plus row-normalised, as a figure. ROC and PR curves per class.

### 3.2 Confidence intervals
95% stratified bootstrap at **holdout scale**: 6,199 rows with train class proportions (e.g. 39 Optimize rows), 1,000 replicates; 200 replicates for the AUCs. This is the expected spread of a holdout score from sampling alone. For Optimize it is wide, roughly ±0.1 F1.

### 3.3 p-values
| Test | Question |
|---|---|
| `p(F1≤0.8)`, one-sided bootstrap | Does this class meet the bar? |
| Mann–Whitney U | Is ROC-AUC > 0.5? |
| Permutation (200) | Is macro-F1 above chance? |

### 3.4 Comparing two models (`compare()`)
- Paired stratified bootstrap of the per-class F1 difference and the macro-F1 difference (CI and p).
- McNemar's exact test on per-row correctness.

With about 40 experiments compared, p-values are read as evidence, not proof. Decisive comparisons (choosing the final model) are reported with the Holm-corrected p-value.

### 3.5 Results registry
Every run appends one row to `reports/results.csv`: exp_id, features, model, params, runtime (fit + predict), and all §3.1 numbers. `reports/03_experiments.md` is generated from it.

## 4. Features

| ID | Feature | Details | Where / cost |
|---|---|---|---|
| F-W | TF-IDF words 1–3-grams | sublinear tf, min_df=2, max_df=0.9, unicode word tokens, about 1M features before min_df | Local, < 1 min |
| F-C | TF-IDF char 3–5-grams (`char_wb`) | Robust to typos, variants and non-English text | Local, about 2 min |
| F-WC | F-W ⊕ F-C | Sparse hstack | Local |
| E-BGE | BGE-M3 dense (1024-d) | max_len 1024, CLS pooling, L2-normalised | Local, about 3–6 min |
| E-Q06 | Qwen3-Embedding-0.6B (1024-d) | Last-token pooling; with and without task instruction ("Instruct: Classify the developer request by type of work\nQuery: …") | Local, about 9–15 min |
| E-Q4 | Qwen3-Embedding-4B (2560-d) | Same, max_len 1024 | **Colab**, 5–15 min on A100/L4 |
| E-Q8 | Qwen3-Embedding-8B (4096-d) | Same | **Colab**, needs ≥ 24 GB GPU |
| E-NC | Nomic Embed Code 7B (3584-d) | Query prefix as on the model card. Code-retrieval model; expected weaker on intent, kept as a measured contrast | **Colab** |
| J | JEV probabilities (10-d) | open-jev-deberta-v3-large, one *choice* question over 10 labels and their TASK.md definitions | Local, minutes (to benchmark) |
| COMB-early | Sparse TF-IDF ⊕ dense embedding | Embedding block scaled by a tuned weight α so neither block dominates | Local |
| COMB-late | Average or stack of class probabilities from separate models | See §5.4 | Local |

Embeddings are computed once, cached in `data/emb/<model>.npy` for train and holdout (gitignored), and reused by every classifier. Runtime measures are in `reports/03_embed_benchmark.md`.

## 5. Models

### 5.1 Linear models on fixed features (local, 5-fold, seconds to minutes each)
| ID | Model | Grid | Notes |
|---|---|---|---|
| RIDGE | `RidgeClassifier` | alpha ∈ {0.3, 1, 3, 10} | One-vs-rest regression on ±1; the decision function serves as the score for the AUCs |
| LR | `LogisticRegression` (multinomial, saga/lbfgs) | C ∈ {0.3, 1, 3, 10}, class_weight ∈ {None, balanced} | Gives probabilities |
| SVM | `LinearSVC` | C ∈ {0.03, 0.1, 0.3, 1}, class_weight ∈ {None, balanced} | Scores by decision function; Platt calibration only if needed for stacking |
| KNN | Cosine kNN on embeddings | k ∈ {5, 15, 50}, distance-weighted | Also a direct probe of teacher self-consistency |

Grids are tuned by inner CV on the training part of each fold, so the OOF score is not tuned on its own rows. If too slow, the grid is fixed on fold 0 and declared.

### 5.2 Fine-tuned encoders (Colab GPU; on this Mac about 40–60 min per run)
| ID | Model | Setup |
|---|---|---|
| FT-DEB | DeBERTa-v3-base (then large if time) + FC head | 3–4 epochs, lr 2e-5, max_len 512, batch 16, warmup 10%, weighted CE **computed in fp32** (TASK.md collapse), bf16 autocast for the forward pass only |
| FT-MB | ModernBERT-large + FC head | Same recipe, max_len 1024 (native 8k context, code in pretraining) |
| FT-MM | mmBERT-base (multilingual ModernBERT) | Only if non-English errors are material |
| FT-QL | Qwen3-0.6B/1.7B (or Qwen3-Embedding-0.6B) + classification head, LoRA r=16 | Decoder as classifier; strong on noisy intent labels, long prompts and code |
| SETFIT | SetFit on BGE-M3 or Qwen3-0.6B | Contrastive fine-tune then LR head; targets small classes |

**Compute policy:** run fold 0 first for each candidate (selection). Run all 5 folds only for the 1–2 candidates that enter the final ensemble, so their OOF scores exist for stacking and thresholds.

### 5.3 JEV and LLM classifiers
| ID | Model | Setup |
|---|---|---|
| JEV-ZS | open-jev-deberta-v3-large, zero-shot | Scored on all 24,792 train rows (no training, so no fold needed). Limits: 256-token state, English only |
| JEV-SJ | simple-jev on Qwen3.5-0.8B, zero-shot, then LoRA (RFDT) on folds | Colab; only if JEV-ZS or J-features are promising |
| LLM | Claude Haiku 4.5, few-shot with the label table | **Pending decision** (cost, production fit, "train only on train.jsonl"). If approved: a 2,000-row stratified train sample to estimate agreement first |

### 5.4 Ensemble and decision layer
- **Stacking:** a multinomial LR meta-model on the concatenated OOF scores of the best 2–4 base models, cross-fitted on the same folds.
- **Per-class decision bias (§6)** applied last.

## 6. Decision layer: per-class thresholds and biases

Argmax favours large classes. We learn an additive bias vector b (10 values) on log-probabilities, prediction = argmax(log p + b), chosen by coordinate ascent to maximise **min-F1 first, macro-F1 second**.
- To avoid an optimistic estimate it is **cross-fitted**: b is learned on 4 folds' OOF and applied to the 5th. The reported score is the cross-fitted one.
- The final b is learned on all OOF and applied to the holdout.
- It uses train only. The holdout class mix is not used.

## 7. Class imbalance handling (compared on the best feature set)

None vs `class_weight=balanced` vs (fine-tunes) weighted CE / focal loss (γ=2) / logit adjustment (τ=1). All losses are computed in fp32.

## 8. Label-noise and error analysis (feeds `note.md`)

1. **Confident learning (cleanlab):** on the best OOF probabilities, list rows whose teacher label is likely inconsistent, with counts per class pair.
2. **Near-duplicate collisions:** pairs with cosine ≥ 0.9 and different teacher labels (EDA §4: 28% of near-duplicate pairs). Report how many of the model's errors fall on such rows.
3. **Per hard pair** (Bug fix/Feature dev, Review/Researching, Refactoring/Optimize, Architecting/Feature dev, Setup/Bug fix): confusion counts, 10 sampled errors each, classified by hand as a *model error* or a *label collision* (both labels defensible).
4. **Optional:** train on cleaned or soft labels and measure the effect on OOF scored against the **original** labels. The holdout keeps noisy labels, so cleaning may not raise the score.
5. **Per-class verdict for the note:** expected holdout F1 with 95% CI, `p(F1≤0.8)`, and whether a shortfall is attributable to collisions.

## 9. Runtime optimisation

- Embeddings computed once and cached. Batches sorted by length to minimise padding. fp16 on MPS/GPU.
- Sparse TF-IDF fitted inside each fold (no IDF leakage) and cached per fold. Linear models in parallel across folds (`joblib`, n_jobs=5).
- Grids kept small (§5.1).
- Fine-tunes and ≥ 4B embeddings on Colab via notebooks in `notebooks/` that read the jsonl files from the fork and write `.npy` results to download into `data/`.
- Every experiment logs wall-clock fit and predict time to `reports/results.csv`.

## 10. Experiment queue and timeline (remaining ≈ 6h20m)

| # | Block | Experiments | Where | Est. time | Gate to next |
|---|---|---|---|---|---|
| 1 | TF-IDF baselines | F-W, F-C, F-WC × RIDGE, LR, SVM | Local | 30 min | Reference table |
| 2 | Local embeddings | E-BGE, E-Q06 (± instruction) × RIDGE, LR, SVM, KNN; COMB-early with best TF-IDF | Local | 45 min | Pick best fixed-feature model |
| 3 | JEV zero-shot | JEV-ZS on train; J as extra features | Local | 30 min | Keep J if it adds a significant gain |
| 4 | Colab, run in parallel with 2–3 | Notebook A: E-Q4, E-Q8, E-NC embeddings. Notebook B: FT-DEB / FT-MB / FT-QL fold 0 | Colab | 1–2 h wall | Shay runs the notebooks |
| 5 | Large embeddings | E-Q4 / E-Q8 / E-NC × LR, SVM, COMB | Local | 30 min | — |
| 6 | Best fine-tune, 5 folds | 1–2 winners of block 4 | Colab | 1–1.5 h | — |
| 7 | Ensemble + decision layer | Stacking, cross-fitted per-class bias, imbalance variants | Local | 45 min | Final model chosen by §1 criterion |
| 8 | Error and noise analysis | §8 | Local | 45 min | — |
| 9 | Deliverables | Holdout predictions (format checks: 6,199 unique ids, valid labels, all 10 labels used), `note.md`, PR | Local | 45 min | Hard stop 17:28 |

Checkpoints with Shay after blocks 1, 2+3, 5+6 and 7.

## 11. Open decisions

1. Colab GPU tier (T4 / L4 / A100). This determines whether E-Q8 and DeBERTa-large are feasible.
2. LLM classifier (Claude Haiku) in or out of scope.
3. Acceptable inference cost for the production model (does a 7–8B embedding model qualify for the chart?).

## 12. Reproducibility

- Seeds fixed (42 for folds, 0 for bootstrap).
- `uv.lock` pins every package.
- One numbered script per block in `src/`, plus Colab notebooks in `notebooks/`.
- Large artefacts (embeddings, model weights) are gitignored but reproducible from the scripts.
