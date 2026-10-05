# Work-type labels — note

Full step-by-step record: `LOG.md` · plan: `PLAN.md` · reports and figures: `reports/` · code: `src/`, `notebooks/`.

## Result

`predictions.jsonl` labels all 6,199 holdout prompts. All ten labels are used, and the predicted mix matches the training
mix to within 0.8 percentage points per class.

Measured out of fold on all 24,792 training rows, the model reaches **macro-F1 0.794 (95% CI 0.777–0.812), min-F1 0.665
and accuracy 0.854**, against 0.603 / 0.423 / 0.695 for the DistilBERT reference in TASK.md.

**Six of the ten labels reach 0.8. Four do not: Refactoring, Architecting, Optimize and Review.** I do not expect them to
on the holdout either. In those four classes the teacher's own labels collide: near-identical prompts carry different
labels about a third of the time (see "Where the number is real").

## Per-class F1 I believe I earned

These are out-of-fold F1 scores on train. The 95% interval is what a holdout-sized sample (6,199 rows, bootstrap, train
class mix) would show. The holdout was drawn row by row from the same conversations as train (EDA §5), so this is my
estimate of the holdout score.

| Label | Expected F1 | 95% CI | P(F1 ≤ 0.8) | Verdict |
|---|---|---|---|---|
| Other | 0.905 | 0.894–0.915 | 0.00 | ✅ real |
| Bug fix | 0.879 | 0.863–0.894 | 0.00 | ✅ real |
| Feature dev | 0.863 | 0.849–0.875 | 0.00 | ✅ real |
| Researching | 0.855 | 0.841–0.869 | 0.00 | ✅ real |
| Testing | 0.834 | 0.779–0.885 | 0.11 | ✅ likely; small class (436 train rows) |
| Setup | 0.802 | 0.768–0.834 | 0.48 | ≈ at the bar: a coin flip on the holdout |
| Refactoring | 0.746 | 0.703–0.787 | 0.99 | ✗ label collision with Feature dev |
| Architecting | 0.711 | 0.664–0.752 | 1.00 | ✗ label collision with Researching and Feature dev |
| Optimize | 0.685 | 0.568–0.800 | 0.98 | ✗ collision with Refactoring, Review and Researching; ≈ 40 holdout rows |
| Review | 0.665 | 0.613–0.723 | 1.00 | ✗ label collision with Feature dev, Bug fix and Researching |

## The pipeline that worked

```
prompt text ─┬─► Qwen3-8B + LoRA classifier (fine-tuned) ────────────► 10 probabilities ─┐
             │                                                                          ├─► meta LogReg ─► label
             └─► Qwen3-Embedding-8B (task instruction) ─► LogReg ────► 10 probabilities ─┘
```

1. **Fine-tuned classifier** (`src/finetune.py`, Colab A100).
   - `Qwen/Qwen3-8B` with a new linear head over the last token: 4096 → 10.
   - The base is frozen in bf16. LoRA adapters (r = 16, α = 32, dropout 0.05) sit on every linear layer; the adapters and
     the head train in fp32.
   - Input: 512 tokens, head + last 128 tokens for longer prompts.
   - Training: 3 epochs, lr 5e-5, linear decay, 10% warm-up, effective batch 16.
   - Loss: cross-entropy with inverse-frequency class weights, computed in fp32 on fp32 logits (the failure mode TASK.md warns about).
   - About 1 h 43 min per fold on one A100 (40 GB).
   - Out of fold alone: 0.789 / 0.654.
2. **Embedding + linear model** (`src/embed.py`, `src/05_embedding_models.py`).
   - `Qwen/Qwen3-Embedding-8B` with the instruction *"Classify the developer's request to a coding assistant by the type
     of software work it asks for"*: 1,024 tokens, 4096-d, L2-normalised.
   - Multinomial LogReg on top (C = 100, no class weights; chosen on fold 0).
   - Out of fold alone: 0.740 / 0.573.
3. **Stacking** (`src/06_stack.py`).
   - A multinomial LogReg on the standardised log-probabilities of the two models (20 features), cross-fitted on the same 5 folds.
   - It learns per class how much to trust each model. It leans on the embedding model for the rare classes (Optimize,
     Refactoring, Setup) and weighs both equally for the large ones.
   - Gain over the fine-tune alone: +0.005 macro-F1 (not significant, p = 0.16), but it gets significantly more rows right
     (McNemar p ≈ 6e-7) and lifts Setup over 0.8.
4. **Holdout** (`src/10_final_holdout.py`).
   - For each fold k, fine-tune k and an embedding LR trained on the same 4 folds score the holdout.
   - The meta-model is applied to that pair, and the 5 outputs are averaged.
   - This keeps the meta-model's inputs the same as in training. Feeding it the averaged fine-tune and a full-train LR
     instead shifted 2.9% of the labels.

## How I evaluated

- **Folds:** 5-fold stratified CV on train. Every model produces out-of-fold scores for all rows, and every comparison uses them.
- **Metrics:** per class precision, recall, F1, ROC-AUC, PR-AUC and the confusion matrix. Bootstrap CIs at holdout size.
- **Tests:** `P(F1 ≤ 0.8)` per class. Paired bootstrap and McNemar between models.
- **No holdout information was used** beyond running the final model on its text. That includes the "support" column in
  TASK.md, whose meaning I could not verify.
- **Conversation order was not used.** The rows are in conversation order, and the holdout rows sit between train rows of
  the same conversations (EDA §5), so neighbours would leak labels. I did not use them: the classifier has to label one
  prompt, as in production.

## What I tried

| Step | Pipeline | Best macro / min F1 |
|---|---|---|
| Bag of words | TF-IDF words 1–3 + chars 3–5 → LogReg / linear SVM / Ridge | 0.558 / 0.267 |
| Small embeddings | BGE-M3, Qwen3-Emb-0.6B (± instruction) → LogReg / SVM / Ridge / kNN | 0.602 / 0.368 |
| Early fusion | TF-IDF ⊕ 0.6B embedding in one LogReg | 0.640 / 0.410 |
| Large embeddings | Qwen3-Emb-4B, 8B, 8B + instruction, nomic-embed-code → same | 0.740 / 0.573 |
| Fine-tune 0.6B | Qwen3-0.6B + LoRA (same recipe, lr 1e-4) | 0.726 / 0.603 |
| Stack without 8B fine-tune | 8B-emb LR + SVM, TF-IDF, 0.6B-emb, 0.6B LoRA | 0.766 / 0.635 |
| **Fine-tune 8B** | Qwen3-8B + LoRA | **0.789 / 0.654** |
| **Final stack** | 8B LoRA + 8B-emb LR | **0.794 / 0.665** |

**What moved the number:**
1. Model scale: embeddings 0.6B → 8B gave +0.15; fine-tune 0.6B → 8B gave +0.06.
2. The task instruction for the embedding model: +0.07 macro, +0.16 min-F1 at 8B.
3. Fine-tuning over frozen embeddings.
4. Stacking models that make different errors (gains shrink once the 8B fine-tune is in).

## Where the number is real and where the labels collide

The target is the teacher's labels, and in four classes they disagree with themselves.

Evidence: `reports/07_error_analysis.md`, EDA §4 and §4b.
- **Near-duplicates.** For each prompt I found its most similar other prompt (Qwen3-8B cosine ≥ 0.95). The two share a
  label 97% of the time for Other and 88% for Bug fix, but only **64–68% for Architecting, Refactoring and Review**, and
  72% for Optimize. There are 212 such pairs with different labels.
- **Examples** (more in EDA §4b):
  - "can this code be improved?" → **Review**; "can you think of any ways to optimize this code?" → **Optimize**.
  - "Combine the last code with the initial code." → **Refactoring**; "now integrate this code into the previous code
    and put it together" → **Feature dev**.
  - "What attributes does an Air Shipment have to have?" → **Architecting**; "…a parcel shipment have?" →
    **Researching**; "…a full truckload need to have?" → **Other**.
  - "ModuleNotFoundError: No module named 'panel'" → **Setup**; "…'evdev'" → **Bug fix**.
- **Confident learning (cleanlab).** It flags 16–20% of Review, Optimize, Architecting and Refactoring rows as likely
  label errors, against 4–7% for Other, Feature dev and Bug fix. Most of the model's false positives in those classes
  (68–77%) fall on flagged rows.
- **Self-consistent rows.** With the 8% of rows that are flagged or collide set aside, an earlier stack scores ≥ 0.80 on
  every class except Review (0.76). The holdout keeps its noisy labels, so this is not a holdout estimate. It does show
  where the gap sits: in the labels more than in the model.
- **Follow-ups.** Many remaining errors are short follow-ups whose intent depends on the conversation ("what did you
  change?", "Is there another, non-blocking way?"). A single-prompt classifier cannot recover it.

## What I would not repeat

- **A per-class decision bias tuned to maximise min-F1.** Every time I cross-fitted it, it lowered both macro and min F1.
  It chases the noisiest class (Optimize: 31 rows per fold).
- **Early fusion** of sparse TF-IDF and a dense embedding in one LogReg. It took up to 90 minutes per run and gave the
  same score as stacking the two models, which takes one minute.
- **XGBoost as the meta-model.** No gain on any stack (Δ macro-F1 −0.008 to 0.000): the inputs are already calibrated
  scores that combine linearly.
- **kNN and a code-retrieval embedding** (nomic-embed-code). kNN was the weakest family because of label noise.
  nomic-embed-code was no better than a general embedding of similar size.
- **Engineering:**
  - `Trainer.predict()` with length-grouped batches: transformers 5.x silently reorders predictions. It was caught by a
    chance-level validation score and repaired from the deterministic order (`src/fix_ft_order.py`).
  - GB-sized checkpoints on Google Drive.
  - Smoke tests too short to catch a wrong prediction order.
- **Full fine-tuning of the 8B model was not possible on one 40 GB A100** (≈ 130 GB needed), so LoRA was the only option at that size.

## What I would do next

- **Give the model the previous prompt of the session.** Zuzai has it in production, and many collisions are
  context-dependent follow-ups.
- **Re-label the colliding near-duplicate clusters with the teacher**, using the label definitions and the conversation.
  0.8 on every label cannot be measured against labels that disagree with themselves.
- **Production model:** one fine-tune on all of train (the holdout here averages the 5 fold models). One 8B forward pass
  per prompt plus one embedding call.

## Reproduce

- `src/01_eda.py`: EDA.
- `src/02_folds.py`: folds and evaluation check.
- `src/04_tfidf_baselines.py`: TF-IDF baselines.
- `src/embed.py`, then `src/05_embedding_models.py`: embeddings and linear models.
- `src/finetune.py` via `notebooks/colab_finetune.ipynb`, then `src/import_colab.py`, then `src/08_collect_ft.py`:
  fine-tuning.
- `src/06_stack.py`: stacking.
- `src/07_error_analysis.py`: error analysis.
- `src/10_final_holdout.py`, then `src/09_predict.py stack-H-perfold`: `predictions.jsonl`.

Environment: `uv sync`.
