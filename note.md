# Work-type labels — note

> **DRAFT.** Numbers marked ⟨TBD⟩ are filled in once the Qwen3-8B + LoRA folds 1–4 are back. Everything else is
> final. Full step-by-step record: `LOG.md`; plan: `PLAN.md`; reports: `reports/`.

## Result in one paragraph

The submitted model is ⟨TBD: final model⟩. Measured out-of-fold on all 24,792 training rows, it reaches **macro-F1
⟨TBD⟩, min-F1 ⟨TBD⟩, accuracy ⟨TBD⟩**, against 0.603 / 0.423 / 0.695 for the DistilBERT reference in TASK.md.
**⟨TBD⟩ of the ten labels reach 0.8. The others do not**, and I do not expect them to on the holdout. For Review,
Architecting, Refactoring and Optimize, most of the gap sits on prompts where the teacher's own labels collide: near-identical
prompts carry different labels about a third of the time in exactly these classes (§4). All ten labels are predicted, and
the predicted mix on the holdout matches the training mix to within 0.6 percentage points per class.

## Per-class F1 I believe I earned

Out-of-fold F1 on train, with the 95% interval expected for a holdout-sized sample (6,199 rows, bootstrap). The holdout is
the same distribution, sampled row by row from the same conversations (EDA §5), so these are my estimate of the holdout score.

| Label | Expected F1 | 95% CI | vs 0.8 | Why |
|---|---|---|---|---|
| Other | ⟨TBD⟩ | | ✅ real | Teacher consistent (97% agreement on near-duplicates) |
| Bug fix | ⟨TBD⟩ | | ✅ real | |
| Feature dev | ⟨TBD⟩ | | ✅ real | |
| Researching | ⟨TBD⟩ | | ✅ real | |
| Testing | ⟨TBD⟩ | | ≈ at the bar | 436 train rows; wide CI |
| Setup | ⟨TBD⟩ | | short | Setup / Bug fix / Researching overlap ("ModuleNotFoundError" is both) |
| Refactoring | ⟨TBD⟩ | | short — label collision | 65% teacher self-agreement on near-duplicates |
| Optimize | ⟨TBD⟩ | | short — label collision + 155 rows | 72% self-agreement; CI ±0.12 |
| Architecting | ⟨TBD⟩ | | short — label collision | 64% self-agreement |
| Review | ⟨TBD⟩ | | short — label collision | 68% self-agreement; 20% of its rows flagged by cleanlab |

*(Current best, stack-E, for reference: Other 0.895, Bug fix 0.866, Feature dev 0.838, Researching 0.830, Testing 0.799,
Setup 0.762, Refactoring 0.719, Architecting 0.670, Optimize 0.648, Review 0.635.)*

## What I did

**Evaluation first.**
- 5-fold stratified CV on train; every model produces out-of-fold scores for all rows.
- Per class: precision, recall, F1, ROC-AUC, PR-AUC and the confusion matrix, with bootstrap CIs at holdout scale.
- p-values: `p(F1 ≤ 0.8)` per class, and paired bootstrap plus McNemar tests between models.
- Nothing from the holdout was used beyond running the final model on its text. That includes the support column in
  TASK.md, whose meaning I could not verify.
- Conversation order would have helped (rows are in conversation order; EDA §5), but I did not use it: the classifier
  must label a single prompt, as it will in production.

| Step | Pipeline | Best macro / min F1 |
|---|---|---|
| Bag of words | TF-IDF words 1–3 + chars 3–5 → LogReg / SVM / Ridge | 0.558 / 0.267 |
| Small embeddings | BGE-M3, Qwen3-Emb-0.6B (± task instruction) → LogReg / SVM / Ridge / kNN | 0.602 / 0.368 |
| Large embeddings | Qwen3-Emb-4B / 8B / 8B + instruction, nomic-embed-code → same classifiers | 0.740 / 0.573 (8B + instruction, LogReg) |
| Fine-tune 0.6B | Qwen3-0.6B + LoRA, classification head, 512 tokens, 3 epochs, class-weighted loss in fp32 | 0.726 / 0.603 |
| Fine-tune 8B | Qwen3-8B + LoRA, same recipe | fold 0: 0.779 / 0.607; 5 folds ⟨TBD⟩ |
| Stacking | Out-of-fold scores of the base models → cross-fitted LogReg meta-model | 0.766 / 0.635 (stack-E); ⟨TBD with 8B⟩ |

**What moved the number, in order:**
1. **Model scale.** Embedding 0.6B → 8B: +0.15 macro-F1. Fine-tune 0.6B → 8B: +0.05 on fold 0.
2. **The task instruction for the embedding model:** +0.07 macro and +0.16 min-F1 at 8B.
3. **Fine-tuning over frozen embeddings.**
4. **Stacking models that make different errors.**

## Where the number is real and where the labels collide

The teacher labels are the target, and they are not self-consistent in the weak classes. Evidence (`reports/07_error_analysis.md`, EDA §4/§4b):
- **Near-duplicates.** Among prompts whose nearest neighbour is a near-duplicate (Qwen3-8B cosine ≥ 0.95), the two carry the same teacher label
  97% of the time for Other and 88% for Bug fix, but only **64–68% for Architecting, Refactoring and Review**.
- **Examples of collisions:**
  - "can this code be improved?" is **Review**; "can you think of any ways to optimize this code?" is **Optimize**.
  - "Combine the last code with the initial code." is **Refactoring**; "now integrate this code into the previous code"
    is **Feature dev**.
  - "What attributes does an Air Shipment have to have?" is **Architecting**; "…a parcel shipment have?" is **Researching**;
    "…a full truckload need to have?" is **Other**.
- **Confident learning (cleanlab)** flags 16–20% of Review, Optimize, Architecting and Refactoring rows as likely label
  errors, against 4–7% for Other, Feature dev and Bug fix. 68–77% of the model's false positives in those classes fall on flagged rows.
- **On rows whose labels are self-consistent** (8% excluded), the earlier best model scores ≥ 0.80 on every class except Review
  (0.76). That is not a holdout estimate, since the holdout keeps its noisy labels, but it locates the gap: it is mostly in
  the labels, not in the model.
- **Many errors are short follow-ups** ("what did you change?", "Is there another, non-blocking way?") whose intent depends on the
  conversation. A single-prompt classifier cannot recover it, and the teacher may have seen the conversation.

## What I would not repeat

- **Tuning a per-class decision bias to maximise min-F1.** Cross-fitted, it lowered both macro and min F1 every time: it
  chases the noisiest class (Optimize: 31 rows per fold).
- **Early fusion** (sparse TF-IDF and a dense embedding in one LogReg). It took up to 90 minutes per run and gave the
  same score as stacking the two models' predictions, which takes one minute.
- **XGBoost as the meta-model.** No gain over LogReg on any stack (Δ macro-F1 −0.008 to 0.000): the inputs are already
  calibrated class scores.
- **kNN and code-retrieval embeddings** (nomic-embed-code). Weakest family, and no better than a general embedding of the
  same size.
- **Engineering:**
  - `Trainer.predict()` with length-grouped batches: transformers 5.x silently reorders predictions. It was caught by a
    chance-level validation score and repaired deterministically.
  - Writing GB-sized checkpoints to Google Drive.
  - Smoke tests too short to detect a wrong prediction order.

## What I would do next

- **Context:** feed the previous prompt of the session to the model (Zuzai has it in production). Many collisions are
  context-dependent follow-ups.
- **Teacher consistency:** re-label the colliding near-duplicate clusters with the teacher, using the label definitions
  and the conversation, before training. The bar cannot be met against labels that disagree with themselves.
- **Production model:** one fine-tune on all of train (the holdout predictions here average the 5 fold models).

## Reproduce

Commands are in `LOG.md`.
- `src/01_eda.py`: EDA.
- `src/02_folds.py`: folds and evaluation check.
- `src/04_tfidf_baselines.py`, `src/05_embedding_models.py`: blocks 1, 2 and 5.
- `src/embed.py`, `src/finetune.py`, `notebooks/colab_*.ipynb`: Colab jobs, resumable.
- `src/08_collect_ft.py`: collect fine-tuning folds.
- `src/06_stack.py`: stacking.
- `src/07_error_analysis.py`: error analysis.
- `src/09_predict.py <exp_id>`: writes `predictions.jsonl` with format checks.
