# Work-type labels

You have nine hours. Submit once.

## Why Zuzai is doing this

Zuzai sits in front of the coding agents a company already uses (Cursor, Claude Code, and the others on the same screen). The home view tells a customer where the money and the time went. Cost, speed, and policy violations are meter readings. They do not say what the developers were actually doing.

The missing piece is the mix of work over 30 days. One label on each prompt a developer typed. The chart compares slice sizes. A customer should be able to see that a team spent the month on bug fixing, or on feature work, or on research, and not on a guess. The slice the product calls Debugging is **Bug fix** in this data. These ten labels are the full set. Do not add another one.

The chart is only useful if every slice is real. A model that is right on Bug fix and Feature dev and wrong on Review and Optimize draws a donut that lies about the small slices. That is the bar below.

## Labels

| Label | Meaning |
|---|---|
| Bug fix | Something is broken: errors, crashes, why it does not work, and the patch. |
| Feature dev | Add or implement behavior. |
| Refactoring | Restructure without changing what it does. |
| Architecting | Design, boundaries, tradeoffs, how it should be built. |
| Researching | Explain, read, compare, how something works. |
| Testing | Write or run tests. |
| Review | Is this right, audit, critique a diff. |
| Optimize | Make it faster or smaller. A refactor that is really about speed. |
| Setup | Install, config, CI, deploy. |
| Other | Not a coding task. |

If two labels both fit, pick the main request. `Other` is only for text that is not a coding task.

## What we already ran

Use this. Do not spend the day rediscovering it.

The teacher labels are from a strong model, with `Other` reserved for text that is not a coding task. A bag-of-words model (fastText) on this train set lands near **0.50** macro F1 and **0.62** accuracy. A small encoder is the step that moved the number. DistilBERT, three epochs, max length 256, class weights inverse to how often each label appears, scored on this same holdout:

| Label | F1 | Support |
|---|---:|---:|
| Bug fix | 0.749 | 885 |
| Feature dev | 0.684 | 1,381 |
| Refactoring | 0.496 | 262 |
| Architecting | 0.474 | 247 |
| Researching | 0.701 | 1,221 |
| Testing | 0.649 | 128 |
| Review | 0.423 | 171 |
| Optimize | 0.455 | 49 |
| Setup | 0.581 | 316 |
| Other | 0.822 | 1,539 |

Accuracy **0.695**. Macro F1 **0.603**.

We also generated about 4,300 extra prompts in the shape of the weak classes and trained fastText again. Macro F1 moved from 0.494 to 0.501. More text of that kind is not the lever. A regex over words like "refactor" and "faster" is not either. Those words sit on both sides of the hard pairs.

One run of a larger encoder collapsed to predicting Bug fix for every row. The cause was mixed precision together with the class-weight loss (`Half` vs `Float` in the loss). If you fine-tune, keep that loss in float32.

## Files

- `train.jsonl` — 24,792 prompts with `id`, `text`, and `label`.
- `holdout.jsonl` — 6,199 prompts with `id` and `text`. No labels. Same rows as the table above.

## What you turn in

`predictions.jsonl` — one row per holdout id:

```json
{"id": "...", "label": "Bug fix"}
```

Every holdout id appears once. Every label is one of the ten names above.

`note.md` — what you did, what you would not do again, and the per-class F1 you believe you earned. You do not have the holdout labels. We score you.

## Bar

**0.8 F1 on every label.** All ten. The holdout is scored once.

Train only on `train.jsonl`. You may add training text you create. Do not put holdout rows into training. Do not drop a label to raise the average.
