# Work-type labels

You have nine hours. Submit once.

A product chart shows the mix of developer work over 30 days. Each prompt gets one label. Viewers compare slice sizes. The labels below are the full set the chart uses. In this data, the slice called Debugging is **Bug fix**. Do not add another label.

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

## Files

- `train.jsonl` — 24,792 prompts with `id`, `text`, and `label`.
- `holdout.jsonl` — 6,199 prompts with `id` and `text`. No labels.

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
