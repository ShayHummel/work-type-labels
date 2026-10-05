# Holdout predictions — `stack-E`

Written to `data/predictions_dryrun_stack-E.jsonl`: 6199 rows, 6199 unique ids, 10 of 10 labels used. All format checks passed.

## Predicted class mix (holdout) vs teacher mix (train)

|              |   holdout predicted |   train (teacher) |   holdout predicted % |   train % |
|:-------------|--------------------:|------------------:|----------------------:|----------:|
| Bug fix      |                 898 |              3536 |                  14.5 |      14.3 |
| Feature dev  |                1432 |              5574 |                  23.1 |      22.5 |
| Refactoring  |                 215 |               971 |                   3.5 |       3.9 |
| Architecting |                 215 |              1016 |                   3.5 |       4.1 |
| Researching  |                1238 |              4782 |                  20   |      19.3 |
| Testing      |                 107 |               436 |                   1.7 |       1.8 |
| Review       |                 155 |               736 |                   2.5 |       3   |
| Optimize     |                  48 |               155 |                   0.8 |       0.6 |
| Setup        |                 300 |              1214 |                   4.8 |       4.9 |
| Other        |                1591 |              6372 |                  25.7 |      25.7 |

## Expected per-class F1 (out-of-fold on train, 95% CI for a holdout-sized sample)

|              |   expected F1 (OOF) | 95% CI (holdout scale)   |   p(F1≤0.8) |
|:-------------|--------------------:|:-------------------------|------------:|
| Bug fix      |               0.866 | 0.849–0.881              |       0     |
| Feature dev  |               0.838 | 0.824–0.851              |       0     |
| Refactoring  |               0.719 | 0.674–0.766              |       1     |
| Architecting |               0.67  | 0.623–0.718              |       1     |
| Researching  |               0.83  | 0.814–0.846              |       0     |
| Testing      |               0.799 | 0.741–0.853              |       0.557 |
| Review       |               0.635 | 0.577–0.693              |       1     |
| Optimize     |               0.648 | 0.515–0.767              |       0.996 |
| Setup        |               0.762 | 0.725–0.799              |       0.981 |
| Other        |               0.895 | 0.884–0.905              |       0     |

|                       |   value |      lo |      hi |   p-value |
|:----------------------|--------:|--------:|--------:|----------:|
| accuracy              |   0.833 |   0.824 |   0.842 |   nan     |
| macro-F1              |   0.766 |   0.749 |   0.785 |     0.005 |
| min-F1 (bar)          |   0.635 |   0.515 |   0.670 |     1.000 |
| classes with F1 ≥ 0.8 |   4.000 | nan     | nan     |   nan     |
| macro ROC-AUC         |   0.983 | nan     | nan     |   nan     |
| macro PR-AUC          |   0.844 | nan     | nan     |   nan     |
