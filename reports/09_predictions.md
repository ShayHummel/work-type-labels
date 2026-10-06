# Holdout predictions — `stack-H-perfold`

Written to `predictions.jsonl`: 6199 rows, 6199 unique ids, 10 of 10 labels used. All format checks passed.

## Predicted class mix (holdout) vs teacher mix (train)

|              |   holdout predicted |   train (teacher) |   holdout predicted % |   train % |
|:-------------|--------------------:|------------------:|----------------------:|----------:|
| Bug fix      |                 897 |              3536 |                  14.5 |      14.3 |
| Feature dev  |                1417 |              5574 |                  22.9 |      22.5 |
| Refactoring  |                 226 |               971 |                   3.6 |       3.9 |
| Architecting |                 216 |              1016 |                   3.5 |       4.1 |
| Researching  |                1244 |              4782 |                  20.1 |      19.3 |
| Testing      |                 110 |               436 |                   1.8 |       1.8 |
| Review       |                 163 |               736 |                   2.6 |       3   |
| Optimize     |                  44 |               155 |                   0.7 |       0.6 |
| Setup        |                 304 |              1214 |                   4.9 |       4.9 |
| Other        |                1578 |              6372 |                  25.5 |      25.7 |

## Expected per-class F1 (out-of-fold on train, 95% CI for a holdout-sized sample)

|              |   expected F1 (OOF) | 95% CI (holdout scale)   |   p(F1≤0.8) |
|:-------------|--------------------:|:-------------------------|------------:|
| Bug fix      |               0.879 | 0.863–0.894              |       0     |
| Feature dev  |               0.863 | 0.849–0.875              |       0     |
| Refactoring  |               0.746 | 0.703–0.787              |       0.992 |
| Architecting |               0.711 | 0.664–0.752              |       1     |
| Researching  |               0.855 | 0.841–0.869              |       0     |
| Testing      |               0.834 | 0.779–0.885              |       0.114 |
| Review       |               0.665 | 0.613–0.723              |       1     |
| Optimize     |               0.685 | 0.568–0.800              |       0.979 |
| Setup        |               0.802 | 0.768–0.834              |       0.48  |
| Other        |               0.905 | 0.894–0.915              |       0     |

|                       |   value |      lo |      hi |   p-value |
|:----------------------|--------:|--------:|--------:|----------:|
| accuracy              |   0.854 |   0.844 |   0.862 |   nan     |
| macro-F1              |   0.794 |   0.777 |   0.812 |     0.005 |
| min-F1 (bar)          |   0.665 |   0.568 |   0.703 |     1.000 |
| classes with F1 ≥ 0.8 |   6.000 | nan     | nan     |   nan     |
| macro ROC-AUC         |   0.985 | nan     | nan     |   nan     |
| macro PR-AUC          |   0.868 | nan     | nan     |   nan     |
