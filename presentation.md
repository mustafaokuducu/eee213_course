# Presentation

## Week 10 — Polynomial regression (method #2)

Slides: [slides_week10.pdf](slides_week10.pdf)

- Task: regression — predict net output PE (MW) from ambient temperature AT (°C).
- Hyperparameter: polynomial degree d = 1…6, chosen by 5-fold cross-validation on the training rows: d = 4.
- Result: test MAE 3.99 MW (baseline 14.72 MW, Week 09 straight line 4.33 MW).

Figures are produced by `python lab/w10_figures.py`; numbers by `python lab/w10_poly.py`.

[Back to home](index.md)
