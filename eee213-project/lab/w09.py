"""EEE213 Week 09 lab - machine learning on the power-plant (CCPP) data.

Run from the repository folder:  python lab/w09.py
Data: data/ccpp.csv (Tufekci and Kaya, UCI ML Repository, doi:10.24432/C5002N, CC BY 4.0)
Split (fixed by the course): first 7,654 rows = training, last 1,914 rows = test.
"""
import numpy as np

# ---------- load ----------
d = np.genfromtxt("data/ccpp.csv", delimiter=",", skip_header=1)
names = ["AT", "V", "AP", "RH", "PE"]
units = ["degC", "cm Hg", "mbar", "%", "MW"]

# ---------- Section 3: data-quality checks ----------
print("=== Section 3: data quality ===")
print("rows:", d.shape[0], " columns:", d.shape[1])
print("missing values:", int(np.isnan(d).sum()))
n_unique = np.unique(d, axis=0).shape[0]
print("exact duplicate rows:", d.shape[0] - n_unique)
for j in range(5):
    print(f"{names[j]}: min {d[:, j].min():.2f}  max {d[:, j].max():.2f} {units[j]}")

# ---------- Section 4: summary statistics and correlation (full dataset) ----------
print("\n=== Section 4: statistics (all rows) ===")
for j in [0, 4]:
    col = d[:, j]
    print(f"{names[j]}: mean {col.mean():.2f}  median {np.median(col):.2f}  "
          f"std {col.std(ddof=1):.2f} {units[j]}")
for j in range(4):
    r = np.corrcoef(d[:, j], d[:, 4])[0, 1]
    print(f"r({names[j]}, PE) = {r:.3f}")

# ---------- Section 5: models, trained on training rows, tested once ----------
n_train = 7654
train, test = d[:n_train], d[n_train:]
y_train, y_test = train[:, 4], test[:, 4]
print("\n=== Section 5: models ===")
print("training rows:", len(train), " test rows:", len(test))


def mae(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))


# 5.1 baseline: always predict the training mean
mean_pe = y_train.mean()
print(f"baseline: training mean PE = {mean_pe:.2f} MW, test MAE = {mae(y_test, mean_pe):.2f} MW")

# Week 09 reference line from AT
a_at, b_at = np.polyfit(train[:, 0], y_train, 1)
print(f"line from AT: PE = {a_at:.3f}*AT + {b_at:.2f}, "
      f"test MAE = {mae(y_test, a_at * test[:, 0] + b_at):.2f} MW")

# 5.2 line from V (exhaust vacuum)
a, b = np.polyfit(train[:, 1], y_train, 1)
pred = a * test[:, 1] + b
print(f"line from V:  PE = {a:.4f}*V + {b:.2f}, test MAE = {mae(y_test, pred):.2f} MW")

# 5.5 one prediction by hand: first test row (file line = index + 2, header is line 1)
i = 0
line_in_file = n_train + i + 2
x, true = test[i, 1], test[i, 4]
p = a * x + b
print(f"hand check: file line {line_in_file}, V = {x:.2f} cm Hg, "
      f"predicted {p:.2f} MW, true {true:.2f} MW, |error| {abs(true - p):.2f} MW")
