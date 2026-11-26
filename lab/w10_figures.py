"""EEE213 Week 10 - figures for the polynomial-regression presentation.

Run from the repository folder:  python lab/w10_figures.py
Writes figures/fit.png, figures/cv.png, figures/compare.png
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BLUE, ORANGE, GRAY, TEXT, TEXT2 = "#257de0", "#f47230", "#b8b8b8", "#191919", "#666666"
plt.rcParams.update({"font.size": 13, "axes.edgecolor": "#cccccc", "axes.labelcolor": TEXT,
                     "xtick.color": TEXT2, "ytick.color": TEXT2, "axes.spines.top": False,
                     "axes.spines.right": False, "figure.dpi": 150})
os.makedirs("figures", exist_ok=True)

d = np.genfromtxt("data/ccpp.csv", delimiter=",", skip_header=1)
n_train = 7654
x_tr, y_tr = d[:n_train, 0], d[:n_train, 4]
x_te, y_te = d[n_train:, 0], d[n_train:, 4]
mae = lambda a, b: np.mean(np.abs(a - b))

# ---- Figure 1: data with degree-1 line and degree-4 curve ----
xs = np.linspace(x_tr.min(), x_tr.max(), 300)
c1, c4 = np.polyfit(x_tr, y_tr, 1), np.polyfit(x_tr, y_tr, 4)
fig, ax = plt.subplots(figsize=(8, 5))
ax.scatter(x_tr, y_tr, s=4, color=GRAY, alpha=0.5, linewidths=0, label="Training hours")
ax.plot(xs, np.polyval(c1, xs), color=BLUE, lw=2.5, label="Degree 1 (straight line)")
ax.plot(xs, np.polyval(c4, xs), color=ORANGE, lw=2.5, label="Degree 4 (chosen)")
ax.set_xlabel("Ambient temperature AT (°C)")
ax.set_ylabel("Net output PE (MW)")
ax.grid(axis="y", color="#eeeeee")
ax.legend(frameon=False, loc="upper right", markerscale=3)
fig.tight_layout(); fig.savefig("figures/fit.png"); plt.close(fig)

# ---- Figure 2: training and cross-validation error against degree ----
folds = np.array_split(np.arange(n_train), 5)
degs, tr_err, cv_err = range(1, 7), [], []
for deg in degs:
    e = []
    for k in range(5):
        fit = np.concatenate([folds[j] for j in range(5) if j != k])
        c = np.polyfit(x_tr[fit], y_tr[fit], deg)
        e.append(mae(y_tr[folds[k]], np.polyval(c, x_tr[folds[k]])))
    cv_err.append(np.mean(e))
    tr_err.append(mae(y_tr, np.polyval(np.polyfit(x_tr, y_tr, deg), x_tr)))
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(degs, tr_err, color=BLUE, lw=2, marker="o", ms=8, label="Training MAE")
ax.plot(degs, cv_err, color=ORANGE, lw=2, marker="o", ms=8, label="5-fold CV MAE")
best = int(np.argmin(cv_err))
ax.annotate(f"chosen: degree {best + 1}\nCV MAE {cv_err[best]:.2f} MW",
            xy=(best + 1, cv_err[best]), xytext=(best + 0.2, cv_err[best] + 0.12),
            color=TEXT, fontsize=12, arrowprops=dict(arrowstyle="-", color=TEXT2))
ax.set_xlabel("Polynomial degree d")
ax.set_ylabel("Mean absolute error (MW)")
ax.set_xticks(list(degs))
ax.grid(axis="y", color="#eeeeee")
ax.legend(frameon=False)
fig.tight_layout(); fig.savefig("figures/cv.png"); plt.close(fig)

# ---- Figure 3: test MAE comparison ----
cv_ = np.polyfit(d[:n_train, 1], y_tr, 1)
names = ["Baseline (training mean)", "Line from V", "Line from AT", "Polynomial, degree 4"]
vals = [mae(y_te, y_tr.mean()), mae(y_te, np.polyval(cv_, d[n_train:, 1])),
        mae(y_te, np.polyval(c1, x_te)), mae(y_te, np.polyval(c4, x_te))]
fig, ax = plt.subplots(figsize=(8, 4))
bars = ax.barh(names[::-1], vals[::-1], color=[ORANGE, BLUE, BLUE, BLUE], height=0.55)
for b, v in zip(bars, vals[::-1]):
    ax.text(v + 0.2, b.get_y() + b.get_height() / 2, f"{v:.2f} MW", va="center", color=TEXT)
ax.set_xlabel("Test MAE on the last 1,914 hours (MW) — shorter is better")
ax.set_xlim(0, 17)
ax.spines["left"].set_visible(False); ax.tick_params(axis="y", length=0)
fig.tight_layout(); fig.savefig("figures/compare.png"); plt.close(fig)

print("test MAE:", ", ".join(f"{n} {v:.2f}" for n, v in zip(names, vals)))
print("degree-4 coefficients (highest power first):", np.round(c4, 6))
