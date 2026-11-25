"""EEE213 Week 10 - polynomial regression on the power-plant data.

Run from the repository folder:  python lab/w10_poly.py
Model: PE = c_d*AT^d + ... + c_1*AT + c_0  (one feature: ambient temperature AT)
Hyperparameter: degree d, chosen by 5-fold cross-validation on the training rows only.
Test set (last 1,914 rows) is used once, at the end.
"""
import numpy as np

d = np.genfromtxt("data/ccpp.csv", delimiter=",", skip_header=1)
n_train = 7654
x_train, y_train = d[:n_train, 0], d[:n_train, 4]   # AT (degC), PE (MW)
x_test, y_test = d[n_train:, 0], d[n_train:, 4]


def mae(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))


# 5-fold cross-validation: split training rows into 5 consecutive parts
folds = np.array_split(np.arange(n_train), 5)
print("degree | train MAE | CV MAE  (MW)")
best_deg, best_cv = None, np.inf
for deg in range(1, 7):
    cv_errors = []
    for k in range(5):
        val = folds[k]
        fit = np.concatenate([folds[j] for j in range(5) if j != k])
        c = np.polyfit(x_train[fit], y_train[fit], deg)
        cv_errors.append(mae(y_train[val], np.polyval(c, x_train[val])))
    c_all = np.polyfit(x_train, y_train, deg)
    train_mae = mae(y_train, np.polyval(c_all, x_train))
    cv_mae = np.mean(cv_errors)
    print(f"  {deg}    |  {train_mae:.2f}    |  {cv_mae:.2f}")
    if cv_mae < best_cv:
        best_deg, best_cv = deg, cv_mae

# final model: chosen degree, fitted on all training rows, tested once
c = np.polyfit(x_train, y_train, best_deg)
print(f"\nchosen degree (lowest CV MAE): {best_deg}")
print(f"test MAE, degree {best_deg}: {mae(y_test, np.polyval(c, x_test)):.2f} MW")
print(f"baseline test MAE (training mean): {mae(y_test, y_train.mean()):.2f} MW")
