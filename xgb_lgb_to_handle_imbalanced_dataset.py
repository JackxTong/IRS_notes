# --- Imports
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score, average_precision_score, accuracy_score,
    balanced_accuracy_score, precision_recall_curve
)
from sklearn.utils.class_weight import compute_class_weight

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

# -------------------------------------------------------------------
# 1) Helpers
# -------------------------------------------------------------------
def compute_class_weight_dict(y_binary):
    classes = np.array([0, 1])
    w = compute_class_weight(class_weight='balanced', classes=classes, y=y_binary)
    return {int(c): float(weight) for c, weight in zip(classes, w)}  # e.g., {0: 2.8, 1: 0.5}

def build_sample_weight(y_binary, class_weight_dict):
    return y_binary.map(class_weight_dict).astype(float).values

def oof_proba(model, X, y, sample_weight=None, n_splits=5, random_state=42):
    """
    Generate out-of-fold probabilities (for fair evaluation) and collect per-fold models.
    Supports passing sample_weight into fit.
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    oof = np.zeros(len(y), dtype=float)
    models = []

    for tr_idx, va_idx in skf.split(X, y):
        X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
        y_tr, y_va = y.iloc[tr_idx], y.iloc[va_idx]

        if sample_weight is not None:
            sw_tr = sample_weight[tr_idx]
            model.fit(X_tr, y_tr, sample_weight=sw_tr)
        else:
            model.fit(X_tr, y_tr)

        proba = model.predict_proba(X_va)[:, 1]
        oof[va_idx] = proba
        models.append(model)

    return oof, models

def choose_best_threshold(y_true, prob, metric="accuracy"):
    """
    Find threshold that maximizes the requested metric (default: accuracy).
    """
    thresholds = np.unique(prob)
    best_thr, best_val = 0.5, -np.inf

    for thr in thresholds:
        pred = (prob >= thr).astype(int)
        if metric == "accuracy":
            val = accuracy_score(y_true, pred)
        elif metric == "balanced_accuracy":
            val = balanced_accuracy_score(y_true, pred)
        else:
            raise ValueError("Unsupported metric choice.")
        if val > best_val:
            best_val, best_thr = val, float(thr)
    return best_thr, best_val

def evaluate_predictions(y_true, prob, name="model"):
    # Default threshold 0.5
    pred_05 = (prob >= 0.5).astype(int)
    acc_05 = accuracy_score(y_true, pred_05)
    bacc_05 = balanced_accuracy_score(y_true, pred_05)
    auc = roc_auc_score(y_true, prob)
    pr_auc = average_precision_score(y_true, prob)

    # Best accuracy threshold
    thr_acc, best_acc = choose_best_threshold(y_true, prob, metric="accuracy")

    summary = {
        "model": name,
        "roc_auc": auc,
        "pr_auc": pr_auc,
        "accuracy@0.5": acc_05,
        "balanced_acc@0.5": bacc_05,
        "best_accuracy": best_acc,
        "best_threshold_for_accuracy": thr_acc
    }
    return summary

# -------------------------------------------------------------------
# 2) Class weights & sample weights
# -------------------------------------------------------------------
y_binary = y_bin.astype(int)
cw = compute_class_weight_dict(y_binary)            # e.g., {0: weight_for_neg, 1: weight_for_pos}
sw = build_sample_weight(y_binary, cw)              # per-sample weights (for models that use sample_weight)

print("Class distribution:", y_binary.value_counts().to_dict())
print("Class weights:", cw)

# -------------------------------------------------------------------
# 3) MODELS
# -------------------------------------------------------------------
# 3A) Logistic Regression (scaled, linear)
log_reg = make_pipeline(
    StandardScaler(with_mean=True, with_std=True),
    LogisticRegression(
        max_iter=1000,
        class_weight=cw,   # or 'balanced'
        solver="lbfgs",
        n_jobs=-1
    )
)

# 3B) XGBoost
xgb = XGBClassifier(
    n_estimators=800,
    learning_rate=0.05,
    max_depth=5,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    objective="binary:logistic",
    eval_metric="logloss",
    tree_method="hist",
    n_jobs=-1,
    random_state=42
)
# NOTE: For XGB we use sample_weight (so the minority class is upweighted).

# 3C) LightGBM
lgb = LGBMClassifier(
    n_estimators=1200,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    num_leaves=63,
    reg_lambda=1.0,
    class_weight=cw,   # LightGBM accepts dict class_weight directly
    random_state=42,
    n_jobs=-1
)

# 3D) CatBoost
cat = CatBoostClassifier(
    iterations=1200,
    learning_rate=0.05,
    depth=6,
    loss_function="Logloss",
    eval_metric="Logloss",
    verbose=False,
    random_seed=42,
    class_weights=[cw[0], cw[1]]  # order is [weight_for_class0, weight_for_class1]
)

# -------------------------------------------------------------------
# 4) OOF evaluation (fair), threshold tuning
# -------------------------------------------------------------------
results = []

# Logistic (class_weight inside model; no sample_weight needed)
oof_lr, _ = oof_proba(log_reg, X, y_binary, sample_weight=None, n_splits=5, random_state=42)
results.append(evaluate_predictions(y_binary, oof_lr, name="LogisticRegression"))

# XGBoost (pass sample_weight to fit)
oof_xgb, _ = oof_proba(xgb, X, y_binary, sample_weight=sw, n_splits=5, random_state=42)
results.append(evaluate_predictions(y_binary, oof_xgb, name="XGBoost"))

# LightGBM (class_weight dict in params)
oof_lgb, _ = oof_proba(lgb, X, y_binary, sample_weight=None, n_splits=5, random_state=42)
results.append(evaluate_predictions(y_binary, oof_lgb, name="LightGBM"))

# CatBoost (class_weights in params)
oof_cat, _ = oof_proba(cat, X, y_binary, sample_weight=None, n_splits=5, random_state=42)
results.append(evaluate_predictions(y_binary, oof_cat, name="CatBoost"))

# Show results
perf_df = pd.DataFrame(results).sort_values("best_accuracy", ascending=False)
print(perf_df)

# -------------------------------------------------------------------
# 5) Pick thresholds per model (by accuracy), refit on all data for deployment
# -------------------------------------------------------------------
thr_lr, _ = choose_best_threshold(y_binary, oof_lr, metric="accuracy")
thr_xgb, _ = choose_best_threshold(y_binary, oof_xgb, metric="accuracy")
thr_lgb, _ = choose_best_threshold(y_binary, oof_lgb, metric="accuracy")
thr_cat, _ = choose_best_threshold(y_binary, oof_cat, metric="accuracy")

# Fit final models on ALL data
log_reg.fit(X, y_binary)
xgb.fit(X, y_binary, sample_weight=sw)
lgb.fit(X, y_binary)                 # class_weight already inside
cat.fit(X, y_binary)                 # class_weights already inside

# Store thresholds for later prediction
thresholds = {
    "LogisticRegression": thr_lr,
    "XGBoost": thr_xgb,
    "LightGBM": thr_lgb,
    "CatBoost": thr_cat
}
print("Chosen thresholds (maximize accuracy):", thresholds)

# -------------------------------------------------------------------
# 6) Example: predict on new data with tuned thresholds
# -------------------------------------------------------------------
def predict_with_threshold(model, X_new, thr):
    prob = model.predict_proba(X_new)[:, 1]
    pred = (prob >= thr).astype(int)
    return pred, prob
    
    
import numpy as np

def get_logit(model, X, eps=1e-15):
    """
    Return the logit (log-odds) of predicted probabilities.
    logit(p) = log(p / (1 - p))
    """
    prob = model.predict_proba(X)[:, 1]
    prob = np.clip(prob, eps, 1 - eps)  # avoid log(0)
    return np.log(prob / (1 - prob))

# Example usage on your trained models:
logit_lr  = get_logit(log_reg, X)
logit_xgb = get_logit(xgb, X)
logit_lgb = get_logit(lgb, X)
logit_cat = get_logit(cat, X)

# Example usage:
# pred_lr, prob_lr = predict_with_threshold(log_reg, X_new, thresholds["LogisticRegression"])