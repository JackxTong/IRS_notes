# pip install scikit-learn patsy
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import log_loss, brier_score_loss, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import SplineTransformer

# ---------------------------
# 0) Inputs: X (DataFrame), y (Series)
# ---------------------------
# X: columns include 'x1' and others ('x2'...'xp')
# y: realized Y (same length as X)

assert 'x1' in X.columns, "X must contain column 'x1'"

# ---------------------------
# 1) Build the binary target: Z = 1{Y > x1}
# ---------------------------
Z = (y.values > X['x1'].values).astype(int)

# ---------------------------
# 2) Train/valid split (stratify to balance classes)
# ---------------------------
X_train, X_test, Z_train, Z_test = train_test_split(
    X, Z, test_size=0.25, random_state=42, stratify=Z
)

# ---------------------------
# 3) Identify column types
#    (Assumes mostly numeric; extend 'cat_cols' if you have categoricals)
# ---------------------------
numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = [c for c in X.columns if c not in numeric_cols]

# Ensure x1 is numeric
if 'x1' in cat_cols:
    raise ValueError("'x1' must be numeric to use spline features")

# ---------------------------
# 4) Build transformers
#    - Spline on x1 (flexible shape for the dominant driver)
#    - Impute+scale numerics
#    - Optional: one-hot on categoricals
# ---------------------------
spline_x1 = Pipeline(steps=[
    ('selector', ColumnTransformer(
        transformers=[('take_x1', 'passthrough', ['x1'])],
        remainder='drop'
    )),
    ('imputer', SimpleImputer(strategy='median')),
    ('spline', SplineTransformer(n_knots=7, degree=3, include_bias=False))
])

other_nums = Pipeline(steps=[
    ('selector', ColumnTransformer(
        transformers=[('take_num', 'passthrough', [c for c in numeric_cols if c != 'x1'])],
        remainder='drop'
    )),
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

cats = Pipeline(steps=[
    ('selector', ColumnTransformer(
        transformers=[('take_cat', 'passthrough', cat_cols)],
        remainder='drop'
    )),
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
]) if cat_cols else None

# Column-wise assembly
transformers = [
    ('x1_spline', spline_x1, X.columns),    # will internally pick only 'x1'
    ('other_nums', other_nums, X.columns)   # will internally pick numeric != x1
]
if cats:
    transformers.append(('cats', cats, X.columns))  # will internally pick cat_cols

features = ColumnTransformer(transformers=transformers, remainder='drop')

# ---------------------------
# 5) Optional interactions
#    Applying PolynomialFeatures after the column transformer creates
#    pairwise interactions among the transformed features (including
#    interactions between spline(x1) and the other predictors).
#    Set 'add_interactions' = False to disable.
# ---------------------------
add_interactions = True
poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False) if add_interactions else 'passthrough'

# ---------------------------
# 6) Classifier
# ---------------------------
logit = LogisticRegression(
    solver='saga', penalty='l2', C=1.0, max_iter=5000, n_jobs=-1
)

# ---------------------------
# 7) Full pipeline
# ---------------------------
pipe = Pipeline(steps=[
    ('features', features),
    ('interact', poly),
    ('clf', logit),
])

# ---------------------------
# 8) Hyperparameter search (lightweight)
#    - Tune number of spline knots for x1
#    - Tune regularization strength C
# ---------------------------
param_grid = {
    # Access nested step via named steps:
    'features__x1_spline__spline__spline__n_knots': [5, 7, 9, 11],
    'clf__C': [0.1, 0.3, 1.0, 3.0]
}

# Note: If this param path is confusing, you can skip GridSearchCV and just fit pipe.
# The path is: features (ColumnTransformer) -> x1_spline (Pipeline)
# -> 'spline' step inside that pipeline -> SplineTransformer's n_knots

search = GridSearchCV(
    estimator=pipe,
    param_grid=param_grid,
    scoring='neg_log_loss',   # log loss is a good proper scoring rule
    cv=5,
    n_jobs=-1,
    verbose=0
)

search.fit(X_train, Z_train)

best_model = search.best_estimator_

# ---------------------------
# 9) Evaluation
# ---------------------------
proba_val = best_model.predict_proba(X_test)[:, 1]
logloss = log_loss(Z_test, proba_val)
brier = brier_score_loss(Z_test, proba_val)
auc = roc_auc_score(Z_test, proba_val)

print("Best params:", search.best_params_)
print(f"LogLoss: {logloss:.4f}")
print(f"Brier:   {brier:.4f}")
print(f"AUC:     {auc:.4f}")

# Example: predict for new X_new (same columns as X)
# proba_new = best_model.predict_proba(X_new)[:, 1]
