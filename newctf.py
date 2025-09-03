import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from scipy.special import logit, expit

# X: all feature columns; y_bin: 1 if mifid_spread > 0 else 0
features = [c for c in df.columns if c != 'mifid_spread']
X = df[features]
y_bin = (df['mifid_spread'] > 0).astype(int)

# Probabilistic classifier (any calibrated model works; start simple)
clf = Pipeline([
    ('scaler', StandardScaler()),
    ('lr', LogisticRegression(max_iter=2000, class_weight='balanced'))
]).fit(X, y_bin)

def predict_b(X_new):
    p0 = clf.predict_proba(X_new)[:, 1]        # estimate P(Y>0 | z)
    p0 = np.clip(p0, 1e-6, 1-1e-6)             # numerical safety
    return logit(p0)                           # b = logit(p0)

# Example: get b for a new row and compute CTF at quote spread x
a = -0.8  # your fixed slope (should be negative)
row = X.iloc[[0]]
b_hat = predict_b(row)[0]

x_quote = 1.2
ctf = expit(a * x_quote + b_hat)