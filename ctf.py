import numpy as np
import pandas as pd
from scipy.optimize import minimize

# Suppose df is your DataFrame:
# columns: ['feature1', 'feature2', ..., 'MIFID']
# features = all columns except 'MIFID'

# Extract features and target
features = [c for c in df.columns if c != "MIFID"]
X = df[features].values   # shape (n_samples, n_features)
y = df["MIFID"].values    # MIFID price

# Add intercept to features
X = np.column_stack([np.ones(len(X)), X])  # prepend column of 1s
n_features = X.shape[1]

# Logistic pdf log-likelihood
def nll(params):
    beta_mu = params[:-1]        # coefficients for mu(z)
    log_s = params[-1]           # log scale (to ensure positivity)
    s = np.exp(log_s)

    mu = X @ beta_mu             # linear predictor for mu(z)
    z = (y - mu) / s

    # log pdf of logistic distribution
    log_pdf = -z - np.log(s) - 2*np.log1p(np.exp(-z))
    return -np.sum(log_pdf)

# Initialize params: beta_mu zeros, log_s=0
init_params = np.zeros(n_features + 1)

res = minimize(nll, init_params, method="BFGS")
beta_mu = res.x[:-1]
s = np.exp(res.x[-1])

print("Fitted beta_mu:", beta_mu)
print("Fitted scale s:", s)

# Functions to get slope & intercept at new features z
def get_ab(z_row):
    """z_row: 1d array of features (without intercept)"""
    z_row = np.insert(z_row, 0, 1.0)  # add intercept
    mu = z_row @ beta_mu
    a = -1.0 / s
    b = mu / s
    return a, b
    
    
    
    
    
z_new = df[features].iloc[0].values
a, b = get_ab(z_new)
print("Slope a:", a, "Intercept b:", b)


from scipy.special import expit  # logistic sigmoid

ctf = expit(a * x_quote + b)
