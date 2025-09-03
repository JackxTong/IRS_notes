Create a per-row binary label

\( Z_i = 1\{ Y_i > x_{1i} \}, \)

and fit a logistic/probit model

\[
\Pr(Z_i = 1 \mid x_{1i}, \ldots, x_{pi}) = 
\sigma\!\big( f_1(x_{1i}) + \beta_2 x_{2i} + \cdots + \beta_p x_{pi} + \text{interactions} \big).
\]

Notes:

- Use a **flexible \( f_1 \)** for the dominant \(x_1\) (splines/GAM), and allow **interactions \(x_1 \times x_j\)** if plausible.
- Enforce **monotonicity in \(x_1\)** if theory says \(\Pr(Y > x_1)\) decreases with \(x_1\) (monotone GAMs, XGBoost monotone constraints, isotonic pieces).
- Evaluate with **log loss / Brier score, AUC, and calibration** (reliability curve).

This directly estimates the probability you want at each observed \(x_1\); no threshold bootstrapping required.


Notes & options:

- **Monotonicity in \(x_1\)** (probability should decrease as \(x_1\) increases) isn’t enforced by plain logistic + splines. If you need it, use:
  - **XGBoost** with monotonic constraints on \(x_1\), or
  - a **monotone GAM** library, or
  - post-hoc **isotonic calibration** on predictions along \(x_1\).
- If you don’t want global interactions, set `add_interactions = False`.
- If \(x_1\) is so dominant that other features add tiny signal, consider **L2 (ridge)** or **elastic-net** to stabilize.
