import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

# ===================================================================
# SIMPLE & PRACTICAL REGULARIZED POLYNOMIAL REGRESSION
# Lasso (L1) + Ridge (L2) using Coordinate Descent
# Goal: show how λ shrinks coefficients and Lasso creates sparsity
#       → prevents overfitting in high-degree polynomials
# ===================================================================

np.random.seed(0)

# -----------------------------
# 1. DATA (non-linear, noisy)
# -----------------------------
X = np.linspace(-3, 3, 30).reshape(-1, 1)
y = 0.5 * X**3 - X**2 + 2 + np.random.randn(30, 1) * 3

# -----------------------------
# 2. HIGH-DEGREE POLYNOMIAL FEATURES
# -----------------------------
degree = 15                                 # high enough to clearly overfit
poly = PolynomialFeatures(degree=degree, include_bias=True)
X_poly = poly.fit_transform(X)              # shape: (30, 16) → [1, x, x², ..., x¹⁵]

# -----------------------------
# 3. IMPORTANT: SCALE FEATURES (except intercept)
#    → makes λ meaningful across powers
# -----------------------------
scaler = StandardScaler()
X_scaled = np.hstack([
    X_poly[:, [0]],                         # keep intercept unscaled
    scaler.fit_transform(X_poly[:, 1:])     # scale x → x¹⁵
])

# -----------------------------
# 4. COORDINATE DESCENT HELPERS (clean & efficient)
# -----------------------------
def ridge_cd(X, y, lam=0.0, tol=1e-6, max_epochs=2000):
    n, d = X.shape
    w = np.zeros((d, 1))
    
    for epoch in range(max_epochs):
        w_old = w.copy()
        
        for j in range(d):
            # residual without current feature
            residual = y - X @ w + X[:, [j]] * w[j]
            rho = X[:, [j]].T @ residual
            
            if j == 0:                                      # ← NEVER regularize intercept
                denom = np.sum(X[:, j]**2) or 1
                w[j] = rho / denom
            else:
                denom = np.sum(X[:, j]**2) + lam
                w[j] = rho / denom
        
        # early stopping
        if np.max(np.abs(w - w_old)) < tol:
            break
            
    return w


def lasso_cd(X, y, lam=0.0, tol=1e-6, max_epochs=2000):
    n, d = X.shape
    w = np.zeros((d, 1))
    
    for epoch in range(max_epochs):
        w_old = w.copy()
        
        for j in range(d):
            residual = y - X @ w + X[:, [j]] * w[j]
            rho = X[:, [j]].T @ residual
            
            if j == 0:                                      # ← no regularization on bias
                denom = np.sum(X[:, j]**2) or 1
                w[j] = rho / denom
            else:
                denom = np.sum(X[:, j]**2)
                if rho < -lam:
                    w[j] = (rho + lam) / denom
                elif rho > lam:
                    w[j] = (rho - lam) / denom
                else:
                    w[j] = 0
        
        if np.max(np.abs(w - w_old)) < tol:
            break
            
    return w


# -----------------------------
# 5. TEST GRID (smooth curve)
# -----------------------------
X_test = np.linspace(-3, 3, 200).reshape(-1, 1)
X_test_poly = poly.transform(X_test)
X_test_scaled = np.hstack([
    X_test_poly[:, [0]],
    scaler.transform(X_test_poly[:, 1:])
])

# -----------------------------
# 6. TRY DIFFERENT λ VALUES
#    λ = 0  → no regularization (overfits)
#    λ > 0  → shrinkage / sparsity
# -----------------------------
lambdas = [0.0, 0.01, 0.1, 1.0, 10.0]

plt.figure(figsize=(12, 5))

for lam in lambdas:
    # train both models
    w_ridge = ridge_cd(X_scaled, y, lam=lam)
    w_lasso = lasso_cd(X_scaled, y, lam=lam)
    
    # predict
    y_ridge = X_test_scaled @ w_ridge
    y_lasso = X_test_scaled @ w_lasso
    
    # plot
    plt.plot(X_test, y_ridge, label=f'Ridge λ={lam}', linewidth=2)
    plt.plot(X_test, y_lasso, '--', label=f'Lasso λ={lam}', linewidth=2)

plt.scatter(X, y, color='black', label='Training data', zorder=5)
plt.title(f'High-Degree Polynomial (degree={degree}) → Regularization prevents overfitting')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# -----------------------------
# 7. COEFFICIENT ANALYSIS (the key insight)
# -----------------------------
print("\n" + "="*70)
print("COEFFICIENT SHRINKAGE & SPARSITY (how λ prevents overfitting)")
print("="*70)

for lam in lambdas:
    w_ridge = ridge_cd(X_scaled, y, lam=lam)
    w_lasso = lasso_cd(X_scaled, y, lam=lam)
    
    # sparsity = how many feature weights (excluding bias) are exactly zero
    sparsity_lasso = np.sum(np.abs(w_lasso[1:]) < 1e-8)
    
    print(f"\nλ = {lam}")
    print(f"  Ridge  coeffs : {np.round(w_ridge.ravel(), 3)}")
    print(f"  Lasso  coeffs : {np.round(w_lasso.ravel(), 3)}")
    print(f"  Lasso non-zero features : {16 - sparsity_lasso} / 15")   # 15 features + bias

print("\nTakeaway:")
print("   • λ = 0   → wild coefficients → overfitting")
print("   • Ridge   → all coeffs shrink smoothly")
print("   • Lasso   → many coeffs become exactly zero → sparsity + simpler model")
print("   • Higher λ → stronger regularization → smoother predictions")