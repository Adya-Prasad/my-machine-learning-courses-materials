"""
Kernel Ridge Regression (KRR) - Simple Numerical Example
--------------------------------------------------------
This script demonstrates how Kernel Ridge Regression works
using a small dataset, an RBF (Gaussian) kernel, and direct
matrix computations — without relying on ML libraries.

Core formulas implemented:
    α = (K + λI)⁻¹ y
    ŷ(x_new) = k(x_new)ᵀ α

Where:
    K_ij = exp(-||x_i - x_j||² / (2σ²))   # kernel matrix
    k(x_new)_i = exp(-||x_i - x_new||² / (2σ²))

Author: Adya’s Learning Session (Guided by GPT)
"""

import numpy as np

# -------------------------------------------------------
# Step 1. Define RBF (Gaussian) kernel
# -------------------------------------------------------
def rbf_kernel(x1, x2, sigma=1.0):
    """
    Compute the RBF (Gaussian) kernel value between two vectors.
    K(x1, x2) = exp(-||x1 - x2||² / (2σ²))
    """
    diff = np.linalg.norm(x1 - x2)
    return np.exp(-diff**2 / (2 * sigma**2))

# -------------------------------------------------------
# Step 2. Build the kernel matrix K for training data
# -------------------------------------------------------
def compute_kernel_matrix(X, sigma=1.0):
    """
    Compute full kernel matrix for training set X.
    K_ij = RBF(x_i, x_j)
    """
    m = len(X)
    K = np.zeros((m, m))
    for i in range(m):
        for j in range(m):
            K[i, j] = rbf_kernel(X[i], X[j], sigma)
    return K

# -------------------------------------------------------
# Step 3. Fit Kernel Ridge Regression
# -------------------------------------------------------
def fit_krr(X, y, lam=0.1, sigma=1.0):
    """
    Fit Kernel Ridge Regression model.
    
    Args:
        X: Training inputs (m x d)
        y: Target values (m x 1)
        lam: Regularization strength (λ)
        sigma: RBF kernel bandwidth (σ)
    
    Returns:
        alpha: Coefficient vector (m x 1)
        K: Kernel matrix used for fitting
    """
    K = compute_kernel_matrix(X, sigma)
    m = len(X)
    
    # α = (K + λI)^(-1) y
    alpha = np.linalg.inv(K + lam * np.eye(m)) @ y
    return alpha, K

# -------------------------------------------------------
# Step 4. Predict for a new point
# -------------------------------------------------------
def predict_krr(X_train, alpha, x_new, sigma=1.0):
    """
    Predict output for a new input using trained KRR model.
    
    Args:
        X_train: Training inputs (used to compute k(x_new))
        alpha: Coefficients from training
        x_new: New input vector
        sigma: Kernel bandwidth (σ)
    
    Returns:
        y_pred: Predicted scalar value
    """
    k_new = np.array([rbf_kernel(xi, x_new, sigma) for xi in X_train])
    y_pred = k_new.T @ alpha
    return y_pred

# -------------------------------------------------------
# Step 5. Run small demo
# -------------------------------------------------------
if __name__ == "__main__":
    # Tiny 1D dataset
    X_train = np.array([[1.0], [2.0], [3.0]])
    y_train = np.array([1.0, 3.0, 2.0])

    lam = 0.5    # regularization strength (λ)
    sigma = 1.0  # kernel width (σ)

    # Train model
    alpha, K = fit_krr(X_train, y_train, lam, sigma)

    print("Kernel matrix (K):\n", K)
    print("\n(K + λI):\n", K + lam * np.eye(len(X_train)))
    print("\nAlpha coefficients (α):\n", alpha)

    # Predict on new data
    x_new = np.array([2.5])
    y_pred = predict_krr(X_train, alpha, x_new, sigma)

    print(f"\nPrediction for x_new = {x_new[0]:.2f}: {y_pred:.4f}")
