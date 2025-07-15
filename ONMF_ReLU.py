import numpy as np
from sklearn.preprocessing import normalize
from scipy.optimize import nnls

def relu(x):
    return np.maximum(0, x)

def onmf_with_relu(X, r, max_iter=100, lambda_reg=0.1, tol=1e-4, verbose=False):
    m, n = X.shape

    # Initialize W with orthonormal columns using SVD
    U, _, _ = np.linalg.svd(X @ X.T)
    W = np.maximum(U[:, :r], 1e-8)
    W = normalize(W, axis=0)  # ensure orthonormality approximately

    # Initialize H using NNLS
    H = np.zeros((r, n))
    for i in range(n):
        H[:, i], _ = nnls(W, X[:, i])

    for iter in range(max_iter):
        # --- Update H ---
        for i in range(n):
            # Solve: min_h ||X[:, i] - W h||^2 + lambda * ||ReLU(h)||_1 s.t. h >= 0
            # Since h >= 0 and ReLU(h) = h, this reduces to NNLS + L1:
            # min_h ||X[:, i] - W h||^2 + lambda * ||h||_1  s.t. h >= 0
            from scipy.optimize import minimize

            def obj(h):
                return np.linalg.norm(X[:, i] - W @ h)**2 + lambda_reg * np.sum(h)

            bounds = [(0, None)] * r
            res = minimize(obj, H[:, i], bounds=bounds, method='L-BFGS-B')
            H[:, i] = res.x

        # --- Update W ---
        # Solve: min_W ||X - WH||_F^2 s.t. W^T W = I
        # Use SVD-based projection
        A = X @ H.T
        U, _, Vt = np.linalg.svd(A, full_matrices=False)
        W = U @ Vt

        # Enforce nonnegativity
        W = np.maximum(W, 1e-8)

        # Normalize columns to keep orthonormality approximately
        W = normalize(W, axis=0)

        # Compute loss
        reconstruction = W @ H
        loss = np.linalg.norm(X - reconstruction, 'fro')**2 + lambda_reg * np.sum(H)

        if verbose:
            print(f"Iter {iter:03d}: loss = {loss:.4f}")

        if iter > 0 and abs(prev_loss - loss) < tol:
            break
        prev_loss = loss

    return W, H
