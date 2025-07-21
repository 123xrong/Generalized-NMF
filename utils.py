import numpy as np
from sklearn.metrics import accuracy_score, adjusted_rand_score, normalized_mutual_info_score
from sklearn.linear_model import Lasso
from sklearn.cluster import SpectralClustering
from sklearn.metrics.pairwise import cosine_similarity, rbf_kernel
from sklearn.decomposition import NMF as SklearnNMF
from sklearn.utils.linear_assignment_ import linear_assignment
from scipy.optimize import linear_sum_assignment
from numpy.linalg import qr

def ssc_func(X, K, alpha=0.01):
    """Basic SSC implementation with spectral clustering."""
    n = X.shape[0]
    C = np.zeros((n, n))
    for i in range(n):
        x_i = X[i]
        X_others = np.delete(X, i, axis=0)
        lasso = Lasso(alpha=alpha, fit_intercept=False, max_iter=1000)
        lasso.fit(X_others.T, x_i)
        c_i = np.insert(lasso.coef_, i, 0)
        C[i] = c_i
    affinity = np.abs(C) + np.abs(C.T)
    clustering = SpectralClustering(n_clusters=K, affinity='precomputed', random_state=0)
    return clustering.fit_predict(affinity)

def sparse_nmf(X_block, r, l1_reg=0.1, W_init=None, H_init=None, n_iter=200):
    W = np.random.rand(X_block.shape[0], r) if W_init is None else W_init
    H = np.random.rand(r, X_block.shape[1]) if H_init is None else H_init
    for _ in range(n_iter):
        WH = W @ H
        H *= (W.T @ X_block) / (W.T @ WH + l1_reg + 1e-10)
        W *= (X_block @ H.T) / (W @ (H @ H.T) + 1e-10)
    return W, H

def approximate_gpca(X, K, affinity='cosine', gamma=20):
    """
    Approximate GPCA using affinity + spectral clustering.
    """
    if affinity == 'cosine':
        S = cosine_similarity(X.T)
    elif affinity == 'rbf':
        S = rbf_kernel(X.T, gamma=gamma)
    else:
        raise ValueError("Unsupported affinity type")

    np.fill_diagonal(S, 0)
    clustering = SpectralClustering(n_clusters=K, affinity='precomputed', random_state=0)
    return clustering.fit_predict(S)

def projective_nmf(X, r, max_iter=200, tol=1e-4):
    """
    Projective NMF: X ≈ W Wᵗ X, with W ≥ 0 and Wᵗ W = I (softly)
    """
    d, n = X.shape
    W = np.abs(np.random.randn(d, r))

    for it in range(max_iter):
        WtX = W.T @ X
        WWtX = W @ WtX
        num = X @ WtX.T
        denom = WWtX @ WtX.T + 1e-10
        W *= num / denom

        loss = np.linalg.norm(X - W @ W.T @ X, 'fro')**2
        if it % 50 == 0:
            print(f"[ProjNMF] Iter {it}: Loss = {loss:.4f}")
    return W, loss


def projective_nmf_orthogonal(X, r, max_iter=200, tol=1e-4, verbose=False):
    """
    Projective NMF with hard orthogonality constraint: WᵗW = I, W ≥ 0.

    Args:
        X: (d, n) input matrix
        r: rank of factorization
        max_iter: number of iterations
        tol: tolerance for convergence
        verbose: print loss if True

    Returns:
        W: (d, r) nonnegative orthonormal basis matrix
        loss: final reconstruction loss
    """
    d, n = X.shape
    W = np.abs(np.random.randn(d, r))  # non-negative init

    for it in range(max_iter):
        # Compute gradient direction
        WtX = W.T @ X
        grad = -2 * X @ WtX.T + 2 * W @ WtX @ WtX.T

        # Gradient step
        W -= 0.01 * grad
        W = np.maximum(W, 1e-10)  # nonnegativity

        # Enforce orthogonality: QR + rectification
        Q, _ = np.linalg.qr(W)  # orthonormal columns
        W = np.maximum(Q, 0)    # project back to nonnegativity

        # Optional: normalize columns (not strictly necessary)
        W /= np.linalg.norm(W, axis=0, keepdims=True) + 1e-10

        # Compute reconstruction loss
        X_proj = W @ W.T @ X
        loss = np.linalg.norm(X - X_proj, 'fro')**2

        if verbose and it % 50 == 0:
            print(f"[Iter {it}] Loss: {loss:.4f}")

    return W, loss

def remap_accuracy(y_true, y_pred):
    """
    Compute clustering accuracy with optimal label permutation using Hungarian algorithm.
    
    Parameters:
        y_true (array-like): Ground truth labels, shape (n_samples,)
        y_pred (array-like): Predicted cluster labels, shape (n_samples,)
    
    Returns:
        acc (float): Clustering accuracy after optimal label alignment
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    assert y_true.shape == y_pred.shape
    D = max(y_pred.max(), y_true.max()) + 1
    cost_matrix = np.zeros((D, D), dtype=np.int64)

    for i in range(len(y_true)):
        cost_matrix[y_pred[i], y_true[i]] += 1

    # Solve the assignment problem
    row_ind, col_ind = linear_sum_assignment(-cost_matrix)  # maximize total match

    # Create label mapping
    optimal_mapping = dict(zip(row_ind, col_ind))

    # Map predicted labels
    y_pred_aligned = np.array([optimal_mapping[yi] for yi in y_pred])

    # Compute accuracy
    acc = accuracy_score(y_true, y_pred_aligned)
    return acc