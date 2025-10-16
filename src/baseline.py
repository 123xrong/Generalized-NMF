import numpy as np

from src.utils import *
from .gnmf import GNMF
from .nmfbase import NMFBase
from sklearn.cluster import KMeans
from sklearn.decomposition import NMF
from sklearn.preprocessing import normalize
from scipy.optimize import nnls
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

def divide(a, b, eps=1e-10):
    """Element-wise safe division."""
    return a / (b + eps)

def initialize_onmf(X, k, random_state=42):
    m, n = X.shape
    X = np.maximum(X, 1e-8)
    rng = np.random.default_rng(random_state)

    # (A) K-means on columns → G
    km_cols = KMeans(n_clusters=k, n_init=10, random_state=random_state)
    col_labels = km_cols.fit_predict(X.T)
    G = np.zeros((n, k))
    G[np.arange(n), col_labels] = 1.0
    G = G + 0.2  # make strictly positive

    # (B) K-means on rows → F
    km_rows = KMeans(n_clusters=k, n_init=10, random_state=random_state)
    row_labels = km_rows.fit_predict(X)
    F = np.zeros((m, k))
    F[np.arange(m), row_labels] = 1.0
    F = F + 0.2  # strictly positive

    # (C) Initialize S via Eq.(17)
    S = F.T @ X @ G

    return F, S, G


def onmf_ding(X, k, true_labels=None, max_iter=500, tol=1e-5, verbose=False, random_state=42):
    """
    Orthogonal Nonnegative Matrix Tri-Factorization (ONMF-Ding 2006)
    with clustering and evaluation.

    Args
    ----
    X : array (m, n)
        Nonnegative data matrix (features × samples)
    k : int
        Number of clusters / latent factors
    true_labels : array (n,), optional
        Ground-truth labels for evaluation
    max_iter : int
        Maximum iterations
    tol : float
        Convergence tolerance
    verbose : bool
        Print progress every 50 iterations
    random_state : int
        Random seed
    """
    X = np.maximum(X, 1e-8)
    m, n = X.shape

    # --- Initialization (K-means + Eq 17) ---
    F, S, G = initialize_onmf(X, k, random_state)

    F_diff = G_diff = np.inf
    recon_error = np.inf
    it = 0

    while it < max_iter and (F_diff > tol or G_diff > tol):
        # --- Update F ---
        P = X @ G @ S.T
        denom = F @ F.T @ P
        F_next = F * np.sqrt(divide(P, denom))
        F_diff = np.linalg.norm(F - F_next, 'fro') / (np.linalg.norm(F, 'fro') + 1e-12)
        F = np.maximum(F_next, 1e-12)

        # --- Update G ---
        P = X.T @ F @ S
        denom = G @ G.T @ P
        G_next = G * np.sqrt(divide(P, denom))
        G_diff = np.linalg.norm(G - G_next, 'fro') / (np.linalg.norm(G, 'fro') + 1e-12)
        G = np.maximum(G_next, 1e-12)

        # --- Update S ---
        P = F.T @ X @ G
        denom = F.T @ F @ S @ G.T @ G
        S = np.maximum(S * np.sqrt(divide(P, denom)), 1e-12)

        if it % 50 == 0 or it == max_iter - 1:
            recon_error = np.linalg.norm(X - F @ S @ G.T, 'fro')
            if verbose:
                print(f"[Iter {it:03d}] F_diff={F_diff:.2e} G_diff={G_diff:.2e} Recon={recon_error:.4f}")
        it += 1

    # --- Final reconstruction & clustering ---
    X_hat = F @ S @ G.T
    recon_error = np.linalg.norm(X - X_hat, 'fro') / np.linalg.norm(X, 'fro')

    pred_labels = np.argmax(G, axis=1)

    if true_labels is not None:
        acc = remap_accuracy(true_labels, pred_labels)
        ari = adjusted_rand_score(true_labels, pred_labels)
        nmi = normalized_mutual_info_score(true_labels, pred_labels)
    else:
        acc = ari = nmi = None

    return acc, ari, nmi, recon_error


def GNMF_clus(X, K, r, true_labels, max_iter=1000, random_state=None, lmd=10, weight_type='heat-kernel', param=0.2):
    """
    Graph-based Non-negative Matrix Factorization (GNMF) for subspace clustering.
    """
    # base = NMFBase(X, K)
    model = GNMF(X, rank=r*K)
    model.compute_factors(max_iter=max_iter, lmd=lmd, weight_type=weight_type, param=param)

    W = model.W
    H = model.H

    H_array = np.asarray(H)
    predicted_labels = KMeans(n_clusters=K, random_state=random_state).fit_predict(H_array.T)
    print(predicted_labels.shape)

    acc = remap_accuracy(true_labels, predicted_labels)
    ari = adjusted_rand_score(true_labels, predicted_labels)
    nmi = normalized_mutual_info_score(true_labels, predicted_labels)
    reconstruction_error = np.linalg.norm(X - W @ H) / np.linalg.norm(X)

    return acc, ari, nmi, reconstruction_error, predicted_labels, W, H

