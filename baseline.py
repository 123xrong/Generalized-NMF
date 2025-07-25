import numpy as np

from utils import *
from libnmf.gnmf import GNMF
from libnmf.nmfbase import NMFBase
from sklearn.cluster import KMeans
from sklearn.decomposition import NMF
from sklearn.preprocessing import normalize
from scipy.optimize import nnls
from sklearn.metrics import accuracy_score, adjusted_rand_score, normalized_mutual_info_score

# implement orthogonal NMF


def onmf(X, K, true_labels=None, max_iter=100, tol=1e-4, verbose=False, random_state=42):
    """
    Orthogonal NMF (ONMF) with clustering via argmax(H).
    """
    m, n = X.shape
    norm_X = np.linalg.norm(X, 'fro')**2

    # Step 1: Initialize W with nonnegative orthonormal basis
    U, _, _ = np.linalg.svd(X @ X.T)
    W = np.maximum(U[:, :K], 1e-8)
    W = normalize(W, axis=0)

    # Step 2: Initialize H using NNLS
    H = np.zeros((K, n))
    for i in range(n):
        H[:, i], _ = nnls(W, X[:, i])

    prev_loss = None

    for it in range(max_iter):
        # --- Update H ---
        for i in range(n):
            H[:, i], _ = nnls(W, X[:, i])

        # --- Update W via Procrustes (orthogonal update + nonnegativity) ---
        A = X @ H.T
        U, _, Vt = np.linalg.svd(A, full_matrices=False)
        W = np.maximum(U @ Vt, 1e-8)
        W = normalize(W, axis=0)

        # --- Compute normalized loss ---
        rec = W @ H
        rec_loss = np.linalg.norm(X - rec, 'fro')**2
        total_loss = rec_loss / norm_X

        if verbose and it % 10 == 0:
            print(f"[Iter {it}] Normalized Loss: {total_loss:.4f}")

        if prev_loss is not None and abs(prev_loss - total_loss) < tol:
            break
        prev_loss = total_loss

    # --- Clustering via argmax(H) ---
    pred_labels = H.argmax(axis=0)

    acc = remap_accuracy(true_labels, pred_labels) if true_labels is not None else None
    ari = adjusted_rand_score(true_labels, pred_labels) if true_labels is not None else None
    nmi = normalized_mutual_info_score(true_labels, pred_labels) if true_labels is not None else None
    recon_error = np.linalg.norm(X - W @ H) / np.linalg.norm(X)

    return acc, ari, nmi, recon_error


def GNMF_clus(X, K, true_labels, max_iter=1000, random_state=None, lmd=0, weight_type='heat-kernel', param=0.3):
    """
    Graph-based Non-negative Matrix Factorization (GNMF) for subspace clustering.
    """
    # base = NMFBase(X, K)
    model = GNMF(X, K)
    model.compute_factors(max_iter=max_iter, lmd=lmd, weight_type=weight_type, param=param)

    W = model.W
    H = model.H

    H_array = np.asarray(H)
    predicted_labels = KMeans(n_clusters=K, random_state=random_state).fit_predict(H_array.T)

    acc = remap_accuracy(true_labels, predicted_labels)
    ari = adjusted_rand_score(true_labels, predicted_labels)
    nmi = normalized_mutual_info_score(true_labels, predicted_labels)
    reconstruction_error = np.linalg.norm(X - W @ H) / np.linalg.norm(X)

    return acc, ari, nmi, reconstruction_error

