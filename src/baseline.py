import numpy as np

from src.utils import *
from .gnmf import GNMF
from .nmfbase import NMFBase
from sklearn.cluster import KMeans
from sklearn.decomposition import NMF
from sklearn.preprocessing import normalize
from scipy.optimize import nnls
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

def onmf_em(X, K, true_labels=None, n_iter=100, random_state=None):
    """
    Orthogonal NMF (ONMF) using EM-style updates (Pompili et al., 2014).
    
    Args:
        X: (d, n) nonnegative data matrix
        K: number of clusters
        true_labels: (n,) optional ground-truth labels
        n_iter: number of EM iterations
        random_state: for reproducibility
    """
    if random_state is not None:
        np.random.seed(random_state)

    d, n = X.shape
    X = np.maximum(X, 0)

    # --- Initialization ---
    W = np.abs(np.random.randn(d, K))
    H = np.abs(np.random.randn(K, n))
    W = normalize(W, axis=0)
    H = normalize(H, axis=1)

    for it in range(n_iter):
        # --- E-step: update H ---
        H = np.maximum(0, W.T @ X)
        # enforce orthogonality (row-wise normalization)
        for k in range(K):
            norm = np.linalg.norm(H[k, :])
            if norm > 0:
                H[k, :] /= norm

        # --- M-step: update W ---
        W = np.maximum(0, X @ H.T)
        W = normalize(W, axis=0)

    # --- Reconstruction and clustering ---
    X_hat = W @ H
    pred_labels = np.argmax(H, axis=0)

    reconstruction_error = np.linalg.norm(X - X_hat) / np.linalg.norm(X)

    if true_labels is not None:
        acc = remap_accuracy(true_labels, pred_labels)
        ari = adjusted_rand_score(true_labels, pred_labels)
        nmi = normalized_mutual_info_score(true_labels, pred_labels)
    else:
        acc = ari = nmi = None

    return acc, ari, nmi, reconstruction_error


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

