import numpy as np

from utils import *
from libnmf.gnmf import GNMF
from libnmf.nmfbase import NMFBase
from sklearn.cluster import KMeans
from sklearn.decomposition import NMF
from sklearn.preprocessing import normalize
from sklearn.metrics import accuracy_score, adjusted_rand_score, normalized_mutual_info_score

def onmf(X, K, true_labels=None, max_iter=200, random_state=0):
    """
    Orthogonal NMF (ONMF) using SVD-based orthogonal updates and NNLS for H.
    """
    m, n = X.shape
    norm_X = np.linalg.norm(X, 'fro')**2

    # Step 1: SVD initialization for orthogonal W
    U, _, _ = np.linalg.svd(X @ X.T)
    W = np.maximum(U[:, :K], 1e-8)
    W = normalize(W, axis=0)

    # Step 2: Solve H via NNLS (or pseudo-inverse)
    H = np.linalg.pinv(W.T @ W) @ (W.T @ X)
    H = np.maximum(H, 1e-8)

    # Step 3: Cluster using argmax
    pred_labels = H.argmax(axis=0)

    # Step 4: Metrics
    acc = accuracy_score(true_labels, pred_labels) if true_labels is not None else None
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

