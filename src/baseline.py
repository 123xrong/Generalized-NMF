import numpy as np

from src.utils import *
from .gnmf import GNMF
from .nmfbase import NMFBase
from sklearn.cluster import KMeans
from sklearn.decomposition import NMF
from sklearn.preprocessing import normalize
from scipy.optimize import nnls
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

def onmf_em(X, K, true_labels, random_state=None):
    """
    EM-based ONMF from Pompili et al. (2014), adapted to return standard results.
    
    Parameters:
        X: (d, n) nonnegative data matrix
        K: number of clusters
        true_labels: (n,) ground-truth labels
    
    Returns:
        acc: clustering accuracy
        ari: adjusted Rand index
        nmi: normalized mutual information
        reconstruction_error: ||X - WH|| / ||X||
    """
    _, n = X.shape
    X = np.maximum(X, 0)

    # Step 1: EM clustering on the sphere
    asgn_list, W_orth = spherical_k_means(X, K, random_state=random_state)  # W_orth: (d, K)

    # Step 2: Compute H by projecting X onto W
    H = np.zeros((K, n))
    for k in range(K):
        for j in asgn_list[k]:
            H[k, j] = W_orth[:, k].T @ X[:, j]

    X_hat = W_orth @ H
    pred_labels = np.argmax(H, axis=0)

    acc = remap_accuracy(true_labels, pred_labels)
    ari = adjusted_rand_score(true_labels, pred_labels)
    nmi = normalized_mutual_info_score(true_labels, pred_labels)
    reconstruction_error = np.linalg.norm(X - X_hat) / np.linalg.norm(X)

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

