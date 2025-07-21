import numpy as np
from utils import *
from sklearn.preprocessing import normalize
from sklearn.cluster import SpectralClustering
from sklearn.linear_model import Lasso
from sklearn.metrics import accuracy_score, adjusted_rand_score, normalized_mutual_info_score
import torch
import torch.nn as nn
import torch.nn.functional as F

class ShallowEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, hidden_dim)

    def forward(self, x):
        return F.relu(self.linear(x))

def compute_sparse_coeff(X, alpha=0.01):
    """
    Input X: (n_samples, n_features)
    Returns: sparse coefficient matrix C (n_samples x n_samples)
    """
    X = X.T  # (n_features, n_samples)
    n = X.shape[1]
    C = np.zeros((n, n))
    print(f"X.shape: {X.shape}")

    for i in range(n):
        x_i = X[:, i]
        print(f"x_i.shape: {x_i.shape}")
        X_ = np.delete(X, i, axis=1)  # (n_features, n-1)

        clf = Lasso(alpha=alpha, fit_intercept=False, max_iter=1000)

        print(f"shape of X_.T: {X_.T.shape}, shape of x_i: {x_i.shape}")
        clf.fit(X_, x_i)

        coef = clf.coef_
        C[i, :i] = coef[:i]
        C[i, i+1:] = coef[i:]

    return C.T



def nmf_from_sparse(C, rank, max_iter=200):
    """
    Apply NMF to the sparse representation matrix C (n x n)
    Returns: W, H such that C ≈ WH
    """
    n = C.shape[0]
    W = np.random.rand(n, rank)
    H = np.random.rand(rank, n)

    for _ in range(max_iter):
        H *= (W.T @ C) / (W.T @ W @ H + 1e-8)
        W *= (C @ H.T) / (W @ H @ H.T + 1e-8)
        W = np.maximum(W, 1e-8)
        H = np.maximum(H, 1e-8)

    return W, H

def cluster_from_affinity(C, n_clusters):
    """
    Perform spectral clustering from affinity matrix C.
    """
    C_sym = 0.5 * (np.abs(C) + np.abs(C.T))
    clustering = SpectralClustering(n_clusters=n_clusters, affinity='precomputed', assign_labels='kmeans')
    labels = clustering.fit_predict(C_sym)
    return labels

def deep_ssc_then_nmf_pipeline(X, rank, n_clusters, true_labels, alpha=0.01):
    """
    Deep SSC + NMF pipeline: sparse self-expressiveness first, then apply NMF
    """
    X = normalize(X, axis=0)  # normalize columns
    C = compute_sparse_coeff(X, alpha=alpha)  # sparse self-expressiveness
    W, H = nmf_from_sparse(C, rank=rank)      # NMF on sparse graph
    pred_labels = cluster_from_affinity(C, n_clusters)

    acc = remap_accuracy(true_labels, pred_labels)
    ari = adjusted_rand_score(true_labels, pred_labels)
    nmi = normalized_mutual_info_score(true_labels, pred_labels)

    return acc, ari, nmi, C, W, H

def encoder_ssc_then_nmf_pipeline(X, encoder, rank, n_clusters, true_labels, alpha=0.01):
    """
    Variant: encode input using neural network first, then SSC and NMF
    """
    with torch.no_grad():
        X_tensor = torch.tensor(X.T, dtype=torch.float32)  # shape: (n, d)
        H_latent = encoder(X_tensor).numpy()  # shape: (d', n)

    C = compute_sparse_coeff(H_latent, alpha=alpha)
    W, H = nmf_from_sparse(C, rank=rank)
    pred_labels = cluster_from_affinity(C, n_clusters)

    acc = accuracy_score(true_labels, pred_labels)
    ari = adjusted_rand_score(true_labels, pred_labels)
    nmi = normalized_mutual_info_score(true_labels, pred_labels)

    return acc, ari, nmi, C, W, H
