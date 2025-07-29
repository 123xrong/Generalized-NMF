import torch
import torch.nn as nn
import numpy as np
from utils import *
from sklearn.linear_model import Lasso
from sklearn.cluster import SpectralClustering, KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

class BasicNMF(nn.Module):
    def __init__(self, input_dim, rank):
        super(BasicNMF, self).__init__()
        self.W = nn.Parameter(torch.rand(input_dim, rank))
        self.H = None

    def forward(self, X):
        W = torch.clamp(self.W, min=1e-8)
        H = torch.linalg.lstsq(W, X).solution
        H = torch.clamp(H, min=1e-8)
        X_hat = W @ H
        return X_hat, H

def sparse_subspace_clustering(X_np, alpha=0.01):
    n_samples = X_np.shape[1]
    X_centered = X_np - X_np.mean(axis=0, keepdims=True)
    C = np.zeros((n_samples, n_samples))
    for i in range(n_samples):
        x_i = X_centered[:, i]
        X_rest = np.delete(X_centered, i, axis=1)
        lasso = Lasso(alpha=alpha, fit_intercept=False, max_iter=1000)
        lasso.fit(X_rest, x_i)
        c = lasso.coef_
        C[np.arange(n_samples) != i, i] = c
    C = np.maximum(0.5 * (C + C.T), 0)
    return C

def deep_ssc_nmf(X_np, ranks=[256, 128, 64], alpha=0.01, n_iter=100, true_labels=None, device='cpu'):
    X = torch.tensor(X_np, dtype=torch.float32).to(device)
    norm_X = torch.norm(X, p='fro')
    H_input = X.clone().detach()
    input_dim = X.shape[0]
    X_hat_all = torch.zeros_like(X)

    for r in ranks:
        X_np_cpu = H_input.detach().cpu().numpy()
        C = sparse_subspace_clustering(X_np_cpu, alpha=alpha)
        K = len(np.unique(true_labels)) if true_labels is not None else 10
        pred_labels = SpectralClustering(n_clusters=K, affinity='precomputed', assign_labels='kmeans', random_state=0).fit_predict(C)

        H_layer = torch.zeros(r, X.shape[1], device=device)

        for k in range(K):
            idx_k = np.where(pred_labels == k)[0]
            if len(idx_k) == 0:
                continue
            X_k = H_input[:, idx_k]
            nmf_layer = BasicNMF(H_input.shape[0], r).to(device)
            optimizer = torch.optim.Adam(nmf_layer.parameters(), lr=1e-3)
            for _ in range(n_iter):
                optimizer.zero_grad()
                X_hat, H_k = nmf_layer(X_k)
                loss = torch.norm(X_k - X_hat, p='fro')
                loss.backward()
                optimizer.step()
                nmf_layer.W.data.clamp_(min=1e-8)
            with torch.no_grad():
                X_hat, H_k = nmf_layer(X_k)
                H_layer[:, idx_k] = H_k
                if H_input.shape[0] == X.shape[0]:  # Only store reconstruction during first layer
                    X_hat_all[:, idx_k] = X_hat


        H_input = torch.clamp(H_layer.clone().detach(), min=1e-8)

    H_final = H_input.detach().cpu().numpy().T
    H_final = normalize(H_final, axis=0)
    pred_labels = KMeans(n_clusters=K, n_init=10).fit_predict(H_final)
    acc = remap_accuracy(true_labels, pred_labels) if true_labels is not None else None
    ari = adjusted_rand_score(true_labels, pred_labels) if true_labels is not None else None
    nmi = normalized_mutual_info_score(true_labels, pred_labels) if true_labels is not None else None
    recon_error = torch.norm(X - X_hat_all, p='fro') / norm_X

    return acc, ari, nmi, recon_error.item()