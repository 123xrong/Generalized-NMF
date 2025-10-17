import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from src.utils import *
from sklearn.decomposition import NMF
from sklearn.preprocessing import normalize
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, adjusted_rand_score, normalized_mutual_info_score

# class DeepNMF(nn.Module):
#     def __init__(self, input_dim, hidden_dims):
#         super(DeepNMF, self).__init__()
#         self.L = len(hidden_dims)
#         self.W = nn.ParameterList([
#             nn.Parameter(torch.rand(hidden_dims[i], input_dim if i == 0 else hidden_dims[i-1]))
#             for i in range(self.L)
#         ])

#     def encode(self, X):
#         H = X
#         for i in range(self.L):
#             W = torch.clamp(self.W[i], min=1e-8)
#             H = torch.clamp(W @ H, min=1e-8)
#         return H

#     def decode(self, H):
#         for i in reversed(range(self.L)):
#             W = torch.clamp(self.W[i], min=1e-8)
#             H = torch.clamp(W.T @ H, min=1e-8)
#         return H

#     def forward(self, X):
#         H = self.encode(X)
#         X_hat = self.decode(H)
#         return X_hat, H

# def deep_nmf(X_np, r1=256, r2=128, r3=64, n_iter=200, true_labels=None, device='cpu'):
#     # Convert to torch tensor without normalization
#     X_np = normalize(X_np, axis=0)
#     X = torch.tensor(X_np, dtype=torch.float32).to(device)
#     norm_X = torch.norm(X, p='fro')

#     model = DeepNMF(X.shape[0], [r1, r2, r3]).to(device)
#     optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

#     for _ in range(n_iter):
#         optimizer.zero_grad()
#         X_hat, _ = model(X)
#         loss = torch.norm(X - X_hat, p='fro') / norm_X
#         loss.backward()
#         optimizer.step()
#         # Enforce nonnegativity
#         for param in model.parameters():
#             param.data.clamp_(min=1e-8)

#     # Final representation for clustering
#     _, H_final = model(X)
#     H_final = H_final.detach().cpu().numpy().T

#     if true_labels is not None:
#         K = len(np.unique(true_labels))
#         pred_labels = KMeans(n_clusters=K, n_init=10).fit_predict(H_final)
#         acc = remap_accuracy(true_labels, pred_labels)
#         ari = adjusted_rand_score(true_labels, pred_labels)
#         nmi = normalized_mutual_info_score(true_labels, pred_labels)
#     else:
#         acc = ari = nmi = None

#     # Normalized reconstruction error
#     X_hat_final, _ = model(X)
#     X_hat_final = normalize(X_hat_final.detach().cpu().numpy(), axis=0)
#     recon_error = torch.norm(X - X_hat_final, p='fro') / np.linalg.norm(X_np, ord='fro')
#     recon_error = recon_error.item()

#     return acc, ari, nmi, recon_error

def deep_nmf(X, hidden_dims=[256, 128, 64], max_iter=200, tol=1e-4,
             random_state=None, verbose=False, true_labels=None):
    """
    Deep NMF (Trigeorgis et al., 2014): layer-wise pretraining + fine-tuning,
    followed by clustering evaluation (KMeans on top-level H).

    Args
    ----
    X : array, shape (m, n)
        Nonnegative data matrix (features × samples)
    hidden_dims : list of int
        Dimensions of hidden layers (e.g., [256, 128, 64])
    max_iter : int
        Max fine-tuning iterations
    tol : float
        Convergence threshold on reconstruction loss
    random_state : int
        Random seed
    verbose : bool
        Print reconstruction loss periodically
    true_labels : array-like, optional
        Ground-truth labels for evaluation

    Returns
    -------
    acc : float
        Clustering accuracy (if labels provided)
    ari : float
        Adjusted Rand index
    nmi : float
        Normalized mutual information
    recon_error : float
        Final normalized reconstruction error
    W_list : list of np.ndarray
        Learned basis matrices
    H : np.ndarray
        Final representation (top-layer H)
    """

    np.random.seed(random_state)
    X = np.maximum(X, 1e-8)
    m, n = X.shape
    L = len(hidden_dims)

    # ----- 1. Layer-wise pretraining -----
    W_list, H = [], X.copy()
    for r in hidden_dims:
        nmf = NMF(n_components=r, init='nndsvda', max_iter=1000, random_state=random_state)
        H = np.maximum(nmf.fit_transform(H), 1e-8)
        W = np.maximum(nmf.components_.T, 1e-8)
        W_list.append(W)
    H = normalize(H, axis=0)

    # ----- 2. Joint fine-tuning (multiplicative updates) -----
    prev_err, recon_error = np.inf, np.inf
    for it in range(max_iter):
        # Forward reconstruction
        X_hat = W_list[0]
        for l in range(1, L):
            X_hat = X_hat @ W_list[l]
        print(X_hat.shape, H.shape)
        X_hat = X_hat @ H
        recon_error = np.linalg.norm(X - X_hat, 'fro') / np.linalg.norm(X, 'fro')

        if it % 50 == 0 or it == max_iter - 1:
            if verbose:
                print(f"[Iter {it:03d}] Recon error = {recon_error:.4f}")
        if abs(prev_err - recon_error) < tol:
            break
        prev_err = recon_error

        # Backward multiplicative updates
        H = H * (W_list[-1].T @ X) / (W_list[-1].T @ W_list[-1] @ H + 1e-8)
        H = np.maximum(H, 1e-8)
        for l in reversed(range(L)):
            W = W_list[l]
            numer = X @ H.T
            denom = W @ (H @ H.T) + 1e-8
            W_list[l] = np.maximum(W * numer / denom, 1e-8)
            W_list[l] = normalize(W_list[l], axis=0)

    # ----- 3. Clustering step -----
    K = len(np.unique(true_labels)) if true_labels is not None else hidden_dims[-1]
    H_final = H.T  # (samples × latent_dim)
    pred_labels = KMeans(n_clusters=K, n_init=20, random_state=random_state).fit_predict(H_final)

    # ----- 4. Evaluation -----
    if true_labels is not None:
        acc = remap_accuracy(true_labels, pred_labels)
        ari = adjusted_rand_score(true_labels, pred_labels)
        nmi = normalized_mutual_info_score(true_labels, pred_labels)
    else:
        acc = ari = nmi = None

    return acc, ari, nmi, recon_error, W_list, H