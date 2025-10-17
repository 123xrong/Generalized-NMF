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
    X = np.maximum(X, 1e-8)           # ensure nonnegativity
    d, n = X.shape                    # d = features, n = samples
    L = len(hidden_dims)

    # -------------------------------------------------------
    # 1. Layer-wise pretraining
    # -------------------------------------------------------
    W_list = []
    H_current = X.copy()

    for r in hidden_dims:
        nmf = NMF(n_components=r, init='nndsvda',
                  max_iter=500, random_state=random_state)
        # sklearn expects (samples, features)
        H_new = np.maximum(nmf.fit_transform(H_current.T), 1e-8)  # (samples, r)
        W_new = np.maximum(nmf.components_.T, 1e-8)               # (features, r)
        H_current = H_new.T                                       # (r, samples)
        W_list.append(W_new)

    H = normalize(H_current, axis=0)  # (r_L, samples)

    # -------------------------------------------------------
    # 2. Joint fine-tuning (multiplicative updates)
    # -------------------------------------------------------
    prev_err, recon_error = np.inf, np.inf

    for it in range(max_iter):
        # Forward reconstruction: X_hat = W1 @ W2 @ ... @ WL @ H
        W_eff = W_list[0]
        for l in range(1, L):
            W_eff = W_eff @ W_list[l]
        X_hat = W_eff @ H

        # Compute reconstruction error
        recon_error = np.linalg.norm(X - X_hat, 'fro') / np.linalg.norm(X, 'fro')

        if it % 50 == 0 or it == max_iter - 1:
            if verbose:
                print(f"[Iter {it:03d}] Reconstruction error = {recon_error:.5f}")

        if abs(prev_err - recon_error) < tol:
            break
        prev_err = recon_error

        # ---- Update H ----
        H = H * (W_eff.T @ X) / (W_eff.T @ (W_eff @ H) + 1e-8)
        H = np.maximum(H, 1e-8)

        # ---- Optionally update W's (coarse fine-tuning) ----
        W_eff = W_list[0]
        for l in range(1, L):
            W_eff = W_eff @ W_list[l]

        H = H * (W_eff.T @ X) / (W_eff.T @ (W_eff @ H) + 1e-8)
        H = np.maximum(H, 1e-8)

        # ---- Optionally update the effective W ----
        W_eff = W_eff * (X @ H.T) / (W_eff @ (H @ H.T) + 1e-8)
        W_eff = np.maximum(W_eff, 1e-8)
        W_eff = normalize(W_eff, axis=0)

    # -------------------------------------------------------
    # 3. Clustering and evaluation
    # -------------------------------------------------------
    K = len(np.unique(true_labels)) if true_labels is not None else hidden_dims[-1]
    H_final = H.T  # (samples, latent_dim)
    pred_labels = KMeans(n_clusters=K, n_init=20, random_state=random_state).fit_predict(H_final)

    if true_labels is not None:
        acc = remap_accuracy(true_labels, pred_labels)
        ari = adjusted_rand_score(true_labels, pred_labels)
        nmi = normalized_mutual_info_score(true_labels, pred_labels)
    else:
        acc = ari = nmi = None

    return acc, ari, nmi, recon_error, W_list, H