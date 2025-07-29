import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from utils import *
from sklearn.preprocessing import normalize
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, adjusted_rand_score, normalized_mutual_info_score

class DeepNMF(nn.Module):
    def __init__(self, input_dim, hidden_dims):
        super(DeepNMF, self).__init__()
        self.L = len(hidden_dims)
        self.W = nn.ParameterList([
            nn.Parameter(torch.rand(hidden_dims[i], input_dim if i == 0 else hidden_dims[i-1]))
            for i in range(self.L)
        ])

    def forward(self, X):
        H = X
        for i in range(self.L):
            W = torch.clamp(self.W[i], min=1e-8)
            H = torch.clamp(W @ H, min=1e-8)
        
        # Decode: reconstruct back to original dim
        for i in reversed(range(self.L)):
            W = torch.clamp(self.W[i], min=1e-8)
            H = torch.clamp(W.T @ H, min=1e-8)

        return H  # this is now the reconstructed X_hat


def deep_nmf(X_np, r1=256, r2=128, r3=64, n_iter=200, true_labels=None, device='cpu'):
    # Normalize and convert to torch tensor
    X_np = normalize(X_np, axis=0)
    X = torch.tensor(X_np, dtype=torch.float32).to(device)

    model = DeepNMF(X.shape[0], [r1, r2, r3]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

    for _ in range(n_iter):
        optimizer.zero_grad()
        H = model(X)
        X_hat = H
        loss = F.mse_loss(X_hat, X)
        loss.backward()
        optimizer.step()
        # Enforce nonnegativity
        for param in model.parameters():
            param.data.clamp_(min=1e-8)

    # Final representation for clustering
    H_final = model(X).detach().cpu().numpy().T

    if true_labels is not None:
        K = len(np.unique(true_labels))
        pred_labels = KMeans(n_clusters=K, n_init=10).fit_predict(H_final)
        acc = remap_accuracy(true_labels, pred_labels)
        ari = adjusted_rand_score(true_labels, pred_labels)
        nmi = normalized_mutual_info_score(true_labels, pred_labels)
    else:
        acc = ari = nmi = None

    # Reconstruction error
    X_hat = model(X).detach().cpu().numpy()
    recon_error = np.linalg.norm(X_np - X_hat) / np.linalg.norm(X_np)

    return {
        'accuracy': acc,
        'ARI': ari,
        'NMI': nmi,
        'reconstruction_error': recon_error,
        'H': H_final
    }
