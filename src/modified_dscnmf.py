from sklearn.decomposition import NMF
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, adjusted_rand_score, normalized_mutual_info_score

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


def dsc_nmf_baseline(X_full, K, r, true_labels, hidden_dims=[256, 64], epochs=100, lr=1e-3):

    class LinearAE(nn.Module):
        def __init__(self, input_dim, hidden_dims):
            super().__init__()
            layers = []
            dims = [input_dim] + hidden_dims
            for i in range(len(dims) - 1):
                layers.append(nn.Linear(dims[i], dims[i + 1]))
                layers.append(nn.ReLU())
            self.encoder = nn.Sequential(*layers)
            decoder_layers = []
            rev_dims = list(reversed(dims))
            for i in range(len(rev_dims) - 1):
                decoder_layers.append(nn.Linear(rev_dims[i], rev_dims[i + 1]))
                if i < len(rev_dims) - 2:
                    decoder_layers.append(nn.ReLU())
            self.decoder = nn.Sequential(*decoder_layers)

        def forward(self, x):
            z = self.encoder(x)
            x_recon = self.decoder(z)
            return z, x_recon

    class SelfExpression(nn.Module):
        def __init__(self, n_samples):
            super().__init__()
            self.C = nn.Parameter(1e-4 * torch.rand(n_samples, n_samples))

        def forward(self, Z):
            return torch.matmul(self.C, Z)

    class DSC_NMF_Model(nn.Module):
        def __init__(self, input_dim, hidden_dims, n_samples):
            super().__init__()
            self.ae = LinearAE(input_dim, hidden_dims)
            self.self_exp = SelfExpression(n_samples)

        def forward(self, X):
            Z, X_recon = self.ae(X)
            Z_recon = self.self_exp(Z)
            return X_recon, Z, Z_recon, self.self_exp.C

    # Prepare data
    X = X_full.T  # shape: (n_samples, features)
    n_samples = X.shape[0]
    input_dim = X.shape[1]
    X_tensor = torch.tensor(X, dtype=torch.float32)

    model = DSC_NMF_Model(input_dim, hidden_dims, n_samples)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Train
    for epoch in range(epochs):
        X_recon, Z, Z_recon, C = model(X_tensor)
        loss_ae = nn.functional.mse_loss(X_recon, X_tensor)
        loss_selfexp = nn.functional.mse_loss(Z_recon, Z)
        loss_reg = torch.norm(C, p=1)
        loss = loss_ae + 0.1 * loss_selfexp + 1e-3 * loss_reg
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    C_np = C.detach().numpy()
    model_nmf = NMF(n_components=K, init='nndsvda', random_state=42)
    W = model_nmf.fit_transform(np.abs(C_np))
    H = model_nmf.components_
    pred_labels = KMeans(n_clusters=K, random_state=42).fit_predict(H.T)

    acc = accuracy_score(true_labels, pred_labels)
    ari = adjusted_rand_score(true_labels, pred_labels)
    nmi = normalized_mutual_info_score(true_labels, pred_labels)
    numerator = np.linalg.norm(X_tensor.numpy() - X_recon.detach().numpy())
    denominator = np.linalg.norm(X_tensor.numpy())
    recon_error = numerator / (denominator + 1e-8)  # small epsilon to avoid divide-by-zero


    return acc, ari, nmi, recon_error
