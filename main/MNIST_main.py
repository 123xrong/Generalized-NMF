import os
import sys
import inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir) 

import numpy as np
import argparse
import wandb
from sklearn.metrics import adjusted_rand_score
from src.GenNMF import *
from src.modified_dscnmf import *
from src.baseline import *
from src.deepNMF import *
from src.deepSSCNMF import *
from sklearn.linear_model import Lasso
from sklearn.preprocessing import normalize
from src.nmf import *
from sklearn.datasets import fetch_olivetti_faces, fetch_openml 

def arg_parser():
    parser = argparse.ArgumentParser(description="Iterative subspace clustering with NMF")
    # parser.add_argument('--m', type=int, default=50, help='Dimension of the ambient space (default: 50)')
    parser.add_argument('--r', type=int, default=5, help='Dimension (rank) of each subspace (default: 5)')
    parser.add_argument('--n', type=int, default=50, help='Number of points per subspace (default: 100)')
    parser.add_argument('--K', type=int, default=10, help='Number of subspaces (default: 10)')
    parser.add_argument('--sigma', type=float, default=0.0, help='Standard deviation of Gaussian noise (default: 0.0)')
    parser.add_argument('--alpha', type=float, default=1e-2, help='Regularization parameter for ssc')
    parser.add_argument('--max_iter', type=int, default=200, help='Maximum number of iterations (default: 50)')
    parser.add_argument('--tol', type=float, default=1e-6, help='Tolerance for stopping criterion (default: 1e-6)')
    parser.add_argument('--random_state', type=int, default=42, help='Random seed for clustering (default: None)')
    parser.add_argument('--model', type=str, choices=['sscnmf', 'ricc', 'gnmf', 'gpcanmf', 'onmf_relu', 'dscnmf', 'onmf', 'deepnmf', 'deepsscnmf'],
    help='Model to use for clustering')
    parser.add_argument('--l1_reg', type=float, default=0.01, help='L1 regularization parameter for ONMF-ReLU/GPCANMF')
    return parser.parse_args()

def main(model, r, n, K, sigma=0.0, alpha = 0.1, l1_reg=0.01, random_state=None, max_iter=500, tol=1e-6):
    mnist = fetch_openml('mnist_784', version=1)
    X_full = mnist.data.to_numpy() 
    y_full = mnist.target.to_numpy().astype(int) 

    # 2. Subset digits 0-5
    X_list = []
    labels = []

    for digit in range(K):
        idx = np.where(y_full == digit)[0]
        selected_idx = np.random.choice(idx, n, replace=False)
        X_list.append(X_full[selected_idx])
        labels.append(np.full(len(selected_idx), digit))

    X_subset = np.vstack(X_list)
    true_labels = np.concatenate(labels) 
    X_subset = X_subset.T
    # normalize x_subset
    X_subset = normalize(X_subset, axis=1)

    if sigma > 0:
        # Add non-negative Gaussian noise to the data
        noise = np.random.normal(0, sigma, X_subset.shape)
        X_subset += noise
        # 0 truncate negative values
        X_subset = np.maximum(X_subset, 0)

    if model == 'sscnmf':
        project_name = 'sscnmf-MNIST'
    elif model == 'ricc':
        project_name = 'ricc-MNIST'
    elif model == 'gnmf':
        project_name = 'gnmf-MNIST'
    elif model == 'gpcanmf':
        project_name = 'gpcanmf-MNIST'
    elif model == 'dscnmf':
        project_name = 'dscnmf-MNIST'
    elif model == 'onmf':
        project_name = 'onmf-MNIST'
    elif model == 'deepnmf':
        project_name = 'deepnmf-MNIST'
    elif model == 'deepsscnmf':
        project_name = 'deepsscnmf-MNIST'

    wandb.init(
        project="coneClustering",
        name=project_name
    )

    if model == 'sscnmf':
        acc, ARI, NMI, reconstruction_error = ssc_nmf_baseline(
            X_subset, K, r, true_labels=true_labels, alpha=alpha)
    elif model == 'ricc':
        acc, ARI, NMI, reconstruction_error, _ = iter_reg_coneclus_warmstart(
            X_subset, K, r, true_labels=true_labels)
    elif model == 'gnmf':
        acc, ARI, NMI, reconstruction_error, _, _, _ = GNMF_clus(
            X_subset, K=K, r=r, true_labels=true_labels)
    elif model == 'gpcanmf':
        acc, ARI, NMI, reconstruction_error = gpca_nmf(
            X_subset, K, r, true_labels=true_labels, l1_reg=l1_reg)
    elif model == 'dscnmf':
        acc, ARI, NMI, reconstruction_error = dsc_nmf_baseline(
            X_subset, K=K, r=r, true_labels=true_labels)
    elif model == 'onmf':
        acc, ARI, NMI, reconstruction_error = onmf_em(
            X_subset, K=K, true_labels=true_labels)
    elif model == 'deepnmf':    
        acc, ARI, NMI, reconstruction_error = deep_nmf(
            X_subset, true_labels=true_labels)
    elif model == 'deepsscnmf':
        acc, ARI, NMI, reconstruction_error = deep_ssc_nmf(
            X_subset, ranks=[256, 128, 64], alpha=alpha, n_iter=max_iter,
            true_labels=true_labels)

    wandb.log({
        "accuracy": acc,
        "ARI": ARI,
        "NMI": NMI,
        "reconstruction_error": reconstruction_error
    })

    print("\n--- Results ---")
    print(f"Clustering Accuracy: {acc:.4f}")
    print(f"Adjusted Rand Index (ARI): {ARI:.4f}")
    print(f"Normalized Mutual Information (NMI): {NMI:.4f}")
    print(f"Final Reconstruction Loss: {reconstruction_error:.4f}")
    wandb.finish()

if __name__ == "__main__":
    args = arg_parser()
    model = args.model
    r = args.r
    n = args.n
    K = args.K
    sigma = args.sigma
    alpha = args.alpha
    l1_reg = args.l1_reg
    max_iter = args.max_iter
    tol = args.tol
    random_state = args.random_state

    main(model, r, n, K, sigma, alpha, l1_reg, random_state, max_iter, tol)