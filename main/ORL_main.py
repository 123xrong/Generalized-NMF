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
from coneClustering import *
from modified_dscnmf import *
from sklearn.linear_model import Lasso
from sklearn.preprocessing import normalize
from nmf import *
from sklearn.datasets import fetch_olivetti_faces, fetch_openml 

def arg_parser():
    parser = argparse.ArgumentParser(description="Iterative subspace clustering with NMF")
    # parser.add_argument('--m', type=int, default=50, help='Dimension of the ambient space (default: 50)')
    parser.add_argument('--r', type=int, default=5, help='Dimension (rank) of each subspace (default: 5)')
    parser.add_argument('--n', type=int, default=50, help='Number of points per subspace (default: 100)')
    parser.add_argument('--K', type=int, default=4, help='Number of subspaces (default: 3)')
    parser.add_argument('--sigma', type=float, default=0.0, help='Standard deviation of Gaussian noise (default: 0.0)')
    parser.add_argument('--alpha', type=float, default=1e-2, help='Regularization parameter for ssc')
    parser.add_argument('--max_iter', type=int, default=200, help='Maximum number of iterations (default: 50)')
    parser.add_argument('--tol', type=float, default=1e-6, help='Tolerance for stopping criterion (default: 1e-6)')
    parser.add_argument('--random_state', type=int, default=42, help='Random seed for clustering (default: None)')
    parser.add_argument('--model', type=str,
    choices=['sscnmf', 'ricc', 'gnmf', 'gpcanmf', 'onmf_relu', 'dscnmf'],
    help='Model to use for clustering')

    parser.add_argument('--l1_reg', type=float, default=0.01, help='L1 regularization parameter for ONMF-ReLU/GPCANMF')
    return parser.parse_args()

def main(model, r, n, K, sigma=0.0, alpha = 0.1, l1_reg=0.01, random_state=42, max_iter=50, tol=1e-6):
    faces = fetch_olivetti_faces(shuffle=True, random_state=random_state)
    X_full = faces.data.T
    true_labels = faces.target

    if sigma > 0:
        # Add non-negative Gaussian noise to the data
        noise = np.random.normal(0, sigma, X_full.shape)
        X_full += noise
        # 0 truncate negative values
        X_full = np.maximum(X_full, 0)
    
    print(f"Received model: {model}")
    if model == 'sscnmf':
        project_name = 'sscnmf-ORL'
    elif model == 'ricc':
        project_name = 'ricc-ORL'
    elif model == 'gnmf':
        project_name = 'gnmf-ORL'
    elif model == 'gpcanmf':
        project_name = 'gpcanmf-ORL'
    elif model == 'onmf_relu':
        project_name = 'onmf_relu-ORL'
    elif model == 'dscnmf':
        project_name = 'dscnmf-ORL'
    
    print(f"Project name: {project_name}")

    wandb.init(
        project="coneClustering",
        name=project_name
    )

    if model == 'sscnmf':
        acc, ARI, NMI, reconstruction_error = ssc_nmf_baseline(
            X_full, K, r, true_labels=true_labels, alpha=alpha)
    elif model == 'ricc':
        acc, ARI, NMI, reconstruction_error, _ = iter_reg_coneclus_warmstart(
            X_full, K, r, true_labels=true_labels, alpha=alpha)
    elif model == 'gnmf':
        acc, ARI, NMI, reconstruction_error = GNMF_clus(
            X_full, K, true_labels=true_labels, lmd=l1_reg)
    elif model == 'gpcanmf':
        acc, ARI, NMI, reconstruction_error = gpca_nmf(
            X_full, K, r, true_labels=true_labels, l1_reg=l1_reg)
    elif model == 'onmf_relu':
        acc, ARI, NMI, reconstruction_error = onmf_with_relu(
            X_full, K=K, r=r, true_labels=true_labels,
            lambda_reg=l1_reg, tol=1e-4, verbose=False)
    elif model == 'dscnmf':
        acc, ARI, NMI, reconstruction_error = dsc_nmf_baseline(
            X_full, K=K, r=r, true_labels=true_labels,
            alpha=alpha, max_iter=max_iter, tol=tol)

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