import os
import sys
import inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir) 

import re
import numpy as np
import argparse
import wandb
from src.GenNMF import *
from src.modified_dscnmf import *
from src.baseline import *
from src.deepNMF import *
from src.deepSSCNMF import *
from sklearn.datasets import fetch_20newsgroups
from nltk.corpus import stopwords
from nltk import download
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA

def arg_parser():
    parser = argparse.ArgumentParser(description="Iterative subspace clustering with NMF")
    # parser.add_argument('--m', type=int, default=50, help='Dimension of the ambient space (default: 50)')
    parser.add_argument('--r', type=int, default=10, help='Dimension (rank) of each subspace (default: 5)')
    parser.add_argument('--n', type=int, default=50, help='Number of points per subspace (default: 100)')
    parser.add_argument('--K', type=int, default=4, help='Number of subspaces (default: 3)')
    parser.add_argument('--sigma', type=float, default=0.0, help='Standard deviation of Gaussian noise (default: 0.0)')
    parser.add_argument('--alpha', type=float, default=1e-2, help='Regularization parameter for ssc')
    parser.add_argument('--max_iter', type=int, default=200, help='Maximum number of iterations (default: 50)')
    parser.add_argument('--tol', type=float, default=1e-6, help='Tolerance for stopping criterion (default: 1e-6)')
    parser.add_argument('--random_state', type=int, default=42, help='Random seed for clustering (default: None)')
    parser.add_argument('--model', type=str, choices=['sscnmf', 'ricc', 'gnmf', 'gpcanmf', 'onmf_relu', 'dscnmf', 'onmf', 'deepnmf', 'deepsscnmf', 'ssc-omp-nmf'], help='Model to use for clustering')
    parser.add_argument('--l1_reg', type=float, default=0.01, help='L1 regularization parameter for ONMF-ReLU/GPCANMF')
    parser.add_argument('--n_nonzero_coefs', type=int, default=8, help='Number of non-zero coefficients for OMP')
    return parser.parse_args()

def load_TDT2_10(path='data/TDT2_4_preprocessed.npz', normalize_cols=True):
    """
    Loads the preprocessed 4-class TDT2 subset.

    Parameters
    ----------
    path : str
        Path to your saved .npz file (default: 'data/TDT2_4_preprocessed.npz')
    normalize_cols : bool
        Whether to re-normalize columns (recommended for NMF stability).

    Returns
    -------
    X : np.ndarray, shape (n_features, n_samples)
        Column-normalized data matrix (ready for NMF/Deep NMF).
    y : np.ndarray, shape (n_samples,)
        Integer labels (0–3).
    """

    data = np.load(path)
    X = data['X']
    y = data['y']

    if normalize_cols:
        X = normalize(X, axis=0)

    print(f"Loaded TDT2-4: X.shape={X.shape}, unique classes={len(np.unique(y))}")
    return X, y

def main(model, r, n, K, sigma=0.0, alpha=0.01, l1_reg=0.01, random_state=None, max_iter=50, tol=1e-6, n_nonzero_coefs=8):
    X, y = load_TDT2_10('data/TDT2_4_preprocessed.npz')

    # reduce data dimensionality if needed
    if X.shape[0] > 1000:
        pca = PCA(n_components=1000, whiten=True, random_state=random_state)
        X_reduced = pca.fit_transform(X.T).T  # shape (1000, n_samples)
        X = X_reduced
        print(f"Reduced data dimensionality to {X.shape[0]} via PCA.")

    print(f"Received model: {model}")
    if model == 'sscnmf':
        project_name = 'sscnmf-TDT2'
    elif model == 'ssc-omp-nmf':
        project_name = 'ssc-omp-nmf-TDT2'
    elif model == 'ricc':
        project_name = 'ricc-TDT2'
    elif model == 'gnmf':
        project_name = 'gnmf-TDT2'
    elif model == 'gpcanmf':
        project_name = 'gpcanmf-TDT2'
    elif model == 'onmf_relu':
        project_name = 'onmf_relu-TDT2'
    elif model == 'dscnmf':
        project_name = 'dscnmf-TDT2'
    elif model == 'onmf':
        project_name = 'onmf-TDT2'
    elif model == 'deepnmf':
        project_name = 'deepnmf-TDT2'
    elif model == 'deepsscnmf':
        project_name = 'deepsscnmf-TDT2'

    wandb.init(
        project="coneClustering",
        name=project_name
    )

    if model == 'sscnmf':
        acc, ARI, NMI, reconstruction_error, _, _, _ = ssc_nmf_baseline(
            X, K, r, true_labels=y, random_state=random_state, alpha=alpha)
    elif model == 'ssc-omp-nmf':
        acc, ARI, NMI, reconstruction_error, _, _, _ = ssc_omp_nmf_baseline(
            X, K, r, true_labels=y, n_nonzero_coefs=n_nonzero_coefs, random_state=random_state)
    elif model == 'ricc':
        acc, ARI, NMI, reconstruction_error, _, _, _ = iter_reg_coneclus_warmstart(
            X, K, r, true_labels=y, random_state=random_state, alpha=alpha)
    elif model == 'gnmf':
        acc, ARI, NMI, reconstruction_error = GNMF_clus(
            X, K, true_labels=y, random_state=random_state, lmd=l1_reg)
    elif model == 'gpcanmf':
        acc, ARI, NMI, reconstruction_error = gpca_nmf(
            X, K, r, true_labels=y, l1_reg=l1_reg)
    elif model == 'dscnmf':
        acc, ARI, NMI, reconstruction_error = dsc_nmf_baseline(
            X, K=K, r=r, true_labels=y)
    elif model == 'onmf':
        acc, ARI, NMI, reconstruction_error = onmf_ding(
            X, K=K, true_labels=y)
    elif model == 'deepnmf':
        acc, ARI, NMI, reconstruction_error = deep_nmf(
            X, true_labels=y)
    elif model == 'deepsscnmf':
        acc, ARI, NMI, reconstruction_error = deep_ssc_nmf(
            X, ranks=[256, 128, 64], alpha=alpha, n_iter=max_iter,
            true_labels=y)

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
    main(
        model=args.model,
        r=args.r,
        n=args.n,
        K=args.K,
        sigma=args.sigma,
        alpha=args.alpha,
        l1_reg=args.l1_reg,
        random_state=args.random_state,
        max_iter=args.max_iter,
        tol=args.tol,
        n_nonzero_coefs=args.n_nonzero_coefs
    )