import os
import sys
import inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir) 

import numpy as np
import argparse
from sklearn.decomposition import NMF
from sklearn.metrics import adjusted_rand_score
from scipy.optimize import nnls
from coneClustering import *
from nmf import *

def arg_parser():
    parser = argparse.ArgumentParser(description="Iterative subspace clustering with NMF")
    parser.add_argument('--m', type=int, default=50, help='Dimension of the ambient space (default: 50)')
    parser.add_argument('--r', type=int, default=5, help='Dimension (rank) of each subspace (default: 5)')
    parser.add_argument('--n', type=int, default=100, help='Number of points per subspace (default: 100)')
    parser.add_argument('--K', type=int, default=4, help='Number of subspaces (default: 3)')
    parser.add_argument('--sigma', type=float, default=0.0, help='Standard deviation of Gaussian noise (default: 0.0)')
    parser.add_argument('--max_iter', type=int, default=200, help='Maximum number of iterations (default: 50)')
    parser.add_argument('--random_state', type=int, default=42, help='Random seed for clustering (default: None)')
    parser.add_argument('--NMF_method', choices=['anls', 'NMF'], default='anls', help='NMF method to use')
    parser.add_argument('--NMF_solver', choices=['cd', 'mu'], default='cd', help='Solver for NMF')
    parser.add_argument('--alpha', type=float, default=1, help='Regularization parameter for ReLU regularization')
    parser.add_argument('--ord', type=int, default=2, help='Order of the regularization (default: 2)')
    return parser.parse_args()


def main(m, r, n_k, K, NMF_method='anls', sigma=0.0, random_state=None, max_iter=50, alpha=2.3, ord=2):

    wandb.init(
        project="coneClustering",
        name = "RICC-synthetic"
    )
    # 1. Generate distinct subspace data
    X, true_labels = data_simulation(m, r, n_k, K, sigma=sigma, random_state=random_state)

    # 2. Run iterative subspace clustering
    accuracy, ARI, NMI, reconstruction_error, _ = iter_reg_coneclus_sparse_nmf(
        X, K, r, true_labels, max_iter=50, random_state=None,
                                 alpha=0.01, ord=2, l1_reg=0.1)

    # 3. Log results
    wandb.log({
        "accuracy": accuracy,
        "ARI": ARI,
        "NMI": NMI,
        "reconstruction_error": reconstruction_error
    })

    print("\n--- Results ---")
    print(f"Clustering Accuracy (Accuracy): {accuracy:.4f}")
    print(f"Adjusted Rand Index (ARI): {ARI:.4f}")
    print(f"Normalized Mutual Information (NMI): {NMI:.4f}")
    print(f"Final Reconstruction Loss: {reconstruction_error:.4f}")

if __name__ == "__main__":
    args = arg_parser()
    m = args.m
    r = args.r
    n_k = args.n
    K = args.K
    NMF_method = args.NMF_method
    sigma = args.sigma
    random_state = args.random_state
    max_iter = args.max_iter
    alpha = args.alpha
    ord = args.ord
    main(m, r, n_k, K, NMF_method=NMF_method, sigma=sigma, random_state=random_state, max_iter=max_iter, alpha=alpha, ord=ord)
