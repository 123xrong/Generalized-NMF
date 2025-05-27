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
    parser.add_argument('--n_neighbors', type=int, default=5, help='Number of neighbors for kNN graph (default: 10)')
    parser.add_argument('--n', type=int, default=100, help='Number of points per subspace (default: 100)')
    parser.add_argument('--K', type=int, default=4, help='Number of subspaces (default: 3)')
    parser.add_argument('--sigma', type=float, default=0.0, help='Standard deviation of Gaussian noise (default: 0.0)')
    parser.add_argument('--max_iter', type=int, default=200, help='Maximum number of iterations (default: 50)')
    parser.add_argument('--random_state', type=int, default=42, help='Random seed for clustering (default: None)')
    parser.add_argument('--alpha', type=float, default=2.3, help='Regularization parameter for ReLU regularization')
    return parser.parse_args()

def main(m, r, n_k, K, n_neighbors, sigma=0.0, random_state=None, max_iter=50):

    wandb.init(
        project="coneClustering",
        name = "GNMF-synthetic"
    )
    # 1. Generate distinct subspace data
    X, true_labels = data_simulation(m, r, n_k, K, sigma=sigma, random_state=random_state)

    acc, ARI, NMI, reconstruction_error = GNMF(
        X, K, n_neighbors=n_neighbors, true_labels=true_labels, max_iter=max_iter, random_state=random_state
    )

    wandb.log({
    "accuracy": acc,
    "ARI": ARI,
    "NMI": NMI,
    "reconstruction_error": reconstruction_error
    })

    print("\n--- Results ---")
    print(f"Clustering Accuracy (ARI): {acc:.4f}")
    print(f"Adjusted Rand Index (ARI): {ARI:.4f}")
    print(f"Normalized Mutual Information (NMI): {NMI:.4f}")
    print(f"Final Reconstruction Loss: {reconstruction_error:.4f}")

if __name__ == "__main__":
    args = arg_parser()
    m = args.m
    r = args.r
    n_k = args.n
    K = args.K
    n_neighbors = args.n_neighbors
    sigma = args.sigma
    max_iter = args.max_iter
    random_state = args.random_state
    alpha = args.alpha

    main(m, r, n_k, K, n_neighbors, sigma, random_state, max_iter, ord=2)

