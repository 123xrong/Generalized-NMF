import numpy as np
import argparse
from sklearn.metrics import adjusted_rand_score
from coneClustering import *
import wandb

def arg_parser():
    parser = argparse.ArgumentParser(description="Iterative subspace clustering with NMF")
    parser.add_argument('--m', type=int, default=50, help='Dimension of the ambient space (default: 50)')
    parser.add_argument('--r', type=int, default=5, help='Dimension (rank) of each subspace (default: 5)')
    parser.add_argument('--n', type=int, default=100, help='Number of points per subspace (default: 100)')
    parser.add_argument('--K', type=int, default=4, help='Number of subspaces (default: 3)')
    parser.add_argument('--sigma', type=float, default=0.0, help='Standard deviation of Gaussian noise (default: 0.0)')
    parser.add_argument('--alpha', type=float, default=0.01, help='Regularization parameter for ssc')
    parser.add_argument('--max_iter', type=int, default=200, help='Maximum number of iterations (default: 50)')
    parser.add_argument('--tol', type=float, default=1e-6, help='Tolerance for stopping criterion (default: 1e-6)')
    parser.add_argument('--random_state', type=int, default=42, help='Random seed for clustering (default: None)')
    return parser.parse_args()

def main(m, r, n_k, K, sigma=0.0, alpha = 0.01, random_state=None, max_iter=50, tol=1e-6):

    wandb.init(
        project="coneClustering",
        name = "ssc-Baseline"
    )

    # Simulate your data matrix X of shape (m, n_k * K)
    X, true_labels = data_simulation(m, r, n_k, K, sigma, random_state)  # Make sure this is defined
    pre_labels, accuracy = baseline_ssc(X, alpha, max_iter=max_iter, tol=tol, random_state=random_state)

    wandb.log({
        "accuracy": accuracy,
    })

    wandb.finish()

if __name__ == "__main__":
    args = arg_parser()
    m = args.m
    r = args.r
    n_k = args.n
    K = args.K
    sigma = args.sigma
    max_iter = args.max_iter
    tol = args.tol
    random_state = args.random_state
    alpha = args.alpha

    main(m, r, n_k, K, sigma, alpha, random_state, max_iter, tol)