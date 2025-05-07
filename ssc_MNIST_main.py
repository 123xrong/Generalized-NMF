import numpy as np
import argparse
from sklearn.metrics import adjusted_rand_score
from coneClustering import *
import wandb
from sklearn.linear_model import Lasso
from sklearn.datasets import fetch_olivetti_faces, fetch_openml  # Similar, or use your ORL data

def arg_parser():
    parser = argparse.ArgumentParser(description="Iterative subspace clustering with NMF")
    # parser.add_argument('--m', type=int, default=50, help='Dimension of the ambient space (default: 50)')
    parser.add_argument('--r', type=int, default=5, help='Dimension (rank) of each subspace (default: 5)')
    parser.add_argument('--n', type=int, default=100, help='Number of points per subspace (default: 100)')
    parser.add_argument('--K', type=int, default=4, help='Number of subspaces (default: 3)')
    parser.add_argument('--sigma', type=float, default=0.0, help='Standard deviation of Gaussian noise (default: 0.0)')
    parser.add_argument('--alpha', type=float, default=1.7, help='Regularization parameter for ssc')
    parser.add_argument('--max_iter', type=int, default=200, help='Maximum number of iterations (default: 50)')
    parser.add_argument('--tol', type=float, default=1e-6, help='Tolerance for stopping criterion (default: 1e-6)')
    parser.add_argument('--random_state', type=int, default=42, help='Random seed for clustering (default: None)')
    return parser.parse_args()

def main(r, n, K, sigma=0.0, alpha = 1.7, random_state=None, max_iter=50, tol=1e-6):
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
    X_subset = X_subset / 255.0  # Still non-negative

    print(X_subset.shape)
    print(true_labels.shape)

    if sigma > 0:
        # Add non-negative Gaussian noise to the data
        noise = np.random.normal(0, sigma, X_subset.shape)
        X_subset += noise
        # 0 truncate negative values
        X_subset = np.maximum(X_subset, 0)

    wandb.init(
        project="coneClustering",
        name = "ssc-Baseline"
    )

    pre_labels, accuracy = baseline_ssc(X_subset, true_labels = true_labels, alpha=alpha)

    wandb.log({
        "accuracy": accuracy,
    })

    wandb.finish()

if __name__ == "__main__":
    args = arg_parser()
    r = args.r
    n_k = args.n
    K = args.K
    sigma = args.sigma
    max_iter = args.max_iter
    tol = args.tol
    random_state = args.random_state
    alpha = args.alpha

    main(r, n_k, K, sigma, alpha, random_state, max_iter, tol)