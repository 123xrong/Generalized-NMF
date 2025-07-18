import os
import sys
import inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir) 

import coneClustering
import numpy as np
from sklearn.datasets import fetch_olivetti_faces, fetch_openml  # Similar, or use your ORL data
import argparse
import wandb


def arg_parser():
    parser = argparse.ArgumentParser(description="Itern bative subspace clustering with NMF")
    # parser.add_argument('--m', type=int, default=50, help='Dimension of the ambient space (default: 50)')
    parser.add_argument('--r', type=int, default=5, help='Dimension (rank) of each subspace (default: 5)')
    parser.add_argument('--n', type=int, default=50, help='Number of points per subspace (default: 100)')
    parser.add_argument('--K', type=int, default=4, help='Number of subspaces (default: 3)')
    parser.add_argument('--sigma', type=float, default=0.0, help='Standard deviation of Gaussian noise (default: 0.0)')
    parser.add_argument('--alpha', type=float, default=0.01, help='Regularization parameter for ssc')
    parser.add_argument('--max_iter', type=int, default=1000, help='Maximum number of iterations (default: 50)')
    parser.add_argument('--random_state', type=int, default=42, help='Random seed for clustering (default: None)')
    parser.add_argument('--dataset', choices=['mnist', 'YaleB', 'ORL'], default='mnist', help='Dataset to use (default: mnist)')
    return parser.parse_args()

def load_dataset(name):
    if name == 'mnist':
        print("Loading MNIST...")
        mnist = fetch_openml('mnist_784', version=1, as_frame=False)
        X_full = mnist.data / 255.0  # normalize to [0,1]
        y_full = mnist.target.astype(int)

        # Subset: select digits 0–5 for faster experiments
        mask = y_full < 6
        X = X_full[mask].T  # shape (784, n)
        y = y_full[mask]

    elif name == 'ORL':
        print("Loading ORL (Olivetti Faces)...")
        faces = fetch_olivetti_faces(shuffle=True, random_state=42)
        X_full = faces.data.T  # shape (4096, n)
        y_full = faces.target

        # Subset: select only classes 0–5 (6 people × 10 = 60 samples)
        mask = y_full < 6
        X = faces.data[mask].T  # shape (4096, 60)
        y = y_full[mask]

    else:
        raise ValueError(f"Unsupported dataset: {name}")

    print(f"Loaded {name.upper()} | X shape: {X.shape}, y shape: {y.shape}")
    return X, y

def main(r, K, n, sigma=0.0, alpha = 0.01, random_state=None, max_iter=1000, dataset='mnist'):
    X_subset, true_labels = load_dataset(dataset)
    X_subset = X_subset.T
    X_subset = X_subset / 255.0

    if sigma > 0:
        # Add non-negative Gaussian noise to the data
        noise = np.random.normal(0, sigma, X_subset.shape)
        X_subset += noise
        # 0 truncate negative values
        X_subset = np.maximum(X_subset, 0)

    log_name = f'sscnmf-{dataset}'
    wandb.init(
        project="coneClustering",
        name = log_name
    )
    accuracy, ARI, NMI, reconstruction_error = coneClustering.ssc_nmf_baseline(X_subset, r, K, true_labels = true_labels, alpha=alpha, max_iter=max_iter)

    wandb.log({
        "accuracy": accuracy,
        "ARI": ARI,
        "NMI": NMI,
        "reconstruction_error": reconstruction_error
    })

    wandb.finish()

    print("\n--- Results ---")
    print(f"Clustering Accuracy (ARI): {accuracy:.4f}")
    print(f"Adjusted Rand Index (ARI): {ARI:.4f}")
    print(f"Normalized Mutual Information (NMI): {NMI:.4f}")
    print(f"Final Reconstruction Loss: {reconstruction_error:.4f}")

if __name__ == "__main__":
    args = arg_parser()
    r = args.r
    K = args.K
    n = args.n
    sigma = args.sigma
    max_iter = args.max_iter
    random_state = args.random_state
    alpha = args.alpha
    dataset = args.dataset

    main(r, K, n, sigma, alpha, random_state, max_iter, dataset)