import coneClustering
import numpy as np
from sklearn.datasets import fetch_olivetti_faces, fetch_openml  # Similar, or use your ORL data
import argparse
import wandb

def arg_parser():
    parser = argparse.ArgumentParser(description="Iterative subspace clustering with NMF")
    # parser.add_argument('--m', type=int, default=50, help='Dimension of the ambient space (default: 50)')
    parser.add_argument('--r', type=int, default=5, help='Dimension (rank) of each subspace (default: 5)')
    parser.add_argument('--n', type=int, default=50, help='Number of points per subspace (default: 100)')
    parser.add_argument('--K', type=int, default=5, help='Number of subspaces (default: 3)')
    parser.add_argument('--sigma', type=float, default=0.0, help='Standard deviation of Gaussian noise (default: 0.0)')
    parser.add_argument('--max_iter', type=int, default=200, help='Maximum number of iterations (default: 50)')
    parser.add_argument('--random_state', type=int, default=42, help='Random seed for clustering (default: None)')
    parser.add_argument('--NMF_method', choices=['anls', 'NMF'], default='anls', help='NMF method to use')
    parser.add_argument('--NMF_solver', choices=['cd', 'mu'], default='cd', help='Solver for NMF')
    parser.add_argument('--alpha', type=float, default=0.5, help='Regularization parameter for ReLU regularization')
    parser.add_argument('--ord', type=int, default=2, help='Order of the regularization (default: 2)')
    parser.add_argument('--dataset', choices=['mnist', 'YaleB', 'ORL'], default='mnist', help='Dataset to use (default: mnist)')
    return parser.parse_args()

def load_dataset(name, n, K):
    if name == 'mnist':
        print("Loading MNIST...")
        mnist = fetch_openml('mnist_784', version=1, as_frame=False)
        X_full = mnist.data / 255.0  # normalize to [0,1]
        y_full = mnist.target.astype(int)

        X_list = []
        labels = []
        for digit in range(K+1):
            idx = np.where(y_full == digit)[0]
            selected_idx = np.random.choice(idx, n, replace=False)
            X_list.append(X_full[selected_idx])
            labels.append(np.full(len(selected_idx), digit))
        X = np.vstack(X_list)
        y = np.concatenate(labels)

    elif name == 'ORL':
        print("Loading ORL (Olivetti Faces)...")
        faces = fetch_olivetti_faces(shuffle=True, random_state=42)
        X_full = faces.data.T  # shape (4096, n)
        y_full = faces.target

        # Subset: select only classes 0–5 (6 people × 10 = 60 samples)
        mask = y_full < K+1
        X = faces.data[mask].T  # shape (4096, 60)
        y = y_full[mask]

    else:
        raise ValueError(f"Unsupported dataset: {name}")

    print(f"Loaded {name.upper()} | X shape: {X.shape}, y shape: {y.shape}")
    return X, y

def main(r, K, n, NMF_method='anls', sigma=0.0, random_state=None, max_iter=200, alpha=0.5, ord=2, dataset='mnist'):
    X_subset, true_labels = load_dataset(dataset, n, K)

    X_subset = X_subset.T
    X_preprocessed = X_subset / 255.0  # normalize to [0, 1]
    means = X_preprocessed.mean(axis=1, keepdims=True)
    X_preprocessed = X_preprocessed - means
    X_preprocessed = np.maximum(X_preprocessed, 0)  # truncate negatives 

    if sigma > 0:
        # Add non-negative Gaussian noise to the data
        noise = np.random.normal(0, sigma, X_preprocessed.shape)
        X_preprocessed += noise
        # 0 truncate negative values
        X_preprocessed = np.maximum(X_preprocessed, 0)
    
    log_name = f'RICC-{dataset}'

    wandb.init(
        project="coneClustering",
        name = log_name,
    )

    accuracy, ARI, NMI, reconstruction_error, _ = coneClustering.iter_reg_coneclus_warmstart(
    X_preprocessed,
    K, 
    r,
    true_labels=true_labels,
    max_iter=max_iter,
    NMF_method=NMF_method,
    random_state=42,
    alpha=alpha,
    ord=2
    )

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
    NMF_method = args.NMF_method
    sigma = args.sigma
    random_state = args.random_state
    max_iter = args.max_iter
    alpha = args.alpha
    ord = args.ord
    dataset = args.dataset

    main(r, K, n, NMF_method=NMF_method, sigma=sigma, random_state=random_state, max_iter=max_iter, alpha=alpha, ord=ord, dataset=dataset)