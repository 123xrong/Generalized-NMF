import numpy as np
import argparse
import wandb
from sklearn.datasets import make_blobs
from onmf_relu import onmf_with_relu

def arg_parser():
    parser = argparse.ArgumentParser(description="Orthogonal NMF with ReLU regularization")

    parser.add_argument('--m', type=int, default=50, help='Ambient dimension (default: 50)')
    parser.add_argument('--r', type=int, default=5, help='Factorization rank (default: 5)')
    parser.add_argument('--n', type=int, default=100, help='Number of points per subspace (default: 100)')
    parser.add_argument('--K', type=int, default=4, help='Number of subspaces (default: 4)')
    parser.add_argument('--sigma', type=float, default=0.0, help='Gaussian noise standard deviation (default: 0.0)')
    parser.add_argument('--max_iter', type=int, default=200, help='Maximum number of iterations (default: 200)')
    parser.add_argument('--random_state', type=int, default=42, help='Random seed (default: 42)')
    parser.add_argument('--alpha', type=float, default=0.1, help='ReLU regularization weight λ (default: 0.1)')

    return parser.parse_args()

def generate_synthetic_data(m, r, n_k, K, sigma=0.0, random_state=0):
    total_points = n_k * K
    X, labels_true = make_blobs(n_samples=total_points, centers=K,
                                 n_features=m, cluster_std=sigma,
                                 random_state=random_state)
    X = np.abs(X)  # Ensure nonnegativity
    return X.T, labels_true  # shape (m, n), matching ONMF

def main(m, r, n_k, K, sigma, random_state, max_iter, alpha):
    wandb.init(
        project="coneClustering",
        name="ONMF-ReLU",
        config={
            "m": m,
            "r": r,
            "n_k": n_k,
            "K": K,
            "sigma": sigma,
            "lambda": alpha,
            "max_iter": max_iter,
            "random_state": random_state,
        }
    )

    # 1. Generate synthetic data
    X, labels_true = generate_synthetic_data(m, r, n_k, K, sigma=sigma, random_state=random_state)

    # 2. Run ONMF-ReLU algorithm
    result = onmf_with_relu(X, K=K, r=r, true_labels=labels_true,
                            lambda_reg=alpha, max_iter=max_iter, verbose=True)

    # 3. Log results
    wandb.log({
        "accuracy": result["accuracy"],
        "ARI": result["ARI"],
        "NMI": result["NMI"],
        "normalized_loss": result["normalized_loss"],
        "reconstruction_error": result["reconstruction_error"],
        "proportion_negatives": result["proportion_negatives"],
    })

    print("\n--- Clustering Results ---")
    print(f"Accuracy:             {result['accuracy']:.4f}")
    print(f"Adjusted Rand Index:  {result['ARI']:.4f}")
    print(f"NMI:                  {result['NMI']:.4f}")
    print(f"Normalized Loss:      {result['normalized_loss']:.6f}")
    print(f"Reconstruction Error: {result['reconstruction_error']:.4f}")
    print(f"Negative Proportion:  {result['proportion_negatives']:.6f}")

if __name__ == "__main__":
    args = arg_parser()
    main(
        m=args.m,
        r=args.r,
        n_k=args.n,
        K=args.K,
        sigma=args.sigma,
        random_state=args.random_state,
        max_iter=args.max_iter,
        alpha=args.alpha
    )
