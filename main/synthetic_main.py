import os
import sys
import inspect
import argparse
import numpy as np
import wandb

# Path setup
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

# Imports from your repo
from src.GenNMF import *
from src.modified_dscnmf import *
from src.baseline import *
from sklearn.preprocessing import normalize

def parse_args():
    parser = argparse.ArgumentParser(description="Subspace clustering with NMF on synthetic data")
    parser.add_argument('--m', type=int, default=50, help='Ambient space dimension')
    parser.add_argument('--r', type=int, default=5, help='Subspace/NMF rank')
    parser.add_argument('--n', type=int, default=50, help='Samples per cluster')
    parser.add_argument('--K', type=int, default=4, help='Number of clusters/subspaces')
    parser.add_argument('--sigma', type=float, default=0.0, help='Noise level')
    parser.add_argument('--alpha', type=float, default=1e-2, help='Regularization strength')
    parser.add_argument('--l1_reg', type=float, default=0.01, help='L1 regularization for ONMF/GPCANMF')
    parser.add_argument('--max_iter', type=int, default=1000, help='Maximum iterations')
    parser.add_argument('--random_state', type=int, default=42, help='Random seed')
    parser.add_argument('--model', type=str, required=True,
                        choices=['sscnmf', 'ricc', 'gnmf', 'gpcanmf', 'dscnmf', 'onmf'],
                        help='Model to run')
    parser.add_argument('--NMF_method', type=str, default='anls', choices=['mu', 'cd', 'anls'], help='NMF solver')
    return parser.parse_args()

def main(args):
    # Generate synthetic data
    X, true_labels = data_simulation(
        m=args.m, r=args.r, n=args.n, K=args.K,
        sigma=args.sigma, random_state=args.random_state
    )

    # Dispatch model execution
    model_dispatch = {
        'sscnmf': lambda: ssc_nmf_baseline(X, K=args.K, r=args.r, true_labels=true_labels, alpha=args.alpha),
        'ricc': lambda: iter_reg_coneclus_warmstart(
            X, K=args.K, r=args.r, true_labels=true_labels, alpha=args.alpha,
            max_iter=args.max_iter, NMF_method=args.NMF_method, ord=2, random_state=args.random_state
        )[0:4],
        'gnmf': lambda: GNMF_clus(X, K=args.K, true_labels=true_labels, lmd=args.l1_reg),
        'gpcanmf': lambda: gpca_nmf(X, K=args.K, r=args.r, true_labels=true_labels, l1_reg=args.l1_reg),
        'onmf_relu': lambda: onmf_with_relu(X, K=args.K, r=args.r, true_labels=true_labels,
                                            lambda_reg=args.l1_reg, tol=1e-4, verbose=False),
        'dscnmf': lambda: dsc_nmf_baseline(X, K=args.K, r=args.r, true_labels=true_labels),
        'onmf': lambda: onmf_ding(X, K=args.K, true_labels=true_labels, random_state=args.random_state)
    }

    # Run model
    acc, ARI, NMI, reconstruction_error = model_dispatch[args.model]()

    # Log to wandb
    wandb.init(project="coneClustering", name=f"{args.model}-synthetic")
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
    args = parse_args()
    main(args)
