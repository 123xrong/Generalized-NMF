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
from deep_ssc_nmf import *
from sklearn.linear_model import Lasso
from sklearn.preprocessing import normalize
from nmf import *
from sklearn.datasets import fetch_olivetti_faces, fetch_openml 

def arg_parser():
    parser = argparse.ArgumentParser(description="DSSCNMF")
    parser.add_argument('--n_components', type=int, default=10, help='Number of components for NMF')
    parser.add_argument('--alpha', type=float, default=0.01, help='Regularization parameter for SSC')
    parser.add_argument('--max_iter', type=int, default=200, help='Maximum number of iterations for NMF')
    parser.add_argument('--tol', type=float, default=1e-4, help='Tolerance for stopping criterion')
    parser.add_argument('--random_state', type=int, default=42, help='Random seed for NMF')
    parser.add_argument('--hidden_dim', type=int, default=64, help='Hidden dimension for encoder')
    return parser.parse_args()

def main(n_components, alpha=0.01, max_iter=200, tol=1e-4, random_state=42, hidden_dim=64):
    faces = fetch_olivetti_faces(shuffle=True, random_state=random_state)
    X_full = faces.data.T  # shape (4096, 400)
    true_labels = faces.target

    input_dim = X_full.shape[0]
    encoder = ShallowEncoder(input_dim=input_dim, hidden_dim=hidden_dim)

    acc, ari, nmi, C, W, H = encoder_ssc_then_nmf_pipeline(
        X_full, encoder, rank=n_components, n_clusters=40,
        true_labels=true_labels, alpha=alpha)

    print(f"Accuracy: {acc:.4f}, ARI: {ari:.4f}, NMI: {nmi:.4f}")

if __name__ == '__main__':
    args = arg_parser()
    main(**vars(args))