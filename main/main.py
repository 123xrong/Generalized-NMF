import os
import sys
import inspect

# Set parent directory in path
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

# Core imports
import argparse
import numpy as np
import wandb

# Data and preprocessing
from sklearn.datasets import fetch_20newsgroups, fetch_openml, fetch_olivetti_faces
from sklearn.preprocessing import normalize
from sklearn.feature_extraction.text import TfidfVectorizer
from torchvision import datasets, transforms
from torch_geometric.datasets import WebKB
from nltk.corpus import stopwords
from nltk import download
from scipy.io import loadmat

# Project models
from nmf import *
from GenNMF import *
from modified_dscnmf import *
from baseline import *
from deepNMF import *
from deepSSCNMF import *

def parse_args():
    parser = argparse.ArgumentParser(description="Subspace clustering with NMF variants")
    parser.add_argument('--r', type=int, default=5, help='Subspace dimension / NMF rank')
    parser.add_argument('--n', type=int, default=50, help='Samples per class (only used in MNIST/CIFAR)')
    parser.add_argument('--K', type=int, default=4, help='Number of clusters/subspaces')
    parser.add_argument('--sigma', type=float, default=0.0, help='Std. dev. of Gaussian noise')
    parser.add_argument('--alpha', type=float, default=1e-2, help='Regularization strength')
    parser.add_argument('--max_iter', type=int, default=1000, help='Max iterations for clustering')
    parser.add_argument('--random_state', type=int, default=42, help='Random seed')
    parser.add_argument('--dataset', type=str, required=True,
                        choices=['20newsgroups', 'olivetti_faces', 'webkb', 'mnist', 'cifar10', 'fashion_mnist', 'coil20'])
    parser.add_argument('--model', type=str, required=True,
                        choices=['sscnmf', 'ricc', 'gnmf', 'gpcanmf', 'onmf', 'dscnmf', 'deepnmf', 'deepsscnmf'])
    parser.add_argument('--l1_reg', type=float, default=0.01, help='L1 regularization for ONMF/GPCANMF')
    parser.add_argument('--NMF_method', type=str, default='anls', choices=['mu', 'cd', 'anls'],
                        help='NMF optimization method (default: anls)')
    return parser.parse_args()

def load_dataset(name, K, n, random_state):
    if name == '20newsgroups':
        download('stopwords')
        newsgroups = fetch_20newsgroups(subset='all', remove=('headers', 'footers', 'quotes'))
        vectorizer = TfidfVectorizer(stop_words=stopwords.words('english'), max_features=5000)
        X = vectorizer.fit_transform(newsgroups.data).toarray()
        return X.T, newsgroups.target

    elif name == 'olivetti_faces':
        faces = fetch_olivetti_faces(shuffle=True, random_state=random_state)
        X, y = faces.data.T, faces.target
        selected_idx = [i for subj in range(10) for i in range(subj*10, subj*10+10)]
        return X[:, selected_idx], y[selected_idx]

    elif name == 'webkb':
        dataset = WebKB(root='~/data/WebKB', name='Cornell')
        data = dataset[0]
        return data.x.numpy().T, data.y.numpy()

    elif name == 'mnist':
        mnist = fetch_openml('mnist_784', version=1)
        X_full = mnist.data.to_numpy()
        y_full = mnist.target.astype(int)
        return subset_digits(X_full, y_full, K, n, random_state)

    elif name == 'cifar10':
        cifar = datasets.CIFAR10(root='./data', train=True, download=True, transform=transforms.ToTensor())
        X = cifar.data.reshape(len(cifar), -1) / 255.0
        y = np.array(cifar.targets)
        idx = np.random.RandomState(random_state).choice(len(X), size=500, replace=False)
        return X[idx].T, y[idx]

    elif name == 'fashion_mnist':
        data = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transforms.ToTensor())
        X = data.data.numpy().reshape(len(data), -1) / 255.0
        y = data.targets.numpy()
        idx = np.random.RandomState(random_state).choice(len(X), size=1000, replace=False)
        return X[idx].T, y[idx]
    elif name == 'coil20':
        coil20_data = loadmat('data/COIL20.mat')
        X_full = coil20_data['fea'].T  # shape (feature_dim, num_samples)
        y = coil20_data['gnd'].flatten() - 1  # Convert
        X = normalize(X_full, axis=0)

def subset_digits(X, y, K, n, seed):
    rng = np.random.RandomState(seed)
    X_list, y_list = [], []
    for digit in range(K):
        idx = np.where(y == digit)[0]
        chosen = rng.choice(idx, n, replace=False)
        X_list.append(X[chosen])
        y_list.append(np.full(n, digit))
    X_concat = np.vstack(X_list).astype(float)
    return normalize(X_concat.T, axis=1), np.concatenate(y_list)

def main():
    args = parse_args()
    X, true_labels = load_dataset(args.dataset, args.K, args.n, args.random_state)

    if args.sigma > 0:
        noise = np.random.normal(0, args.sigma, X.shape)
        X += noise
        X = normalize(X, axis=1)

    # Run selected model
    model_map = {
        'sscnmf': lambda: ssc_nmf_baseline(X, K=args.K, r=args.r, true_labels=true_labels, alpha=args.alpha),
        'ricc': lambda: iter_reg_coneclus_warmstart(X, K=args.K, r=args.r, true_labels=true_labels,
                                                    alpha=args.alpha, max_iter=args.max_iter,
                                                    NMF_method=args.NMF_method, ord=2, random_state=args.random_state)[0:4],
        'gnmf': lambda: GNMF_clus(X, K=args.K, true_labels=true_labels, max_iter=args.max_iter),
        'gpcanmf': lambda: gpca_nmf(X, K=args.K, r=args.r, true_labels=true_labels),
        'onmf': lambda: onmf_em(X, K=args.K, true_labels=true_labels),
        'dscnmf': lambda: dsc_nmf_baseline(X, K=args.K, r=args.r, true_labels=true_labels),
        'deepnmf': lambda: dsc_nmf_baseline(X, K=args.K, r=args.r, true_labels=true_labels),
        'deepsscnmf': lambda: deep_ssc_nmf(X, ranks=[256, 128, 64], alpha=args.alpha,
                                           n_iter=args.max_iter, true_labels=true_labels)
    }

    acc, ARI, NMI, recon_err = model_map[args.model]()

    wandb.init(project="coneClustering", name=f"{args.model}-{args.dataset}")
    wandb.log({
        "accuracy": acc,
        "ARI": ARI,
        "NMI": NMI,
        "reconstruction_error": recon_err
    })

    print("\n--- Results ---")
    print(f"Accuracy: {acc:.4f}")
    print(f"ARI: {ARI:.4f}")
    print(f"NMI: {NMI:.4f}")
    print(f"Reconstruction Error: {recon_err:.4f}")

if __name__ == "__main__":
    main()


