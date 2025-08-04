import os
import sys
import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir) 

import re
import numpy as np
import argparse
import wandb
from nmf import *
from coneClustering import *
from modified_dscnmf import *
from baseline import *
from deepNMF import *
from deepSSCNMF import *
from scipy.io import loadmat
from nltk.corpus import stopwords
from nltk import download
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch_geometric.datasets import WebKB
from sklearn.datasets import fetch_20newsgroups, fetch_olivetti_faces, fetch_openml
from sklearn.preprocessing import normalize
from sklearn.feature_extraction.text import TfidfVectorizer

def arg_parser():
    parser = argparse.ArgumentParser(description="Iterative subspace clustering with NMF")
    parser.add_argument('--m', type=int, default=50, help='Dimension of the ambient space (default: 50)')
    parser.add_argument('--r', type=int, default=5, help='Dimension (rank) of each subspace (default: 5)')
    parser.add_argument('--n', type=int, default=50, help='Number of points per subspace (default: 100)')
    parser.add_argument('--K', type=int, default=4, help='Number of subspaces (default: 3)')
    parser.add_argument('--sigma', type=float, default=0.0, help='Standard deviation of Gaussian noise (default: 0.0)')
    parser.add_argument('--alpha', type=float, default=1e-2, help='Regularization parameter for ssc')
    parser.add_argument('--max_iter', type=int, default=1000, help='Maximum number of iterations (default: 50)')
    parser.add_argument('--tol', type=float, default=1e-6, help='Tolerance for stopping criterion (default: 1e-6)')
    parser.add_argument('--random_state', type=int, default=42, help='Random seed for clustering (default: None)')
    parser.add_argument('--dataset', type=str, choices=['20newsgroups', 'olivetti_faces', 'webkb', 'mnist', 'cifar10', 'fashion_mnist'], help='Dataset to use for clustering')
    parser.add_argument('--model', type=str, choices=['sscnmf', 'ricc', 'gnmf', 'gpcanmf', 'lrrnmf', 'onmf'], help='Model to use for clustering')
    parser.add_argument('--l1_reg', type=float, default=0.01, help='L1 regularization parameter for ONMF-ReLU/GPCANMF')
    parser.add_argument('--NMF_method', type=str, default='anls', choices=['mu', 'cd', 'anls'], help='NMF solver method (default: anls)')
    return parser.parse_args()

def main(model, m, r, n, K, sigma=0.0, alpha=0.01, l1_reg=0.01, random_state=42, max_iter=50, tol=1e-6, NMF_method='anls', dataset='20newsgroups'):
    if dataset == '20newsgroups':
        download('stopwords')
        newsgroups = fetch_20newsgroups(subset='all', remove=('headers', 'footers', 'quotes'))
        vectorizer = TfidfVectorizer(stop_words=stopwords.words('english'), max_features=5000)
        X = vectorizer.fit_transform(newsgroups.data).toarray()
        true_labels = newsgroups.target

    elif dataset == 'olivetti_faces':
        faces = fetch_olivetti_faces(shuffle=True, random_state=random_state)
        X_full = faces.data.T
        true_labels = faces.target

        n_subjects = 10
        n_images_per_subject = 10

        # Select only the first `n_subjects` (each subject has 10 images in order)
        selected_indices = []
        for subject_id in range(n_subjects):
            start_idx = subject_id * 10
            end_idx = start_idx + n_images_per_subject
            selected_indices.extend(range(start_idx, end_idx))

        # Subset data and labels
        X = X_full[selected_indices].T  # shape (feature_dim, num_samples)
        true_labels = true_labels[selected_indices]

    elif dataset == 'webkb':
        dataset = WebKB(root='~/data/WebKB', name='Cornell')
        data = dataset[0]
        X = data.x.numpy().T  # feature matrix: (features, samples)
        true_labels = data.y.numpy()    # labels (0-4 for five classes)

    elif dataset == 'mnist':
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
        X = np.vstack(X_list)
        true_labels = np.concatenate(labels) 
        X = normalize(X.T, axis=1)

    elif dataset == 'cifar10':
        transform = transforms.ToTensor()
        cifar = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
        X_cifar = cifar.data.reshape(len(cifar), -1) / 255.0  # Shape: (50000, 3072)
        y_cifar = np.array(cifar.targets)
        # Subsample 500 images
        np.random.seed(42)
        subset_idx = np.random.choice(len(X_cifar), size=500, replace=False)
        X = X_cifar[subset_idx]
        true_labels = y_cifar[subset_idx]

    elif dataset == 'fashion_mnist':
        transform = transforms.ToTensor()
        full_data = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
        X_full = full_data.data.numpy().reshape(len(full_data), -1) / 255.0
        y_full = full_data.targets.numpy()

        subset_idx = np.random.choice(len(X_full), size=1000, replace=False, random_state=random_state)
        X = X_full[subset_idx]
        true_labels = y_full[subset_idx]
        X = X.T

    else:
        raise ValueError(f"Unknown dataset: {dataset}")
    
    if sigma > 0:
        # Add non-negative Gaussian noise to the data
        noise = np.random.normal(0, sigma, X.shape)
        X += noise
        X = normalize(X, axis=1)
    
    if model == 'sscnmf':
        project_name = f'sscnmf-{dataset}'
        acc, ARI, NMI, reconstruction_error = ssc_nmf_baseline(
            X, K=K, r=r, true_labels=true_labels, alpha=alpha)
    elif model == 'ricc':
        project_name = f'ricc-{dataset}'
        acc, ARI, NMI, reconstruction_error, _ = iter_reg_coneclus_warmstart(
            X, K=K, r=r, true_labels=true_labels,
            alpha=alpha, max_iter=max_iter, NMF_method=NMF_method, ord=2, random_state=random_state)
    elif model == 'gnmf':
        project_name = f'gnmf-{dataset}'
        acc, ARI, NMI, reconstruction_error = GNMF_clus(
            X, K=K, true_labels=true_labels, max_iter=max_iter)
    elif model == 'gpcanmf':
        project_name = f'gpcanmf-{dataset}'
        acc, ARI, NMI, reconstruction_error = gpca_nmf(
            X, K=K, r=r, true_labels=true_labels)
    elif model == 'dscnmf':
        project_name = f'dscnmf-{dataset}'
        acc, ARI, NMI, reconstruction_error = dsc_nmf_baseline(
            X, K=K, r=r, true_labels=true_labels)
    elif model == 'onmf':
        project_name = f'onmf-{dataset}'
        acc, ARI, NMI, reconstruction_error = onmf_em(
            X, K=K, true_labels=true_labels)
    elif model == 'deepnmf':
        project_name = f'deepnmf-{dataset}'
        acc, ARI, NMI, reconstruction_error = dsc_nmf_baseline(
            X, K=K, r=r, true_labels=true_labels)
    elif model == 'deepsscnmf':
        project_name = f'deepsscnmf-{dataset}'
        acc, ARI, NMI, reconstruction_error = deep_ssc_nmf(
            X, ranks=[256, 128, 64], alpha=alpha, n_iter=max_iter,
            true_labels=true_labels)
    else:
        raise ValueError(f"Unknown model: {model}")

    wandb.init(
        project="coneClustering",
        name=project_name
    )
    wandb.log({
        "accuracy": acc,
        "ARI": ARI,
        "NMI": NMI,
        "reconstruction_error": reconstruction_error
    })
    print("\n--- Results ---")
    print(f"Accuracy: {acc}")
    print(f"ARI: {ARI}")
    print(f"NMI: {NMI}")            

if __name__ == "__main__":
    args = arg_parser()
    main(model=args.model, m=args.m, r=args.r, n=args.n, K=args.K, sigma=args.sigma,
         alpha=args.alpha, l1_reg=args.l1_reg, random_state=args.random_state,
         max_iter=args.max_iter, tol=args.tol, NMF_method=args.NMF_method, dataset=args.dataset)


