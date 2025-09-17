import numpy as np
import wandb
from sklearn.metrics import accuracy_score, adjusted_rand_score, normalized_mutual_info_score
from sklearn.decomposition import NMF
from nmf import *
from sklearn.linear_model import Lasso
from sklearn import cluster
from libnmf.gnmf import GNMF
from libnmf.nmfbase import NMFBase
from sklearn.cluster import KMeans
from sklearn.neighbors import kneighbors_graph
from scipy.sparse import csgraph
from scipy.sparse.linalg import svds
from utils import *
from scipy.optimize import nnls, minimize
from sklearn.preprocessing import normalize

def data_simulation(m, r, n_k, K, sigma=0.0, random_state=None):
    """
    Simulate K distinct subspaces in R^m, each with dimension r,
    ensuring they differ by generating orthonormal bases.

    Parameters
    ----------
    m : int
        Dimension of the ambient space.
    r : int
        Dimension (rank) of each subspace.
    n_k : int
        Number of points per subspace.
    K : int
        Number of subspaces.
    sigma : float
        Standard deviation of the Gaussian noise to add to each subspace (optional).
    random_state : int or None
        Random seed for reproducibility.

    Returns
    -------
    X : ndarray of shape (m, K*n)
        Concatenated data for all subspaces.
    labels : ndarray of shape (K*n,)
        True cluster labels indicating the subspace for each column.
    """
    np.random.seed(random_state)

    X_list = []
    labels_list = []

    # Generate K distinct orthonormal bases
    for k in range(K):
        # Orthonormal basis for the k-th subspace
        U_k = np.abs(np.random.rand(m, r))
        V_k = np.abs(np.random.rand(r, n_k))
        X_k = np.dot(U_k, V_k)
        if sigma > 0:
            noise = np.random.normal(0, sigma, X_k.shape)
            X_k += noise
        # Ensure non-negativity
        X_k = np.maximum(X_k, 0)
        X_list.append(X_k)
        labels_list.append(np.full(n_k, k, dtype=int))

    # Concatenate subspace data (columns) and labels
    X = np.concatenate(X_list, axis=1)       # shape (m, K*n_k)
    labels = np.concatenate(labels_list)     # shape (K*n_k,)

    return X, labels

def baseline_nmf(X, r, max_iter=1000, tol=1e-6, random_state=42):
    np.random.seed(random_state)
    U, V = anls(X, r, max_iter=max_iter, tol=tol)
    X_new = U @ V
    reconstruction_error = np.linalg.norm(X_new - X) / np.linalg.norm(X)
    print(f"Baseline NMF reconstruction error: {reconstruction_error}")
    return U, V, reconstruction_error

def baseline_ksubspace(X, r, K, true_labels, max_iter=1000, tol=1e-6, random_state=None):
    np.random.seed(random_state)
    n = X.shape[1]
    # 1) Initialize cluster labels randomly
    cluster_labels = np.tile(np.arange(K), n//K)
    np.random.shuffle(cluster_labels)   
    iter = 0

    while iter < max_iter:
        sub_datasets = []
        subspace_bases = []
        for k_ in range(K):
            idx_k = np.where(cluster_labels == k_)[0]
            sub_datasets.append(X[:, idx_k])

            U_k, _, _ = np.linalg.svd(sub_datasets[k_], full_matrices=False)
            # zero truncate U_k
            U_k = np.where(U_k > 0, U_k, 0)
            subspace_bases.append(U_k[:, :r])
        
        new_labels = np.zeros_like(cluster_labels)
        for i in range(n):
            x_i = X[:, i]
            best_k = 0
            best_dist = np.inf
            for k_ in range(K):
                U_k = subspace_bases[k_]
                proj_i = U_k @ np.linalg.pinv(U_k.T @ U_k) @ (U_k.T @ x_i)
                dist = np.linalg.norm(x_i - proj_i)
                if dist < best_dist:
                    best_dist = dist
                    best_k = k_
            new_labels[i] = best_k

        num_changed = np.sum(new_labels != cluster_labels)
        if num_changed == 0:
            break
        cluster_labels = new_labels.copy()
        iter += 1

    acc = remap_accuracy(true_labels, cluster_labels)
    ARI = adjusted_rand_score(true_labels, cluster_labels)
    NMI = normalized_mutual_info_score(true_labels, cluster_labels)
    
    return cluster_labels, acc, ARI, NMI

def baseline_ssc(X, true_labels, alpha):
    # X: [number of samples, dimension]
    n_samples = X.shape[1]
    X = X - X.mean(axis=1, keepdims=True)

    C = np.zeros((n_samples, n_samples))

    # Perform Lasso regression for each sample
    for i in range(n_samples):
        x_i = X[:, i]
        X_rest = np.delete(X, i, axis=1)

        # Solve Lasso problem: minimize ||x_i - X_rest * c||^2 + alpha * ||c||_1
        lasso = Lasso(alpha=alpha, fit_intercept=False)#, max_iter=1000)
        lasso.fit(X_rest, x_i)
        c = lasso.coef_

        # Insert c into C matrix
        C[np.arange(n_samples) != i, i] = c
    # 0 truncate C
    C[C < 0] = 0
    # Symmetrize the affinity matrix
    C = 0.5 * (C + C.T)

    # Run spectral clustering on C
    n_clusters = len(np.unique(true_labels))
    spectral = cluster.SpectralClustering(n_clusters=n_clusters, affinity='precomputed', assign_labels='kmeans', random_state=42)
    cluster_labels = spectral.fit_predict(C)
    print("length of cluster_labels: ", len(cluster_labels))

    # Return ARI
    acc = remap_accuracy(true_labels, cluster_labels)
    ARI = adjusted_rand_score(true_labels, cluster_labels)
    NMI = normalized_mutual_info_score(true_labels, cluster_labels)

    return cluster_labels, acc, ARI, NMI

def ksub_nmf_baseline(X, r, K, true_labels, max_iter=1000, tol=1e-6, random_state=None):
    np.random.seed(random_state)
    n = X.shape[1]
    # 1) Initialize cluster labels randomly
    cluster_labels = np.tile(np.arange(K), n//K)
    np.random.shuffle(cluster_labels)

    # run k-subspace clustering once
    new_labels = np.zeros_like(cluster_labels)
    for i in range(n):
        x_i = X[:, i]
        best_k = 0
        best_dist = np.inf
        for k_ in range(K):
            U_k = np.linalg.svd(X[:, cluster_labels == k_], full_matrices=False)[0]
            U_k = np.where(U_k > 0, U_k, 0)
            proj_i = U_k @ np.linalg.pinv(U_k.T @ U_k) @ (U_k.T @ x_i)
            dist = np.linalg.norm(x_i - proj_i)
            if dist < best_dist:
                best_dist = dist
                best_k = k_
        new_labels[i] = best_k

    acc = remap_accuracy(true_labels, cluster_labels)
    ARI = adjusted_rand_score(true_labels, cluster_labels)
    NMI = normalized_mutual_info_score(true_labels, cluster_labels)

    sub_datasets = []
    subspace_bases = []
    X_new = np.zeros_like(X)
    for k_ in range(K):
        idx_k = np.where(new_labels == k_)[0]
        sub_datasets.append(X[:, idx_k])

        if len(sub_datasets[k_]) == 0:
            subspace_bases.append(None)
            continue
        else:
            U_k = NMF(n_components=r, random_state=random_state, max_iter=1000).fit_transform(sub_datasets[k_])
            U_k = np.where(U_k > 0, U_k, 0)
            subspace_bases.append(U_k[:, :r])

        X_k = sub_datasets[k_]
        X_new[:, idx_k] = U_k @ np.linalg.pinv(U_k.T @ U_k) @ (U_k.T @ X_k)
    # calculate reconstruction error
    reconstruction_error = np.linalg.norm(X_new - X) / np.linalg.norm(X)

    return reconstruction_error, acc, ARI, NMI

def ssc_nmf_baseline(X, r, K, true_labels, max_iter=1000, random_state=None, alpha=0.01):
    np.random.seed(random_state)

    # Step 1: SSC clustering
    pred_labels, acc, ARI, NMI = baseline_ssc(X, true_labels, alpha=alpha)

    # Step 2: Initialize containers
    sub_datasets = []
    subspace_bases = []
    X_new = np.zeros_like(X)

    # Step 3: Per-cluster NMF
    for k_ in range(K):
        idx_k = np.where(pred_labels == k_)[0]
        X_k = X[:, idx_k]
        sub_datasets.append(X_k)

        if X_k.shape[1] == 0:
            subspace_bases.append(None)
            continue

        # Fit NMF and store basis
        U_k = NMF(n_components=r, random_state=random_state, max_iter=max_iter).fit_transform(X_k)
        U_k = np.maximum(U_k, 0)  # optional ReLU
        subspace_bases.append(U_k[:, :r])

        # Project back into subspace
        X_new[:, idx_k] = U_k @ np.linalg.pinv(U_k.T @ U_k) @ (U_k.T @ X_k)

    # Step 4: Evaluate reconstruction error
    reconstruction_error = np.linalg.norm(X_new - X) / np.linalg.norm(X)

    return acc, ARI, NMI, reconstruction_error


def coneClus_iterative(X, K, r, true_labels, max_iter=50, random_state=None, nmf_method='anls', nmf_solver='cd'):
    """
    Iterative subspace clustering with NMF until convergence.
    
    Parameters
    ----------
    X : ndarray of shape (m, n)
        The data matrix with m features and n data points.
    K : int
        Number of clusters (subspaces).
    r : int
        Rank for NMF.
    true_labels : ndarray of shape (n,)
        True labels for evaluation (if available).
    max_iter : int
        Maximum number of iterations.
    tol : float
        Tolerance for stopping criterion (change in labels).
    random_state : int or None
        Random seed for reproducibility.
        
    Returns
    -------
    X_new : ndarray of shape (m, n)
        Reconstructed/Transformed data from the last iteration.
    cluster_labels : ndarray of shape (n,)
        Cluster labels assigned to each column of X in the last iteration.
    errors : list of float
        Reconstruction errors per cluster from the last iteration.
    accuracy : float
        Adjusted Rand Index comparing final cluster labels to true labels.
    proportion_negatives : float
        Fraction of negative entries in the final reconstructed data matrix.
    n_iter : int
        Number of iterations before convergence or max_iter is reached.
    """
    np.random.seed(random_state)
    n = X.shape[1] 
    
    # 1) Initialize cluster labels randomly
    # cluster_labels = np.random.randint(low=0, high=K, size=n)
    cluster_labels = np.tile(np.arange(K), n//K)
    np.random.shuffle(cluster_labels)   

    n_iter = 0
    X_new = None

    for iteration in range(max_iter):
        # 2) Partition X by cluster labels
        sub_datasets = []
        for k_ in range(K):
            idx_k = np.where(cluster_labels == k_)[0]
            sub_datasets.append(X[:, idx_k])

        # 3) NMF on each partition
        nmf_bases = []
        nmf_components = []
        reconstructed_subs = []

        for k_ in range(K):
            x_k = sub_datasets[k_]
            if x_k.shape[1] == 0:
                # Empty cluster
                nmf_bases.append(None)
                nmf_components.append(None)
                reconstructed_subs.append(None)
                continue
            if nmf_method == 'anls':
                U_k, V_k = anls(x_k, r, max_iter=1000, tol=1e-10)
            elif nmf_method == 'NMF':
                model = NMF(n_components=r, init='nndsvda', random_state=random_state, max_iter=1000, solver=nmf_solver)
                x_k = np.maximum(x_k, 0)  # Ensure non-negativity]
                U_k = model.fit_transform(x_k)
                V_k = model.components_
            x_k_new = U_k @ V_k

            err_k = np.linalg.norm(x_k_new - x_k)

            nmf_bases.append(U_k)      # shape (m, r)
            nmf_components.append(V_k) # shape (r, #cols in partition)
            reconstructed_subs.append(x_k_new)

        # 4) Rebuild a full reconstructed matrix X_new
        X_new = np.zeros_like(X)
        for k_ in range(K):
            idx_k = np.where(cluster_labels == k_)[0]
            if reconstructed_subs[k_] is not None:
                X_new[:, idx_k] = reconstructed_subs[k_]

        # 5) Reassign cluster labels
        new_labels = np.zeros_like(cluster_labels)
        for j in range(n):
            x_j = X_new[:, j]
            best_k = 0
            best_dist = np.inf
            for k_ in range(K):
                if nmf_bases[k_] is None:
                    continue
                U_k = nmf_bases[k_] # shape (m, r)
                U_k = np.where(U_k > 0, U_k, 0)
                proj_j = U_k @ np.linalg.pinv(U_k.T @ U_k) @ (U_k.T @ x_j)
                dist = np.linalg.norm(x_j - proj_j)
                # print(f"Dist to cluster {k_}: {dist}")
                if dist < best_dist:
                    best_dist = dist
                    best_k = k_
                # print(f'Point {j} distance to cluster {k_}: {dist:.4f}')
            new_labels[j] = best_k

        # 6) Check convergence
        num_changed = np.sum(new_labels != cluster_labels)
        if num_changed == 0:
            break
    
        # Otherwise, update and continue
        cluster_labels = new_labels.copy()

        n_iter += 1

    # Final metrics
    negatives = np.sum(X_new < 0)
    proportion_negatives = negatives / X_new.size
    reconstruction_error = np.linalg.norm(X_new - X) / np.linalg.norm(X)  # Final loss of reconstruction
    acc = remap_accuracy(true_labels, cluster_labels)
    ARI = adjusted_rand_score(true_labels, cluster_labels)
    NMI = normalized_mutual_info_score(true_labels, cluster_labels)

    return acc, ARI, NMI, reconstruction_error, proportion_negatives


def iter_reg_coneclus(X, K, r, true_labels, max_iter=50, random_state=None,
                      NMF_method='anls', NMF_solver='cd', alpha=0.01, ord=2):
    np.random.seed(random_state)
    n = X.shape[1]
    m = X.shape[0]

    # 1) Initialize cluster labels randomly
    cluster_labels = np.tile(np.arange(K), n // K + 1)[:n]
    np.random.shuffle(cluster_labels)

    n_iter = 0

    for iteration in range(max_iter):
        # 2) Partition X by cluster labels
        sub_datasets = [X[:, cluster_labels == k_] for k_ in range(K)]

        # 3) NMF on each partition
        nmf_bases = [] # bases
        nmf_components = [] # coefficients

        for k_ in range(K):
            x_k = sub_datasets[k_]

            if x_k.shape[1] == 0:
                nmf_bases.append(None)
                nmf_components.append(None)
                continue

            if NMF_method == 'anls':
                U_k, V_k = anls(x_k, r, max_iter=1000, tol=1e-10)
            elif NMF_method == 'NMF':
                model = NMF(n_components=r, init='nndsvda', random_state=random_state,
                            max_iter=1000, solver=NMF_solver)
                x_k = np.maximum(x_k, 0)
                U_k = model.fit_transform(x_k)
                V_k = model.components_
            else:
                raise ValueError(f"Unknown NMF method: {NMF_method}")

            nmf_bases.append(U_k)
            nmf_components.append(V_k)

        # 4) Reassign cluster labels (without reconstructing full X_new)
        new_labels = np.zeros_like(cluster_labels)

        all_dists = np.zeros((K, n))
        for k_ in range(K):
            U_k = nmf_bases[k_]
            if U_k is None or U_k.shape[1] == 0:
                all_dists[k_] = np.inf  # Assign large distance for all points
                continue
            C_k = np.linalg.lstsq(U_k, X, rcond=None)[0]  # (r, n)

            # ReLU and projection
            C_k_relu = np.where(C_k > 0, C_k, 0)
            X_proj_k = U_k @ C_k # shape (m, n)

            # Compute distances
            distances_k = np.linalg.norm(X - X_proj_k, axis=0) + alpha * np.linalg.norm(C_k_relu, ord=ord, axis=0)

            all_dists[k_] = distances_k
            new_labels = np.argmin(all_dists, axis=0)

        # 5) Check convergence
        num_changed = np.sum(new_labels != cluster_labels)
        if num_changed == 0:
            break

        cluster_labels = new_labels.copy()
        n_iter += 1

    X_reconstructed = np.zeros_like(X)
    for k_ in range(K):
        idx_k = np.where(cluster_labels == k_)[0]
        if nmf_bases[k_] is not None and nmf_components[k_] is not None:
            x_k = X[:, idx_k]
            if x_k.shape[1] > 0:
                if NMF_method == 'anls':
                    U_k, V_k = anls(x_k, r, max_iter=1000, tol=1e-10)
                elif NMF_method == 'NMF':
                    model = NMF(n_components=r, init='nndsvda', random_state=random_state,
                                max_iter=1000, solver=NMF_solver)
                    x_k = np.maximum(x_k, 0)
                    U_k = model.fit_transform(x_k)
                    V_k = model.components_
                else:
                    raise ValueError(f"Unknown NMF method: {NMF_method}")
                
                X_reconstructed[:, idx_k] = U_k @ V_k


    reconstruction_error = np.linalg.norm(X_reconstructed - X) / np.linalg.norm(X)
    proportion_negatives = np.sum(X_reconstructed < 0) / X_reconstructed.size
    acc = remap_accuracy(true_labels, cluster_labels)
    ARI = adjusted_rand_score(true_labels, cluster_labels)
    NMI = normalized_mutual_info_score(true_labels, cluster_labels)

    return acc, ARI, NMI, reconstruction_error, proportion_negatives

def iter_reg_coneclus_optimized(X, K, r, true_labels, max_iter=50, random_state=None,
                                NMF_method='anls', NMF_solver='cd', alpha=0.01, ord=2):
    np.random.seed(random_state)
    n = X.shape[1]
    m = X.shape[0]

    # Initialize cluster labels
    cluster_labels = np.tile(np.arange(K), n // K + 1)[:n]
    np.random.shuffle(cluster_labels)

    # Pre-allocate structures
    nmf_bases = [None] * K
    nmf_components = [None] * K
    n_iter = 0

    for iteration in range(max_iter):
        # Partition data by clusters
        sub_datasets = [X[:, cluster_labels == k_] for k_ in range(K)]

        # NMF on each partition
        for k_ in range(K):
            x_k = sub_datasets[k_]
            if x_k.shape[1] == 0:
                nmf_bases[k_] = None
                nmf_components[k_] = None
                continue

            if NMF_method == 'anls':
                # Placeholder for ANLS implementation
                U_k, V_k = anls(x_k, r, max_iter=1000, tol=1e-10)
            elif NMF_method == 'NMF':
                model = NMF(n_components=r, init='nndsvda', random_state=random_state,
                            max_iter=300, solver=NMF_solver)
                x_k = np.maximum(x_k, 0)
                U_k = model.fit_transform(x_k)
                V_k = model.components_
            else:
                raise ValueError(f"Unknown NMF method: {NMF_method}")

            nmf_bases[k_] = U_k
            nmf_components[k_] = V_k

        # Label reassignment using projection distances
        all_dists = np.full((K, n), np.inf)

        for k_ in range(K):
            U_k = nmf_bases[k_]
            if U_k is None or U_k.shape[1] == 0:
                continue
            C_k, _, _, _ = np.linalg.lstsq(U_k, X, rcond=None)
            C_k_relu = np.where(C_k > 0, C_k, 0)
            X_proj_k = U_k @ C_k
            distances_k = np.linalg.norm(X - X_proj_k, axis=0) + alpha * np.linalg.norm(C_k_relu, ord=ord, axis=0)
            all_dists[k_] = distances_k

        new_labels = np.argmin(all_dists, axis=0)
        num_changed = np.sum(new_labels != cluster_labels)

        if num_changed == 0:
            break

        cluster_labels = new_labels.copy()
        n_iter += 1

    # Final reconstruction
    X_reconstructed = np.zeros_like(X)
    for k_ in range(K):
        idx_k = np.where(cluster_labels == k_)[0]
        if len(idx_k) == 0:
            continue
        x_k = X[:, idx_k]
        if NMF_method == 'anls':
            U_k, V_k = anls(x_k, r, max_iter=1000, tol=1e-10)
        elif NMF_method == 'NMF':
            model = NMF(n_components=r, init='nndsvda', random_state=random_state,
                        max_iter=300, solver=NMF_solver)
            x_k = np.maximum(x_k, 0)
            U_k = model.fit_transform(x_k)
            V_k = model.components_
        else:
            raise ValueError(f"Unknown NMF method: {NMF_method}")
        X_reconstructed[:, idx_k] = U_k @ V_k

    # Metrics
    reconstruction_error = np.linalg.norm(X_reconstructed - X) / np.linalg.norm(X)
    proportion_negatives = np.sum(X_reconstructed < 0) / X_reconstructed.size
    acc = remap_accuracy(true_labels, cluster_labels)
    ARI = adjusted_rand_score(true_labels, cluster_labels)
    NMI = normalized_mutual_info_score(true_labels, cluster_labels)

    return acc, ARI, NMI, reconstruction_error, proportion_negatives

def GNMF_clus(X, K, true_labels, max_iter=1000, random_state=None, lmd=0.1, weight_type='heat-kernel', param=0.3):
    """
    Graph-based Non-negative Matrix Factorization (GNMF) for subspace clustering.
    """
    # base = NMFBase(X, K)
    model = GNMF(X, K)
    model.compute_factors(max_iter=max_iter, lmd=lmd, weight_type=weight_type, param=param)

    W = model.W
    H = model.H

    H_array = np.asarray(H)
    predicted_labels = KMeans(n_clusters=K, random_state=random_state).fit_predict(H_array.T)

    acc = remap_accuracy(true_labels, predicted_labels)
    ari = adjusted_rand_score(true_labels, predicted_labels)
    nmi = normalized_mutual_info_score(true_labels, predicted_labels)
    reconstruction_error = np.linalg.norm(X - W @ H) / np.linalg.norm(X)

    return acc, ari, nmi, reconstruction_error

def iter_reg_coneclus_warmstart(X, K, r, true_labels, max_iter=50, random_state=42,
                                NMF_method='anls', alpha=0.01, ord=2):
    np.random.seed(random_state)
    n = X.shape[1]
    m = X.shape[0]

    # 1. Random init of cluster labels
    cluster_labels = np.tile(np.arange(K), n // K + 1)[:n]
    np.random.shuffle(cluster_labels)

    # 2. Warm start: track old cluster assignment
    prev_labels = np.full(n, -1)
    
    # 3. Store W, H for each cluster
    nmf_bases = [None] * K
    nmf_components = [None] * K

    model_template = lambda: NMF(n_components=r, init='random', random_state=random_state,
                                 max_iter=1000, solver='cd')

    for iteration in range(max_iter):
        # Flag to check if cluster content changed
        cluster_changed = [False] * K

        # 4. Partition data by clusters
        sub_datasets = [X[:, cluster_labels == k_] for k_ in range(K)]

        # 5. NMF on each cluster with warm start
        for k_ in range(K):
            x_k = sub_datasets[k_]
            if x_k.shape[1] == 0:
                nmf_bases[k_] = None
                nmf_components[k_] = None
                continue

            # Check if cluster content changed
            idx = (cluster_labels == k_)
            cluster_changed[k_] = not np.array_equal(prev_labels[idx], cluster_labels[idx])

            if not cluster_changed[k_] and nmf_bases[k_] is not None:
                # Skip recomputing if cluster didn't change
                continue

            model = model_template()
            x_k = np.maximum(x_k, 0)

            if nmf_bases[k_] is not None and nmf_components[k_] is not None:
                # Warm start
                try:
                    U_k = model.fit_transform(x_k, W=nmf_bases[k_], H=nmf_components[k_])
                except:
                    U_k = model.fit_transform(x_k)  # fallback
            else:
                U_k = model.fit_transform(x_k)

            V_k = model.components_
            nmf_bases[k_] = U_k
            nmf_components[k_] = V_k

        # 6. Reassign labels using projection distances
        all_dists = np.full((K, n), np.inf)

        for k_ in range(K):
            U_k = nmf_bases[k_]
            if U_k is None or U_k.shape[1] == 0:
                continue
            try:
                # Precompute (UᵀU)^(-1) UᵀX for efficiency
                C_k, _, _, _ = np.linalg.lstsq(U_k, X, rcond=None)
                C_k_relu = np.where(C_k > 0, C_k, 0)
                X_proj_k = U_k @ C_k_relu
                dist_k = np.linalg.norm(X - X_proj_k, axis=0) + alpha * np.linalg.norm(C_k_relu, ord=ord, axis=0)
                all_dists[k_] = dist_k
            except np.linalg.LinAlgError:
                continue

        new_labels = np.argmin(all_dists, axis=0)
        num_changed = np.sum(new_labels != cluster_labels)

        if num_changed == 0:
            break

        prev_labels = cluster_labels.copy()
        cluster_labels = new_labels.copy()

    # 7. Final reconstruction (optional: reuse warm-start W, H)
    X_reconstructed = np.zeros_like(X)
    for k_ in range(K):
        idx_k = np.where(cluster_labels == k_)[0]
        if len(idx_k) == 0:
            continue
        x_k = X[:, idx_k]
        x_k = np.maximum(x_k, 0)
        model = model_template()
        U_k = model.fit_transform(x_k)
        V_k = model.components_
        X_reconstructed[:, idx_k] = U_k @ V_k

    # 8. Evaluation
    reconstruction_error = np.linalg.norm(X_reconstructed - X) / np.linalg.norm(X)
    proportion_negatives = np.sum(X_reconstructed < 0) / X_reconstructed.size
    acc = remap_accuracy(true_labels, cluster_labels)
    ARI = adjusted_rand_score(true_labels, cluster_labels)
    NMI = normalized_mutual_info_score(true_labels, cluster_labels)

    return acc, ARI, NMI, reconstruction_error, proportion_negatives

def iter_ssc_nmf(X, K, r, true_labels, max_iter=50, random_state=None,
                 alpha_ssc=0.01, l1_reg=0.1, use_sparse=True):
    np.random.seed(random_state)
    n = X.shape[1]
    m = X.shape[0]

    cluster_labels = ssc_func(X.T, K, alpha=alpha_ssc)
    prev_labels = np.full(n, -1)

    nmf_bases = [None] * K
    nmf_components = [None] * K

    def run_nmf_block(X_block, r, W_init=None, H_init=None):
        if use_sparse:
            return sparse_nmf(X_block, r, l1_reg=l1_reg, W_init=W_init, H_init=H_init)
        else:
            model = NMF(n_components=r, init='random', random_state=random_state,
                               max_iter=500)
            W = model.fit_transform(X_block)
            H = model.components_
            return W, H

    for iteration in range(max_iter):
        cluster_changed = [False] * K
        sub_datasets = [X[:, cluster_labels == k_] for k_ in range(K)]

        for k_ in range(K):
            x_k = sub_datasets[k_]
            if x_k.shape[1] == 0:
                nmf_bases[k_] = None
                nmf_components[k_] = None
                continue

            idx = (cluster_labels == k_)
            cluster_changed[k_] = not np.array_equal(prev_labels[idx], cluster_labels[idx])
            if not cluster_changed[k_] and nmf_bases[k_] is not None:
                continue

            x_k = np.maximum(x_k, 0)
            W_init = nmf_bases[k_]
            H_init = nmf_components[k_]
            if H_init is not None and H_init.shape[1] != x_k.shape[1]:
                H_init = None

            Wk, Hk = run_nmf_block(x_k, r, W_init=W_init, H_init=H_init)
            nmf_bases[k_] = Wk
            nmf_components[k_] = Hk

        # Recluster using SSC on fresh NMF latent features
        X_latent = np.zeros((r, n))
        for k_ in range(K):
            idx_k = np.where(cluster_labels == k_)[0]
            if len(idx_k) == 0:
                continue
            x_k = X[:, idx_k]
            x_k = np.maximum(x_k, 0)
            _, Hk = run_nmf_block(x_k, r)
            if Hk.shape[1] != len(idx_k):
                continue
            X_latent[:, idx_k] = Hk

        new_labels = ssc_func(X_latent.T, K, alpha=alpha_ssc)
        num_changed = np.sum(new_labels != cluster_labels)
        if num_changed == 0:
            break

        prev_labels = cluster_labels.copy()
        cluster_labels = new_labels.copy()

    # Final reconstruction
    X_reconstructed = np.zeros_like(X)
    for k_ in range(K):
        idx_k = np.where(cluster_labels == k_)[0]
        if len(idx_k) == 0:
            continue
        x_k = X[:, idx_k]
        x_k = np.maximum(x_k, 0)
        W_init = nmf_bases[k_]
        H_init = nmf_components[k_]
        if H_init is not None and H_init.shape[1] != x_k.shape[1]:
            H_init = None

        Wk, Hk = run_nmf_block(x_k, r, W_init=W_init, H_init=H_init)
        X_reconstructed[:, idx_k] = Wk @ Hk

    reconstruction_error = np.linalg.norm(X_reconstructed - X) / np.linalg.norm(X)
    proportion_negatives = np.sum(X_reconstructed < 0) / X_reconstructed.size
    acc = remap_accuracy(true_labels, cluster_labels)
    ARI = adjusted_rand_score(true_labels, cluster_labels)
    NMI = normalized_mutual_info_score(true_labels, cluster_labels)

    return acc, ARI, NMI, reconstruction_error, proportion_negatives

def ssc_sparse_nmf_baseline(X, r, K, true_labels, max_iter=1000, random_state=None, alpha=0.01, l1_reg=0.1):
    np.random.seed(random_state)
    
    # 1. SSC clustering
    pred_labels, acc, ARI, NMI = baseline_ssc(X, true_labels, alpha=alpha)

    sub_datasets = []
    subspace_bases = []
    X_reconstructed = np.zeros_like(X)

    for k_ in range(K):
        idx_k = np.where(pred_labels == k_)[0]
        X_k = X[:, idx_k]
        sub_datasets.append(X_k)

        if X_k.shape[1] == 0:
            subspace_bases.append(None)
            continue

        # 2. Sparse NMF on each cluster
        W, H = sparse_nmf(X_k, r=r, l1_reg=l1_reg)
        subspace_bases.append(W)

        # 3. Reconstruct each cluster
        X_reconstructed[:, idx_k] = W @ H

    # 4. Compute reconstruction error
    reconstruction_error = np.linalg.norm(X_reconstructed - X) / np.linalg.norm(X)

    return acc, ARI, NMI, reconstruction_error

def iter_reg_coneclus_sparse_nmf(X, K, r, true_labels, max_iter=50, random_state=None,
                                 alpha=0.01, ord=2, l1_reg=0.1):
    np.random.seed(random_state)
    n = X.shape[1]
    m = X.shape[0]

    # 1. Random init of cluster labels
    cluster_labels = np.tile(np.arange(K), n // K + 1)[:n]
    np.random.shuffle(cluster_labels)

    # 2. Warm start: track old cluster assignment
    prev_labels = np.full(n, -1)
    
    # 3. Store W, H for each cluster
    nmf_bases = [None] * K
    nmf_components = [None] * K

    def sparse_nmf(X_block, r, W_init=None, H_init=None, n_iter=200):
        W = np.random.rand(X_block.shape[0], r) if W_init is None else W_init
        H = np.random.rand(r, X_block.shape[1]) if H_init is None else H_init
        for _ in range(n_iter):
            WH = W @ H
            H *= (W.T @ X_block) / (W.T @ WH + l1_reg + 1e-10)
            W *= (X_block @ H.T) / (W @ (H @ H.T) + 1e-10)
        return W, H

    for iteration in range(max_iter):
        cluster_changed = [False] * K
        sub_datasets = [X[:, cluster_labels == k_] for k_ in range(K)]

        for k_ in range(K):
            x_k = sub_datasets[k_]
            if x_k.shape[1] == 0:
                nmf_bases[k_] = None
                nmf_components[k_] = None
                continue

            idx = (cluster_labels == k_)
            cluster_changed[k_] = not np.array_equal(prev_labels[idx], cluster_labels[idx])
            if not cluster_changed[k_] and nmf_bases[k_] is not None:
                continue

            x_k = np.maximum(x_k, 0)
            W_init = nmf_bases[k_]
            H_init = nmf_components[k_]
            if H_init is not None and H_init.shape[1] != x_k.shape[1]:
                H_init = None

            Wk, Hk = sparse_nmf(x_k, r=r, W_init=W_init, H_init=H_init, n_iter=200)
            nmf_bases[k_] = Wk
            nmf_components[k_] = Hk

        # 6. Reassign labels using projection distances
        all_dists = np.full((K, n), np.inf)

        for k_ in range(K):
            U_k = nmf_bases[k_]
            if U_k is None or U_k.shape[1] == 0:
                continue
            try:
                # Project entire X onto each U_k
                C_k, _, _, _ = np.linalg.lstsq(U_k, X, rcond=None)
                C_k_relu = np.maximum(C_k, 0)
                X_proj_k = U_k @ C_k_relu
                dist_k = np.linalg.norm(X - X_proj_k, axis=0) + alpha * np.linalg.norm(C_k_relu, ord=ord, axis=0)
                all_dists[k_] = dist_k
            except np.linalg.LinAlgError:
                continue

        new_labels = np.argmin(all_dists, axis=0)
        num_changed = np.sum(new_labels != cluster_labels)
        if num_changed == 0:
            break

        prev_labels = cluster_labels.copy()
        cluster_labels = new_labels.copy()

    # 7. Final reconstruction
    X_reconstructed = np.zeros_like(X)
    for k_ in range(K):
        idx_k = np.where(cluster_labels == k_)[0]
        if len(idx_k) == 0:
            continue
        x_k = X[:, idx_k]
        x_k = np.maximum(x_k, 0)
        Wk, Hk = sparse_nmf(x_k, r=r, n_iter=200)
        X_reconstructed[:, idx_k] = Wk @ Hk

    # 8. Evaluation
    reconstruction_error = np.linalg.norm(X_reconstructed - X) / np.linalg.norm(X)
    proportion_negatives = np.sum(X_reconstructed < 0) / X_reconstructed.size
    acc = remap_accuracy(true_labels, cluster_labels)
    ARI = adjusted_rand_score(true_labels, cluster_labels)
    NMI = normalized_mutual_info_score(true_labels, cluster_labels)

    return acc, ARI, NMI, reconstruction_error, proportion_negatives

def gpca_nmf(X, K, r, true_labels, use_sparse=False, l1_reg=0.1, random_state=None):
    """
    GPCA + NMF pipeline using sklearn-friendly tools.
    """
    np.random.seed(random_state)
    n = X.shape[1]
    X_reconstructed = np.zeros_like(X)

    # Step 1: GPCA clustering
    pred_labels = approximate_gpca(X, K)

    # Step 2: Cluster-wise NMF
    for k in range(K):
        idx_k = np.where(pred_labels == k)[0]
        X_k = X[:, idx_k]
        if X_k.shape[1] == 0:
            continue
        X_k = np.maximum(X_k, 0)

        if use_sparse:
            W = np.random.rand(X_k.shape[0], r)
            H = np.random.rand(r, X_k.shape[1])
            for _ in range(200):
                WH = W @ H
                H *= (W.T @ X_k) / (W.T @ WH + l1_reg + 1e-10)
                W *= (X_k @ H.T) / (W @ (H @ H.T) + 1e-10)
        else:
            model = SklearnNMF(n_components=r, init='random', max_iter=1000, random_state=random_state)
            W = model.fit_transform(X_k)
            H = model.components_

        X_reconstructed[:, idx_k] = W @ H

    # Step 3: Evaluation
    acc = remap_accuracy(true_labels, pred_labels)
    ARI = adjusted_rand_score(true_labels, pred_labels)
    NMI = normalized_mutual_info_score(true_labels, pred_labels)
    reconstruction_error = np.linalg.norm(X_reconstructed - X) / np.linalg.norm(X)

    return acc, ARI, NMI, reconstruction_error

def ssc_projnmf(X, K, r, true_labels, alpha=0.01, max_iter=500):
    """
    SSC + Projective NMF pipeline for representation learning.
    
    Parameters:
        X: (d, n) data matrix
        K: number of clusters
        r: NMF rank
        true_labels: ground-truth labels (n,)
        max_iter: iterations for projective NMF
    
    Returns:
        acc: clustering accuracy from SSC
        ARI: adjusted Rand index from SSC
        NMI: normalized mutual info from SSC
        reconstruction_error: ||X - \hat{X}|| / ||X||
    """
    n = X.shape[1]
    X_reconstructed = np.zeros_like(X)

    # Step 1: SSC clustering
    pred_labels, acc, ARI, NMI = baseline_ssc(X, true_labels, alpha=alpha)

    # Step 2: Cluster-wise Projective NMF
    for k in range(K):
        idx_k = np.where(pred_labels == k)[0]
        X_k = X[:, idx_k]
        if X_k.shape[1] == 0:
            continue

        W_k, loss_k = projective_nmf_orthogonal(X_k, r=r, max_iter=max_iter)
        X_reconstructed[:, idx_k] = W_k @ W_k.T @ X_k

    # Step 3: Evaluate
    reconstruction_error = np.linalg.norm(X - X_reconstructed) / np.linalg.norm(X)

    print("Norm of X:", np.linalg.norm(X))
    print("Norm of X_reconstructed:", np.linalg.norm(X_reconstructed))
    print("Absolute error:", np.linalg.norm(X - X_reconstructed))
    print("Relative error:", np.linalg.norm(X - X_reconstructed) / np.linalg.norm(X))

    return acc, ARI, NMI, reconstruction_error

def onmf_with_relu(X, K, true_labels, r=None, max_iter=100, lambda_reg=0.1, tol=1e-4, verbose=False):
    """
    Orthogonal NMF with ReLU penalty (soft nonnegativity).

    Parameters:
        X: (d, n) data matrix
        K: number of clusters
        true_labels: ground-truth labels (n,)
        r: rank of factorization
        lambda_reg: ReLU penalty weight
        max_iter: maximum iterations
        tol: stopping tolerance
        verbose: print intermediate loss

    Returns:
        acc, ARI, NMI, reconstruction_error
    """
    m, n = X.shape
    if r is None:
        r = K

    # Ensure nonnegative input and normalize
    X = np.maximum(X, 0)
    if X.max() > 1.0:
        X = X / X.max()

    norm_X = np.linalg.norm(X, 'fro')

    # Initialize W via truncated SVD
    U, _, _ = np.linalg.svd(X @ X.T)
    W = np.abs(U[:, :r])
    W = normalize(W, axis=0)

    # Initialize H with NNLS (nonneg only for init)
    H = np.zeros((r, n))
    for i in range(n):
        H[:, i], _ = nnls(W, X[:, i])

    prev_loss = None
    for it in range(max_iter):
        # Update H with soft ReLU penalty (no nonneg bounds)
        for i in range(n):
            def obj(h):
                residual = np.linalg.norm(X[:, i] - W @ h)**2
                relu_penalty = lambda_reg * np.sum(np.maximum(-h, 0))  # ReLU(-h)
                return residual + relu_penalty

            res = minimize(obj, H[:, i], method='L-BFGS-B')
            H[:, i] = res.x

        # Update W using orthogonal Procrustes
        A = X @ H.T
        U, _, Vt = np.linalg.svd(A, full_matrices=False)
        W = U @ Vt
        W = np.abs(W)
        W = normalize(W, axis=0)

        # Total normalized loss
        reconstruction = W @ H
        rec_loss = np.linalg.norm(X - reconstruction, 'fro')
        relu_penalty_total = lambda_reg * np.sum(np.maximum(-H, 0))
        total_loss = (rec_loss**2 + relu_penalty_total) / (norm_X**2)

        if verbose and it % 10 == 0:
            print(f"[Iter {it}] Normalized Loss: {total_loss:.6f} | Recon Loss: {rec_loss:.4f}")

        if prev_loss is not None and abs(prev_loss - total_loss) < tol:
            break
        prev_loss = total_loss

    # Clustering by argmax on H
    labels_pred = H.argmax(axis=0)
    acc = remap_accuracy(true_labels, labels_pred)
    ARI = adjusted_rand_score(true_labels, labels_pred)
    NMI = normalized_mutual_info_score(true_labels, labels_pred)
    recon_error = np.linalg.norm(X - W @ H) / np.linalg.norm(X)

    return acc, ARI, NMI, recon_error