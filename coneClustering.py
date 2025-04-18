import numpy as np
from sklearn.metrics import adjusted_rand_score
from sklearn.decomposition import NMF
from nmf import *
from sklearn.linear_model import Lasso
from sklearn import cluster
import wandb


# def random_orthonormal(m, r):
#     """
#     Generate an m x r*K random matrix with orthonormal columns via QR decomposition.
#     """
#     A = np.random.rand(m, r)
#     Q, _ = np.linalg.qr(A)
#     return np.abs(Q)

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
        X_list.append(X_k)
        labels_list.append(np.full(n_k, k, dtype=int))

    # Concatenate subspace data (columns) and labels
    X = np.concatenate(X_list, axis=1)       # shape (m, K*n_k)
    labels = np.concatenate(labels_list)     # shape (K*n_k,)

    return X, labels

def baseline_nmf(X, r, max_iter=1000, tol=1e-6, random_state=None):
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
    
    accuracy = adjusted_rand_score(cluster_labels, true_labels)
    print(f"Baseline K-Subspace clustering distribution: {dict(zip(*np.unique(cluster_labels, return_counts=True)))}")
    print(f"Baseline K-Subspace clustering accuracy (ARI): {accuracy:.4f}")
    
    return cluster_labels, accuracy

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
    ari = adjusted_rand_score(true_labels, cluster_labels)

    return cluster_labels, ari

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

    accuracy = adjusted_rand_score(true_labels, new_labels)
    # run NMF on each partition
    sub_datasets = []
    subspace_bases = []
    X_new = np.zeros_like(X)
    for k_ in range(K):
        idx_k = np.where(new_labels == k_)[0]
        sub_datasets.append(X[:, idx_k])

        if len(sub_datasets[k_]) == 0:
            # Empty cluster
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

    return reconstruction_error, accuracy

def ssc_nmf_baseline(X, r, K, true_labels, max_iter=1000, random_state=None, alpha=0.01):
    np.random.seed(random_state)
    # run sparse subspace clustering once
    pred_labels, accuracy = baseline_ssc(X, true_labels, alpha=alpha)
    # run NMF on each partition
    sub_datasets = []
    subspace_bases = []
    X_new = np.zeros_like(X)
    for k_ in range(K):
        idx_k = np.where(pred_labels == k_)[0]
        sub_datasets.append(X[:, idx_k])

        if len(sub_datasets[k_]) == 0:
            # Empty cluster
            subspace_bases.append(None)
            continue
        else:
            U_k = NMF(n_components=r, random_state=random_state, max_iter=max_iter).fit_transform(sub_datasets[k_])
            U_k = np.where(U_k > 0, U_k, 0)
            subspace_bases.append(U_k[:, :r])

        X_k = sub_datasets[k_]
        X_new[:, idx_k] = U_k @ np.linalg.pinv(U_k.T @ U_k) @ (U_k.T @ X_k)
    reconstruction_error = np.linalg.norm(X_new - X) / np.linalg.norm(X)

    return reconstruction_error, accuracy

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
    accuracy = adjusted_rand_score(true_labels, cluster_labels)
    negatives = np.sum(X_new < 0)
    proportion_negatives = negatives / X_new.size
    reconstruction_error = np.linalg.norm(X_new - X) / np.linalg.norm(X)  # Final loss of reconstruction

    return accuracy, reconstruction_error, proportion_negatives


def iter_reg_coneclus(X, K, r, true_labels, max_iter=50, random_state=None,
                      nmf_method='anls', nmf_solver='cd', alpha=0.01, ord=2):
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
        nmf_bases = []
        nmf_components = []
        UtU_inv_list = []

        for k_ in range(K):
            x_k = sub_datasets[k_]

            if x_k.shape[1] == 0:
                nmf_bases.append(None)
                nmf_components.append(None)
                UtU_inv_list.append(None)
                continue

            if nmf_method == 'anls':
                U_k, V_k = anls(x_k, r, max_iter=1000, tol=1e-10)
            elif nmf_method == 'NMF':
                model = NMF(n_components=r, init='nndsvda', random_state=random_state,
                            max_iter=1000, solver=nmf_solver)
                x_k = np.maximum(x_k, 0)
                U_k = model.fit_transform(x_k)
                V_k = model.components_
            else:
                raise ValueError(f"Unknown NMF method: {nmf_method}")

            nmf_bases.append(U_k)
            nmf_components.append(V_k)

            # Precompute pseudoinverse term
            # UtU = U_k.T @ U_k
            # UtU_inv = np.linalg.pinv(UtU)
            # UtU_inv_list.append(UtU_inv)

        # 4) Reassign cluster labels (without reconstructing full X_new)
        new_labels = np.zeros_like(cluster_labels)

        for j in range(n):
            x_j = X[:, j]
            best_k = 0
            best_dist = np.inf

            for k_ in range(K):
                U_k = nmf_bases[k_]
                UtU_inv = UtU_inv_list[k_]

                if U_k is None or UtU_inv is None:
                    continue

                # Project x_j onto subspace
                proj_coeff, *_ = np.linalg.lstsq(U_k, x_j, rcond=None)
                proj_coeff_relu = np.where(proj_coeff > 0, proj_coeff, 0)
                proj_j = U_k @ proj_coeff_relu

                # Distance + regularization
                dist = np.linalg.norm(x_j - proj_j) + alpha * np.linalg.norm(proj_coeff_relu, ord=ord)

                if dist < best_dist:
                    best_dist = dist
                    best_k = k_

            new_labels[j] = best_k

        # 5) Check convergence
        num_changed = np.sum(new_labels != cluster_labels)
        if num_changed == 0:
            break

        cluster_labels = new_labels.copy()
        n_iter += 1

    # Final reconstruction error (optional)
    X_reconstructed = np.zeros_like(X)
    for k_ in range(K):
        idx_k = np.where(cluster_labels == k_)[0]
        if nmf_bases[k_] is not None and nmf_components[k_] is not None:
            X_reconstructed[:, idx_k] = nmf_bases[k_] @ nmf_components[k_][:, :len(idx_k)]

    reconstruction_error = np.linalg.norm(X_reconstructed - X) / np.linalg.norm(X)
    proportion_negatives = np.sum(X_reconstructed < 0) / X_reconstructed.size
    accuracy = adjusted_rand_score(true_labels, cluster_labels)

    return accuracy, reconstruction_error, proportion_negatives