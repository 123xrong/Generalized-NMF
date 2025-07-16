import numpy as np
from sklearn.preprocessing import normalize
from scipy.optimize import nnls, minimize
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans


def onmf_with_relu(X, r, max_iter=100, lambda_reg=0.1, tol=1e-4, verbose=False):
    m, n = X.shape

    # --- Initialize W via SVD ---
    U, _, _ = np.linalg.svd(X @ X.T)
    W = np.maximum(U[:, :r], 1e-8)
    W = normalize(W, axis=0)

    # --- Initialize H via NNLS ---
    H = np.zeros((r, n))
    for i in range(n):
        H[:, i], _ = nnls(W, X[:, i])

    for iter in range(max_iter):
        # --- Update H ---
        for i in range(n):
            def obj(h):
                return np.linalg.norm(X[:, i] - W @ h)**2 + lambda_reg * np.sum(h)

            bounds = [(0, None)] * r
            res = minimize(obj, H[:, i], bounds=bounds, method='L-BFGS-B')
            H[:, i] = res.x

        # --- Update W via orthogonal Procrustes ---
        A = X @ H.T
        U, _, Vt = np.linalg.svd(A, full_matrices=False)
        W = U @ Vt
        W = np.maximum(W, 1e-8)
        W = normalize(W, axis=0)

        # --- Compute loss ---
        reconstruction = W @ H
        loss = np.linalg.norm(X - reconstruction, 'fro')**2 + lambda_reg * np.sum(H)

        if verbose:
            print(f"Iter {iter:03d}: loss = {loss:.4f}")

        if iter > 0 and abs(prev_loss - loss) < tol:
            break
        prev_loss = loss

    return W, H


def cluster_accuracy(y_true, y_pred):
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    row_ind, col_ind = linear_sum_assignment(w.max() - w)
    return w[row_ind, col_ind].sum() / y_pred.size


def evaluate_clustering(H, labels_true, method="kmeans"):

    if method == "kmeans":
        labels_pred = KMeans(n_clusters=len(np.unique(labels_true)), n_init=10).fit(H.T).labels_
    elif method == "argmax":
        labels_pred = H.argmax(axis=0)
    else:
        raise ValueError("Unsupported clustering method.")

    ari = adjusted_rand_score(labels_true, labels_pred)
    nmi = normalized_mutual_info_score(labels_true, labels_pred)
    acc = cluster_accuracy(labels_true, labels_pred)

    return acc, ari, nmi
