import numpy as np
from sklearn.preprocessing import normalize
from scipy.optimize import nnls, minimize
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans


def onmf_with_relu(X, K, true_labels=None, r=None, max_iter=100, lambda_reg=0.1, tol=1e-4, verbose=False):
    m, n = X.shape
    if r is None:
        r = K  # default to number of clusters

    norm_X = np.linalg.norm(X, 'fro')**2
    X = np.maximum(X, 0)

    # --- Initialize W ---
    U, _, _ = np.linalg.svd(X @ X.T)
    W = np.maximum(U[:, :r], 1e-8)
    W = normalize(W, axis=0)

    # --- Initialize H ---
    H = np.zeros((r, n))
    for i in range(n):
        H[:, i], _ = nnls(W, X[:, i])

    for iteration in range(max_iter):
        # --- Update H ---
        for i in range(n):
            def obj(h):
                return np.linalg.norm(X[:, i] - W @ h)**2 + lambda_reg * np.sum(h)

            bounds = [(0, None)] * r
            res = minimize(obj, H[:, i], bounds=bounds, method='L-BFGS-B')
            H[:, i] = res.x

        # --- Update W (Procrustes) ---
        A = X @ H.T
        U, _, Vt = np.linalg.svd(A, full_matrices=False)
        W = U @ Vt
        W = np.maximum(W, 1e-8)
        W = normalize(W, axis=0)

        # --- Compute normalized loss ---
        reconstruction = W @ H
        rec_loss = np.linalg.norm(X - reconstruction, 'fro')**2
        l1_penalty = lambda_reg * np.sum(H)
        total_loss = (rec_loss + l1_penalty) / norm_X

        if verbose:
            print(f"Iter {iteration:03d}: norm. loss = {total_loss:.6f}")

        if iteration > 0 and abs(prev_loss - total_loss) < tol:
            break
        prev_loss = total_loss

    # --- Clustering via H ---
    labels_pred = H.argmax(axis=0)

    # --- Final metrics ---
    acc = adjusted_rand_score(true_labels, labels_pred) if true_labels is not None else None
    ari = adjusted_rand_score(true_labels, labels_pred) if true_labels is not None else None
    nmi = normalized_mutual_info_score(true_labels, labels_pred) if true_labels is not None else None
    recon_error = np.linalg.norm(X - W @ H) / np.linalg.norm(X)
    prop_negative = np.sum(W @ H < 0) / (m * n)

    return {
        'W': W,
        'H': H,
        'labels': labels_pred,
        'accuracy': acc,
        'ARI': ari,
        'NMI': nmi,
        'reconstruction_error': recon_error,
        'proportion_negatives': prop_negative,
        'normalized_loss': total_loss
    }


def cluster_accuracy(y_true, y_pred):
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    row_ind, col_ind = linear_sum_assignment(w.max() - w)
    return w[row_ind, col_ind].sum() / y_pred.size
