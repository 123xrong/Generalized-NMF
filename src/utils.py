import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, adjusted_rand_score, normalized_mutual_info_score
from sklearn.linear_model import Lasso
from sklearn.cluster import SpectralClustering
from sklearn.metrics.pairwise import cosine_similarity, rbf_kernel
from sklearn.decomposition import NMF as SklearnNMF
from scipy.optimize import linear_sum_assignment
from numpy.linalg import qr

def ssc_func(X, K, alpha=0.01):
    """Basic SSC implementation with spectral clustering."""
    n = X.shape[0]
    C = np.zeros((n, n))
    for i in range(n):
        x_i = X[i]
        X_others = np.delete(X, i, axis=0)
        lasso = Lasso(alpha=alpha, fit_intercept=False, max_iter=1000)
        lasso.fit(X_others.T, x_i)
        c_i = np.insert(lasso.coef_, i, 0)
        C[i] = c_i
    affinity = np.abs(C) + np.abs(C.T)
    clustering = SpectralClustering(n_clusters=K, affinity='precomputed', random_state=0)
    return clustering.fit_predict(affinity)

def sparse_nmf(X_block, r, l1_reg=0.1, W_init=None, H_init=None, n_iter=200):
    W = np.random.rand(X_block.shape[0], r) if W_init is None else W_init
    H = np.random.rand(r, X_block.shape[1]) if H_init is None else H_init
    for _ in range(n_iter):
        WH = W @ H
        H *= (W.T @ X_block) / (W.T @ WH + l1_reg + 1e-10)
        W *= (X_block @ H.T) / (W @ (H @ H.T) + 1e-10)
    return W, H

def approximate_gpca(X, K, affinity='cosine', gamma=20, random_state=None):
    """
    Approximate GPCA using affinity + spectral clustering.
    """
    if affinity == 'cosine':
        S = cosine_similarity(X.T)
    elif affinity == 'rbf':
        S = rbf_kernel(X.T, gamma=gamma)
    else:
        raise ValueError("Unsupported affinity type")

    np.fill_diagonal(S, 0)
    clustering = SpectralClustering(n_clusters=K, affinity='precomputed', random_state=random_state)
    return clustering.fit_predict(S)

def projective_nmf(X, r, max_iter=200, tol=1e-4):
    """
    Projective NMF: X ≈ W Wᵗ X, with W ≥ 0 and Wᵗ W = I (softly)
    """
    d, n = X.shape
    W = np.abs(np.random.randn(d, r))

    for it in range(max_iter):
        WtX = W.T @ X
        WWtX = W @ WtX
        num = X @ WtX.T
        denom = WWtX @ WtX.T + 1e-10
        W *= num / denom

        loss = np.linalg.norm(X - W @ W.T @ X, 'fro')**2
        if it % 50 == 0:
            print(f"[ProjNMF] Iter {it}: Loss = {loss:.4f}")
    return W, loss


def projective_nmf_orthogonal(X, r, max_iter=200, tol=1e-4, verbose=False):
    """
    Projective NMF with hard orthogonality constraint: WᵗW = I, W ≥ 0.

    Args:
        X: (d, n) input matrix
        r: rank of factorization
        max_iter: number of iterations
        tol: tolerance for convergence
        verbose: print loss if True

    Returns:
        W: (d, r) nonnegative orthonormal basis matrix
        loss: final reconstruction loss
    """
    d, n = X.shape
    W = np.abs(np.random.randn(d, r))  # non-negative init

    for it in range(max_iter):
        # Compute gradient direction
        WtX = W.T @ X
        grad = -2 * X @ WtX.T + 2 * W @ WtX @ WtX.T

        # Gradient step
        W -= 0.01 * grad
        W = np.maximum(W, 1e-10)  # nonnegativity

        # Enforce orthogonality: QR + rectification
        Q, _ = np.linalg.qr(W)  # orthonormal columns
        W = np.maximum(Q, 0)    # project back to nonnegativity

        # Optional: normalize columns (not strictly necessary)
        W /= np.linalg.norm(W, axis=0, keepdims=True) + 1e-10

        # Compute reconstruction loss
        X_proj = W @ W.T @ X
        loss = np.linalg.norm(X - X_proj, 'fro')**2

        if verbose and it % 50 == 0:
            print(f"[Iter {it}] Loss: {loss:.4f}")

    return W, loss

def remap_accuracy(y_true, y_pred):
    """
    Compute clustering accuracy with optimal label permutation using Hungarian algorithm.
    
    Parameters:
        y_true (array-like): Ground truth labels, shape (n_samples,)
        y_pred (array-like): Predicted cluster labels, shape (n_samples,)
    
    Returns:
        acc (float): Clustering accuracy after optimal label alignment
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    assert y_true.shape == y_pred.shape
    D = max(y_pred.max(), y_true.max()) + 1
    cost_matrix = np.zeros((D, D), dtype=np.int64)

    for i in range(len(y_true)):
        cost_matrix[y_pred[i], y_true[i]] += 1

    # Solve the assignment problem
    row_ind, col_ind = linear_sum_assignment(-cost_matrix)  # maximize total match

    # Create label mapping
    optimal_mapping = dict(zip(row_ind, col_ind))

    # Map predicted labels
    y_pred_aligned = np.array([optimal_mapping[yi] for yi in y_pred])

    # Compute accuracy
    acc = accuracy_score(y_true, y_pred_aligned)
    return acc

def divide (N, P):
    out = np.copy(N)
    np.divide(out, P, where = np.logical_not(np.isclose(P, 0)), out = out)
    return out

def normalize(M, axis):
    M_sq = M * M
    M_norm = np.sqrt(np.sum(M_sq, axis = axis, keepdims = True))
    ret = np.copy(M)
    np.divide(ret, M_norm, where = np.logical_not(np.isclose(M_norm, 0)), out = ret)
    return ret

# Clustering utility functions for ONMF-EM algorithm
def spherical_k_means(X, K, max_iter=100, random_state=None):
    """
    Spherical k-means used for initializing ONMF centroids.
    Returns:
        - asgn_list: list of sample indices per cluster
        - centers: (d, K) orthonormal basis vectors
    """
    d, n = X.shape
    asgn_list = [[] for _ in range(K)]
    if random_state is not None:
        np.random.seed(random_state)
    for i in range(n):
        asgn_list[np.random.randint(K)].append(i)

    asgn = []
    converged = False
    iter_count = 0

    while not converged and iter_count < max_iter:
        iter_count += 1
        centers = np.random.rand(d, K)
        centers = normalize(centers, axis=0)

        for k in range(K):
            if asgn_list[k]:
                subX = X[:, asgn_list[k]]
                u, _, _ = np.linalg.svd(subX, full_matrices=False)
                centers[:, k] = np.abs(u[:, 0])
            else:
                r = np.random.randint(n)
                centers[:, k] = divide(X[:, r], np.linalg.norm(X[:, r]))

        old_asgn = asgn
        asgn = []
        asgn_list = [[] for _ in range(K)]

        dots = X.T @ centers
        max_dots = np.max(dots, axis=1, keepdims=True)
        # Break ties randomly
        asgn = np.argmax(np.isclose(dots, max_dots) * np.random.random((n, K)), axis=1)

        if len(old_asgn) > 0:
            for i in range(n):
                if np.isclose(dots[i, old_asgn[i]], max_dots[i]):
                    asgn[i] = old_asgn[i]

        for i in range(n):
            asgn_list[asgn[i]].append(i)

        if same_assignment(asgn, old_asgn):
            converged = True

    return asgn_list, centers

def same_assignment(a, b):
    if len(a) != len(b):
        return False
    return all(a[i] == b[i] for i in range(len(a)))

def plot_affinity_heatmap(H, true_labels=None, title="Affinity Matrix"):
    """
    Plot affinity (cosine similarity) matrix from coefficient matrix H.

    H: (r, n) coefficient matrix
    true_labels: optional, reorder samples by labels for block structure
    title: plot title
    """
    # Cosine similarity between samples
    A = cosine_similarity(H.T)
    
    # Reorder by labels if provided
    if true_labels is not None:
        order = np.argsort(true_labels)
        A = A[order][:, order]
    
    plt.figure(figsize=(5, 4))
    sns.heatmap(A, cmap="viridis", cbar=True)
    plt.title(title)
    plt.xlabel("Samples")
    plt.ylabel("Samples")
    plt.tight_layout()
    plt.show()

def plot_reconstructions(X, W, H, n_samples=10, img_shape=None, title_prefix="", random_state=0):
    """
    Plot original vs reconstruction for a subset of samples.

    X: (m, n) original data matrix
    W, H: NMF factors (X ≈ W @ H)
    n_samples: number of samples to show
    img_shape: (h, w) if data are images
    title_prefix: label for the method (e.g., "Global NMF")
    """
    # Randomly select samples
    np.random.seed(random_state)
    idx = np.random.choice(X.shape[1], n_samples, replace=False)
    X_hat = W @ H

    fig, axes = plt.subplots(2, n_samples, figsize=(1.5*n_samples, 3))
    
    for i, j in enumerate(idx):
        orig = X[:, j]
        rec = X_hat[:, j]
        
        if img_shape is not None:
            orig = orig.reshape(img_shape)
            rec = rec.reshape(img_shape)
        
        # Original
        axes[0, i].imshow(orig, cmap="gray")
        axes[0, i].axis("off")
        if i == 0: axes[0, i].set_ylabel("Original")
        
        # Reconstruction
        axes[1, i].imshow(rec, cmap="gray")
        axes[1, i].axis("off")
        if i == 0: axes[1, i].set_ylabel(title_prefix)
    
    plt.suptitle(f"Reconstructions with {title_prefix}")
    plt.tight_layout()
    plt.show()


def assemble_global_H(coeffs, labels, K, r, n):
    """
    Assemble a global H (block-diagonal style) from cluster-wise coefficient matrices.

    coeffs: list of H_k, each of shape (r, n_k) for cluster k
    labels: array of length n, cluster assignment for each sample
    K: number of clusters
    r: rank per cluster
    n: total number of samples

    Returns:
        H_global: (K*r, n) coefficient matrix
    """
    H_global = np.zeros((K * r, n))
    
    for k in range(K):
        H_k = coeffs[k]    # shape (r, n_k)
        if H_k is None:
            continue
        idx_k = np.where(labels == k)[0]
        H_global[k*r:(k+1)*r, idx_k] = H_k

    return H_global

def assemble_global_W(bases, labels, K, r, n):
    """
    Assemble a global W (block-diagonal style) from cluster-wise basis matrices.

    bases: list of W_k, each of shape (d, r) for cluster k
    labels: array of length n, cluster assignment for each sample
    K: number of clusters
    r: rank per cluster
    n: total number of samples

    Returns:
        W_global: (d, K*r) basis matrix
    """
    d = bases[0].shape[0]
    W_global = np.zeros((d, K * r))
    
    for k in range(K):
        W_k = bases[k]    # shape (d, r)
        if W_k is None:
            continue
        W_global[:, k*r:(k+1)*r] = W_k

    return W_global

def visualize_nmf_parts(W, H, img_idx, img_shape, n_bases=5):
    """
    Visualize NMF parts for a chosen image.

    W: (m, r) basis matrix
    H: (r, n) coefficient matrix
    img_idx: index of the image to visualize
    img_shape: tuple (h, w) to reshape vectors into images
    n_bases: number of bases to show
    """

    # Original image
    x_orig = W @ H[:, img_idx]
    x_orig = x_orig.reshape(img_shape)

    fig, axes = plt.subplots(4, n_bases, figsize=(2*n_bases, 8))

    # 1. Basis images (columns of W)
    for i in range(n_bases):
        axes[0, i].imshow(W[:, i].reshape(img_shape), cmap="gray")
        axes[0, i].set_title(f"Basis {i+1}")
        axes[0, i].axis("off")
    axes[0, 0].set_ylabel("Basis", rotation=0, labelpad=40)

    # 2. Contribution to chosen image (rank-1 outer product)
    for i in range(n_bases):
        contrib = (W[:, i] * H[i, img_idx]).reshape(img_shape)
        axes[1, i].imshow(contrib, cmap="gray")
        axes[1, i].axis("off")
    axes[1, 0].set_ylabel("Contribution", rotation=0, labelpad=40)

    # 3. Progressive reconstruction (add parts one by one)
    partial_recons = []
    for i in range(n_bases):
        recon_i = W[:, :i+1] @ H[:i+1, img_idx]
        partial_recons.append(recon_i.reshape(img_shape))
        axes[2, i].imshow(partial_recons[-1], cmap="gray")
        axes[2, i].axis("off")
    axes[2, 0].set_ylabel("Progressive", rotation=0, labelpad=40)

    # 4. Full reconstruction vs original
    for i in range(n_bases):
        if i == 0:
            axes[3, i].imshow(x_orig, cmap="gray")
            axes[3, i].set_title("Full Recon")
        else:
            axes[3, i].axis("off")
    axes[3, 0].set_ylabel("Final", rotation=0, labelpad=40)

    plt.tight_layout()
    plt.show()


def visualize_cluster_nmf_parts(W_k, H_k, img_local_idx, img_shape, n_bases=5):
    """
    Visualize NMF parts for a chosen image within a single cluster.
    
    W_k: (m, r) basis matrix for cluster k
    H_k: (r, n_k) coefficient matrix for cluster k
    img_local_idx: index of the image *within this cluster*
    img_shape: tuple (h, w)
    n_bases: number of bases to show
    """
    x_orig = (W_k @ H_k[:, img_local_idx]).reshape(img_shape)

    fig, axes = plt.subplots(4, n_bases, figsize=(2*n_bases, 8))

    # 1. Basis images
    for i in range(n_bases):
        axes[0, i].imshow(W_k[:, i].reshape(img_shape), cmap="gray")
        axes[0, i].set_title(f"Basis {i+1}")
        axes[0, i].axis("off")
    axes[0, 0].set_ylabel("Basis", rotation=0, labelpad=40)

    # 2. Contributions
    for i in range(n_bases):
        contrib = (W_k[:, i] * H_k[i, img_local_idx]).reshape(img_shape)
        axes[1, i].imshow(contrib, cmap="gray")
        axes[1, i].axis("off")
    axes[1, 0].set_ylabel("Contribution", rotation=0, labelpad=40)

    # 3. Progressive reconstruction
    for i in range(n_bases):
        recon_i = (W_k[:, :i+1] @ H_k[:i+1, img_local_idx]).reshape(img_shape)
        axes[2, i].imshow(recon_i, cmap="gray")
        axes[2, i].axis("off")
    axes[2, 0].set_ylabel("Progressive", rotation=0, labelpad=40)

    # 4. Full reconstruction
    for i in range(n_bases):
        if i == 0:
            axes[3, i].imshow(x_orig, cmap="gray")
            axes[3, i].set_title("Full Recon")
        else:
            axes[3, i].axis("off")
    axes[3, 0].set_ylabel("Final", rotation=0, labelpad=40)

    plt.tight_layout()
    plt.show()
