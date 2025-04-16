import numpy as np
from scipy.optimize import nnls

def anls(V, num_components, max_iter=1000, tol=1e-10):
    """
    Alternating Non-negative Least Squares (ANLS) for NMF.

    Args:
        V (numpy.ndarray): The input non-negative matrix.
        num_components (int): The number of components (rank) for factorization.
        max_iter (int, optional): Maximum number of iterations. Defaults to 1000.
        tol (float, optional): Tolerance for convergence. Defaults to 1e-6.

    Returns:
        tuple: Matrices W and H such that V ≈ WH.
    """
    n_rows, n_cols = V.shape
    
    # Initialize W and H randomly with non-negative values
    W = np.random.rand(n_rows, num_components)
    H = np.random.rand(num_components, n_cols)
    
    for _ in range(max_iter):
        H_old = H.copy()
        
        # Update W
        for i in range(n_rows):
            W[i], _ = nnls(H.T, V[i])
        
        # Update H
        for j in range(n_cols):
            H[:, j], _ = nnls(W, V[:, j])
            
        # Check for convergence
        if np.linalg.norm(H - H_old) < tol:
            break
    
    return W, H

def projected_GD(V, num_components, max_iter=1000, tol=1e-10, eta=0.001):
    """
    Perform Non-negative Matrix Factorization using Projected Gradient Descent.

    Args:
        V (numpy.ndarray): The input non-negative matrix.
        num_components (int): The number of components (rank) for factorization.
        max_iter (int): Maximum number of iterations.
        tol (float): Tolerance for convergence.
        eta (float): Learning rate for updates.

    Returns:
        W (numpy.ndarray): Basis matrix where V ≈ WH.
        H (numpy.ndarray): Coefficient matrix where V ≈ WH.
    """
    n_rows, n_cols = V.shape

    # Initialize W and H with random non-negative values
    W = np.random.rand(n_rows, num_components)
    H = np.random.rand(num_components, n_cols)

    for iteration in range(max_iter):
        # Current approximation
        WH = np.dot(W, H)

        # Calculate the gradients
        grad_W = np.dot((WH - V), H.T)
        grad_H = np.dot(W.T, (WH - V))

        # Update W and H
        W -= eta * grad_W
        H -= eta * grad_H

        # Projection to enforce non-negativity
        W = np.maximum(W, 0)
        H = np.maximum(H, 0)

        # Calculate the objective function
        cost = np.linalg.norm(V - np.dot(W, H), 'fro')

        # Convergence check
        if cost < tol:
            print(f"Converged at iteration {iteration} with cost {cost}")
            break

        # Optionally, print progress
        if iteration % 100 == 0:
            print(f"Iteration {iteration}: cost {cost}")

    return W, H