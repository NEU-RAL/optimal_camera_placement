from typing import List, Optional, Tuple
import numpy as np
from scipy.optimize import linprog, minimize_scalar
import gtsam
import scipy
from enum import Enum
from enum import Enum
import utilities
import FIM as infmat
import numpy as np
import matplotlib.pyplot as plt
from functools import partial
from typing import List, Optional
from gtsam.utils import plot
# from Experiments import exp_utils
from scipy.optimize import minimize, Bounds, LinearConstraint
from numpy import linalg as la
from scipy.sparse import csr_matrix, identity
from scipy.sparse.linalg import eigsh, spsolve, inv
from joblib import Parallel, delayed
import scipy.sparse as sp
from functools import partial
import time
from gurobipy import Model, GRB, quicksum

L = gtsam.symbol_shorthand.L
X = gtsam.symbol_shorthand.X

class Metric(Enum):
    LOGDET = 1
    MIN_EIG = 2
    MSE = 3

def compute_schur_complement(x, inf_mats, H0, num_poses):
    """
    Computes the Schur complement and auxiliary matrices for use in the objective and gradient computations.

    Args:
        x (np.ndarray): Continuous selection vector.
        inf_mats (List[scipy.sparse.csr_matrix]): List of sparse information matrices.
        H0 (scipy.sparse.csr_matrix): Prior information matrix.
        num_poses (int): Number of poses.

    Returns:
        Tuple[scipy.sparse.csc_matrix, np.ndarray, scipy.sparse.csc_matrix, scipy.sparse.csc_matrix]:
        - H_schur: The Schur complement matrix.
        - min_eig_vec: Eigenvector corresponding to the smallest eigenvalue of H_schur.
        - Hll: Submatrix from combined FIM.
        - X: Dense intermediate matrix from solving Hll * X = Hlx.
    """
    # Combine the Fisher Information Matrices
    combined_fim = compute_combined_fim(x, inf_mats, H0)

    # Extract submatrices
    pose_dim = 6
    num_pose_elements = num_poses * pose_dim
    measurement_dim = combined_fim.shape[0] - num_pose_elements
    Hll = combined_fim[:measurement_dim, :measurement_dim].tocsc()
    Hlx = combined_fim[:measurement_dim, measurement_dim:].tocsc()
    Hxx = combined_fim[measurement_dim:, measurement_dim:].tocsc()

    # Compute Schur complement
    try:
        X = spsolve(Hll, Hlx)
    except Exception as e:
        print("Linear solver failed:", e)
        X = Hll.inverse().dot(Hlx).toarray()

    H_schur = Hxx - Hlx.transpose().dot(X)

    # Compute smallest eigenvalue and eigenvector
    try:
        min_eig_val, min_eig_vec = eigsh(H_schur, k=1, which='SA')
    except Exception as e:
        print("Eigenvalue solver failed:", e)
        min_eig_vec = np.zeros(H_schur.shape[1])

    return H_schur, min_eig_val, min_eig_vec, Hll, X


def compute_combined_fim(x, inf_mats, H0):
    combined_fim = H0.copy()
    for xi, Hi in zip(x, inf_mats):
        combined_fim += xi * Hi
    return combined_fim    

def greedy_selection(
    inf_mats: List[np.ndarray],
    prior: np.ndarray,
    Nc: int,
    metric: Metric = Metric.MIN_EIG,
    num_runs: int = 1,
    num_poses: int = None
) -> Tuple[np.ndarray, float, np.ndarray]:
    """
    Greedy selection algorithm to maximize information gain using the Schur complement.

    Args:
        inf_mats (List[np.ndarray or sparse matrix]): List of information matrices for each sensor.
        prior (np.ndarray): Prior information matrix.
        Nc (int): Number of sensors to select.
        metric (Metric): Metric to use for selection.
        num_runs (int): Number of runs (default is 1).
        num_poses (int): Number of poses in the problem (must be provided).

    Returns:
        Tuple[np.ndarray, float, np.ndarray]: Selection vector, best score, and availability vector.
    """
    if num_poses is None:
        raise ValueError("num_poses must be provided and cannot be None.")

    best_selection_indices = []
    best_score = float('-inf')
    avail_cand = np.ones(len(inf_mats), dtype=int)

    # Initialize the combined information matrix with the prior
    combined_inf_mat = prior.copy()

    for run in range(num_runs):
        for i in range(Nc):
            max_inf = float('-inf')
            selected_cand = None
            start_time = time.time()

            for j in range(len(inf_mats)):
                if avail_cand[j] == 1:
                    # Tentatively add the candidate sensor's information matrix
                    temp_inf_mat = combined_inf_mat + inf_mats[j]

                    # Compute the Schur complement
                    total_size = temp_inf_mat.shape[0]
                    pose_dim = 6
                    num_pose_elements = num_poses * pose_dim
                    measurement_dim = total_size - num_pose_elements

                    # Ensure the dimensions of submatrices are correct
                    if measurement_dim <= 0:
                        raise ValueError(f"Invalid measurement dimension: {measurement_dim}")

                    Hll = temp_inf_mat[:measurement_dim, :measurement_dim]
                    Hlx = temp_inf_mat[:measurement_dim, measurement_dim:]
                    Hxx = temp_inf_mat[measurement_dim:, measurement_dim:]

                    # Convert Hll to dense if it is sparse
                    if scipy.sparse.issparse(Hll):
                        Hll = Hll.toarray()

                    # Regularize and compute the pseudoinverse of Hll
                    reg_term = 1e-8 * np.eye(Hll.shape[0])
                    try:
                        Hll_inv = np.linalg.pinv(Hll + reg_term)
                    except np.linalg.LinAlgError as e:
                        raise ValueError(f"Failed to compute pseudoinverse for candidate {j}: {e}")

                    # Compute the Schur complement
                    H_schur = Hxx - Hlx.T @ Hll_inv @ Hlx

                    # Ensure H_schur is symmetric
                    H_schur = (H_schur + H_schur.T) / 2

                    # Compute the minimum eigenvalue of the Schur complement
                    eigvals = np.linalg.eigvalsh(H_schur)
                    min_eig_val = eigvals[0]
                    score = min_eig_val

                    if score > max_inf:
                        max_inf = score
                        selected_cand = j
            
            elapsed_time = time.time() - start_time
            print(f"Iteration {i} compute time: {elapsed_time:.4f} seconds - Min eigen: {max_inf:.4f}")

            if selected_cand is not None:
                best_score = max_inf
                best_selection_indices.append(selected_cand)
                avail_cand[selected_cand] = 0

                # Update the combined information matrix
                combined_inf_mat += inf_mats[selected_cand]

    print("Selected candidates are:", best_selection_indices)

    selection_vector = np.zeros(len(inf_mats))
    selection_vector[best_selection_indices] = 1
    return selection_vector, best_score, avail_cand


def solve_lmo(grad, A, b):
    """
    Solves the Linear Minimization Oracle (LMO) problem for Frank-Wolfe optimization.

    Args:
        grad (np.ndarray): Gradient vector (objective coefficients for the linear program).
        A (np.ndarray or scipy.sparse.csr_matrix): Inequality constraint matrix.
        b (np.ndarray): Inequality constraint bounds.

    Returns:
        np.ndarray or None: Solution vector if the optimization succeeds; otherwise, None.
    """
    num_sensors = len(grad)

    # Define bounds for all variables between 0 and 1
    bounds = [(0, 1) for _ in range(num_sensors)]

    # Use `linprog` from scipy to solve the LMO
    res = linprog(c=grad, A_ub=A, b_ub=b, bounds=bounds, method='highs')

    if res.success:
        return res.x
    else:
        print(f"LMO failed: {res.message}")
        return None


def frank_wolfe_optimization(
    inf_mats: List[csr_matrix],
    H0: csr_matrix,
    selection_init: np.ndarray,
    num_poses: int,
    A: np.ndarray,
    b: np.ndarray
) -> Tuple[np.ndarray, float, int]:
    """
    Performs Frank-Wolfe optimization to select sensors that maximize the smallest eigenvalue of the Schur complement.

    Args:
        inf_mats (List[csr_matrix]): List of sparse information matrices.
        prior (csr_matrix): Prior information matrix.
        n_iters (int): Number of iterations.
        selection_init (np.ndarray): Initial selection vector.
        k (int): Number of sensors to select.
        num_poses (int): Number of poses.
        A (np.ndarray): Inequality constraint matrix.
        b (np.ndarray): Inequality constraint bounds.

    Returns:
        Tuple[np.ndarray, float, int]: Final selection vector, best score (smallest eigenvalue), and number of iterations.
    """
    # Initialize selection vector as a continuous variable (float)
    selection_cur = selection_init.copy().astype(float)
    prev_min_eig = -np.inf  # Initialize to negative infinity for maximization
    
    for iteration in range(1000):
        # Compute the Schur complement and minimum eigenvalue
        try:
            _, min_eig_val, min_eig_vec, Hll, X = compute_schur_complement(selection_cur, inf_mats, H0, num_poses)

        except Exception as e:
            print(f"Error during Schur complement or eigenvalue computation at iteration {iteration}: {e}")
            break

        # Check for convergence
        if abs(min_eig_val - prev_min_eig) < 1e-2:
            print(f"Converged at iteration {iteration}")
            break
        prev_min_eig = min_eig_val

        # Compute gradient
        try:
            t2 = X.transpose()

            # Parallelized gradient computation
            measurement_dim = Hll.shape[0]
            results = Parallel(n_jobs=-1)(
                delayed(compute_grad_parallel)(idx, Hi, Hll, X, t2, min_eig_vec, measurement_dim)
                for idx, Hi in enumerate(inf_mats)
            )

            grad = np.zeros_like(selection_cur)
            for idx, grad_value in results:
                grad[idx] = grad_value
        except Exception as e:
            print(f"Error during gradient computation at iteration {iteration}: {e}")
            break

        # Solve the Linear Minimization Oracle (LMO)
        s = solve_lmo(grad, A, b)
        if s is None:
            print(f"LMO failed to find a feasible solution at iteration {iteration}.")
            break

        # Update step size (classic Frank-Wolfe step size: 2/(t+2))
        alpha = 2 / (iteration + 2)

        # Update the selection vector
        selection_cur = selection_cur + alpha * (s - selection_cur)
        selection_cur = np.clip(selection_cur, 0, 1)

    return selection_cur, min_eig_val, iteration + 1


def roundsolution(selection, k, inf_mats, H0):
   """
   Selects the top `k` elements in the `selection` vector and sets them to 1 in `rounded_sol`.
   Additionally, computes and returns the objective score (smallest eigenvalue).

   Args:
       selection (np.ndarray): Array of selection scores for each candidate.
       k (int): Number of elements to select.
       inf_mats (List[np.ndarray]): Information matrices for each candidate.
       H0 (np.ndarray): Prior information matrix.
       num_poses (int): Number of poses.

   Returns:
       Tuple[np.ndarray, float]: Binary vector where the top `k` elements in `selection` are marked as 1, and the objective score.
   """
   idx = np.argpartition(selection, -k)[-k:]
   rounded_sol = np.zeros(len(selection))
 
   if k > 0:
       rounded_sol[idx] = 1.0
   # Compute the objective score after rounding
   objective_score = evaluate_solution(inf_mats, H0, rounded_sol)
   
   return rounded_sol, objective_score

def roundsolution_breakties(selection, k, all_mats, H0):
    """
    Selects the top `k` elements in the `selection` vector, breaking ties by using the smallest eigenvalue
    of a matrix formed by adding each candidate matrix in `all_mats` to the prior `H0`. This method ensures
    more robust selection by accounting for eigenvalue differences.

    Args:
        selection (np.ndarray): Array of selection scores for each candidate.
        k (int): Number of elements to select.
        all_mats (List[np.ndarray]): List of candidate matrices to compute eigenvalues for tie-breaking.
        H0 (np.ndarray): Prior information matrix added to each candidate matrix.

    Returns:
        np.ndarray: Binary vector where the top `k` elements in `selection`, based on both selection values and
                    eigenvalues, are marked as 1, others as 0.
    """
    s_rnd = np.round(selection, decimals=5)
    all_eigs = []
    for i, m in enumerate(all_mats):
        # Compute the smallest eigenvalue of the matrix H0 + m for each candidate
        m_p = H0 + m

        # Ensure symmetry
        m_p = (m_p + m_p.T) / 2  # Enforce numerical symmetry

        # Add a small regularization term for stability
        reg_term = 1e-8 * np.eye(m_p.shape[0])
        m_p += reg_term

        try:
            # Attempt sparse computation of the smallest eigenvalue
            eigval, _ = eigsh(m_p, k=1, which='SA', maxiter=2000)
        except Exception as e:
            print(f"ARPACK failed for matrix {i}, falling back to dense computation: {e}")
            if m_p.shape[0] <= 1000:  # Use dense computation for smaller matrices
                eigvals = np.linalg.eigh(m_p)[0]
                eigval = eigvals[:1]  # Take the smallest eigenvalue
            else:
                raise RuntimeError(f"Eigenvalue computation failed for large matrix {i}")

        all_eigs.append(eigval[0])  # Store the smallest eigenvalue
    all_eigs = np.array(all_eigs)

    # Combine selection scores and eigenvalues for tie-breaking
    zipped_vals = np.array([(s_rnd[i], all_eigs[i]) for i in range(len(s_rnd))],
                           dtype=[('w', 'float'), ('weight', 'float')])
    idx = np.argpartition(zipped_vals, -k, order=['w', 'weight'])[-k:]

    rounded_sol = np.zeros(len(s_rnd))
    if k > 0:
        rounded_sol[idx] = 1.0
    return rounded_sol


# def roundsolution_madow(selection, k):
#     """
#     Implements a weighted probabilistic rounding to select `k` elements
#     while preserving the relative importance of the `selection` scores.

#     Args:
#         selection (np.ndarray): Array of selection scores for each candidate (non-negative values).
#         k (int): Number of elements to select.

#     Returns:
#         np.ndarray: Binary vector with exactly `k` elements selected probabilistically based on their weights.
#     """
#     num = len(selection)
#     if k > num:
#         raise ValueError("k cannot be greater than the number of candidates.")
    
#     # Normalize the selection scores to form probabilities
#     normalized_selection = selection / np.sum(selection)

#     # Perform weighted random sampling without replacement
#     selected_indices = np.random.choice(
#         np.arange(num), size=k, replace=False, p=normalized_selection
#     )

#     # Construct the binary solution vector
#     rounded_sol = np.zeros(num, dtype=int)
#     rounded_sol[selected_indices] = 1

#     return rounded_sol

def roundsolution_madow(selection, k, inf_mats, H0):
   """
   Uses a probabilistic approach to select `k` candidates based on the cumulative sum of the selection values.
   Additionally, computes and returns the objective score (smallest eigenvalue).

   Args:
       selection (np.ndarray): Array of selection scores for each candidate.
       k (int): Number of elements to select.
       inf_mats (List[np.ndarray]): Information matrices for each candidate.
       H0 (np.ndarray): Prior information matrix.
       num_poses (int): Number of poses.

   Returns:
       Tuple[np.ndarray, float]: Binary vector with exactly `k` elements selected probabilistically based on their cumulative weights, and the objective score.
   """
   num = len(selection)
   phi = np.zeros(num + 1)  
   rounded_sol = np.zeros(num)
   phi[1:] = np.cumsum(selection)  # Cumulative sum of selection scores
   u = np.random.rand()  # Random number for probabilistic selection

   for i in range(k):
       for j in range(num):
           # Check if the random value falls within the cumulative range
           if (phi[j] <= u + i) and (u + i < phi[j + 1]):
               if rounded_sol[j] == 1:  # Ensure the same element isn't selected twice
                   continue
               rounded_sol[j] = 1
               break

   # Compute the objective score after rounding
   objective_score = evaluate_solution(inf_mats, H0, rounded_sol)
   return rounded_sol, objective_score

def evaluate_solution(inf_mats, H0, solution):
   """
   Evaluates a binary solution by computing the smallest eigenvalue of the 
   information matrix constructed from selected sensors.
   
   Args:
       inf_mats (List[scipy.sparse.csr_matrix]): Information matrices for each sensor.
       H0 (scipy.sparse.csr_matrix): Prior information matrix.
       solution (np.ndarray): Binary selection vector for sensors.
   
   Returns:
       float: The smallest eigenvalue of the resulting information matrix.
   """
   # Ensure H0 is in CSC format
   if not scipy.sparse.isspmatrix_csc(H0):
       H0 = H0.tocsc()

   # Initialize the combined information matrix with H0
   combined_fim = H0.copy()

   # Add selected matrices from inf_mats
   for i, selected in enumerate(solution):
       if selected:
           mat = inf_mats[i]
           # Ensure matrix is in CSC format
           if not scipy.sparse.isspmatrix_csc(mat):
               mat = mat.tocsc()
           combined_fim += mat

   # Convert the combined FIM to a dense array for eigenvalue computation
   combined_fim_dense = combined_fim.toarray()

   # Compute the smallest eigenvalue of the combined information matrix
   try:
       eigvals = np.linalg.eigvalsh(combined_fim_dense)
       min_eig_val = eigvals[0]
   except Exception as e:
       print("Error in eigenvalue computation:", e)
       min_eig_val = float('-inf')  # Assign a large negative value for error cases

   return min_eig_val


# Define a function to parallelize the computation for a single Hi
def compute_grad_parallel(idx, Hi, Hll, X, t2, min_eig_vec, measurement_dim):
    """
    Compute gradient contribution for a single Hi matrix.

    Args:
        idx (int): Index of the matrix.
        Hi (scipy.sparse.csr_matrix): Information matrix.
        Hll (scipy.sparse.csc_matrix): Submatrix from combined FIM.
        X (np.ndarray): Dense intermediate matrix from Schur computation.
        t2 (np.ndarray): Transpose of X.
        min_eig_vec (np.ndarray): Smallest eigenvector of Schur complement.
        measurement_dim (int): Dimensionality of the measurement space.

    Returns:
        Tuple[int, float]: Index and gradient contribution.
    """
    # Extract submatrices from Hi (keep sparse)
    Hll_i = Hi[:measurement_dim, :measurement_dim].tocsc()
    Hlx_i = Hi[:measurement_dim, measurement_dim:].tocsc()
    Hxx_i = Hi[measurement_dim:, measurement_dim:].tocsc()

    # Compute grad_schur = Hxx_i - Hlx_i^T * X - t2 * Hll_i * Y + t2 * Hlx_i
    try:
        Y = spsolve(Hll, Hlx_i)
    except Exception as e:
        print(f"Linear solver failed for Hi index {idx}:", e)
        return idx, 0.0

    grad_schur = Hxx_i - Hlx_i.transpose().dot(X) - t2.dot(Hll_i.dot(Y)) + t2.dot(Hlx_i)

    grad_value = -min_eig_vec.flatten().dot(grad_schur.dot(min_eig_vec).flatten())
    return idx, grad_value

def min_eig_grad_vectorized(x, inf_mats, H0, num_poses):
    """
    Computes the gradient of the objective function (Jacobian) using vectorization.

    Args:
        x (np.ndarray): Continuous selection vector.
        inf_mats (List[sp.csr_matrix]): List of sparse information matrices.
        H0 (sp.csr_matrix): Prior information matrix.
        num_poses (int): Number of poses.

    Returns:
        np.ndarray: Gradient vector.
    """
    start_time = time.time()
    
    # Compute Schur complement and get min_eig_vec
    _, _, min_eig_vec, Hll, X = compute_schur_complement(x, inf_mats, H0, num_poses)
    t2 = X.transpose()
    measurement_dim = Hll.shape[0]
    
    n = len(inf_mats)
    
    # Preallocate gradient array
    grad = np.zeros_like(x)
    
    # Convert Hll to dense for faster computations
    Hll_dense = Hll.toarray()
    
    # Precompute Y for all Hi
    # Since Hll_i = Hll (assuming Hll_i same for all Hi), which may not be true
    # If Hll_i are different, this approach won't work
    # Assuming Hll_i are the same as Hll for all Hi (common in some formulations)
    
    # Convert all Hlx_i and Hxx_i to dense
    Hlx_dense = np.array([Hi[:measurement_dim, measurement_dim:].toarray() for Hi in inf_mats])  # Shape: (n, m_l, m_x)
    Hxx_dense = np.array([Hi[measurement_dim:, measurement_dim:].toarray() for Hi in inf_mats])  # Shape: (n, m_x, m_x)
    
    # Solve Y_i = spsolve(Hll, Hlx_i) for all i
    # Vectorization not possible directly, use list comprehension for speed
    Y_dense = np.array([spsolve(Hll, Hi[:measurement_dim, measurement_dim:]).toarray().flatten() for Hi in inf_mats])  # Shape: (n, m_x)
    
    # Compute grad_schur for all Hi
    # grad_schur_i = Hxx_i - Hlx_i^T * X - t2 * Hll_i * Y_i + t2 * Hlx_i
    # Assuming Hll_i is the same as Hll
    # Compute Hlx_i^T * X for all i
    Hlx_T_X = np.einsum('nij,jk->nik', Hlx_dense, X)  # Shape: (n, m_x, m_x)
    
    # Compute t2 * Hll_i * Y_i for all i
    # t2 is (m_x, m_l)
    # Y_i is (n, m_x)
    # Hll_i * Y_i is (n, m_l)
    # t2 * (Hll_i * Y_i) is (n, m_x)
    Hll_Y = Hll_dense @ Y_dense.T  # Shape: (m_l, n)
    t2_Hll_Y = t2 @ Hll_Y  # Shape: (m_x, n)
    t2_Hll_Y = t2_Hll_Y.T  # Shape: (n, m_x)
    
    # Compute t2 * Hlx_i for all i
    t2_Hlx = np.einsum('ij,jk->ik', t2, Hlx_dense)  # Shape: (n, m_x)
    
    # Compute grad_schur for all Hi
    grad_schur = Hxx_dense - Hlx_T_X - t2_Hll_Y + t2_Hlx  # Shape: (n, m_x, m_x)
    
    # Compute grad_value for all Hi
    # grad_value_i = -min_eig_vec^T * grad_schur_i * min_eig_vec
    # min_eig_vec is (m_x,)
    # grad_schur_i is (m_x, m_x)
    # Use einsum for batch computation
    min_eig_vec = min_eig_vec.flatten()
    grad_values = -np.einsum('ij,j,k->i', grad_schur, min_eig_vec, min_eig_vec)  # Shape: (n,)
    
    # Assign to grad vector
    grad = grad_values
    
    run_time = time.time() - start_time
    print(f"Vectorized gradient computation time: {run_time:.4f} seconds")
    return grad

'''
################################################################
Scipy optimization methods
'''
def min_eig_obj(x, inf_mats, H0, num_poses):
    """
    Computes the objective function value (negative smallest eigenvalue of Schur complement).

    Args:
        x (np.ndarray): Continuous selection vector.
        inf_mats (List[scipy.sparse.csr_matrix]): List of sparse information matrices.
        H0 (scipy.sparse.csr_matrix): Prior information matrix.
        num_poses (int): Number of poses.

    Returns:
        float: Objective function value.
    """
    H_schur, _, _, _,_ = compute_schur_complement(x, inf_mats, H0, num_poses)
    # Compute smallest eigenvalue
    try:
        min_eig_val, _ = eigsh(H_schur, k=1, which='SA')
    except Exception as e:
        print("Eigenvalue solver failed:", e)
        min_eig_val = [0.0]

    return -min_eig_val[0]

def min_eig_grad(x, inf_mats, H0, num_poses):
    """
    Computes the gradient of the objective function (Jacobian).

    Args:
        x (np.ndarray): Continuous selection vector.
        inf_mats (List[scipy.sparse.csr_matrix]): List of sparse information matrices.
        H0 (scipy.sparse.csr_matrix): Prior information matrix.
        num_poses (int): Number of poses.

    Returns:
        np.ndarray: Gradient vector.
    """
    start_time = time.time()
    _, _,min_eig_vec, Hll, X = compute_schur_complement(x, inf_mats, H0, num_poses)
    # Precompute constants for gradient
    t2 = X.transpose()

    # Parallelized gradient computation
    measurement_dim = Hll.shape[0]
    results = Parallel(n_jobs=-1)(
        delayed(compute_grad_parallel)(idx, Hi, Hll, X, t2, min_eig_vec, measurement_dim)
        for idx, Hi in enumerate(inf_mats)
    )

    grad = np.zeros_like(x)
    for idx, grad_value in results:
        grad[idx] = grad_value
    
    run_time = time.time() - start_time
    return grad

def scipy_minimize(inf_mats, H0, selection_init, num_poses, A, b):
    """
    Uses `scipy.optimize.minimize` with inequality constraints to solve a sensor selection problem.
    The objective is to maximize the smallest eigenvalue of the FIM.

    Args:
        inf_mats (List[np.ndarray]): List of information matrices for each candidate sensor configuration.
        H0 (np.ndarray): Prior information matrix.
        selection_init (np.ndarray): Initial continuous selection vector.
        k (int): Exact number of sensors to select.
        num_poses (int): Number of poses in the problem.
        A (np.ndarray): Matrix defining inequality constraints.
        b (np.ndarray): Vector defining inequality constraints.

    Returns:
        Tuple[np.ndarray, float]: Continuous solution vector, maximum minimum eigenvalue.
    """
    # Optimization function that maximizes the smallest eigenvalue
    # Define bounds for all variables between 0 and 1
    bounds = Bounds([0] * len(selection_init), [1] * len(selection_init))

    # Define linear constraint Ax <= b
    linear_constraint = LinearConstraint(A, -np.inf, b)

    # Partial functions for objective and gradient
    obj_fun = partial(min_eig_obj, inf_mats=inf_mats, H0=H0, num_poses=num_poses)
    grad_fun = partial(min_eig_grad, inf_mats=inf_mats, H0=H0, num_poses=num_poses)

    # Optimization function that maximizes the smallest eigenvalue
    res = minimize(
        fun=obj_fun,
        x0=selection_init,  # Initial guess for the optimization variables
        method='SLSQP',  # Use Sequential Least Squares Quadratic Programming
        jac=grad_fun,
        constraints=[linear_constraint], 
        bounds=bounds,  # Provide bounds
        options={'disp': True, 'maxiter': 10000, 'ftol': 1e-1} )

    # Get the minimum eigenvalue of the continuous solution
    min_eig_val_unr, _, _ = infmat.find_min_eig_pair(inf_mats, res.x, H0, num_poses)

    return res.x, min_eig_val_unr

'''
################################################################
Scipy optimization methods with Log sum exponential
'''

def min_eig_obj_lse(x, inf_mats, H0, num_poses, beta=5.0):
    """
    Computes the objective function value using the Log-Sum-Exp (LSE) approximation.
    """
    # Use the helper function to compute the Schur complement and auxiliary matrices
    H_schur, _, _, _, _ = compute_schur_complement(x, inf_mats, H0, num_poses)

    # Compute all eigenvalues of H_schur
    try:
        if H_schur.shape[0] <= 500:  # Threshold to determine when to switch to dense
            # Convert sparse matrix to dense for full eigenvalue computation
            eigvals, _ = np.linalg.eigh(H_schur.toarray())
        else:
            # For very large matrices, use eigsh for performance and check fallback
            eigvals, _ = eigsh(H_schur, k=H_schur.shape[0] - 1, which="SA")
    except Exception as e:
        print(f"Eigenvalue computation failed: {e}")
        return np.inf 

    # Stabilize Log-Sum-Exp computation
    eigvals_shifted = eigvals - eigvals.min()  # Shift to stabilize
    scaled_exp_eigvals = np.exp(-beta * eigvals_shifted)
    weight_sum = np.sum(scaled_exp_eigvals) + 1e-12

    # Objective value using Log-Sum-Exp
    f = (-1 / beta) * np.log(weight_sum) + eigvals.min()
    return -f

def min_eig_grad_lse(x, inf_mats, H0, num_poses, beta=5.0):
    """
    Computes the gradient of the objective function using the Log-Sum-Exp (LSE) approximation.
    """
    # Use the helper function to compute the Schur complement and auxiliary matrices
    H_schur, _,_, Hll, X = compute_schur_complement(x, inf_mats, H0, num_poses)

    try:
        if H_schur.shape[0] <= 500:
            eigvals, eigvecs = np.linalg.eigh(H_schur.toarray())
        else:
            eigvals, eigvecs = eigsh(H_schur, k=H_schur.shape[0] - 1, which="SA")
    except Exception as e:
        print(f"Eigenvalue computation failed: {e}")
        return np.zeros_like(x)  # Return zero gradient if computation fails

    eigvals_shifted = eigvals - eigvals.min()
    scaled_exp_eigvals = np.exp(-beta * eigvals_shifted)
    weight_sum = np.sum(scaled_exp_eigvals) + 1e-12
    softmax_weights = scaled_exp_eigvals / weight_sum

    # Parallelized gradient computation
    def compute_grad_lse(idx, H_j, eigvecs, softmax_weights):
        Hll_j = H_j[:Hll.shape[0], :Hll.shape[0]].tocsc()
        Hlx_j = H_j[:Hll.shape[0], Hll.shape[0]:].tocsc()
        Hxx_j = H_j[Hll.shape[0]:, Hll.shape[0]:].tocsc()

        try:
            Hll_j_inv = spsolve(Hll, sp.identity(Hll.shape[0]).tocsc())
        except Exception as e:
            print(f"Linear solver failed for H_j index {idx}: {e}")
            Hll_j_inv = Hll.inverse()

        H_schur_j = Hxx_j - Hlx_j.T @ Hll_j_inv @ Hlx_j
        H_schur_j = (H_schur_j + H_schur_j.T) / 2  # Ensure symmetry

        lambda_derivatives = np.array([
            eigvecs[:, i].T @ H_schur_j @ eigvecs[:, i]
            for i in range(len(eigvals))
        ])
        return idx, np.sum(softmax_weights * lambda_derivatives)

    results = Parallel(n_jobs=-1)(
        delayed(compute_grad_lse)(idx, H_j, eigvecs, softmax_weights)
        for idx, H_j in enumerate(inf_mats)
    )

    # Collect results into the gradient vector
    grad = np.zeros_like(x)
    for idx, grad_value in results:
        grad[idx] = grad_value

    return -grad

def scipy_minimize_lse(inf_mats, H0, selection_init, num_poses, A, b):
    """
    Uses `scipy.optimize.minimize` with inequality constraints to solve a sensor selection problem
    using the Log-Sum-Exp (LSE) approximation, separating the objective and gradient calculations.
    """
    # Set bounds for each variable in x (between 0 and 1)
    bounds = [(0, 1) for _ in range(selection_init.shape[0])]

    # Define constraints
    cons = [{'type': 'ineq', 'fun': lambda x: b - A @ x}]

    # Define objective and gradient as separate functions
    def objective(x):
        return min_eig_obj_lse(x, inf_mats, H0, num_poses)

    def gradient(x):
        return min_eig_grad_lse(x, inf_mats, H0, num_poses)

    # Optimization function
    res = minimize(
        fun=objective,
        x0=selection_init,
        method='SLSQP',
        jac=gradient,
        constraints=cons,
        bounds=bounds,
        options={'disp': True, 'maxiter': 1000, 'ftol': 1e-2}
    )

    # Get the approximated minimum eigenvalue
    f_opt = min_eig_obj_lse(res.x, inf_mats, H0, num_poses)
    approx_min_eig_val = -f_opt

    return res.x, approx_min_eig_val

'''
################################################################
GUROBI Implementation
'''

# Define the callback for Branch and Cut
def branch_and_cut_callback(model, where):
    if where == GRB.Callback.MIPSOL:
        # Retrieve the solution values as a list
        sol = model.cbGetSolution([v for v in model._x.values()])
        
        # Identify selected indices
        selected_indices = [i for i, val in enumerate(sol) if val > 0.5]
        if not selected_indices:
            return

        # Combine the FIMs based on selected indices
        combined_fim = model._H0.copy()
        for idx in selected_indices:
            combined_fim += model._inf_mats[idx]

        # Compute Schur complement and minimum eigenvalue
        try:
            pose_dim = 6
            num_pose_elements = model._num_poses * pose_dim
            measurement_dim = combined_fim.shape[0] - num_pose_elements

            Hll = combined_fim[:measurement_dim, :measurement_dim].tocsc()
            Hlx = combined_fim[:measurement_dim, measurement_dim:].tocsc()
            Hxx = combined_fim[measurement_dim:, measurement_dim:].tocsc()

            X = spsolve(Hll, Hlx)
            H_schur = Hxx - Hlx.T @ X

            # Ensure H_schur is symmetric
            H_schur = (H_schur + H_schur.T) / 2

            # Compute the minimum eigenvalue
            min_eig_val, _ = eigsh(H_schur, k=1, which='SA')
        except Exception as e:
            return

        # Add the lazy constraint
        model.cbLazy(model._min_eig_var <= min_eig_val[0])

def gurobi_branch_and_cut(inf_mats, H0, num_sensors, k, num_poses, A, b, upper_bound=1e6):
    """
    Solves the sensor selection problem using Gurobi's MIP solver with Branch and Cut.

    Args:
        inf_mats (List[scipy.sparse.csr_matrix]): List of sparse information matrices.
        H0 (scipy.sparse.csr_matrix): Prior information matrix.
        num_sensors (int): Number of candidate sensors.
        k (int): Number of sensors to select.
        num_poses (int): Number of poses.
        A (scipy.sparse.csr_matrix): Inequality constraint matrix.
        b (np.ndarray): Inequality constraint bounds.
        upper_bound (float): Upper bound for min_eig_var to prevent unboundedness.

    Returns:
        Tuple[List[int], float]: Selected sensors and the best score (maximum minimum eigenvalue).
    """
    # Ensure H0 is in CSC format for efficient arithmetic operations
    H0 = H0.tocsc()
    
    # Ensure all inf_mats are in CSC format
    inf_mats = [Hi.tocsc() for Hi in inf_mats]

    # Optionally, precompute an upper bound for min_eig_var
    try:
        combined_fim_max = H0.copy()
        for Hi in inf_mats:
            combined_fim_max += Hi
        max_eig_val, _ = eigsh(combined_fim_max, k=1, which='LA')
        upper_bound = max_eig_val[0] * 2  # Set upper bound slightly above the maximum eigenvalue
    except Exception as e:
        return
        
    # Initialize Gurobi model
    model = Model("SensorSelection")
    model.Params.LazyConstraints = 1

    # Add decision variables for sensor selection
    x = model.addVars(num_sensors, vtype=GRB.BINARY, name="x")

    # Add a continuous variable for minimum eigenvalue with an upper bound
    min_eig_var = model.addVar(vtype=GRB.CONTINUOUS, name="min_eig_var", ub=upper_bound)

    # Set the objective to maximize the minimum eigenvalue
    model.setObjective(min_eig_var, GRB.MAXIMIZE)
    
    # Add constraint to select exactly k sensors
    model.addConstr(x.sum() == k+1, name="sensor_selection")
    

    # Attach additional data to the model
    model._x = x
    model._min_eig_var = min_eig_var
    model._inf_mats = inf_mats
    model._H0 = H0
    model._num_poses = num_poses

    # Set the callback
    model.optimize(branch_and_cut_callback)

    # Check if an optimal solution was found
    if model.status == GRB.OPTIMAL or model.status == GRB.SUBOPTIMAL:
        # Extract the solution
        selected_sensors = [j for j in range(num_sensors) if x[j].X > 0.5]
        best_score = min_eig_var.X
        return selected_sensors, best_score
    else:
        print(f"Optimization was not successful. Status code: {model.status}")
        return None, None
