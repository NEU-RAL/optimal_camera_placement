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
from scipy.sparse import csr_matrix, identity, diags
from scipy.sparse.linalg import eigsh, spsolve
from joblib import Parallel, delayed


L = gtsam.symbol_shorthand.L
X = gtsam.symbol_shorthand.X

class Metric(Enum):
    LOGDET = 1
    MIN_EIG = 2
    MSE = 3

def compute_combined_fim(x, inf_mats, H0):
    combined_fim = H0.copy()
    for xi, Hi in zip(x, inf_mats):
        combined_fim += xi * Hi
    return combined_fim    

def compute_gradient(idx, Hi, Hll_factorized, Hlx, X, t2, min_eig_vec, measurement_dim):
    """
    Computes the gradient for a single index in the loop.
    Args:
        idx (int): Index of the current matrix.
        Hi (sparse matrix): Current sparse information matrix.
        Hll_factorized (SuperLU object): Factorized Hll matrix for efficient solving.
        Hlx (sparse matrix): Landmark-to-pose cross information matrix.
        X (sparse matrix): Intermediate matrix from Schur complement computation.
        t2 (sparse matrix): Precomputed intermediate matrix.
        min_eig_vec (np.ndarray): Eigenvector corresponding to the smallest eigenvalue.
        measurement_dim (int): Dimensionality of the measurement space.

    Returns:
        float: Gradient value for the current index.
    """
    # Extract submatrices from Hi
    Hll_i = Hi[:measurement_dim, :measurement_dim]
    Hlx_i = Hi[:measurement_dim, measurement_dim:]
    Hxx_i = Hi[measurement_dim:, measurement_dim:]

    # Solve Hll * Y = Hlx_i using the pre-factorized Hll
    Y = Hll_factorized.solve(Hlx_i.toarray())

    # Compute grad_schur = Hxx_i - Hlx_i.T @ X - t2 @ Hll_i @ Y + t2 @ Hlx_i
    grad_schur = (
        Hxx_i - Hlx_i.transpose().dot(X)
        - t2.dot(Hll_i.dot(Y))
        + t2.dot(Hlx_i)
    )

    # Flatten vectors to ensure proper alignment
    grad_val = -min_eig_vec.flatten().dot(grad_schur.dot(min_eig_vec).flatten())
    return grad_val


def compute_gradients_parallel(inf_mats, Hll, Hlx, X, t2, min_eig_vec, measurement_dim):
    """
    Parallel computation of gradients using joblib.
    Args:
        inf_mats (List[sparse matrix]): List of sparse information matrices.
        Hll (sparse matrix): Landmark-to-landmark information matrix.
        Hlx (sparse matrix): Landmark-to-pose cross information matrix.
        X (sparse matrix): Intermediate matrix from Schur complement computation.
        t2 (sparse matrix): Precomputed intermediate matrix.
        min_eig_vec (np.ndarray): Eigenvector corresponding to the smallest eigenvalue.
        measurement_dim (int): Dimensionality of the measurement space.

    Returns:
        np.ndarray: Gradient vector for all indices.
    """
    # Factorize Hll once for efficient solving
    Hll_factorized = splu(Hll)

    # Parallel computation
    grad = Parallel(n_jobs=-1)(
        delayed(compute_gradient)(
            idx, Hi, Hll_factorized, Hlx, X, t2, min_eig_vec, measurement_dim
        ) for idx, Hi in enumerate(inf_mats)
    )

    return np.array(grad)


def gradient_main_function(inf_mats, Hll, Hlx, X, t2, min_eig_vec, measurement_dim):
    """
    Main function for computing gradients in parallel.
    Args:
        inf_mats (List[sparse matrix]): List of sparse information matrices.
        Hll (sparse matrix): Landmark-to-landmark information matrix.
        Hlx (sparse matrix): Landmark-to-pose cross information matrix.
        X (sparse matrix): Intermediate matrix from Schur complement computation.
        t2 (sparse matrix): Precomputed intermediate matrix.
        min_eig_vec (np.ndarray): Eigenvector corresponding to the smallest eigenvalue.
        measurement_dim (int): Dimensionality of the measurement space.

    Returns:
        np.ndarray: Gradient vector for all indices.
    """
    grad = compute_gradients_parallel(
        inf_mats, Hll, Hlx, X, t2, min_eig_vec, measurement_dim
    )
    return grad

def greedy_selection(
    inf_mats: List[np.ndarray],
    prior: np.ndarray,
    Nc: int,
    metric: Metric = Metric.MIN_EIG,
    num_runs: int = 1,
    num_poses: int = None  # Make num_poses a required argument
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


def solve_lmo(grad, A, b, k):
    num_sensors = len(grad)
    c = grad
    bounds = [(0, 1) for _ in range(num_sensors)]
    A_eq = [np.ones(num_sensors)]
    b_eq = [k]
    res = linprog(c, A_ub=A, b_ub=b, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')
    if res.success:
        return res.x
    else:
        print("LMO failed to find a feasible solution.")
        return None

def frank_wolfe_optimization(
    inf_mats: List[np.ndarray],
    prior: np.ndarray,
    n_iters: int,
    selection_init: np.ndarray,
    k: int,
    num_poses: int,
    A: np.ndarray,
    b: np.ndarray
) -> Tuple[np.ndarray, float, int]:
    selection_cur = selection_init.copy()
    num_sensors = len(selection_cur)
    prev_min_eig = float('inf')

    for iteration in range(n_iters):
        # Compute the minimum eigenvalue and eigenvector
        min_eig_val, min_eig_vec, final_inf_mat = infmat.find_min_eig_pair(
            inf_mats, selection_cur, prior, num_poses
        )

        # Compute the gradient
        grad = np.zeros(num_sensors)
        # Adjust slicing
        total_size = final_inf_mat.shape[0]
        pose_dim = 6  # Adjust as needed
        expected_pose_elements = num_poses * pose_dim
        measurement_dim = total_size - expected_pose_elements

        Hll = final_inf_mat[:measurement_dim, :measurement_dim]
        Hlx = final_inf_mat[:measurement_dim, measurement_dim:]
        Hxx = final_inf_mat[measurement_dim:, measurement_dim:]

        Hll_inv = np.linalg.pinv(Hll)

        t0 = Hlx.T
        t1 = Hll_inv
        t2 = t0 @ t1

        for idx in range(num_sensors):
            Hc = inf_mats[idx]
            Hll_c = Hc[:measurement_dim, :measurement_dim]
            Hlx_c = Hc[:measurement_dim, measurement_dim:]
            Hxx_c = Hc[measurement_dim:, measurement_dim:]

            grad_schur = Hxx_c - (
                Hlx_c.T @ Hll_inv @ Hlx
                - t2 @ Hll_c @ Hll_inv @ Hlx
                + t2 @ Hlx_c
            )
            grad[idx] = -min_eig_vec.T @ grad_schur @ min_eig_vec

        # Solve the LMO
        s = solve_lmo(grad, A, b, k)
        if s is None:
            print("LMO failed to find a feasible solution.")
            break

        # Step size
        alpha = 1.0 / (iteration + 2.0)

        # Update the current selection
        selection_cur = selection_cur + alpha * (s - selection_cur)
        print(f"Objective value: {min_eig_val}")

        # Check for convergence
        if abs(min_eig_val - prev_min_eig) < 1e-4:
            print(f"Converged at iteration {iteration}")
            break
        prev_min_eig = min_eig_val

    # Final solution
    final_solution = selection_cur.copy()
    min_eig_val_unrounded, _, _ = infmat.find_min_eig_pair(
        inf_mats, selection_cur, prior, num_poses
    )
    min_eig_val, _, _ = infmat.find_min_eig_pair(
        inf_mats, final_solution, prior, num_poses
    )
    return selection_cur, min_eig_val, iteration + 1

def roundsolution(selection, k):
    """
    Selects the top `k` elements in the `selection` vector and sets them to 1 in `rounded_sol`.
    This method does not handle ties specifically, so it may arbitrarily choose elements if
    there are multiple candidates with the same value around the `k`-th element.

    Args:
        selection (np.ndarray): Array of selection scores for each candidate.
        k (int): Number of elements to select.

    Returns:
        np.ndarray: Binary vector where the top `k` elements in `selection` are marked as 1, others as 0.
    """
    idx = np.argpartition(selection, -k)[-k:]
    rounded_sol = np.zeros(len(selection))
    if k > 0:
        rounded_sol[idx] = 1.0
    return rounded_sol

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
    for m in all_mats:
        # Compute the smallest eigenvalue of the matrix H0 + m for each candidate
        m_p = H0 + m
        assert utilities.check_symmetric(m_p)  # Ensure symmetry of the matrix
        eigvals, _ = la.eigh(m_p)
        all_eigs.append(eigvals[0])  # Store the smallest eigenvalue
    all_eigs = np.array(all_eigs)

    # Combine selection scores and eigenvalues for tie-breaking
    zipped_vals = np.array([(s_rnd[i], all_eigs[i]) for i in range(len(s_rnd))], dtype=[('w', 'float'), ('weight', 'float')])
    idx = np.argpartition(zipped_vals, -k, order=['w', 'weight'])[-k:]
    
    rounded_sol = np.zeros(len(s_rnd))
    if k > 0:
        rounded_sol[idx] = 1.0
    return rounded_sol

def roundsolution_madow(selection, k):
    """
    Uses a probabilistic approach to select `k` candidates based on the cumulative sum of the selection values.
    This method introduces randomness to the rounding process, which is useful if a non-deterministic selection is preferred.

    Args:
        selection (np.ndarray): Array of selection scores for each candidate.
        k (int): Number of elements to select.

    Returns:
        np.ndarray: Binary vector with exactly `k` elements selected probabilistically based on their cumulative weights.
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
    # print("Number of candidates selected after rounding:", np.sum(rounded_sol))
    return rounded_sol

def evaluate_solution(inf_mats, H0, solution, num_poses):
    """
    Evaluates a binary solution by computing the smallest eigenvalue of the 
    information matrix constructed from selected sensors.
    
    Args:
        inf_mats (List[np.ndarray]): Information matrices for each sensor.
        H0 (np.ndarray): Prior information matrix.
        solution (np.ndarray): Binary selection vector for sensors.
        num_poses (int): Number of poses.
    
    Returns:
        float: The smallest eigenvalue of the resulting information matrix.
    """
    # Build the combined information matrix based on the selected sensors
    combined_fim = H0.copy()
    for i, selected in enumerate(solution):
        if selected:
            combined_fim += inf_mats[i]
    
    # Compute the smallest eigenvalue of the combined information matrix
    eigvals = np.linalg.eigvalsh(combined_fim)
    min_eig_val = eigvals[0]
    
    return min_eig_val


'''
################################################################
Scipy optimization methods
'''
def min_eig_obj_with_jac(x, inf_mats, H0, num_poses):
    """
    Computes the objective function and its Jacobian (gradient) for use with scipy.optimize.minimize.
    The objective is the negative of the smallest eigenvalue of the Schur complement of the information matrix.

    Args:
        x (np.ndarray): Continuous selection vector.
        inf_mats (List[scipy.sparse.csr_matrix]): List of sparse information matrices.
        H0 (scipy.sparse.csr_matrix): Prior information matrix.
        num_poses (int): Number of poses.

    Returns:
        Tuple[float, np.ndarray]: Objective function value and gradient.
    """
    # Combine the Fisher Information Matrices
    combined_fim = H0.copy()
    for xi, Hi in zip(x, inf_mats):
        combined_fim += xi * Hi

    # Extract submatrices based on the correct dimensions
    pose_dim = 6
    num_pose_elements = num_poses * pose_dim
    total_size = combined_fim.shape[0]
    measurement_dim = total_size - num_pose_elements

    # Slice the combined_fim to get Hll, Hlx, Hxx
    Hll = combined_fim[:measurement_dim, :measurement_dim].tocsc()
    Hlx = combined_fim[:measurement_dim, measurement_dim:].tocsc()
    Hxx = combined_fim[measurement_dim:, measurement_dim:].tocsc()

    # Compute the inverse of Hll (solve Hll * X = Hlx)
    try:
        X = spsolve(Hll, Hlx)
    except Exception as e:
        print("Linear solver failed:", e)
        X = Hll.inverse().dot(Hlx)  

    # Compute the Schur complement: H_schur = Hxx - Hlx^T * X
    H_schur = Hxx - Hlx.transpose().dot(X)

    # Compute the smallest eigenvalue and corresponding eigenvector using eigsh
    try:
        min_eig_val, min_eig_vec = eigsh(H_schur, k=1, which='SA')
    except Exception as e:
        print("Eigenvalue solver failed:", e)
        min_eig_val = 0.0
        min_eig_vec = np.zeros(H_schur.shape[1])

    # Objective function value (negative smallest eigenvalue)
    f = -min_eig_val[0]

    # Precompute constants for gradient
    t2 = X.transpose()

    # Compute the gradient
    grad = np.zeros_like(x)
    for idx, Hi in enumerate(inf_mats):
        # Extract submatrices from Hi
        Hll_i = Hi[:measurement_dim, :measurement_dim].tocsc()
        Hlx_i = Hi[:measurement_dim, measurement_dim:].tocsc()
        Hxx_i = Hi[measurement_dim:, measurement_dim:].tocsc()

        # Compute grad_schur = Hxx_i - Hlx_i^T * X - t2 * Hll_i * Y + t2 * Hlx_i
        try:
            Y = spsolve(Hll, Hlx_i)
        except Exception as e:
            print(f"Linear solver failed for Hi index {idx}:", e)
            Y = Hll.inverse().dot(Hlx_i)  # Fallback to inverse if spsolve fails

        grad_schur = Hxx_i - Hlx_i.transpose().dot(X) - t2.dot(Hll_i.dot(Y)) + t2.dot(Hlx_i)

        # Flatten vectors to ensure proper alignment
        grad[idx] = -min_eig_vec.flatten().dot(grad_schur.dot(min_eig_vec).flatten())

    return f, grad

def scipy_minimize(inf_mats, H0, selection_init, k, num_poses, A, b):
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
    bounds = Bounds([0] * len(selection_init), [1] * len(selection_init))  # Bounds between 0 and 1 for all variables
    linear_constraint = LinearConstraint(A, -np.inf, b)  # Ax <= b

    # Define a callable that returns the identity matrix
    def hess_identity_callable(x):
        return identity(len(x), format='csr')

    # Optimization function that maximizes the smallest eigenvalue
    res = minimize(
        fun=lambda x: min_eig_obj_with_jac(x, inf_mats, H0, num_poses),
        x0=selection_init,
        method='trust-constr',
        jac=True, 
        hess=hess_identity_callable, 
        constraints=linear_constraint,
        bounds=bounds,
        options={'disp': True, 'maxiter': 10000, 'verbose': 0}  # Verbose for detailed output
    )

    # Get the minimum eigenvalue of the continuous solution
    min_eig_val_unr, _, _ = infmat.find_min_eig_pair(inf_mats, res.x, H0, num_poses)

    return res.x, min_eig_val_unr

'''
################################################################
Scipy optimization methods with Log sum exponential
'''

def min_eig_obj_lse_with_jac(x, inf_mats, H0, num_poses):
    """
    Computes the objective function and its Jacobian (gradient) using the Log-Sum-Exp approximation.
    The objective is the negative of the approximated smallest eigenvalue of the information matrix.

    Args:
        x (np.ndarray): Continuous selection vector.
        inf_mats (List[np.ndarray]): List of information matrices for each sensor.
        H0 (np.ndarray): Prior information matrix.
        num_poses (int): Number of poses.

    Returns:
        Tuple[float, np.ndarray]: The objective function value and its gradient with respect to x.
    """
    # Build the combined information matrix based on the selected sensors
    combined_fim = H0.copy()
    for xi, Hi in zip(x, inf_mats):
        combined_fim += xi * Hi

    # Extract submatrices based on the correct dimensions
    pose_dim = 6  # Adjust as needed for your problem
    num_pose_elements = num_poses * pose_dim
    total_size = combined_fim.shape[0]
    measurement_dim = total_size - num_pose_elements

    Hll = combined_fim[:measurement_dim, :measurement_dim]
    Hlx = combined_fim[:measurement_dim, measurement_dim:]
    Hxx = combined_fim[measurement_dim:, measurement_dim:]

    # Compute the inverse of Hll (ensure it's invertible)
    try:
        Hll_inv = np.linalg.inv(Hll)
    except np.linalg.LinAlgError:
        Hll_inv = np.linalg.pinv(Hll)
        print("Warning: Hll is singular; using pseudoinverse.")

    # Compute the Schur complement
    H_schur = Hxx - Hlx.T @ Hll_inv @ Hlx
    # Ensure H_schur is symmetric
    H_schur = (H_schur + H_schur.T) / 2

    # Compute all eigenvalues and eigenvectors of H_schur
    eigvals, eigvecs = np.linalg.eigh(H_schur)
    beta = 5.0 

    # Stabilize weights computation using Log-Sum-Exp trick
    eigvals_shifted = eigvals - eigvals.min()  # Shift to prevent underflow
    scaled_exp_eigvals = np.exp(-beta * eigvals_shifted)
    weight_sum = np.sum(scaled_exp_eigvals) + 1e-12  
    softmax_weights = scaled_exp_eigvals / weight_sum  # This is σ_α(x)

    # Compute the objective function value using the stabilized LSE approximation
    f = (-1 / beta) * np.log(weight_sum) + eigvals.min()

    # Compute the gradient
    grad = np.zeros_like(x)
    for idx, H_j in enumerate(inf_mats):
        # Build the Schur complement of H_j
        Hll_j = H_j[:measurement_dim, :measurement_dim]
        Hlx_j = H_j[:measurement_dim, measurement_dim:]
        Hxx_j = H_j[measurement_dim:, measurement_dim:]

        try:
            Hll_j_inv = np.linalg.inv(Hll_j)
        except np.linalg.LinAlgError:
            Hll_j_inv = np.linalg.pinv(Hll_j)
            print(f"Warning: Hll_j is singular at index {idx}; using pseudoinverse.")

        H_schur_j = Hxx_j - Hlx_j.T @ Hll_j_inv @ Hlx_j
        # Ensure H_schur_j is symmetric
        H_schur_j = (H_schur_j + H_schur_j.T) / 2

        # Compute derivative of eigenvalues with respect to x_j
        lambda_derivatives = np.array([
            eigvecs[:, i].T @ H_schur_j @ eigvecs[:, i]
            for i in range(len(eigvals))
        ])

        # Compute gradient component using softmax weights
        grad[idx] = np.sum(softmax_weights * lambda_derivatives)

    return -f, -grad

def scipy_minimize_lse(inf_mats, H0, selection_init, num_poses, A, b):
    """
    Uses `scipy.optimize.minimize` with inequality constraints to solve a sensor selection problem.
    The objective is to maximize the smallest eigenvalue approximation using the Log-Sum-Exp method.

    Args:
        inf_mats (List[np.ndarray]): List of information matrices for each candidate sensor configuration.
        H0 (np.ndarray): Prior information matrix.
        selection_init (np.ndarray): Initial continuous selection vector.
        num_poses (int): Number of poses in the problem.
        A (np.ndarray): Matrix defining inequality constraints.
        b (np.ndarray): Vector defining inequality constraints.

    Returns:
        Tuple[np.ndarray, float]: Continuous solution vector, approximated maximum minimum eigenvalue.
    """
    # Set bounds for each variable in x (between 0 and 1) to represent selection probabilities
    bounds = [(0, 1) for _ in range(selection_init.shape[0])]

    # Define constraints to include inequality constraints
    cons = [
        {'type': 'ineq', 'fun': lambda x: b - A @ x}     # Inequality constraints Ax <= b
    ]

    # Optimization function that maximizes the smallest eigenvalue approximation
    res = minimize(
        fun=lambda x: min_eig_obj_lse_with_jac(x, inf_mats, H0, num_poses),
        x0=selection_init,
        method='SLSQP',
        jac=True,
        constraints=cons,
        bounds=bounds,
        options={'disp': True, 'maxiter': 1000, 'ftol': 1e-4}
    )

    # Get the approximated minimum eigenvalue of the continuous solution
    f_opt, _ = min_eig_obj_lse_with_jac(res.x, inf_mats, H0, num_poses)
    approx_min_eig_val = -f_opt

    return res.x, approx_min_eig_val