from typing import List, Optional, Tuple
import numpy as np
from scipy.optimize import linprog, minimize_scalar
import gtsam
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
from scipy.optimize import minimize, Bounds
from numpy import linalg as la
from nsopy.methods.subgradient import SubgradientMethod
from nsopy.loggers import GenericDualMethodLogger


L = gtsam.symbol_shorthand.L
X = gtsam.symbol_shorthand.X

class Metric(Enum):
    LOGDET = 1
    MIN_EIG = 2
    MSE = 3

def greedy_selection(
    inf_mats: List[np.ndarray],
    prior: np.ndarray,
    Nc: int,
    metric: Metric = Metric.MIN_EIG,
    num_runs: int = 1,
    num_poses: int = 0  # Ensure this is passed when calling the function
) -> Tuple[np.ndarray, float, np.ndarray]:
    """
    Greedy selection algorithm to maximize information gain.
    """
    best_selection_indices = []
    best_score = float('-inf')
    avail_cand = np.ones(len(inf_mats), dtype=int)

    for run in range(num_runs):
        for i in range(Nc):
            max_inf = float('-inf')
            selected_cand = None

            for j in range(len(inf_mats)):
                if avail_cand[j] == 1:
                    candidate_selection = np.zeros(len(inf_mats))
                    candidate_selection[best_selection_indices + [j]] = 1

                    # Compute the minimum eigenvalue using the external function
                    min_eig_val, _, _ = infmat.find_min_eig_pair(
                        inf_mats, candidate_selection, prior, num_poses
                    )
                    score = min_eig_val

                    if score > max_inf:
                        max_inf = score
                        selected_cand = j

            if selected_cand is not None:
                best_score = max_inf
                best_selection_indices.append(selected_cand)
                avail_cand[selected_cand] = 0
                print(f"Best Score till now: {best_score}")
                print(f"Next best sensor index: {selected_cand}")

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

        # Check for convergence
        if abs(min_eig_val - prev_min_eig) < 1e-7:
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
    print("Number of candidates selected after rounding:", np.sum(rounded_sol))
    return rounded_sol


def branch_and_bound_with_cuts(
    inf_mats, H0, num_poses, k, 
    relaxed_solution, 
    best_score=float('inf'), 
    best_solution=None, 
    depth=0, 
    cut_threshold=0.2,  
    fractional_cut_ratio=0.3 
):
    """
    Combined Branch and Bound with Cuts function to optimize sensor selection.
    Dynamically decides when to add cutting planes based on the fractionality
    of the relaxed solution.

    Args:
        inf_mats (List[np.ndarray]): Information matrices for each sensor.
        H0 (np.ndarray): Prior information matrix.
        num_poses (int): Number of poses.
        k (int): Maximum number of sensors to select.
        relaxed_solution (np.ndarray): Continuous solution from a relaxed solver (e.g., Frank-Wolfe).
        best_score (float): The current best score achieved by an integer solution.
        best_solution (np.ndarray): The best binary selection solution found so far.
        depth (int): Current depth of the recursion.
        cut_threshold (float): Threshold for deciding when to consider a value fractional.
        fractional_cut_ratio (float): Fractional value ratio for applying cuts (e.g., 0.3 for 30%).

    Returns:
        Tuple[np.ndarray, float]: The best binary selection solution and its sinfmat.
    """

    # Round the relaxed solution to get an initial feasible solution at the root depth
    if depth == 0:
        initial_solution = np.round(relaxed_solution).astype(int)
        if np.sum(initial_solution) <= k:
            initial_score = evaluate_solution(inf_mats, H0, initial_solution, num_poses)
            if initial_score < best_score:
                best_score = initial_score
                best_solution = initial_solution

    # Check if we should add cuts
    def should_add_cut(solution):
        # Count fractional values
        fractional_indices = np.where((solution > cut_threshold) & (solution < 1 - cut_threshold))[0]
        fractional_count = len(fractional_indices)
        
        # Decide to add cut based on the ratio of fractional values
        return fractional_count > len(solution) * fractional_cut_ratio, fractional_indices

    # Apply cutting planes if needed
    add_cut, cut_indices = should_add_cut(relaxed_solution)
    if add_cut:
        # Simple heuristic cut by rounding fractional values close to 0 or 1
        for idx in cut_indices:
            relaxed_solution[idx] = 0 if relaxed_solution[idx] < 0.5 else 1

    # Branching phase
    fractional_indices = np.where((relaxed_solution > 0) & (relaxed_solution < 1))[0]
    if len(fractional_indices) == 0:
        return best_solution, best_score

    # Select the most fractional value to branch on
    idx_to_branch = fractional_indices[np.argmin(np.abs(relaxed_solution[fractional_indices] - 0.5))]

    # Create two branches: one setting the selected index to 0 and one setting it to 1
    for branch_value in [0, 1]:
        new_solution = relaxed_solution.copy()
        new_solution[idx_to_branch] = branch_value

        # Ensure that the number of selected sensors does not exceed `k`
        if np.sum(new_solution) > k:
            continue  # Skip this branch if it doesn’t satisfy the "at most k" constraint

        # Evaluate this new solution in its relaxed form
        current_score = evaluate_solution(inf_mats, H0, new_solution, num_poses)

        # Bounding: If this branch cannot yield a better result, prune it
        if current_score >= best_score:
            continue  # Prune this branch

        # Recursively call branch_and_bound_with_cuts on this new solution
        candidate_solution, candidate_score = branch_and_bound_with_cuts(
            inf_mats, H0, num_poses, k, 
            new_solution, 
            best_score=best_score, 
            best_solution=best_solution, 
            depth=depth + 1, 
            cut_threshold=cut_threshold,
            fractional_cut_ratio=fractional_cut_ratio
        )

        # Update the best solution if we found a better one
        if candidate_score < best_score:
            best_score = candidate_score
            best_solution = candidate_solution

    return best_solution, best_score

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
        inf_mats (List[np.ndarray]): List of information matrices for each sensor.
        H0 (np.ndarray): Prior information matrix.
        num_poses (int): Number of poses.

    Returns:
        Tuple[float, np.ndarray]: The objective function value and its gradient with respect to x.
    """

    combined_fim = H0.copy()
    for xi, Hi in zip(x, inf_mats):
        combined_fim += xi * Hi

    # Extract submatrices based on the correct dimensions
    pose_dim = 6
    num_pose_elements = num_poses * pose_dim
    total_size = combined_fim.shape[0]
    measurement_dim = total_size - num_pose_elements

    Hll = combined_fim[:measurement_dim, :measurement_dim]
    Hlx = combined_fim[:measurement_dim, measurement_dim:]
    Hxx = combined_fim[measurement_dim:, measurement_dim:]

    # Compute the inverse or pseudoinverse of Hll
    try:
        Hll_inv = np.linalg.inv(Hll)
    except np.linalg.LinAlgError:
        Hll_inv = np.linalg.pinv(Hll)

    # Compute the Schur complement
    H_schur = Hxx - Hlx.T @ Hll_inv @ Hlx

    # Compute eigenvalues and eigenvectors of H_schur
    eigvals, eigvecs = np.linalg.eigh(H_schur)
    min_eig_index = np.argmin(eigvals)
    min_eig_val = eigvals[min_eig_index]
    min_eig_vec = eigvecs[:, min_eig_index]  # Shape: (num_pose_elements,)

    # Objective function value (negative smallest eigenvalue)
    f = -min_eig_val

    # Precompute constants
    t0 = Hlx.T
    t1 = Hll_inv
    t2 = t0 @ t1

    # Compute the gradient
    grad = np.zeros_like(x)
    for idx, Hi in enumerate(inf_mats):
        # Corrected submatrix extraction
        Hll_i = Hi[:measurement_dim, :measurement_dim]
        Hlx_i = Hi[:measurement_dim, measurement_dim:]
        Hxx_i = Hi[measurement_dim:, measurement_dim:]

        # Compute grad_schur using the same formula as in Frank-Wolfe
        grad_schur = Hxx_i - (
            Hlx_i.T @ Hll_inv @ Hlx
            - t2 @ Hll_i @ Hll_inv @ Hlx
            + t2 @ Hlx_i
        )

        # Gradient component using the pose eigenvector
        grad[idx] = -min_eig_vec.T @ grad_schur @ min_eig_vec

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
    # Set bounds for each variable in x (between 0 and 1) to represent selection probabilities
    bounds = [(0, 1) for _ in range(selection_init.shape[0])]

    # Define constraints to include both the sum constraint and inequality constraints
    cons = [
        {'type': 'ineq', 'fun': lambda x: b - A @ x}     # Inequality constraints Ax <= b
    ]

    # Optimization function that maximizes the smallest eigenvalue
    res = minimize(
        fun=lambda x: min_eig_obj_with_jac(x, inf_mats, H0, num_poses),
        x0=selection_init,
        method='trust-constr',
        jac=True,
        constraints=cons,
        bounds=bounds,
        options={'disp': True}
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
    beta = 10.0  # Parameter for the LSE approximation

    # Stabilize weights computation using Log-Sum-Exp trick
    eigvals_shifted = eigvals - eigvals.min()  # Shift to prevent underflow
    scaled_exp_eigvals = np.exp(-beta * eigvals_shifted)
    weight_sum = np.sum(scaled_exp_eigvals) + 1e-12  # Avoid division by zero
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
        options={'disp': True, 'maxiter': 1000, 'ftol': 1e-9}
    )

    # Get the approximated minimum eigenvalue of the continuous solution
    f_opt, _ = min_eig_obj_lse_with_jac(res.x, inf_mats, H0, num_poses)
    approx_min_eig_val = -f_opt

    return res.x, approx_min_eig_val