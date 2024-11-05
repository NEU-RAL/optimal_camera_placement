from typing import List, Optional, Tuple
import numpy as np
from scipy.optimize import linprog, minimize_scalar
import gtsam
from enum import Enum
from enum import Enum
import utilities
import FIM as infmat
import visualize
import numpy as np
import matplotlib.pyplot as plt
from functools import partial
from typing import List, Optional
import time
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
) -> Tuple[np.ndarray, float, np.ndarray]:
    """
    Greedy selection algorithm to maximize information gain.
    """
    best_selection_indices = []
    best_score = float('-inf')
    avail_cand = np.ones(len(inf_mats), dtype=int)

    # Run the greedy selection for each trajectory/run
    for run in range(num_runs):
        for i in range(Nc):
            max_inf = float('-inf')
            selected_cand = None

            # Iterate through each available candidate
            for j in range(len(inf_mats)):
                if avail_cand[j] == 1:
                    # Build the full information matrix with prior
                    candidate_fim = prior.copy()
                    for idx in best_selection_indices + [j]:
                        candidate_fim += inf_mats[idx]

                    # Calculate objective score based on metric
                    if metric == Metric.LOGDET:
                        sign, logdet = np.linalg.slogdet(candidate_fim)
                        score = sign * logdet
                    elif metric == Metric.MIN_EIG:
                        eigvals = np.linalg.eigvalsh(candidate_fim)
                        score = eigvals[0]  # Smallest eigenvalue
                    else:  # Metric.MSE
                        score = -np.trace(np.linalg.pinv(candidate_fim))

                    # Update the best candidate if the score is higher
                    if score > max_inf:
                        max_inf = score
                        selected_cand = j

            # Update best score and mark selected candidate as unavailable
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
    """
    Frank-Wolfe optimization method with constraints.

    Args:
        inf_mats: List of information matrices for each sensor.
        prior: Prior information matrix.
        n_iters: Number of iterations.
        selection_init: Initial selection vector.
        k: Number of sensors to select.
        num_poses: Number of poses.
        A: Constraint matrix.
        b: Constraint vector.

    Returns:
        selection_cur: Final continuous sensor selection.
        min_eig_val_score: Minimum eigenvalue score of the final solution.
        num_iterations: Number of iterations performed.
    """
    selection_cur = selection_init.copy()
    num_sensors = len(selection_cur)
    for i in range(n_iters):
        # Compute the gradient at the current selection
        min_eig_val, min_eig_vec, final_inf_mat = infmat.find_min_eig_pair(
            inf_mats, selection_cur, prior, num_poses
        )

        # Gradient computation
        grad = np.zeros_like(selection_cur)
        for idx in range(num_sensors):
            Hi = inf_mats[idx]
            grad[idx] = -min_eig_vec.T @ Hi @ min_eig_vec  # Negative because we minimize

        # Solve the linear minimization oracle (LMO)
        c = grad  # Minimize c^T x
        # Include the equality constraint sum(x) = k
        A_eq = np.ones((1, num_sensors))
        b_eq = np.array([k])
        bounds = [(0, 1) for _ in range(num_sensors)]
        res = linprog(c=c, A_ub=A, b_ub=b, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')

        if res.status != 0:
            print(f"Linprog failed at iteration {i}, status {res.status}: {res.message}")
            break

        s = res.x  # Direction from LMO

        # Compute step size (line search)
        delta_x = s - selection_cur
        numerator = -grad @ delta_x
        denominator = np.linalg.norm(infmat.combine_inf_mats(inf_mats, delta_x)) ** 2
        if denominator == 0:
            alpha = 0
        else:
            alpha = min(1, numerator / denominator)
            alpha = max(0, alpha)  

        # Update the current selection
        selection_cur = selection_cur + alpha * delta_x

        # Stopping criterion based on the duality gap
        duality_gap = -grad @ delta_x
        if duality_gap < 1e-2:
            print(f"Converged at iteration {i}, duality gap: {duality_gap}")
            break

    # Compute the final minimum eigenvalue score
    min_eig_val_score, _, _ = infmat.find_min_eig_pair(
        inf_mats, selection_cur, prior, num_poses
    )

    return selection_cur, min_eig_val_score, i + 1

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
    cut_threshold=0.2,  # Threshold for deciding when to add cuts
    fractional_cut_ratio=0.3  # Ratio to determine if a cut is needed
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
        Tuple[np.ndarray, float]: The best binary selection solution and its score.
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
        return best_solution, best_score  # No more branches possible

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
# Not poses - dimensions
def min_eig_obj_with_jac(x, inf_mats, H0, num_poses):
    """
    Computes the objective function and its Jacobian (gradient) for use with scipy.optimize.minimize.
    The objective is the negative of the smallest eigenvalue of the information matrix.

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

    # Compute eigenvalues and eigenvectors
    eigvals, eigvecs = np.linalg.eigh(combined_fim)
    min_eig_index = np.argmin(eigvals)
    min_eig_val = eigvals[min_eig_index]
    min_eig_vec = eigvecs[:, min_eig_index]

    # Objective function value (negative smallest eigenvalue)
    f = -min_eig_val

    # Extract submatrices based on the correct dimensions
    pose_dim = 6  # Assuming each pose contributes 6 dimensions
    num_pose_elements = num_poses * pose_dim
    total_size = combined_fim.shape[0]
    measurement_dim = total_size - num_pose_elements

    Hxx = combined_fim[-num_pose_elements:, -num_pose_elements:]
    Hll = combined_fim[:measurement_dim, :measurement_dim]
    Hlx = combined_fim[:measurement_dim, -num_pose_elements:]

    # Compute the inverse of Hll (ensure it's invertible)
    Hll_inv = np.linalg.inv(Hll)

    # Compute the Schur complement
    H_schur = Hxx - Hlx.T @ Hll_inv @ Hlx

    # Compute the gradient
    grad = np.zeros_like(x)
    for idx, Hi in enumerate(inf_mats):
        # Split Hi into submatrices
        Hxx_i = Hi[-num_pose_elements:, -num_pose_elements:]
        Hll_i = Hi[:measurement_dim, :measurement_dim]
        Hlx_i = Hi[:measurement_dim, -num_pose_elements:]

        # Compute the Schur complement of Hi
        H_schur_i = Hxx_i - Hlx_i.T @ np.linalg.inv(Hll_i) @ Hlx_i

        # Gradient component
        grad[idx] = -min_eig_vec.T @ H_schur_i @ min_eig_vec

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
        {'type': 'eq', 'fun': lambda x: np.sum(x) - k},  # Sum constraint
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
NSOPy optimization methods
'''

def nsopy_optimize(inf_mats, H0, selection_init, num_poses, k, max_iters=100):


    def min_eig_oracle(lambda_k):
        x_k = lambda_k  # Primal variable

        # Compute the combined information matrix
        combined_inf_mat = H0.copy()
        for xi, Hi in zip(x_k, inf_mats):
            combined_inf_mat += xi * Hi

        # Compute the smallest eigenvalue and corresponding eigenvector
        eigvals, eigvecs = np.linalg.eigh(combined_inf_mat)
        min_eig_index = np.argmin(eigvals)
        smallest_eigval = eigvals[min_eig_index]
        corresponding_eigvec = eigvecs[:, min_eig_index]

        # Compute the subgradient
        subgradient = np.array([
            corresponding_eigvec.T @ Hi @ corresponding_eigvec for Hi in inf_mats
        ])

        d_k = smallest_eigval  # Dual function value

        return x_k, d_k, subgradient

    def projection_onto_simplex(v, s):
        n = len(v)
        v_sorted = np.sort(v)[::-1]
        cssv = np.cumsum(v_sorted)
        rho = np.nonzero(v_sorted + (s - cssv) / np.arange(1, n+1) > 0)[0][-1]
        theta = (cssv[rho] - s) / (rho + 1.0)
        w = np.maximum(v - theta, 0)
        return w

    def projection_function(x_k):
        # Project onto the simplex defined by sum(x_k) = k and x_k >= 0
        x_projected = projection_onto_simplex(x_k, s=k)
        return x_projected

    # Initialize the method
    method = SubgradientMethod(
        oracle=min_eig_oracle,
        projection_function=projection_function,
        stepsize_0=0.1,
        stepsize_rule='1/sqrt(k)',
        sense='min',
        dimension=len(selection_init)
    )

    # Use a feasible initial point
    selection_init = projection_onto_simplex(selection_init, s=k)
    method.lambda_k = selection_init.copy()

    logger = GenericDualMethodLogger(method)

    for iteration in range(max_iters):
        method.step()
        current_selection = method.lambda_k
        current_objective = logger.d_k_iterates[-1]  # Minimization problem
        print(f"Iteration {iteration + 1}: Objective Value = {current_objective}")

        # Stopping criterion
        if iteration > 0 and abs(logger.d_k_iterates[-1] - logger.d_k_iterates[-2]) < 1e-5:
            print(f"Converged at iteration {iteration + 1}")
            break

    final_selection = method.lambda_k
    final_objective = logger.d_k_iterates[-1]
    return final_selection, final_objective
