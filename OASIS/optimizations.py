from typing import List, Optional, Tuple
import numpy as np
from scipy.optimize import linprog
import gtsam
from enum import Enum
from enum import Enum
from . import utilities
from . import visualize
import numpy as np
import matplotlib.pyplot as plt
from functools import partial
from typing import List, Optional
import time
from gtsam.utils import plot
from ProblemBuilder import FIM as infmat
from Experiments import exp_utils
from scipy.optimize import minimize, Bounds
from numpy import linalg as la

L = gtsam.symbol_shorthand.L
X = gtsam.symbol_shorthand.X

class Metric(Enum):
    LOGDET = 1
    MIN_EIG = 2
    MSE = 3

def greedy_selection(
    inf_mat: np.ndarray,
    prior: np.ndarray,
    Nc: int,
    metric: Metric = Metric.LOGDET,
    num_runs: int = 1,
) -> Tuple[List[int], float, np.ndarray]:
    """
    Greedy selection algorithm to maximize information gain.

    Args:
        inf_mat: Precomputed information matrix (FIM) for each candidate sensor configuration.
        prior: Prior information matrix to be added for each selection.
        Nc: Number of sensors to select.
        metric: Objective metric to optimize (LOGDET, MIN_EIG, MSE).
        num_runs: Number of runs or trajectories.

    Returns:
        best_selection_indices: Indices of the selected sensor configurations.
        best_score: Score of the best configuration based on the specified metric.
        avail_cand: Array indicating availability of candidates after selection.
    """
    best_selection_indices = []
    best_score = 0.0
    avail_cand = np.ones(len(inf_mat), dtype=int)

    # Run the greedy selection for each trajectory/run
    for run in range(num_runs):
        for i in range(Nc):
            max_inf = 0.0
            selected_cand = None

            # Iterate through each available candidate
            for j in range(len(inf_mat)):
                if avail_cand[j] == 1:
                    # Build the full information matrix with prior
                    candidate_fim = inf_mat[j] + prior

                    # Calculate objective score based on metric
                    if metric == Metric.LOGDET:
                        sign, score = np.linalg.slogdet(candidate_fim)
                        score *= sign
                    elif metric == Metric.MIN_EIG:
                        score = np.linalg.eigvalsh(candidate_fim)[0]
                    else:  # Metric.MSE
                        score = np.trace(np.linalg.inv(candidate_fim))

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

    return best_selection_indices, best_score, avail_cand

def frank_wolfe_optimization(
    inf_mats: List[np.ndarray],
    prior: np.ndarray,
    n_iters: int,
    selection_init: np.ndarray,
    k: int,
    num_poses: int,
    num_runs: int,
    A: np.ndarray,
    b: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, float, float, int]:
    """
    Frank-Wolfe optimization method with constraints for multiple runs.

    Args:
        inf_mats: Information matrices for each trajectory.
        prior: Prior information matrix.
        n_iters: Number of iterations.
        selection_init: Initial selection of sensors.
        k: Number of sensors to select.
        num_poses: Number of poses.
        num_runs: Number of trajectories.
        A: Constraint matrix.
        b: Constraint vector.

    Returns:
        final_solution: Final sensor selection.
        selection_cur: Current selection at the end of optimization.
        min_eig_val_rounded: Minimum eigenvalue for the rounded solution.
        min_eig_val_unrounded: Minimum eigenvalue for the unrounded solution.
        i: Final iteration count.
    """
    selection_cur = selection_init
    prev_min_eig_score = 0

    for i in range(n_iters):
        grad = np.zeros(selection_cur.shape)
        min_eig_val_score = 0.0

        for traj_ind in range(num_runs):
            min_eig_val, min_eig_vec, final_inf_mat = infmat.find_min_eig_pair(
                inf_mats[traj_ind], selection_cur, prior, num_poses
            )
            min_eig_val_score += min_eig_val

            Hxx = final_inf_mat[-num_poses * 6:, -num_poses * 6:]
            Hll = final_inf_mat[0: -num_poses * 6, 0: -num_poses * 6]
            Hlx = final_inf_mat[0: -num_poses * 6, -num_poses * 6:]

            for ind in range(selection_cur.shape[0]):
                Hc = inf_mats[traj_ind][ind]
                grad_schur = Hxx - (Hlx.T @ np.linalg.pinv(Hll) @ Hlx) + Hc
                grad[ind] += min_eig_vec.T @ grad_schur @ min_eig_vec

        result = linprog(c=-grad, A_ub=A, b_ub=b, bounds=(0, 1), method='highs')
        rounded_sol = result.x

        # Stopping criterion
        if abs(min_eig_val_score - prev_min_eig_score) < 1e-4:
            break

        # Step size and update
        alpha = 1.0 / (i + 3.0)
        prev_min_eig_score = min_eig_val_score
        selection_cur = selection_cur + alpha * (rounded_sol - selection_cur)

    # Final rounding
    final_solution = np.round(selection_cur)[:k]
    min_eig_val_unrounded = sum(
        infmat.find_min_eig_pair(inf, selection_cur, prior, num_poses)[0] for inf in inf_mats
    )
    min_eig_val_rounded = sum(
        infmat.find_min_eig_pair(inf, final_solution, prior, num_poses)[0] for inf in inf_mats
    )

    return final_solution, selection_cur, min_eig_val_rounded, min_eig_val_unrounded, i


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
    phi = np.zeros(num)
    rounded_sol = np.zeros(num)
    phi[1:] = np.cumsum(selection[1:])  # Cumulative sum of selection scores
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

def min_eig_obj_with_jac(x, inf_mats, H0, num_poses):
    """
    Objective function for minimizing the smallest eigenvalue with respect to sensor selection.
    This function computes the minimum eigenvalue of the information matrix (FIM) and its gradient
    for optimization. It’s used as the objective function for `scipy_minimize`.

    Args:
        x (np.ndarray): Continuous selection vector for sensors.
        inf_mats (List[np.ndarray]): List of information matrices for each candidate sensor configuration.
        H0 (np.ndarray): Prior information matrix.
        num_poses (int): Number of poses in the problem.

    Returns:
        Tuple[float, np.ndarray]: The negative minimum eigenvalue (objective value) and its gradient.
    """
    # Compute the minimum eigenvalue and eigenvector for the given selection vector x
    min_eig_val, min_eig_vec, final_inf_mat = find_min_eig_pair(inf_mats, x, H0, num_poses)
    
    # Initialize the gradient
    grad = np.zeros(x.shape)
    
    # Compute the Schur complement gradient components
    Hxx = final_inf_mat[-num_poses * 6:, -num_poses * 6:]
    Hll = final_inf_mat[0: -num_poses * 6, 0: -num_poses * 6:]
    Hlx = final_inf_mat[0: -num_poses * 6, -num_poses * 6:]
    
    for ind in range(x.shape[0]):
        Hc = inf_mats[ind]
        Hxx_c = Hc[-num_poses * 6:, -num_poses * 6:]
        Hll_c = Hc[0: -num_poses * 6, 0: -num_poses * 6:]
        Hlx_c = Hc[0: -num_poses * 6, -num_poses * 6:]
        
        # Compute the Schur complement gradient
        t0 = Hlx.T
        t1 = np.linalg.pinv(Hll)
        t2 = t0 @ t1
        grad_schur = Hxx_c - (Hlx_c.T @ t1 @ t0.T - t2 @ Hll_c @ t1 @ t0.T + t2 @ Hlx_c)
        
        # Update the gradient with respect to the minimum eigenvalue direction
        grad[ind] = min_eig_vec.T @ grad_schur @ min_eig_vec

    # Return the objective value and gradient for minimization
    return -1.0 * min_eig_val, -1.0 * grad


def scipy_minimize(inf_mats, H0, selection_init, k, num_poses):
    """
    Uses `scipy.optimize.minimize` to solve a constrained optimization problem for selecting sensors.
    The objective is to minimize the smallest eigenvalue of the FIM by selecting an optimal subset of sensors.
    This method explicitly enforces constraints on the number of selected sensors.

    Args:
        inf_mats (List[np.ndarray]): List of information matrices for each candidate sensor configuration.
        H0 (np.ndarray): Prior information matrix.
        selection_init (np.ndarray): Initial continuous selection vector.
        k (int): Exact number of sensors to select.
        num_poses (int): Number of poses in the problem.

    Returns:
        Tuple[np.ndarray, np.ndarray, float, float]: The rounded solution, continuous solution,
                                                     minimum eigenvalues for rounded and unrounded solutions.
    """
    # Set bounds for each variable in x (between 0 and 1) to represent selection probabilities
    bounds = tuple([(0, 1) for _ in range(selection_init.shape[0])])
    
    # Define constraint to enforce exactly `k` selections (sum of selection vector equals k)
    cons = (
        {'type': 'eq', 'fun': lambda x: np.sum(x) - k}
    )
    
    # Run the constrained optimization using the `trust-constr` method, passing the objective and gradient
    res = minimize(
        min_eig_obj_with_jac, 
        selection_init, 
        method='trust-constr', 
        jac=True, 
        args=(inf_mats, H0, num_poses),
        constraints=cons, 
        bounds=bounds, 
        options={'disp': True}
    )
    
    # Print the continuous solution from the optimization
    print("Continuous solution from optimization:", res.x)
    
    # Round the solution to select exactly k elements by using `roundsolution`
    rounded_sol = roundsolution(res.x, k)
    print("Rounded solution:", rounded_sol)
    
    # Calculate minimum eigenvalues for both the continuous (`res.x`) and rounded (`rounded_sol`) solutions
    min_eig_val_unr, _, _ = find_min_eig_pair(inf_mats, res.x, H0, num_poses)
    min_eig_val_rounded, _, _ = find_min_eig_pair(inf_mats, rounded_sol, H0, num_poses)
    
    return rounded_sol, res.x, min_eig_val_rounded, min_eig_val_unr