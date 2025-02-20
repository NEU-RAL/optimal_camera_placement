from typing import List, Optional, Tuple
import numpy as np
from scipy.optimize import linprog, minimize_scalar, line_search
import gtsam
import scipy
from enum import Enum
from enum import Enum

from sympy.polys.benchmarks.bench_solvers import time_eqs_10x8
from . import utilities
from . import FIM as infmat
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


#Test with dense matrices
def compute_schur_complement_d(x, inf_mats, H0, num_poses):
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
        - T1: Dense intermediate matrix Hxl * inv(Hll)
        - T2: Dense intermediate matrix from solving Hll * T2 = Hlx.
    """
    # Combine the Fisher Information Matrices
    # s= time.time()
    #print("in schur complement")
    combined_fim = H0.copy()
    for xi, Hi in zip(x, inf_mats):
        combined_fim += xi * Hi

    # Extract submatrices
    pose_dim = 6
    num_pose_elements = num_poses * pose_dim
    measurement_dim = combined_fim.shape[0] - num_pose_elements
    Hll = combined_fim[:measurement_dim, :measurement_dim]
    Hlx = combined_fim[:measurement_dim, measurement_dim:]
    Hxx = combined_fim[measurement_dim:, measurement_dim:]

    # s1 = time.time()
    try:
        # # Convert Hll to dense if it is sparse
        if scipy.sparse.issparse(Hll):
            Hll = Hll.toarray()
        # Regularize and compute the pseudoinverse of Hll
        reg_term = 1e-8 * np.eye(Hll.shape[0])

        Hll_inv = np.linalg.pinv(Hll + reg_term)
    except Exception as e:
        raise ValueError(f"Failed to compute pseudoinverse for candidate: {e}")

    # Compute the Schur complement
    H_schur = Hxx - Hlx.T @ Hll_inv @ Hlx

    # Ensure H_schur is symmetric
    H_schur = (H_schur + H_schur.T) / 2
    # T2 = Hll_inv @ Hlx
    # T1 = Hlx.transpose().dot(Hll_inv)
    return H_schur, Hll_inv, Hlx, measurement_dim

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
        - T1: Dense intermediate matrix Hxl * inv(Hll)
        - T2: Dense intermediate matrix from solving Hll * T2 = Hlx.
    """
    # Combine the Fisher Information Matrices
    combined_fim = H0.copy()
    for xi, Hi in zip(x, inf_mats):
        combined_fim += xi * Hi

    # Extract submatrices
    pose_dim = 6
    num_pose_elements = num_poses * pose_dim
    measurement_dim = combined_fim.shape[0] - num_pose_elements
    Hll = combined_fim[:measurement_dim, :measurement_dim].tocsc()
    Hlx = combined_fim[:measurement_dim, measurement_dim:].tocsc()
    Hxx = combined_fim[measurement_dim:, measurement_dim:].tocsc()

    # Compute Schur complement
    #s= time.time()
    try:
        # T2 = spsolve(Hll.tocsc(), Hlx.tocsc())
        Hll_inv = scipy.sparse.linalg.inv(Hll.tocsc())
        T2 = Hll_inv.dot(Hlx)
        T1 = Hlx.transpose().dot(Hll_inv)
    except Exception as e:
        print("Linear solver failed:", e)
        T2= Hll.inverse().dot(Hlx).toarray()

    H_schur = Hxx - Hlx.transpose().dot(T2)
    # e= time.time()
    # execution_time = e - s
    # print(f"execution time sparse schur: {execution_time:.4f} seconds")

    # Compute smallest eigenvalue and eigenvector
    try:
        _, min_eig_vec = eigsh(H_schur, k=1, which='SA')
    except Exception as e:
        print("Eigenvalue solver failed:", e)
        min_eig_vec = np.zeros(H_schur.shape[1])

    return H_schur, min_eig_vec, T1, T2, measurement_dim

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

"""
Frank-wolfe Solver 
"""
'''
Step size selection implementations
1. fixed
2. diminshing
3. adaptive - back tracking line search
'''
def step_size_diminishing(gamma, grad):
    """
    stepsize alpha = gamma * Norm(gradient)
    :return: alpha
    """
    return gamma * np.linalg.norm(grad)

def step_size_backtrack(obj_func, obj_grad, x_k, d_k,*args):
    """
    step size based on back tracking line search that
    satisfies the Armijo rule

    :return: alpha
    """
    #res = line_search_wolfe2(min_eig_obj, min_eig_grad, x_k, -grad,  args=args, maxiter=50)
    res = line_search(obj_func, obj_grad,x_k, d_k, args=args, maxiter=50, amax=1.0, c1=0.001 )


    print("backtracking line search :")
    print (res)
    return res[0]
def step_size_simple_backtrack (obj_func, obj_grad, x_k, d_k,c,b, max_iters, *args):
    """
        custom back tracking line search based on armijo rule
        f(x_k+ gamma d_k) <= f(x_k) + c * gamma * <grad, d_k)
        for frank wolfe d_k = s_k - x_k
    """
    gamma = 1.0
    f_k = obj_func(x_k, *args)
    g_k = obj_grad(x_k, *args)
    f_k_gamma = obj_func(x_k + gamma * d_k, *args)
    rhs = f_k + c * gamma * np.dot(g_k,d_k)

    for i in range(max_iters):
        if  f_k_gamma > rhs:
            gamma = b * gamma
            f_k_gamma = obj_func(x_k + gamma * d_k, *args)
            rhs = f_k + c * gamma * np.dot(g_k, d_k)
        else:
            print(f"found staep size based on armijo rule : {gamma} in iterations:{i}")
            break
    if i == max_iters - 1:
        gamma = None
        print("max iterations reached but no convergence")

    return gamma

def compute_lipschitz_constant(x, inf_mats, H0, num_poses):
    """
    compute lipschitz constant of the minimum eigen value objective
    For now assume simple eigen values
    :param inf_mats:
    :param H0:
    :param num_poses:
    :return:
    """
    sum_op_norm = 0
    combined_fim = H0.copy()
    for xi, Hi in zip(x, inf_mats):
        combined_fim += xi * Hi
    min_eig_val, _ = eigsh(combined_fim.tocsc(), k=2, which='SA')
    delta = min_eig_val[1] - min_eig_val[0]

    for Hi in inf_mats:
        max_eig_val, _ = eigsh(Hi.tocsc(), k=1, which='LA')
        sum_op_norm = sum_op_norm + max_eig_val

    L = 4* (1.0/ delta) * sum_op_norm
    print("lipschitz constant =", L)
    return L


def step_size_lipschitz(x, g_k,d_k, inf_mats, H0, num_poses ):
    L = compute_lipschitz_constant(x, inf_mats, H0, num_poses)
    step_size = min(1, g_k/(L * np.linalg.norm(d_k)**2))
    print(f'step size based on lipschitz constant : {step_size}')
    return step_size
'''
End of step size selection implementation
'''
'''
Checking Stationarity of the solution at current iteration
'''
import cvxpy as cp
def check_stationarity(r, M_primes, A):
    #Define the variables
    #r - multiplicity of the eigen values
    j = A.shape[0] # number of active constraints
    #n = M_primes.shape[0]

    U = cp.Variable(shape=(r, r), PSD=True) # U is a symmetric matrix


    constraints = []
    constraints += [np.eye(r) - U >> 0] # This is 0 <=U<=I or I -U >= 0
    constraints += [cp.trace(U) == 1]  # tr(U) = 1
    if j != 0:
        lamda = cp.Variable(j)
        constraints += [lamda >= 0]
        #objective function
        ATlamda = A.transpose() @ lamda  #this should be a vector of A.shape[1]


    tmp = []
    for M in M_primes:
        tmp.append(cp.trace(M @ U))
    M_res = cp.hstack(tmp) # residual corresponding to subdifferential
    res = M_res  # sum of subdiff residual and norm cone
    if j != 0:
        res = res+ATlamda
        print(f" ATlamda : {ATlamda},Mres : {M_res},")

    objective = cp.Minimize(cp.norm(res, 2)) # minmizing the l2-norm


    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.MOSEK)
    print("status =", prob.status)
    print("optimal value =", prob.value)
    print("U solution =", U.value)
    if j != 0:
        print("Lambda solution =", lamda.value)

'''
End of stationarity check
'''
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
    if A is not None:
        # Convert A to CSC format
        A = sp.csc_matrix(A)
    dim= len(grad)

    # Define bounds for all variables between 0 and 1
    bounds = [(0, 1) for _ in range(dim)]

    # Use `linprog` from scipy to solve the LMO
    res = linprog(c=grad, A_ub=A, b_ub=b, bounds=bounds, method='highs')
    #res = linprog(c=grad, A_eq=A, b_eq=b, bounds=bounds, method='highs')

    if res.success:
        return res.x
    else:
        print(f"LMO failed: {res.message}")
        return None


def frank_wolfe_optimization(
    obj_func,
    obj_grad,
    selection_init: np.ndarray,
    A: np.ndarray,
    b: np.ndarray,
    *args

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
    # intermediate results for plotting
    fw_log = {"iter": [],
              "x": []}
    prev_min_eig = -np.inf  # Initialize to negative infinity for maximization

    # Initialize selection vector as a continuous variable (float)
    selection_cur = selection_init.copy().astype(float)

    for iteration in range(1000):
        # Compute objective and gradient
        min_eig_val = obj_func(selection_cur, *args )
        grad = obj_grad(selection_cur, *args)

        # Solve the Linear Minimization Oracle (LMO),Solve direction finding sub problem
        s = solve_lmo(grad, A, b)

        if s is None:
            print(f"LMO failed to find a feasible solution at iteration {iteration}.")
            break
        fw_log["iter"].append(iteration)
        fw_log["x"].append(selection_cur.tolist())
        d = s - selection_cur

        # Check for convergence based on termination criteria
        # if abs(min_eig_val - prev_min_eig) < 1e-4:
        #     print(f"Converged at iteration {iteration}")
        #     break

        if np.linalg.norm(grad) < 1e-5:
            print(f"Converged at iteration {iteration}, gradient norm: {np.linalg.norm(grad)}")
            break

        if -grad @ d < abs(min_eig_val)*1e-05:
            print(s)
            print(selection_cur)
            print(f"Converged at iteration {iteration}, gradient norm: {np.linalg.norm(grad)}, duality gap minimized")
            break

        prev_min_eig = min_eig_val

        # Get step size (classic Frank-Wolfe step size: 2/(t+2))
        alpha_dim = step_size_diminishing(0.1, grad)
        alpha_bt = 2.0 / (iteration + 2)
        '''
        Different step size calculations
        '''
        #alpha_bt = step_size_backtrack(obj_func, obj_grad, selection_cur, d, *args)
        #alpha_lipschitz = step_size_lipschitz(selection_cur, -grad @ d, d, *args)
        #alpha_bt=step_size_simple_backtrack(obj_func, obj_grad, selection_cur, d, 0.0001, 0.5, 100, *args)
        print(f"Iteration : {iteration}, alpha = {alpha_bt}, f(x): {min_eig_val}, gradient norm = {np.linalg.norm(grad)}, duality gap:{grad @ (s - selection_cur)}")

        # Update the selection vector
        selection_cur = selection_cur + alpha_bt * (s - selection_cur)
        print("solution current")
        print(selection_cur)
        selection_cur = np.clip(selection_cur, 0, 1)
    #check the stationarity of the final solution

    # M_primes is a set of products QT*M_i*Q
    inf_mats = args[0]

    combined_fim = args[1].copy()

    for xi, Hi in zip(selection_cur, inf_mats):
        combined_fim += xi * Hi
    eig_vals, min_eig_vecs = eigsh(combined_fim, k=10, which='SA') # I am just selecting 10, ideally we dont know how many are repeating
    # print(eig_vals)
    # print(min_eig_vecs[:, 0:1].transpose().shape)
    uniq_eigs = np.unique(eig_vals)
    uniq_inds = np.isclose(eig_vals, uniq_eigs)
    uniq_eig_vecs = min_eig_vecs[:, 0:1]
    if np.sum(uniq_inds) > 1:
        uniq_eig_vecs= min_eig_vecs[:, uniq_inds]

    M_primes = []
    for i in inf_mats:
        M_p = - uniq_eig_vecs[:, 0:1].transpose() @ i @ uniq_eig_vecs[:, 0:1]
        M_primes.append(M_p)
    #check which the constraints are active
    active_inds = np.isclose(np.abs(A@selection_cur-b), np.zeros(b.shape[0]), atol=1e-4)
    print("Active constraints =", np.where(active_inds==True))

    check_stationarity(1, M_primes, A[active_inds])
    return selection_cur, min_eig_val, iteration + 1, fw_log


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

# Define a function to parallelize the computation for a single Hi
def compute_grad_parallel(idx, Hi, T1, T2, min_eig_vec, measurement_dim):
    """
    Compute gradient contribution for a single Hi matrix.

    Args:
        idx (int): Index of the matrix.
        Hi (scipy.sparse.csr_matrix): Information matrix.
        Hll (scipy.sparse.csc_matrix): Submatrix from combined FIM.
        T1 (np.ndarray): Dense intermediate matrix from Schur computation - Hxl * inv(Hll)
        T2 (np.ndarray): intermediate matrix from Schur computation -  inv(Hll) * Hlx
        min_eig_vec (np.ndarray): Smallest eigenvector of Schur complement.
        measurement_dim (int): Dimensionality of the measurement space.

    Returns:
        Tuple[int, float]: Index and gradient contribution.
    """
    # Extract submatrices from Hi (keep sparse)
    Hll_i = Hi[:measurement_dim, :measurement_dim].tocsc()
    Hlx_i = Hi[:measurement_dim, measurement_dim:].tocsc()
    Hxx_i = Hi[measurement_dim:, measurement_dim:].tocsc()

    # Compute grad_schur
    # try:
    #     Y = spsolve(Hll, Hlx_i.tocsc())
    # except Exception as e:
    #     print(f"Linear solver failed for Hi index {idx}:", e)
    #     Y = inv(Hll).dot(Hlx_i)  # Sparse fallback
    #     Y = Y.toarray()  # Convert to dense for consistency

    grad_schur = Hxx_i - Hlx_i.transpose().dot(T2) + T1.dot(Hll_i.dot(T2)) - T1.dot(Hlx_i)

    # Flatten vectors to ensure proper alignment
    grad_value = -min_eig_vec.flatten().dot(grad_schur.dot(min_eig_vec).flatten())
    return idx, grad_value

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
    H0 = H0.tocsc()
    inf_mats = [Hi.tocsc() for Hi in inf_mats]
    #H_schur, _, _, _,_ = compute_schur_complement(x, inf_mats, H0, num_poses)
    H_schur = H0.copy()
    for xi, Hi in zip(x, inf_mats):
        H_schur += xi * Hi
    # Compute smallest eigenvalue
    try:
        min_eig_val, _ = eigsh(H_schur, k=2, which='SA')
        # print("eigen values")
        # print(min_eig_val)
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
    H_schur, min_eig_vec, T1, T2, measurement_dim = compute_schur_complement(x, inf_mats, H0, num_poses)

    # Parallelized gradient computation
    results = Parallel(n_jobs=-1)(
        delayed(compute_grad_parallel)(idx, Hi, T1, T2, min_eig_vec, measurement_dim)
        for idx, Hi in enumerate(inf_mats)
    )

    grad = np.zeros_like(x)
    for idx, grad_value in results:
        grad[idx] = grad_value
    
    run_time = time.time() - start_time
    #print(f"jacobian call compute time: {run_time:.4f}")

    return grad


def min_eig_grad_noschur(x, inf_mats, H0, num_poses):
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
    combined_fim = H0.copy()
    for xi, Hi in zip(x, inf_mats):
        combined_fim += xi * Hi
    min_eig_val, min_eig_vec = eigsh(combined_fim, k=1, which='SA')
    grad = np.zeros_like(x)
    for idx, Hi in enumerate(inf_mats):
        grad_value = -min_eig_vec.flatten().dot(Hi.dot(min_eig_vec).flatten())
        grad[idx] = grad_value

    run_time = time.time() - start_time
    # print(f"jacobian call compute time: {run_time:.4f}")

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
        options={'disp': True, 'maxiter': 10000, 'ftol': 1e-4} )

    # Get the minimum eigenvalue of the continuous solution
    min_eig_val_unr, _, _ = infmat.find_min_eig_pair(inf_mats, res.x, H0, num_poses)

    return res.x, min_eig_val_unr

'''
################################################################
Scipy optimization methods with Log sum exponential on dense
'''
def min_eig_obj_lse_d(x, inf_mats, H0, num_poses, beta=5.0):
    """
    Computes the objective function value using the Log-Sum-Exp (LSE) approximation.
    """
    # Use the helper function to compute the Schur complement and auxiliary matrices
    H_schur, T1, T2,measurement_dim = compute_schur_complement_d(x, inf_mats, H0, num_poses)

    # Compute all eigenvalues of H_schur
    try:
        if H_schur.shape[0] <= 500:  # Threshold to determine when to switch to dense
            # Convert sparse matrix to dense for full eigenvalue computation
            if isinstance(H_schur, np.matrix):
                H_schur = np.asarray(H_schur)
            eigvals, eigvecs = np.linalg.eigh(H_schur)
        else:
            # For very large matrices, use eigsh for performance and check fallback
            eigvals, _ = eigsh(H_schur, k=H_schur.shape[0] - 1, which="SA")
    except Exception as e:
        print(f"Eigenvalue computation failed: {e}")
        return np.inf , np.zeros_like(x)

    # Stabilize Log-Sum-Exp computation
    eigvals_shifted = eigvals - eigvals.min()  # Shift to stabilize
    scaled_exp_eigvals = np.exp(-beta * eigvals_shifted)
    weight_sum = np.sum(scaled_exp_eigvals) + 1e-12

    # Objective value using Log-Sum-Exp
    f = (-1 / beta) * np.log(weight_sum) + eigvals.min()
    return -f

def min_eig_grad_lse_d(x, inf_mats, H0, num_poses, beta=5.0):
    """
    Computes the gradient of the objective function using the Log-Sum-Exp (LSE) approximation.
    """
    # Use the helper function to compute the Schur complement and auxiliary matrices
    s=time.time()
    H_schur, H_ll_inv, H_lx, measurement_dim = compute_schur_complement_d(x, inf_mats, H0, num_poses)

    # Compute eigenvalues and eigenvectors
    try:
        if isinstance(H_schur, np.matrix):
            H_schur = np.asarray(H_schur)
        eigvals, eigvecs = np.linalg.eigh(H_schur)

    except Exception as e:
        print(f"Eigenvalue computation failed: {e}")
        return np.zeros_like(x)  # Return zero gradient if computation fails

    eigvals_shifted = eigvals - eigvals.min()  # Stabilize
    scaled_exp_eigvals = np.exp(-beta * eigvals_shifted)
    weight_sum = np.sum(scaled_exp_eigvals) + 1e-12
    softmax_weights = scaled_exp_eigvals / weight_sum

    # Parallelized gradient computation
    def compute_grad_lse_d(idx, H_j, eigvecs, softmax_weights, measurement_dim):

        s1 = time.time()
        Hll_j_d = H_j[:measurement_dim, :measurement_dim]
        Hlx_j_d = H_j[:measurement_dim, measurement_dim:]
        Hxx_j_d = H_j[measurement_dim:, measurement_dim:]

        grad_schur_d = Hxx_j_d - Hlx_j_d.T @ H_ll_inv@H_lx + H_lx.T @ H_ll_inv @ Hll_j_d @ H_ll_inv@H_lx - H_lx.T @ H_ll_inv @ Hlx_j_d
        lambda_derivatives_d = np.array([
            eigvecs[:, i].T @ grad_schur_d @ eigvecs[:, i]
            for i in range(len(eigvals))
        ])
        e1 = time.time()
        execution_time = e1 - s1
        print(f"execution time grad schur computation dense : {execution_time:.4f} seconds")
        s2 = time.time()
        lambda_derivatives_d_fast = np.zeros(len(eigvals))
        for i in range(len(eigvals)):
            v = eigvecs[:, i:i+1]
            tmp1 = H_lx @ v
            tmp2 = H_ll_inv @ tmp1
            grad_schur_d = Hxx_j_d @ v - Hlx_j_d.T @ tmp2 + H_lx.T @ (H_ll_inv @ (Hll_j_d @ tmp2)) - H_lx.T @ (H_ll_inv @ (Hlx_j_d @ v))
            lambda_derivatives_d_fast[i]=  v.T @ grad_schur_d

        assert (np.allclose(lambda_derivatives_d,lambda_derivatives_d_fast))
        tmp_d = np.sum(softmax_weights * lambda_derivatives_d)
        e2 = time.time()
        execution_time = e2 - s2
        print(f"execution time grad schur computation dense fast : {execution_time:.4f} seconds")

        return idx, tmp_d
    s1 = time.time()
    results = Parallel(n_jobs=-1)(
        delayed(compute_grad_lse_d)(idx, H_j.toarray(), eigvecs, softmax_weights, measurement_dim)
        for idx, H_j in enumerate(inf_mats)
    )
    # Collect results into the gradient vector
    grad = np.zeros_like(x)
    for idx, grad_value in results:
        grad[idx] = grad_value
    # e1 = time.time()
    # execution_time = e1 - s1
    # print(f"execution time grad schur computation : {execution_time:.4f} seconds")
    return -grad

def scipy_minimize_lse_d(inf_mats, H0, selection_init, num_poses, A, b):
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
        return min_eig_obj_lse_d(x, inf_mats, H0, num_poses)

    def gradient(x):
        return min_eig_grad_lse_d(x, inf_mats, H0, num_poses)

    # Optimization function
    res = minimize(
        fun=objective,
        x0=selection_init,
        method='SLSQP',
        jac=gradient,
        constraints=cons,
        bounds=bounds,
        options={'disp': True, 'maxiter': 1000, 'ftol': 1e-4}
    )

    # Get the approximated minimum eigenvalue
    f_opt = min_eig_obj_lse_d(res.x, inf_mats, H0, num_poses)
    approx_min_eig_val = -f_opt

    return res.x, approx_min_eig_val

'''
################################################################
Scipy optimization methods with Log sum exponential
'''

def min_eig_obj_lse(x, inf_mats, H0, num_poses, beta=5.0):
    """
    Computes the objective function value using the Log-Sum-Exp (LSE) approximation.
    """
    # Use the helper function to compute the Schur complement and auxiliary matrices
    H_schur, min_eig_vec, T1, T2,measurement_dim = compute_schur_complement(x, inf_mats, H0, num_poses)

    # Compute all eigenvalues of H_schur
    try:
        if H_schur.shape[0] <= 500:  # Threshold to determine when to switch to dense
            # Convert sparse matrix to dense for full eigenvalue computation
            if scipy.sparse.issparse(H_schur):
                H_schur = H_schur.toarray()
            if isinstance(H_schur, np.matrix):
                H_schur = np.asarray(H_schur)
            eigvals, _ = np.linalg.eigh(H_schur)
        else:
            # For very large matrices, use eigsh for performance and check fallback
            eigvals, _ = eigsh(H_schur, k=H_schur.shape[0] - 1, which="SA")
    except Exception as e:
        print(f"Eigenvalue computation failed: {e}")
        return np.inf , np.zeros_like(x)

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
    H_schur, min_eig_vec, T1, T2, measurement_dim = compute_schur_complement(x, inf_mats, H0, num_poses)

    # Compute eigenvalues and eigenvectors
    try:
        if H_schur.shape[0] <= 500:
            if scipy.sparse.issparse(H_schur):
                H_schur = H_schur.toarray()
            if isinstance(H_schur, np.matrix):
                H_schur = np.asarray(H_schur)
            eigvals, eigvecs = np.linalg.eigh(H_schur)
        else:
            eigvals, eigvecs = eigsh(H_schur, k=H_schur.shape[0] - 1, which="SA")
    except Exception as e:
        print(f"Eigenvalue computation failed: {e}")
        return np.zeros_like(x)  # Return zero gradient if computation fails

    eigvals_shifted = eigvals - eigvals.min()  # Stabilize
    scaled_exp_eigvals = np.exp(-beta * eigvals_shifted)
    weight_sum = np.sum(scaled_exp_eigvals) + 1e-12
    softmax_weights = scaled_exp_eigvals / weight_sum

    # Parallelized gradient computation
    def compute_grad_lse(idx, H_j, eigvecs, softmax_weights, measurement_dim):
        # Hll_j = H_j[:measurement_dim, :measurement_dim].tocsc()
        # Hlx_j = H_j[:measurement_dim, measurement_dim:].tocsc()
        # Hxx_j = H_j[measurement_dim:,measurement_dim:].tocsc()

        H_j_d = H_j.toarray()
        Hll_j_d = H_j_d[:measurement_dim, :measurement_dim]
        Hlx_j_d = H_j_d[:measurement_dim, measurement_dim:]
        Hxx_j_d = H_j_d[measurement_dim:, measurement_dim:]
        grad_schur_d = Hxx_j_d - Hlx_j_d.T @ T2 + T1 @ Hll_j_d @ T2 - T1 @ Hlx_j_d
        lambda_derivatives_d = np.array([
            eigvecs[:, i].T @ grad_schur_d @ eigvecs[:, i]
            for i in range(len(eigvals))
        ])
        tmp_d = np.sum(softmax_weights * lambda_derivatives_d)
        return idx, tmp_d

    results = Parallel(n_jobs=-1)(
        delayed(compute_grad_lse)(idx, H_j, eigvecs, softmax_weights, measurement_dim)
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
        options={'disp': True, 'maxiter': 1000, 'ftol': 1e-4}
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
