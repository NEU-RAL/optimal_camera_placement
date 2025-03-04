from typing import List, Optional, Tuple
import numpy as np
from scipy.optimize import linprog, minimize
import gtsam
import scipy
from enum import Enum
import utilities
import FIM as infmat
from functools import partial
from scipy.optimize import Bounds, LinearConstraint
from scipy.sparse import csr_matrix, diags
from scipy.sparse.linalg import eigsh, spsolve
from joblib import Parallel, delayed
import scipy.sparse as sp
import time
import cvxpy as cp
from timeit import default_timer as timer

from collections import namedtuple
import gurobipy as gp
from gurobipy import GRB

# Symbol shorthand from gtsam
L = gtsam.symbol_shorthand.L
X = gtsam.symbol_shorthand.X

class Metric(Enum):
    LOGDET = 1
    MIN_EIG = 2
    MSE = 3

def compute_combined_fim(x, inf_mats, H0):
    """
    Compute combined Fisher Information Matrix given a selection vector x.
    """
    combined_fim = H0.copy()
    for xi, Hi in zip(x, inf_mats):
        combined_fim += xi * Hi
    return combined_fim

def compute_schur_complement(x, inf_mats, H0, num_poses):
    """
    Computes the Schur complement and related components.
    """
    pose_dim = 6
    combined_fim = compute_combined_fim(x, inf_mats, H0)

    total_size = combined_fim.shape[0]
    num_pose_elements = num_poses * pose_dim
    measurement_dim = total_size - num_pose_elements

    assert measurement_dim > 0, f"Invalid measurement dimension: {measurement_dim}"

    Hll = combined_fim[:measurement_dim, :measurement_dim].tocsc()
    Hlx = combined_fim[:measurement_dim, measurement_dim:].tocsc()
    Hxx = combined_fim[measurement_dim:, measurement_dim:].tocsc()

    # Solve for X in Hll * X = Hlx
    try:
        X_sol = spsolve(Hll, Hlx)
    except Exception as e:
        # Fallback: Use pseudo-inverse if spsolve fails
        Hll_dense = Hll.toarray()
        try:
            Hll_inv = np.linalg.pinv(Hll_dense)
            X_sol = Hll_inv.dot(Hlx.toarray())
        except np.linalg.LinAlgError as e2:
            raise RuntimeError(f"Failed to solve system for Schur complement: {e2}")

    H_schur = Hxx - Hlx.transpose().dot(X_sol)

    # Compute smallest eigenvalue and eigenvector
    try:
        min_eig_val, min_eig_vec = eigsh(H_schur, k=1, which='SA')
    except Exception as e:
        # If eigenvalue computation fails, raise an error
        raise RuntimeError(f"Eigenvalue solver failed: {e}")

    return H_schur, min_eig_val, min_eig_vec, Hll, X_sol

def evaluate_solution(inf_mats, H0, solution):
    """
    Evaluates a binary solution by computing the smallest eigenvalue of the 
    information matrix constructed from the selected sensors.

    Parameters:
      inf_mats : list or array of sparse matrices
          Each element is a sensor measurement matrix.
      H0 : sparse matrix or numpy.ndarray
          The prior matrix.
      solution : array-like of {0,1}
          The binary selection vector.

    Returns:
      min_eig_val : float
          The smallest eigenvalue of the combined information matrix.
    """
    # Ensure H0 is a sparse matrix.
    if not sp.isspmatrix(H0):
        # If H0 is a numpy array, convert it to CSR format.
        H0 = sp.csr_matrix(H0)
    # Convert to CSC if it isn't already.
    elif not sp.isspmatrix_csc(H0):
        H0 = H0.tocsc()

    combined_fim = H0.copy()
    for i, selected in enumerate(solution):
        if selected:
            mat = inf_mats[i]
            # Ensure each matrix is sparse.
            if not sp.isspmatrix(mat):
                mat = sp.csr_matrix(mat)
            elif not sp.isspmatrix_csc(mat):
                mat = mat.tocsc()
            combined_fim += mat

    combined_fim_dense = combined_fim.toarray()
    try:
        eigvals = np.linalg.eigvalsh(combined_fim_dense)
        min_eig_val = eigvals[0]
    except Exception as e:
        print("Error in eigenvalue computation:", e)
        min_eig_val = float('-inf')
    return min_eig_val

def greedy_selection(
    inf_mats: List[sp.csr_matrix],
    prior: sp.csr_matrix,
    Nc: int,
    metric: Metric = Metric.MIN_EIG,
    num_runs: int = 1,
    num_poses: int = None
) -> Tuple[np.ndarray, float, np.ndarray]:
    """
    Greedy selection algorithm to maximize information gain using the Schur complement.
    """
    if num_poses is None:
        raise ValueError("num_poses must be provided.")

    best_selection_indices = []
    best_score = float('-inf')
    avail_cand = np.ones(len(inf_mats), dtype=int)
    combined_inf_mat = prior.copy()
    pose_dim = 6

    for run in range(num_runs):
        for i in range(Nc):
            max_inf = float('-inf')
            selected_cand = None
            start_time = time.time()

            for j in range(len(inf_mats)):
                if avail_cand[j] == 1:
                    temp_inf_mat = combined_inf_mat + inf_mats[j]
                    total_size = temp_inf_mat.shape[0]
                    num_pose_elements = num_poses * pose_dim
                    measurement_dim = total_size - num_pose_elements
                    if measurement_dim <= 0:
                        raise ValueError(f"Invalid measurement dimension: {measurement_dim}")

                    Hll = temp_inf_mat[:measurement_dim, :measurement_dim]
                    Hlx = temp_inf_mat[:measurement_dim, measurement_dim:]
                    Hxx = temp_inf_mat[measurement_dim:, measurement_dim:]

                    # Regularize for stability
                    Hll_dense = Hll.toarray() + 1e-8 * np.eye(Hll.shape[0])
                    Hll_inv = np.linalg.pinv(Hll_dense)

                    # Compute Schur complement
                    H_schur = Hxx.toarray() - Hlx.toarray().T @ Hll_inv @ Hlx.toarray()
                    H_schur = (H_schur + H_schur.T) / 2

                    eigvals = np.linalg.eigvalsh(H_schur)
                    min_eig_val = eigvals[0]
                    score = min_eig_val

                    if score > max_inf:
                        max_inf = score
                        selected_cand = j

            elapsed_time = time.time() - start_time
            print(f"Iteration {i}: Min eigen after selection = {max_inf:.4f}, time={elapsed_time:.4f}s")

            if selected_cand is not None:
                best_selection_indices.append(selected_cand)
                avail_cand[selected_cand] = 0
                combined_inf_mat += inf_mats[selected_cand]

    print("Selected candidates:", best_selection_indices)
    selection_vector = np.zeros(len(inf_mats))
    selection_vector[best_selection_indices] = 1
    best_score = evaluate_solution(inf_mats, prior, selection_vector)
    return selection_vector, best_score, avail_cand

def solve_lmo(grad, A, b):
    """
    Solves the Linear Minimization Oracle (LMO) problem for Frank-Wolfe.
    """
    num_sensors = len(grad)
    bounds = [(0, 1) for _ in range(num_sensors)]
    res = linprog(c=grad, A_ub=A, b_ub=b, bounds=bounds, method='highs')
    if res.success:
        return res.x
    else:
        print(f"LMO failed: {res.message}")
        return None

def compute_grad_parallel(idx, Hi, Hll, X, t2, min_eig_vec, measurement_dim):
    """
    Compute gradient contribution for a single Hi matrix in parallel.
    """
    # Extract submatrices
    Hll_i = Hi[:measurement_dim, :measurement_dim].tocsc()
    Hlx_i = Hi[:measurement_dim, measurement_dim:].tocsc()
    Hxx_i = Hi[measurement_dim:, measurement_dim:].tocsc()

    # Solve for Y = Hll^{-1} * Hlx_i
    try:
        Y = spsolve(Hll, Hlx_i)
    except Exception as e:
        print(f"Linear solver failed for Hi index {idx}:", e)
        return idx, 0.0

    grad_schur = Hxx_i - Hlx_i.transpose().dot(X) - t2.dot(Hll_i.dot(Y)) + t2.dot(Hlx_i)
    grad_value = -min_eig_vec.flatten().dot(grad_schur.dot(min_eig_vec).flatten())
    return idx, grad_value

def frank_wolfe_optimization(
    inf_mats: List[csr_matrix],
    H0: csr_matrix,
    selection_init: np.ndarray,
    num_poses: int,
    A: np.ndarray,
    b: np.ndarray,
    max_iters: int = 1000,
    tol: float = 1e-2
) -> Tuple[np.ndarray, float, int]:
    """
    Performs Frank-Wolfe optimization to select sensors.
    """
    selection_cur = selection_init.copy().astype(float)
    prev_min_eig = -np.inf

    for iteration in range(max_iters):
        try:
            _, min_eig_val, min_eig_vec, Hll, X = compute_schur_complement(selection_cur, inf_mats, H0, num_poses)
        except RuntimeError as e:
            print(f"Error at iteration {iteration}: {e}")
            break

        # Convergence check
        if abs(min_eig_val - prev_min_eig) < tol:
            print(f"Converged at iteration {iteration}")
            break
        prev_min_eig = min_eig_val

        t2 = X.transpose()
        measurement_dim = Hll.shape[0]
        try:
            # Parallel gradient computation
            results = Parallel(n_jobs=-1)(
                delayed(compute_grad_parallel)(idx, Hi, Hll, X, t2, min_eig_vec, measurement_dim)
                for idx, Hi in enumerate(inf_mats)
            )
        except Exception as e:
            print(f"Error during gradient computation at iteration {iteration}: {e}")
            break

        grad = np.zeros_like(selection_cur)
        for idx, grad_value in results:
            grad[idx] = grad_value

        s = solve_lmo(grad, A, b)
        if s is None:
            print(f"LMO failed at iteration {iteration}.")
            break

        alpha = 2 / (iteration + 2)
        selection_cur = selection_cur + alpha * (s - selection_cur)
        selection_cur = np.clip(selection_cur, 0, 1)

    return selection_cur, min_eig_val, iteration + 1

def roundsolution(selection, k, inf_mats, H0):
    """
    Select top k elements and compute the score.
    """
    idx = np.argpartition(selection, -k)[-k:]
    rounded_sol = np.zeros(len(selection))
    if k > 0:
        rounded_sol[idx] = 1.0
    objective_score = evaluate_solution(inf_mats, H0, rounded_sol)
    return rounded_sol, objective_score


def roundsolution_breakties(selection, k, all_mats, H0):
    """
    Break ties by smallest eigenvalue computation.
    """
    s_rnd = np.round(selection, decimals=5)
    all_eigs = []
    for i, m in enumerate(all_mats):
        m_p = H0 + m
        m_p = 0.5 * (m_p + m_p.T)
        m_p += 1e-8 * np.eye(m_p.shape[0])
        try:
            eigval, _ = eigsh(m_p, k=1, which='SA', maxiter=2000)
        except Exception:
            eigvals = np.linalg.eigh(m_p.toarray())[0]
            eigval = eigvals[0:1]
        all_eigs.append(eigval[0])
    all_eigs = np.array(all_eigs)

    zipped_vals = np.array([(s_rnd[i], all_eigs[i]) for i in range(len(s_rnd))],
                           dtype=[('w', 'float'), ('weight', 'float')])
    idx = np.argpartition(zipped_vals, -k, order=['w', 'weight'])[-k:]
    rounded_sol = np.zeros(len(s_rnd))
    if k > 0:
        rounded_sol[idx] = 1.0
    return rounded_sol

def roundsolution_madow(selection, k, inf_mats, H0):
    """
    Probabilistic rounding approach.
    """
    num = len(selection)
    if k > num:
        raise ValueError("k cannot be greater than the number of candidates.")

    phi = np.zeros(num + 1)
    rounded_sol = np.zeros(num)
    phi[1:] = np.cumsum(selection)
    u = np.random.rand()

    for i in range(k):
        for j in range(num):
            if (phi[j] <= u + i) and (u + i < phi[j + 1]):
                if rounded_sol[j] == 1:
                    continue
                rounded_sol[j] = 1
                break

    objective_score = evaluate_solution(inf_mats, H0, rounded_sol)
    return rounded_sol, objective_score

def min_eig_obj_lse(x, inf_mats, H0, num_poses, beta=500.0):
    """
    Compute the LSE-based objective over all eigenvalues of the Schur complement.
    The objective is -f, where f = (-1/beta)*log(sum(exp(-beta*(lambda - lambda_min)))) + lambda_min.
    """
    H_schur, _, _, _, _ = compute_schur_complement(x, inf_mats, H0, num_poses)

    # Always compute all eigenvalues to ensure correctness.
    try:
        eigvals, _ = np.linalg.eigh(H_schur.toarray())
    except Exception as e:
        print(f"Eigenvalue computation failed in objective: {e}")
        return np.inf

    # Compute LSE objective
    lambda_min = eigvals.min()
    eigvals_shifted = eigvals - lambda_min
    scaled_exp = np.exp(-beta * eigvals_shifted)
    weight_sum = np.sum(scaled_exp) + 1e-12
    f = (-1.0 / beta) * np.log(weight_sum) + lambda_min

    return -f  


def min_eig_grad_lse(x, inf_mats, H0, num_poses, beta=500.0):
    """
    Compute the gradient of the LSE-based objective with respect to the selection vector x.
    This involves computing all eigenvalues and eigenvectors of the Schur complement.
    """
    H_schur, _, _, Hll, X = compute_schur_complement(x, inf_mats, H0, num_poses)

    # Compute all eigenpairs again for consistency.
    try:
        eigvals, eigvecs = np.linalg.eigh(H_schur.toarray())
    except Exception as e:
        print(f"Eigenvalue computation failed in gradient: {e}")
        return np.zeros_like(x)

    min_eig_val = eigvals.min()
    min_eig_idx = np.argmin(eigvals)
    v_min = eigvecs[:, min_eig_idx]

    eigvals_shifted = eigvals - min_eig_val
    scaled_exp_eigvals = np.exp(-beta * eigvals_shifted)
    weight_sum = np.sum(scaled_exp_eigvals) + 1e-12
    softmax_weights = scaled_exp_eigvals / weight_sum

    measurement_dim = Hll.shape[0]
    Hll_inv = None  # Will be computed per candidate when needed.

    def compute_grad_lse(idx, H_j, eigvecs, softmax_weights, v_min):
        # Extract sub-blocks
        Hll_j = H_j[:measurement_dim, :measurement_dim].tocsc()
        Hlx_j = H_j[:measurement_dim, measurement_dim:].tocsc()
        Hxx_j = H_j[measurement_dim:, measurement_dim:].tocsc()

        # Inversion attempt
        try:
            Hll_j_inv = np.linalg.pinv(Hll_j.toarray())
        except np.linalg.LinAlgError:
            return idx, 0.0

        H_schur_j = Hxx_j.toarray() - (Hlx_j.toarray().T @ Hll_j_inv @ Hlx_j.toarray())
        H_schur_j = 0.5 * (H_schur_j + H_schur_j.T)

        # Compute derivatives of all eigenvalues with respect to x_j
        lambda_derivatives = np.array([
            eigvecs[:, i].T @ H_schur_j @ eigvecs[:, i] for i in range(len(eigvals))
        ])

        # Derivative of min eigenvalue with respect to x_j
        d_lambda_min_dxj = v_min.T @ H_schur_j @ v_min

        grad_j = np.sum(softmax_weights * lambda_derivatives)
        return idx, grad_j

    results = Parallel(n_jobs=-1)(
        delayed(compute_grad_lse)(idx, H_j, eigvecs, softmax_weights, v_min)
        for idx, H_j in enumerate(inf_mats)
    )

    grad = np.zeros_like(x)
    for idx, grad_value in results:
        grad[idx] = grad_value

    return -grad

def scipy_minimize_lse(inf_mats, H0, selection_init, num_poses, A, b):
    """
    Uses scipy.optimize.minimize (SLSQP) with the LSE-based objective and gradient.
    Both objective and gradient now consistently compute all eigenvalues.
    """

    bounds = [(0, 1) for _ in range(selection_init.shape[0])]
    cons = [{'type': 'ineq', 'fun': lambda x: b - A @ x}]

    def objective(x):
        return min_eig_obj_lse(x, inf_mats, H0, num_poses)

    def gradient(x):
        return min_eig_grad_lse(x, inf_mats, H0, num_poses)

    res = minimize(
        fun=objective,
        x0=selection_init,
        method='trust-constr',
        jac=gradient,
        constraints=cons,
        bounds=bounds,
        options={'verbose':1, 'gtol' : 1e-6}
    )

    f_opt = min_eig_obj_lse(res.x, inf_mats, H0, num_poses)
    approx_min_eig_val = -f_opt
    return np.round(res.x, 4), approx_min_eig_val

def cvxpy_optimize_min_eig(
    inf_mats: List[sp.csr_matrix],
    H0: sp.csr_matrix,
    A: np.ndarray,
    b: np.ndarray,
    num_poses: int,
    binary: bool = False,
    solver: Optional[str] = None
) -> Tuple[np.ndarray, float, dict]:
    """
    CVXPY formulation to maximize min eigenvalue via semidefinite constraints.
    """
    n = len(inf_mats)
    x = cp.Variable(n, boolean=binary) if binary else cp.Variable(n)
    t = cp.Variable()

    # Construct H_schur(x)
    H_schur_expr = H0.toarray()
    for i in range(n):
        H_schur_expr = H_schur_expr + x[i] * inf_mats[i].toarray()
    H_schur_expr = 0.5 * (H_schur_expr + H_schur_expr.T)

    constraints = [
        H_schur_expr >> t * np.eye(H_schur_expr.shape[0]),
        A @ x <= b,
        x >= 0, x <= 1
    ]

    objective = cp.Maximize(t)
    prob = cp.Problem(objective, constraints)
    prob.solve(solver=solver, verbose=False)

    if prob.status not in ["optimal", "optimal_inaccurate"]:
        raise ValueError(f"Optimization failed with status: {prob.status}")

    x_opt = abs(np.round(x.value, 4))
    t_opt = t.value
    dual_vars = {}
    try:
        dual_vars['H_schur'] = constraints[0].dual_value
    except:
        dual_vars['H_schur'] = None

    return x_opt, t_opt, dual_vars

def cvxpy_minimize_lse(
    inf_mats: List[sp.csr_matrix],
    H0: sp.csr_matrix,
    selection_init: np.ndarray,
    num_poses: int,
    A: np.ndarray,
    b: np.ndarray,
    binary: bool = False,
    solver: Optional[str] = None
) -> Tuple[np.ndarray, float]:
    """
    Wrapper for cvxpy optimization.
    """
    x_opt, t_opt, _ = cvxpy_optimize_min_eig(
        inf_mats=inf_mats,
        H0=H0,
        A=A,
        b=b,
        num_poses=num_poses,
        binary=binary,
        solver=solver
    )
    return x_opt, t_opt

def solve_branch_and_cut(A, b, time_limit=None, initial_solution=None, cont_solution=None):
        """
        Solve the integer problem using a branch-and-cut approach.
        max lambda_2(L(w)) s.t. sum w_i = k, w_i in {0,1}.
        We introduce a variable z that should be an upper approximation of lambda_2(L(w)).
        Add cutting planes of the form:
        z <= f(w_curr) + grad_f(w_curr)^T (w - w_curr)
        """

        # Gurobi model
        model = gp.Model("branch_and_cut_eigenvalue")
        model.Params.OutputFlag = 1
        model.Params.LazyConstraints = 1
        model.Params.Cuts = -1
        # model.Params.MIPGap = 1.5e-1
        m = len(cont_solution)
        w_vars = model.addVars(m, vtype=GRB.BINARY, name="w")
        z_var = model.addVar(lb=-GRB.INFINITY, name="z")

        # Sum of w_i = k
        expr = gp.LinExpr()
        for i in range(A.shape[0]):
            row_indices = A[i].indices
            row_data = A[i].data
            expr = gp.LinExpr(row_data, [w_vars[j] for j in row_indices])
            model.addConstr(expr <= b[i])

        # Objective: maximize z
        model.setObjective(z_var, GRB.MAXIMIZE)
        model.addConstr(z_var <= evaluate_solution(cont_solution))
        
        print("Using zero vector for initial cut.")
        w0 = np.zeros(m)

        # Compute the initial cut at w0
        f_val0, f_vec0 = evaluate_fiedler_pair(w0)
        grad0 = grad_from_fiedler(f_vec0)
        rhs0 = f_val0 - np.dot(grad0, w0)

        lhs_expr0 = z_var
        for i in range(m):
            if abs(grad0[i]) > 1e-12:
                lhs_expr0 -= grad0[i] * w_vars[i]
        model.addConstr(lhs_expr0 <= rhs0, "initial_cut")

        def callback(model, where):
            if where == GRB.Callback.MIPSOL:
                w_sol_values = [model.cbGetSolution(w_vars[i]) for i in range(m)]
                w_sol = np.array(w_sol_values)
                f_val, f_vec = evaluate_fiedler_pair(w_sol)
                grad = grad_from_fiedler(f_vec)

                dot_product = np.dot(grad, w_sol)
                # Build cut: z ≤ f_val + grad^T(w - w_sol)
                rhs = f_val - dot_product
                lhs_expr = z_var
                for i in range(m):
                    if abs(grad[i]) > 1e-12:
                        lhs_expr -= grad[i] * w_vars[i]

                # Check if violated or can help tighten
                z_val = model.cbGetSolution(z_var)
                lhs_val = z_val - dot_product
                if lhs_val > rhs + 1e-9:
                    model.cbLazy(lhs_expr <= rhs)

        model.optimize(callback)

        if model.Status == GRB.OPTIMAL or model.Status == GRB.TIME_LIMIT:
            w_sol_dict = model.getAttr('x', w_vars)
            w_sol = np.array([w_sol_dict[i] for i in range(m)])
            return w_sol, evaluate_solution(w_sol)
        else:
            raise RuntimeError("Model not solved to optimality or time limit hit without feasible solution.")

def compute_variance_info(A, x_star):
        """
        For each row i of A (i.e. for each constraint), compute and print:
        
        - Expectation: E[s_i] = a_i^T x_star,
        - Variance: Var(s_i) = sum_j (a_ij^2 * x_star[j] * (1 - x_star[j])),
        - Ratio: sqrt(Var(s_i)) / E[s_i]
        
        Parameters
        ----------
        A : numpy.ndarray, shape (p, m)
            The constraint matrix (each row represents the coefficients for one constraint).
        x_star : numpy.ndarray, shape (m,)
            The fractional solution (with entries in [0, 1]).
        
        Returns
        -------
        info : list of tuples
            A list where each element is a tuple (E, Var, ratio) for one row of A.
        """
        p, m = A.shape
        info = []
        
        for i in range(p):
            a_i = A[i, :]          
            E = np.dot(a_i, x_star) 
            Var = np.sum((a_i**2) * x_star * (1 - x_star))  # Variance of s_i
            
            if E > 1e-12:
                ratio = np.sqrt(Var) / E
            else:
                ratio = np.inf  # If the expectation is nearly 0, the ratio is undefined
            
            # Print the values for this constraint row.
            print(f"Constraint row {i}:")
            print(f"  Expectation E[s_{i}] = {E:.4f}")
            print(f"  Variance Var(s_{i})  = {Var:.4f}")
            print(f"  Ratio sqrt(Var)/E    = {ratio:.4f}")
            
            info.append((E, Var, ratio))
        
        return info