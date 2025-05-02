from typing import List, Optional, Tuple, Dict, Any, Callable, Union
import numpy as np
import scipy.sparse as sp
from scipy.optimize import linprog, line_search
from scipy.sparse.linalg import eigsh, lobpcg
from enum import Enum
import logging
import time
import warnings
import networkx as nx
import json
import statistics
import argparse
import os


# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("frank_wolfe")

class Metric(Enum):
    """Enumeration of optimization metrics."""
    LOGDET = 1
    MIN_EIG = 2
    MSE = 3

class StepSizeStrategy(Enum):
    """Enumeration of step size seleci gottion strategies."""
    FIXED = 1
    DIMINISHING = 2
    BACKTRACKING = 3
    LIPSCHITZ = 4

# ======================================================================
# Step Size Selection Methods
# ======================================================================

def step_size_diminishing(iteration: int, gamma: float = 2.0) -> float:
    """
    Compute diminishing step size: gamma/(iteration+2).
    
    Args:
        iteration: Current iteration number (starting from 0)
        gamma: Constant multiplier (default: 2.0)
        
    Returns:
        float: Step size for the current iteration
    """
    return gamma / (iteration + 2)

def step_size_backtracking(
    obj_func: Callable, 
    obj_grad: Callable, 
    x_k: np.ndarray, 
    d_k: np.ndarray, 
    c: float = 0.0001, 
    beta: float = 0.5, 
    max_iters: int = 30,  # Increased from 20
    min_step_size: float = 1e-10,
    # inf_mats=None,  # Explicitly define these parameters
    # H0=None, 
    *args
) -> float:
    """
    Backtracking line search based on Armijo rule:
    f(x_k + gamma d_k) <= f(x_k) + c * gamma * <grad, d_k>
    
    Args:
        obj_func: Objective function
        obj_grad: Gradient function
        x_k: Current point
        d_k: Direction (s_k - x_k for Frank-Wolfe)
        c: Armijo parameter (default: 0.0001)
        beta: Reduction factor (default: 0.5)
        max_iters: Maximum number of backtracking iterations
        min_step_size: Minimum allowable step size
        *args: Additional arguments to pass to objective function
        
    Returns:
        float: Step size that satisfies Armijo condition
    """
    # Start with full step
    gamma = 1.0
    f_k = obj_func(x_k, *args)
    g_k = obj_grad(x_k, *args)
    armijo_product = np.dot(g_k, d_k)
    
    # Early exit if direction doesn't provide descent
    if np.abs(armijo_product) < 1e-10:
        logger.debug("Direction provides negligible descent, using minimum step size")
        return min_step_size
    
    # If needed for convergence, try larger step first
    if armijo_product > 0:  # Positive gradient dot product means we're maximizing
        x_larger = x_k + 2.0 * d_k
        f_larger = obj_func(x_larger, *args)
        if f_larger > f_k + c * 2.0 * armijo_product:
            logger.debug("Larger step (2.0) worked better")
            return 2.0  # Return larger step if it works better
    
    # Standard backtracking loop
    for i in range(max_iters):
        x_new = x_k + gamma * d_k
        f_new = obj_func(x_new, *args)
        
        # Check Armijo condition
        if f_new >= f_k + c * gamma * armijo_product:  # Changed <= to >= for maximization
            logger.debug(f"Backtracking found step size: {gamma} in {i+1} iterations")
            return gamma
        
        # Reduce step size
        gamma *= beta
        
        # Check if step size is too small
        if gamma < min_step_size:
            logger.debug(f"Step size {gamma} below minimum {min_step_size}, using minimum")
            return min_step_size
    
    logger.warning(f"Backtracking did not converge in {max_iters} iterations, using step size: {gamma}")
    return max(gamma, min_step_size)


def compute_lipschitz_constant(
    x: np.ndarray, 
    inf_mats: List[sp.spmatrix], 
    H0: sp.spmatrix, 
    num_eigs: int = 3,
    min_gap: float = 1e-8,  # Changed from 1e-6
    lipschitz_factor: float = 2.0  # Changed from 4.0
) -> float:
    """
    Compute adaptive estimate of Lipschitz constant for the minimum eigenvalue objective.
    
    Args:
        x: Current selection vector
        inf_mats: List of sparse information matrices
        H0: Prior information matrix
        num_eigs: Number of eigenvalues to compute for better stability
        min_gap: Minimum spectral gap to use (for numerical stability)
        lipschitz_factor: Scaling factor for the Lipschitz constant
        
    Returns:
        float: Lipschitz constant estimate
    """
    # Cache for matrix operator norms to avoid recomputing
    if not hasattr(compute_lipschitz_constant, "op_norm_cache"):
        compute_lipschitz_constant.op_norm_cache = {}
    
    # Compute the combined FIM at current point
    combined_fim = H0.copy()
    for xi, Hi in zip(x, inf_mats):
        combined_fim += xi * Hi
    
    # Get smallest eigenvalues to check for clustering
    try:
        min_eig_vals, _ = eigsh(
            combined_fim.tocsc() + 1e-10 * sp.eye(combined_fim.shape[0]), 
            k=min(num_eigs, combined_fim.shape[0]-1), 
            which='SA',
            tol=1e-4  # Improved tolerance for eigenvalue calculation
        )
        
        # Detect if there are clustered eigenvalues
        clustering_present = False
        for i in range(1, len(min_eig_vals)):
            if abs(min_eig_vals[i] - min_eig_vals[0]) < 1e-4:
                clustering_present = True
                break
                
        # Compute spectral gap adaptively
        if clustering_present:
            # If eigenvalues are clustered, use a larger gap to account for this
            for i in range(1, len(min_eig_vals)):
                if abs(min_eig_vals[i] - min_eig_vals[0]) > 1e-4:
                    delta = min_eig_vals[i] - min_eig_vals[0]
                    break
            else:
                # If all eigenvalues are clustered, use default
                delta = min_gap
        else:
            # Normal case - gap between first and second eigenvalue
            delta = min_eig_vals[1] - min_eig_vals[0]
            
        delta = max(delta, min_gap)  # Ensure numerical stability
        
    except Exception as e:
        logger.warning(f"Eigenvalue computation failed: {str(e)}. Using default gap.")
        delta = min_gap
    
    # Compute sum of operator norms of the matrices (with caching)
    sum_op_norm = 0
    for i, Hi in enumerate(inf_mats):
        if i not in compute_lipschitz_constant.op_norm_cache:
            try:
                max_eig_val, _ = eigsh(Hi.tocsc(), k=1, which='LA')
                compute_lipschitz_constant.op_norm_cache[i] = max_eig_val[0]
            except Exception:
                # Fallback method if eigsh fails
                op_norm = np.sqrt(np.sum(Hi.multiply(Hi).data))
                compute_lipschitz_constant.op_norm_cache[i] = op_norm
        
        sum_op_norm += compute_lipschitz_constant.op_norm_cache[i]
    
    # Adaptive Lipschitz factor based on clustering
    factor = lipschitz_factor * (2.0 if clustering_present else 1.0)
    
    # Lipschitz constant based on the spectral gap and operator norms
    L = factor * (1.0 / delta) * sum_op_norm
    logger.debug(f"Lipschitz estimate: {L:.2e} (gap: {delta:.2e}, clustering: {clustering_present})")
    return L


def step_size_lipschitz(
    x: np.ndarray, 
    duality_gap: float, 
    d_k: np.ndarray, 
    inf_mats: List[sp.spmatrix], 
    H0: sp.spmatrix,
    min_step_size: float = 1e-10,
    max_step_size: float = 1.0,
    safety_factor: float = 0.9  # New parameter to scale step size for stability
) -> float:
    """
    Compute step size based on adaptive Lipschitz constant with safety checks.
    
    Args:
        x: Current point
        duality_gap: Current duality gap (-grad @ d_k)
        d_k: Direction
        inf_mats: List of information matrices
        H0: Prior information matrix
        min_step_size: Minimum step size to return
        max_step_size: Maximum step size to return
        safety_factor: Safety scaling factor for numerical stability
        
    Returns:
        float: Step size
    """
    # Ensure duality gap is reasonable (numerical stability)
    if abs(duality_gap) < 1e-10:
        logger.debug("Negligible duality gap, using default step size")
        return 0.1  # Default step size when duality gap is tiny
    
    # Ensure direction has reasonable magnitude
    dir_norm_sq = np.linalg.norm(d_k)**2
    if dir_norm_sq < 1e-10:
        logger.debug("Direction has negligible magnitude, using default step size")
        return 0.1
        
    # Get Lipschitz constant estimate
    L = compute_lipschitz_constant(x, inf_mats, H0)
    
    # Compute step size with safety factor
    step_size = min(max_step_size, safety_factor * duality_gap / (L * dir_norm_sq))
    step_size = max(step_size, min_step_size)
    
    logger.debug(f"Lipschitz-based step size: {step_size:.6f}")
    return step_size
# ======================================================================
# Linear Minimization Oracle (LMO)
# ======================================================================

def solve_lmo(
    grad: np.ndarray, 
    A: Optional[Union[np.ndarray, sp.spmatrix]], 
    b: np.ndarray,
    solver_method: str = 'highs',
    fallback_method: str = 'interior-point'
) -> Optional[np.ndarray]:
    """
    Solves the Linear Minimization Oracle (LMO) problem for Frank-Wolfe optimization.
    
    Args:
        grad: Gradient vector (objective coefficients for the linear program)
        A: Inequality constraint matrix
        b: Inequality constraint bounds
        solver_method: Primary solver method for linprog
        fallback_method: Fallback solver method if primary fails
    
    Returns:
        np.ndarray or None: Solution vector if optimization succeeds; otherwise, None
    """
    dim = len(grad)
    
    # Define bounds for all variables between 0 and 1
    bounds = [(0, 1) for _ in range(dim)]
    
    # Convert A to CSC format if it's not None and is sparse
    if A is not None and sp.issparse(A):
        A = sp.csc_matrix(A)
    
    # Ensure b is a 1D array
    if b.ndim > 1:
        b = b.flatten()
    
    # Try primary solver method
    try:
        res = linprog(
            c=grad,  # For maximization, we minimize -grad 
            A_ub=A, 
            b_ub=b, 
            bounds=bounds, 
            method=solver_method
        )
        
        if res.success:
            return res.x
        else:
            logger.warning(f"Primary LMO solver failed: {res.message}, trying fallback")
    except Exception as e:
        logger.warning(f"Primary LMO solver exception: {str(e)}, trying fallback")
    
    # Try fallback solver method
    try:
        res = linprog(
            c=grad, 
            A_ub=A, 
            b_ub=b, 
            bounds=bounds, 
            method=fallback_method
        )
        
        if res.success:
            return res.x
        else:
            logger.error(f"Fallback LMO solver failed: {res.message}")
            return None
    except Exception as e:
        logger.error(f"Fallback LMO solver exception: {str(e)}")
        return None

# ======================================================================
# Stationarity Check
# ======================================================================

try:
    import cvxpy as cp
    CVXPY_AVAILABLE = True
except ImportError:
    CVXPY_AVAILABLE = False
    warnings.warn("CVXPY not available, stationarity check will be disabled")

def check_stationarity(
    r: int, 
    M_primes: List[np.ndarray], 
    A: Optional[np.ndarray] = None,
    verbose: bool = False
) -> Dict[str, Any]:
    """
    Check stationarity of the solution using CVXPY.
    
    Args:
        r: Multiplicity of the eigenvalues
        M_primes: List of transformed matrices
        A: Active constraint matrix (can be None if no active constraints)
        verbose: Whether to print detailed output
        
    Returns:
        Dict containing stationarity check results
    """
    if not CVXPY_AVAILABLE:
        return {"status": "CVXPY_NOT_AVAILABLE", "value": None}
    
    try:
        # Define the variables
        j = 0 if A is None else A.shape[0]  # Number of active constraints
        U = cp.Variable((r, r), symmetric=True)  # U is a symmetric matrix
        
        # Define constraints - using a simpler approach that avoids PSD constraints
        constraints = []
        
        # Create a diagonal constraint 0 <= U <= I (element-wise)
        for i in range(r):
            for j in range(r):
                if i == j:  # Diagonal elements
                    constraints.append(0 <= U[i, j])
                    constraints.append(U[i, j] <= 1)
                else:  # Off-diagonal elements
                    constraints.append(U[i, j] == U[j, i])  # Symmetry
        
        constraints.append(cp.trace(U) == 1)  # tr(U) = 1
        
        # Add constraints for active inequalities
        if j > 0:
            lamda = cp.Variable(j)
            constraints.append(lamda >= 0)
            ATlamda = A.T @ lamda if A is not None else 0
        else:
            ATlamda = 0
        
        # Build the objective (minimize norm of residual)
        tmp = []
        for M in M_primes:
            tmp.append(cp.trace(M @ U))
        
        if len(tmp) > 0:
            M_res = cp.hstack(tmp)  # Residual corresponding to subdifferential
            res = M_res + ATlamda if j > 0 else M_res
            objective = cp.Minimize(cp.norm2(res))  # Use norm2 instead of norm
        else:
            # Handle empty M_primes case
            objective = cp.Minimize(0)
        
        # Solve the problem
        prob = cp.Problem(objective, constraints)
        
        # Try solvers in order of preference
        solvers = ['OSQP', 'ECOS', 'SCS']  # Try common solvers first
        prob_status = None
        prob_value = None
        U_value = None
        lamda_value = None
        
        for solver in solvers:
            try:
                if verbose:
                    logger.info(f"Trying solver: {solver}")
                prob.solve(solver=solver)
                prob_status = prob.status
                prob_value = prob.value
                U_value = U.value
                if j > 0 and 'lamda' in locals():
                    lamda_value = lamda.value
                break
            except cp.SolverError:
                if verbose:
                    logger.warning(f"Solver {solver} failed")
                continue
            except Exception as e:
                if verbose:
                    logger.warning(f"Solver {solver} error: {str(e)}")
                continue
        
        # If all solvers failed
        if prob_status is None:
            return {"status": "SOLVER_ERROR", "value": None}
        
        # Print results if verbose
        if verbose:
            logger.info(f"Stationarity check status: {prob_status}")
            logger.info(f"Stationarity check optimal value: {prob_value}")
        
        return {
            "status": prob_status,
            "value": prob_value,
            "U": U_value,
            "lambda": lamda_value if j > 0 else None
        }
    except Exception as e:
        logger.error(f"Error in stationarity check: {str(e)}")
        return {"status": "ERROR", "value": None, "error": str(e)}

def reset_min_eig_gradient_cache():
    """
    Clears the cached attributes in min_eigenvalue_gradient so that 
    any dimension-specific caching won't collide with a new problem size.
    """
    cache_attrs = [
        "cached_x",
        "cached_combined_fim",
        "cached_inf_mats",
        "last_eigvec",
    ]
    for attr in cache_attrs:
        if hasattr(min_eigenvalue_gradient, attr):
            delattr(min_eigenvalue_gradient, attr)

# ======================================================================
# Main Frank-Wolfe Optimization
# ======================================================================

def frank_wolfe_optimization(
    obj_func: Callable,
    obj_grad: Callable,
    selection_init: np.ndarray,
    A: np.ndarray,
    b: np.ndarray,
    max_iterations: int = 1000,
    min_iterations: int = 2,
    convergence_tol: float = 2e-2,
    step_size_strategy: str = "diminishing",
    verbose: bool = True,
    args: tuple = (),
    step_size_params: Dict[str, Any] = None
) -> Tuple[np.ndarray, float, int, Dict[str, List]]:
    """
    Performs Frank-Wolfe optimization to maximize an objective function.
    
    Args:
        obj_func: Objective function to maximize
        obj_grad: Gradient function of the objective
        selection_init: Initial selection vector
        A: Inequality constraint matrix
        b: Inequality constraint bounds
        max_iterations: Maximum number of iterations
        min_iterations: Minimum number of iterations to perform
        convergence_tol: Convergence tolerance
        step_size_strategy: Strategy for computing step size ("fixed", "diminishing", or "backtracking")
        verbose: Whether to print detailed output
        args: Additional arguments to pass to objective and gradient functions
        step_size_params: Additional parameters for step size computation
        
    Returns:
        Tuple containing:
        - Final selection vector
        - Best objective value
        - Number of iterations performed
        - Log dictionary with iteration history
    """
    # Set up logging level based on verbosity
    logger.setLevel(logging.INFO if verbose else logging.WARNING)
    
    # Initialize step size parameters
    if step_size_params is None:
        step_size_params = {}
    
    # Initialize log for tracking results
    fw_log = {
        "iter": [],
        "obj_val": [],
        "step_size": [],
        "duality_gap": [],
        "grad_norm": [],
        "time_per_iter": []
    }
    
    # Initialize selection vector as a continuous variable (float)
    selection_cur = selection_init.copy().astype(float)
    
    # Track best objective value for convergence check
    prev_obj_val = obj_func(selection_cur, *args)
    
    # Main optimization loop
    for iteration in range(max_iterations):
        start_time = time.time()
        
        # Compute objective and gradient
        obj_val = obj_func(selection_cur, *args)
        grad = obj_grad(selection_cur, *args)
        
        # Check for NaN or Inf in gradient
        if not np.all(np.isfinite(grad)):
            logger.error("Non-finite values in gradient, terminating")
            break
        
        # Solve the Linear Minimization Oracle (LMO)
        s = solve_lmo(-grad, A, b)  # Note the negation for maximization
        
        if s is None:
            logger.error(f"LMO failed to find a feasible solution at iteration {iteration}.")
            break
        
        # Compute direction and duality gap
        d = s - selection_cur
        duality_gap = grad @ d
        grad_norm = np.linalg.norm(grad)
        
        # Save iteration data
        iter_time = time.time() - start_time
        fw_log["iter"].append(iteration)
        fw_log["obj_val"].append(obj_val)
        fw_log["duality_gap"].append(duality_gap)
        fw_log["grad_norm"].append(grad_norm)
        fw_log["time_per_iter"].append(iter_time)
        
        # Check convergence criteria - only after minimum iterations to avoid premature convergence
        if iteration >= min_iterations:
            if duality_gap < convergence_tol * max(1.0, abs(obj_val)):
                logger.info(f"Converged at iteration {iteration}: duality gap minimized")
                break
                
            if grad_norm < convergence_tol:
                logger.info(f"Converged at iteration {iteration}: gradient norm minimized")
                break
        
        prev_obj_val = obj_val
        
        # Compute step size based on selected strategy
        if step_size_strategy == "fixed":
            alpha = step_size_params.get("fixed_alpha", 0.1)
        elif step_size_strategy == "diminishing":
            alpha = step_size_diminishing(iteration, step_size_params.get("gamma", 2.0))
        elif step_size_strategy == "backtracking":
            alpha = step_size_backtracking(
                obj_func, obj_grad, selection_cur, d,
                step_size_params.get("c", 0.0001),
                step_size_params.get("beta", 0.5),
                step_size_params.get("max_iters", 20),
                *args
            )
        else:
            alpha = 2.0 / (iteration + 2)  # Default to classic Frank-Wolfe
        
        fw_log["step_size"].append(alpha)
        
        # Update the selection vector
        selection_cur = selection_cur + alpha * d
        selection_cur = np.clip(selection_cur, 0, 1)  # Project back to [0,1]^n
        
        # Log iteration results
        if verbose:
            logger.info(
                f"Iteration {iteration}: obj={obj_val:.6f}, "
                f"step={alpha:.6f}, gap={duality_gap:.6e}, "
                f"grad_norm={grad_norm:.6e}, time={iter_time:.3f}s"
            )
    
    # Make sure we've done at least a few iterations
    if iteration < 10:
        logger.warning(f"Algorithm terminated after only {iteration} iterations - check for implementation issues")
    
    # Get final objective value
    final_obj_val = obj_func(selection_cur, *args)
    
    return selection_cur, final_obj_val, iteration + 1, fw_log

# ======================================================================
# Objective Functions and Gradients
# ======================================================================

def min_eigenvalue_objective(
    x: np.ndarray, 
    inf_mats: List[sp.spmatrix], 
    H0: sp.spmatrix
) -> float:
    """
    Compute the minimum eigenvalue of the sum of selected matrices.
    
    Args:
        x: Selection vector
        inf_mats: List of information matrices
        H0: Prior information matrix
        
    Returns:
        float: Minimum eigenvalue
    """
    combined_fim = H0.copy()
    for xi, Hi in zip(x, inf_mats):
        combined_fim += xi * Hi
    
    # Add small regularization for numerical stability
    combined_fim += 1e-10 * sp.eye(combined_fim.shape[0])
    
    try:
        min_eig_val, _ = eigsh(combined_fim.tocsc(), k=1, which='SA', tol=1e-2, ncv = 40)
        return min_eig_val[0]
    except Exception as e:
        logger.error(f"Error computing eigenvalues: {str(e)}")
        # Return a very negative value to indicate failure
        return -1e10

def min_eigenvalue_gradient(
    x: np.ndarray, 
    inf_mats: list,         # List of sparse matrices
    H0: sp.spmatrix,
    skip_threshold: float = 1e-12,
    lobpcg_tol: float = 1e-4,
    lobpcg_maxiter: int = 200
) -> np.ndarray:
    """
    Compute the gradient of the minimum eigenvalue objective 
    (lambda_min of H0 + sum_i x_i * H_i), using:
    
    1) An incremental update for the combined FIM (cached).
    2) LOBPCG every time (no ARPACK), with a warm start from the last eigenvector.
    3) An optional skip if the new x is extremely close to the cached x.
       (We just reuse the last eigenvector in that case, to avoid re-solving.)

    Args:
        x : np.ndarray
            Current selection vector (shape = (n,)).
        inf_mats : list of sp.spmatrix
            List of sparse information matrices H_i.
        H0 : sp.spmatrix
            Prior information matrix (same dimension as each H_i).
        skip_threshold : float
            If ||x - cached_x|| < skip_threshold, skip a new eigen solve 
            and reuse the last eigenvector. Default 1e-12.
        lobpcg_tol : float
            Tolerance for lobpcg solver.
        lobpcg_maxiter : int
            Maximum iterations for lobpcg.

    Returns:
        grad : np.ndarray
            The gradient vector of size (n,), grad[i] = - (v^T H_i v).
    """
    
    # --- 1) Cache the CSC conversion of inf_mats once ---
    if not hasattr(min_eigenvalue_gradient, "cached_inf_mats"):
        min_eigenvalue_gradient.cached_inf_mats = [
            Hi.tocsc() if not sp.isspmatrix_csc(Hi) else Hi
            for Hi in inf_mats
        ]
    cached_inf_mats = min_eigenvalue_gradient.cached_inf_mats
    
    # --- 2) Check if we have a cached x and combined FIM ---
    if not hasattr(min_eigenvalue_gradient, "cached_x"):
        # First call: build combined_fim from scratch
        combined_fim = H0.copy().tocsc()
        for i, xi in enumerate(x):
            if abs(xi) > 1e-12:
                combined_fim += xi * cached_inf_mats[i]
        
        min_eigenvalue_gradient.cached_x = x.copy()
        min_eigenvalue_gradient.cached_combined_fim = combined_fim
        # No last eigenvector yet
        min_eigenvalue_gradient.last_eigvec = None

        # On the first call, we can define diff as something large (so we won't skip)
        diff = x  # or e.g. np.ones_like(x)*9999
    else:
        # We have a cached FIM & x; do incremental update
        old_x = min_eigenvalue_gradient.cached_x
        diff = x - old_x
        combined_fim = min_eigenvalue_gradient.cached_combined_fim

        # If there's any nontrivial change, update the matrix
        if np.linalg.norm(diff) > 1e-12:
            combined_fim = combined_fim.copy()
            for i, d in enumerate(diff):
                if abs(d) > 1e-12:
                    combined_fim += d * cached_inf_mats[i]
            min_eigenvalue_gradient.cached_x = x.copy()
            min_eigenvalue_gradient.cached_combined_fim = combined_fim
    
    # --- 3) Possibly skip a new eigen solve if the x-change is tiny ---
    # (Only skip if we already have a last_eigvec)
    if np.linalg.norm(diff) < skip_threshold and min_eigenvalue_gradient.last_eigvec is not None:
        min_eig_vec = min_eigenvalue_gradient.last_eigvec
    else:
        # Solve for the smallest eigenvalue using LOBPCG with warm start
        n = combined_fim.shape[0]
        if min_eigenvalue_gradient.last_eigvec is not None:
            v0 = min_eigenvalue_gradient.last_eigvec
        else:
            # First solve or no stored eigvec
            v0 = np.random.rand(n)
            v0 /= np.linalg.norm(v0)
        
        X = v0.reshape(-1, 1)
        
        eigvals, eigvecs = lobpcg(
            combined_fim,
            X,
            largest=False,
            tol=lobpcg_tol,
            maxiter=lobpcg_maxiter
        )
        
        min_eig_vec = eigvecs[:, 0]
        min_eigenvalue_gradient.last_eigvec = min_eig_vec
    
    # --- 4) Compute the gradient: grad[i] = - (v^T H_i v) ---
    grad = np.empty(len(cached_inf_mats), dtype=float)
    
    for i, Hi in enumerate(cached_inf_mats):
        temp = Hi.dot(min_eig_vec)
        grad[i] = -float(min_eig_vec.dot(temp))
    
    return grad


# ======================================================================
# Gurobi Branch and Cut Implementation
# ======================================================================

try:
    import gurobipy as gp
    from gurobipy import GRB
    GUROBI_AVAILABLE = True
except ImportError:
    GUROBI_AVAILABLE = False
    warnings.warn("Gurobi not available, branch and cut optimization will be disabled")

def branch_and_cut_gurobi(
    obj_func: Callable,
    obj_grad: Callable,
    A: np.ndarray,
    b: np.ndarray,
    n: int,
    time_limit: int = 600,
    mip_gap: float = 0.0,
    verbose: bool = False,
    cont_solution: np.ndarray = None,
    args: tuple = ()
) -> Tuple[np.ndarray, float, Dict[str, Any]]:
    """
    Solves matrix selection problem using branch and cut with Gurobi.
    
    Args:
        obj_func: Objective function to maximize
        obj_grad: Gradient function to compute subgradients/cuts
        A: Inequality constraint matrix
        b: Inequality constraint bounds
        n: Dimension of the solution vector
        time_limit: Time limit in seconds (default: 600)
        mip_gap: MIP gap for termination (default: 0.01)
        verbose: Whether to print detailed output
        cont_solution: Continuous solution to use for initialization (optional)
        args: Additional arguments to pass to objective and gradient functions
    
    Returns:
        Tuple containing:
        - Selected binary vector
        - Objective value
        - Solution statistics
    """
    if not GUROBI_AVAILABLE:
        logger.error("Gurobi is not available. Cannot perform branch and cut optimization.")
        return None, float('-inf'), {"status": "Gurobi not available"}
    
    # Create a Gurobi model
    with gp.Env(empty=True) as env:
        env.setParam('OutputFlag', 1 if verbose else 0)
        env.start()
        
        with gp.Model("Branch_And_Cut", env=env) as model:
            # Set parameters
            model.setParam('TimeLimit', time_limit)
            model.setParam('MIPGap', mip_gap)
            model.setParam('LazyConstraints', 1)
            model.setParam('Cuts', 2)  # Aggressive cut generation
            
            # Create binary decision variables
            x = model.addVars(n, vtype=GRB.BINARY, name="x")
            
            # Add constraints
            for i in range(A.shape[0]):
                expr = gp.LinExpr()
                for j in range(n):
                    expr.add(x[j], A[i, j])
                model.addConstr(expr <= b[i], f"constraint_{i}")

            # Add auxiliary variable for objective
            t = model.addVar(lb=-GRB.INFINITY, name="t")
            
            # Set the objective to maximize t
            model.setObjective(t, GRB.MAXIMIZE)
            
            # Add initial cut from the zero solution
            if verbose:
                logger.info("Adding initial cut from zero vector")
                
            # Compute objective value and gradient for the zero vector
            zero_vec = np.zeros(n)
            obj_val_0 = obj_func(zero_vec, *args)
            grad_0 = obj_grad(zero_vec, *args)
            
            # Add the cut: t <= obj_val_0 + sum_j grad_0[j] * x[j]
            cut_expr0 = gp.LinExpr()
            for j in range(n):
                cut_expr0.add(x[j], grad_0[j])
            model.addConstr(t <= obj_val_0 + cut_expr0, "initial_cut_zero")
            
            # If we have a continuous solution, use it to generate an upper bound
            if cont_solution is not None:
                if verbose:
                    logger.info("Using continuous solution for initial bound")
                
                # Compute objective value and gradient for continuous solution
                obj_val_cont = obj_func(cont_solution, *args)
                grad_cont = obj_grad(cont_solution, *args)
                
                # Add cut: t <= obj_val_cont + sum_j grad_cont[j] * (x[j] - cont_solution[j])
                rhs_cont = obj_val_cont - np.dot(grad_cont, cont_solution)
                cut_expr_cont = gp.LinExpr()
                for j in range(n):
                    cut_expr_cont.add(x[j], grad_cont[j])
                
                model.addConstr(t <= rhs_cont + cut_expr_cont, "initial_cut_cont")
                
                # Set an upper bound for t based on continuous solution
                t.ub = obj_val_cont
            
            # Callback function to add cutting planes
            def eigenvalue_callback(model, where):
                if where == GRB.Callback.MIPSOL:
                    # Get current integer solution
                    x_vals = model.cbGetSolution([x[j] for j in range(n)])
                    t_val = model.cbGetSolution(t)
                    
                    # Compute the actual objective value and gradient at this solution
                    actual_obj = obj_func(x_vals, *args)
                    grad = obj_grad(x_vals, *args)
                    
                    # If the current t is greater than the actual objective (with some tolerance),
                    # add a cutting plane
                    if t_val > actual_obj + 1e-6:
                        # Add cut: t <= actual_obj + sum_j grad[j] * (x[j] - x_vals[j])
                        rhs = actual_obj - np.dot(grad, x_vals)
                        
                        cut_expr = gp.LinExpr()
                        for j in range(n):
                            cut_expr.add(x[j], grad[j])
                        
                        model.cbLazy(t <= rhs + cut_expr)
                        
                        if verbose:
                            violation = t_val - actual_obj
                            logger.debug(f"Added cut at integer solution with violation {violation:.6f}")
                
                elif where == GRB.Callback.MIPNODE:
                    # Only add cuts at nodes where we have an optimal relaxation
                    if model.cbGet(GRB.Callback.MIPNODE_STATUS) != GRB.OPTIMAL:
                        return
                    
                    # Get the relaxation solution at this node
                    x_vals = model.cbGetNodeRel([x[j] for j in range(n)])
                    t_val = model.cbGetNodeRel(t)
                    
                    # Skip if the solution is nearly binary (let MIPSOL handle it)
                    if all(xi < 0.1 or xi > 0.9 for xi in x_vals):
                        return
                    
                    # Compute the objective value and gradient at this relaxed point
                    actual_obj = obj_func(x_vals, *args)
                    grad = obj_grad(x_vals, *args)
                    
                    # Check if we need to add a cut
                    if t_val > actual_obj + 1e-6:
                        # Add the cut
                        rhs = actual_obj - np.dot(grad, x_vals)
                        
                        cut_expr = gp.LinExpr()
                        for j in range(n):
                            cut_expr.add(x[j], grad[j])
                        
                        model.cbCut(t <= rhs + cut_expr)
                        
                        if verbose:
                            violation = t_val - actual_obj
                            logger.debug(f"Added cut at node relaxation with violation {violation:.6f}")
            
            # Optimize with callback
            if verbose:
                logger.info("Starting branch and cut optimization")
            
            model.optimize(eigenvalue_callback)
            
            # Get solution
            if model.status == GRB.OPTIMAL or model.status == GRB.TIME_LIMIT:
                # Extract solution
                x_solution = np.zeros(n)
                for j in range(n):
                    x_solution[j] = round(x[j].X)  # Round to handle potential numerical issues
                
                # Compute actual objective with the final solution
                obj_val = obj_func(x_solution, *args)
                
                # Collect solution statistics
                stats = {
                    "status": model.status,
                    "runtime": model.Runtime,
                    "mip_gap": model.MIPGap if hasattr(model, 'MIPGap') else None,
                    "obj_bound": model.ObjBound if hasattr(model, 'ObjBound') else None,
                    "num_nodes": model.NodeCount,
                    "num_cuts": model.NumVars - n - 1  # Approximate measure of cuts added
                }
                
                if verbose:
                    logger.info(f"Branch and cut completed with status {model.status}")
                    logger.info(f"Objective value: {obj_val:.6f}")
                    logger.info(f"Runtime: {model.Runtime:.2f} seconds")
                    
                return x_solution, obj_val, stats
            else:
                logger.error(f"Gurobi optimization failed with status {model.status}")
                return None, float('-inf'), {"status": model.status}

def greedy_algorithm_2(
    obj_func: Callable,
    A: np.ndarray,
    b: np.ndarray,
    n: int,
    verbose: bool = True,
    timeout: Optional[float] = None,
    args: tuple = ()
) -> Tuple[np.ndarray, float, Dict[str, Any]]:
    """
    Implementation of Algorithm 2 for maximizing functions under multiple constraints.
    
    This algorithm greedily maximizes a function subject to constraints Ax ≤ b
    by selecting elements based on their marginal gain to cost ratio.
    
    Args:
        obj_func: Objective function to maximize (takes numpy array and returns float)
        A: Constraint matrix
        b: Constraint bounds vector
        n: Problem dimension (number of elements)
        verbose: Whether to print progress information
        timeout: Maximum execution time in seconds (None means no limit)
        args: Additional arguments to pass to objective function
        
    Returns:
        Tuple containing:
        - Selected binary vector
        - Objective value
        - Dictionary with statistics and status information
    """
    start_time = time.time()
    m = A.shape[0]  # Number of constraints
    
    # Statistics collection
    stats = {
        "iterations": 0,
        "obj_evaluations": 0,
        "timed_out": False,
        "runtime": 0.0,
        "obj_history": []
    }
    
    # Initialize with empty selection
    current_selection = np.zeros(n, dtype=int)
    
    # Precompute constraint values for empty selection
    constraint_values = np.zeros(A.shape[0])
    
    # Compute initial objective
    current_obj = obj_func(current_selection, *args)
    stats["obj_evaluations"] += 1
    stats["obj_history"].append(current_obj)
    
    if verbose:
        logger.info(f"Starting greedy selection with initial objective: {current_obj:.6f}")
        if timeout is not None:
            logger.info(f"Timeout set to {timeout:.1f} seconds")
    
    # Set of candidate indices (elements that can potentially be added)
    W = set(range(n))
    
    # Main greedy selection loop
    while W:
        # Check timeout
        current_time = time.time()
        if timeout is not None and (current_time - start_time) > timeout:
            if verbose:
                logger.warning(f"Greedy selection timed out after {current_time - start_time:.1f} seconds")
            stats["timed_out"] = True
            break
            
        stats["iterations"] += 1
        
        # Find the best element and constraint pair
        best_ratio = -float('inf')
        best_element = None
        best_i = None
        
        for v in W:
            # Calculate marginal gain in objective function
            test_selection = current_selection.copy()
            test_selection[v] = 1
            
            new_obj = obj_func(test_selection, *args)
            stats["obj_evaluations"] += 1
            
            delta_f = new_obj - current_obj
            
            # Skip if no improvement in objective
            if delta_f <= 0:
                continue
                
            # Find the best ratio across all constraints where delta_h_i > 0
            for i in range(m):
                delta_h_i = A[i, v]
                
                # Skip if element has no impact on this constraint or impact is negative
                if delta_h_i <= 0:
                    continue
                
                # Calculate ratio: marginal gain / marginal cost
                ratio = delta_f / delta_h_i
                
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_element = v
                    best_i = i
                    
        # If no improvement found, exit
        if best_element is None:
            break
        
        # Check if adding best_element violates any constraint
        will_violate = False
        for i in range(m):
            b_value = float(b[i]) if isinstance(b[i], np.ndarray) else float(b[i])
            if constraint_values[i] + A[i, best_element] > b_value:
                will_violate = True
                break
                
        if not will_violate:
            # Add element to solution
            current_selection[best_element] = 1
            W.remove(best_element)
            
            # Update objective
            current_obj = obj_func(current_selection, *args)
            stats["obj_evaluations"] += 1
            stats["obj_history"].append(current_obj)
            
            # Update constraint values
            for i in range(m):
                constraint_values[i] += A[i, best_element]
            
            if verbose and stats["iterations"] % 10 == 0:
                logger.info(f"Iteration {stats['iterations']}: Added element {best_element}")
                logger.info(f"New objective: {current_obj:.6f}")
        else:
            # Element violates constraints - remove from consideration
            W.remove(best_element)
            if verbose and stats["iterations"] % 10 == 0:
                logger.info(f"Element {best_element} violates constraints, removed from consideration")
    
    # Compute final statistics
    stats["runtime"] = time.time() - start_time
    stats["selected_count"] = np.sum(current_selection)
    stats["final_constraints"] = constraint_values.tolist()
    
    if verbose:
        logger.info(f"Greedy selection complete:")
        logger.info(f"Final objective value: {current_obj:.6f}")
        logger.info(f"Selected {stats['selected_count']} elements")
        logger.info(f"Runtime: {stats['runtime']:.2f} seconds")
        logger.info(f"Objective evaluations: {stats['obj_evaluations']}")
        
        # Report constraint satisfaction
        for i in range(m):
            b_value = float(b[i]) if isinstance(b[i], np.ndarray) else float(b[i])
            utilization = constraint_values[i] / b_value * 100 if b_value != 0 else 0
            logger.info(f"Constraint {i}: {constraint_values[i]:.4f} / {b_value:.4f} ({utilization:.1f}%)")
    
    return current_selection, current_obj, stats


# ======================================================================
# Variance Information Calculation
# ======================================================================

def compute_variance_info(A: np.ndarray, x_star: np.ndarray):
    """
    For each row i of A:
      - Print:
          E[s_i], Var(s_i), ratio = sqrt(Var) / E
      - Collect the same info into a list of dicts for JSON.

    Parameters
    ----------
    A : np.ndarray, shape (p, m)
        The constraint matrix (each row = one constraint).
    x_star : np.ndarray, shape (m,)
        The solution vector (entries in [0,1]).

    Returns
    -------
    A list of dictionaries, each like {"E": ..., "Var": ..., "ratio": ...},
    which is suitable for JSON serialization.
    """
    p, m = A.shape
    variance_info_list = []

    for i in range(p):
        a_i = A[i, :]
        E = np.dot(a_i, x_star)
        Var = np.sum((a_i**2) * x_star * (1 - x_star))
        
        if E > 1e-12:
            ratio = np.sqrt(Var) / E
        else:
            ratio = np.inf
        
        # -- Print to console (same as original) --
        print(f"Constraint row {i}:")
        print(f"  Expectation E[s_{i}] = {E:.4f}")
        print(f"  Variance Var(s_{i})  = {Var:.4f}")
        print(f"  Ratio sqrt(Var)/E    = {ratio:.4f}")
        
        # -- Also store in JSON-friendly dict --
        variance_info_list.append({
            "E": float(E),
            "Var": float(Var),
            "ratio": float(ratio)
        })

    return variance_info_list


# ======================================================================
# Rounding Function for Converting Continuous to Binary
# ======================================================================

def randomized_rounding(
    cont_sol: np.ndarray,
    obj_func: Callable,
    A: np.ndarray,
    b: np.ndarray,
    num_samples: int = 100,
    verbose: bool = False,
    args: tuple = ()
) -> Tuple[np.ndarray, float, Dict[str, Any]]:
    """
    Randomized rounding to convert a continuous solution to a feasible binary solution.
    
    Args:
        cont_sol: Continuous solution vector (in [0,1]^n)
        obj_func: Objective function to evaluate solutions
        A: Inequality constraint matrix
        b: Inequality constraint bounds
        num_samples: Number of randomized rounding samples to try
        verbose: Whether to print detailed output
        args: Additional arguments to pass to objective function
    
    Returns:
        Tuple containing:
        - Best binary solution found
        - Corresponding objective value
        - Statistics dictionary
    """
    n = len(cont_sol)
    
    # Initialize with no solution
    best_binary = None
    best_obj = float('-inf')
    
    # Statistics tracking
    stats = {
        "num_samples": num_samples,
        "num_feasible": 0,
        "feasibility_rate": 0.0,
        "obj_values": [],
        "runtime": 0.0
    }
    
    start_time = time.time()
    
    if verbose:
        logger.info(f"Starting randomized rounding with {num_samples} samples...")
        # Compute variance information for constraints
        if verbose:
            for i in range(A.shape[0]):
                a_i = A[i, :]
                E = np.dot(a_i, cont_sol)
                Var = np.sum((a_i**2) * cont_sol * (1 - cont_sol))
                
                if E > 1e-12:
                    ratio = np.sqrt(Var) / E
                else:
                    ratio = float('inf')
                
                logger.info(f"Constraint row {i}:")
                logger.info(f"  Expectation E[s_{i}] = {E:.4f}")
                logger.info(f"  Variance Var(s_{i})  = {Var:.4f}")
                logger.info(f"  Ratio sqrt(Var)/E    = {ratio:.4f}")
    
    # Try randomized rounding using continuous solution values as probabilities
    for i in range(num_samples):
        binary_sol = np.zeros(n, dtype=int)
        for j in range(n):
            # Each variable has probability cont_sol[j] of being 1
            binary_sol[j] = 1 if np.random.random() < cont_sol[j] else 0
        
        # Check constraints
        feasible = True
        for k in range(A.shape[0]):
            if np.dot(A[k], binary_sol) > b[k] + 1e-9:
                feasible = False
                break
        
        if feasible:
            stats["num_feasible"] += 1
            obj_val = obj_func(binary_sol, *args)
            stats["obj_values"].append(obj_val)
            
            if obj_val > best_obj:
                best_obj = obj_val
                best_binary = binary_sol.copy()
                
                if verbose and (i+1) % (max(1, num_samples // 10)) == 0:
                    logger.info(f"Sample {i+1}: Found better solution with objective {obj_val:.6f}")
    
    stats["runtime"] = time.time() - start_time
    stats["feasibility_rate"] = stats["num_feasible"] / num_samples if num_samples > 0 else 0
    
    # Report statistics if in verbose mode
    if verbose:
        logger.info(f"Rounding complete: {stats['num_feasible']} feasible solutions found ({stats['feasibility_rate']:.2%})")
        if best_binary is not None:
            logger.info(f"Best objective value: {best_obj:.6f}")
    
    # If no feasible solution was found, try a fallback approach
    if best_binary is None:
        if verbose:
            logger.warning("No feasible solution found with randomized rounding. Trying greedy fallback approach.")
        
        # Sort by value (descending to prioritize variables with high probability)
        sorted_indices = np.argsort(-cont_sol)
        
        # Try starting with empty solution and greedily adding variables
        binary_sol = np.zeros(n, dtype=int)
        
        for idx in sorted_indices:
            # Try adding this variable
            binary_sol[idx] = 1
            
            # Check if still feasible
            feasible = True
            for i in range(A.shape[0]):
                if np.dot(A[i], binary_sol) > b[i] + 1e-9:
                    feasible = False
                    break
            
            # If not feasible, revert
            if not feasible:
                binary_sol[idx] = 0
        
        # Check if this solution is valid
        obj_val = obj_func(binary_sol, *args)
        if best_binary is None or obj_val > best_obj:
            best_obj = obj_val
            best_binary = binary_sol.copy()
            
            if verbose:
                logger.info(f"Fallback approach found solution with objective {obj_val:.6f}")
    
    if best_binary is None:
        # If still no solution, return an empty selection as last resort
        logger.warning("Failed to find any feasible solution, returning empty selection")
        return np.zeros(n, dtype=int), obj_func(np.zeros(n, dtype=int), *args), stats
    
    return best_binary, best_obj, stats

def knapsack_cr(
    a_j: np.ndarray,
    b_j: np.ndarray,
    y: np.ndarray,
    verbose: bool = False
) -> np.ndarray:
    """
    KNAPSACK-CR procedure for a single constraint.
    
    Args:
        a_j: Vector of resource requirements for constraint j
        b_j: Capacity bound for constraint j (could be scalar or array)
        y: Binary vector from Bernoulli sampling
        verbose: Whether to print progress
    
    Returns:
        tau_j: Binary vector indicating which elements to keep
    """
    n = len(a_j)
    assert n == len(y), "Dimension mismatch between a_j and y"
    
    # Initialize empty solution
    tau = np.zeros(n, dtype=bool)
    
    # Extract scalar value from b_j if it's an array
    if isinstance(b_j, np.ndarray):
        capacity = float(b_j.item()) if b_j.size == 1 else float(b_j[0])
    else:
        capacity = float(b_j)
    
    # We only consider elements where y[i] = 1
    candidate_indices = np.where(y)[0]
    
    if len(candidate_indices) == 0:
        return tau
    
    # Get resource requirements for candidate elements
    resource_requirements = a_j[candidate_indices]
    
    # Sort candidates by increasing resource requirement
    sorted_idx = np.argsort(resource_requirements)
    sorted_candidates = candidate_indices[sorted_idx]
    
    # Track remaining capacity
    remaining_capacity = capacity
    
    # Greedily add elements in order of increasing resource requirement
    for i in sorted_candidates:
        # Convert resource requirement to float before comparison
        req = float(a_j[i])
        
        # If including this element doesn't violate the constraint
        if req <= remaining_capacity + 1e-9:
            tau[i] = True
            remaining_capacity -= req
    
    if verbose:
        usage = float(a_j @ tau)
        logger.info(f"KNAPSACK-CR for constraint used {usage:.4f}/{capacity:.4f} capacity")
    
    return tau

def algorithm_3_contention_resolution_rounding(
    x_star: np.ndarray,
    A: np.ndarray,
    b: np.ndarray,
    seed: Optional[int] = None,
    num_samples: int = 1,
    verbose: bool = True
) -> np.ndarray:
    """
    Implementation of Algorithm 3: Contention Resolution Rounding.
    
    Given:
      - x_star: Continuous solution in [0,1]^n to be rounded
      - A: Matrix of shape (m, n) for m constraints over n items
      - b: Vector of shape (m,) with constraint bounds
      - seed: Random seed for reproducibility
      - num_samples: Number of rounding attempts to perform (more attempts can improve quality)
      - verbose: Whether to print progress information
    
    Returns:
      - omega: A binary solution in {0,1}^n that satisfies A·omega ≤ b
    """
    if seed is not None:
        np.random.seed(seed)
    
    m, n = A.shape
    assert n == len(x_star), f"Dimension mismatch: A is {m}x{n}, but x_star has length {len(x_star)}"
    assert m == len(b), f"Dimension mismatch: A is {m}x{n}, but b has length {len(b)}"
    
    best_omega = None
    best_objective = float('-inf')
    
    # Try multiple samples to get the best result
    for sample in range(num_samples):
        if verbose and num_samples > 1:
            logger.info(f"Attempt {sample+1}/{num_samples}")
        
        # Step 1: Independent randomized rounding (Bernoulli trials)
        y = np.random.random(n) <= x_star
        
        # Step 2: Apply KNAPSACK-CR for each constraint
        tau_results = []
        
        for j in range(m):
            # Perform knapsack contention resolution for constraint j
            tau_j = knapsack_cr(A[j], b[j], y, verbose=verbose if sample == 0 else False)
            tau_results.append(tau_j)
        
        # Step 3: Compute intersection of all solutions
        omega = np.ones(n, dtype=bool)
        for tau in tau_results:
            omega = np.logical_and(omega, tau)
        
        # Convert boolean array to int array (0s and 1s)
        omega = omega.astype(int)
        
        # Calculate objective value (here we use sum as a simple measure)
        obj_value = omega.sum()
        
        # Keep track of the best solution found
        if obj_value > best_objective:
            best_objective = obj_value
            best_omega = omega.copy()
        
        if verbose and num_samples > 1:
            logger.info(f"  Solution quality: {obj_value}")
    
    # Final solution
    omega = best_omega
    
    if verbose:
        # Check final feasibility
        violations = []
        for j in range(m):
            usage = A[j] @ omega
            if usage > b[j] + 1e-9:
                violations.append((j, usage, b[j]))
        
        logger.info(f"Final solution selects {omega.sum()} elements")
        
        if violations:
            logger.warning("Solution has constraint violations:")
            for j, usage, capacity in violations:
                logger.warning(f"  Constraint {j}: {usage:.4f} > {capacity:.4f}")
        else:
            logger.info("Solution is feasible for all constraints")
    
    return omega

def round_solution_with_cr(
    cont_sol: np.ndarray,
    obj_func: Callable,
    A: np.ndarray,
    b: np.ndarray,
    num_samples: int = 10,
    verbose: bool = True,
    args: tuple = ()
) -> Tuple[np.ndarray, float, Dict[str, Any]]:
    """
    Round a continuous solution using Contention Resolution (CR).
    
    Args:
        cont_sol: Continuous solution in [0,1]^n
        obj_func: Objective function to evaluate solutions
        A: Constraint matrix
        b: Constraint bounds
        num_samples: Number of contention resolution rounding attempts
        verbose: Whether to print detailed output
        args: Additional arguments to pass to objective function
    
    Returns:
        Tuple containing:
        - Best binary solution found
        - Corresponding objective value
        - Statistics dictionary
    """
    # Statistics tracking
    stats = {
        "num_samples": num_samples,
        "runtime": 0.0,
        "obj_values": []
    }
    
    start_time = time.time()
    
    if verbose:
        logger.info(f"Starting contention resolution rounding with {num_samples} samples...")
    
    # Perform contention resolution rounding
    binary_sol = algorithm_3_contention_resolution_rounding(
        x_star=cont_sol,
        A=A,
        b=b,
        num_samples=num_samples,
        verbose=verbose
    )
    
    # Calculate objective value of rounded solution
    obj_val = obj_func(binary_sol, *args)
    stats["obj_values"].append(obj_val)
    
    # Final solution
    stats["runtime"] = time.time() - start_time
    
    if verbose:
        logger.info(f"Rounded solution objective: {obj_val:.6f}")
        logger.info(f"Selected {np.sum(binary_sol)} elements")
    
    return binary_sol, obj_val, stats

def compute_variance_info(A: np.ndarray, x_star: np.ndarray):
    """
    For each row i of A:
      - Compute expectation, variance, and ratio of std to expectation
      - Useful for analyzing randomized rounding behavior
    
    Args:
        A: Constraint matrix (each row = one constraint)
        x_star: Solution vector (entries in [0,1])

    Returns:
        List of dictionaries with statistics for each constraint
    """
    p, m = A.shape
    variance_info_list = []

    for i in range(p):
        a_i = A[i, :]
        E = np.dot(a_i, x_star)
        Var = np.sum((a_i**2) * x_star * (1 - x_star))
        
        if E > 1e-12:
            ratio = np.sqrt(Var) / E
        else:
            ratio = float('inf')
        
        logger.info(f"Constraint row {i}:")
        logger.info(f"  Expectation E[s_{i}] = {E:.4f}")
        logger.info(f"  Variance Var(s_{i})  = {Var:.4f}")
        logger.info(f"  Ratio sqrt(Var)/E    = {ratio:.4f}")
        
        variance_info_list.append({
            "E": float(E),
            "Var": float(Var),
            "ratio": float(ratio)
        })

    return variance_info_list

def compute_suboptimality_bound(
    rounded_obj: float,
    relaxed_obj: float
) -> Tuple[float, float]:
    """
    Compute suboptimality bound using the relaxed solution.
    
    For a maximization problem, we know that:
    OPT <= relaxed_obj
    
    So the relative suboptimality is at most:
    (relaxed_obj - rounded_obj) / relaxed_obj
    
    Args:
        rounded_obj: Objective value of the rounded solution
        relaxed_obj: Objective value of the relaxed solution
        
    Returns:
        Tuple of (absolute gap, relative gap)
    """
    absolute_gap = relaxed_obj - rounded_obj
    relative_gap = absolute_gap / relaxed_obj if relaxed_obj > 0 else float('inf')
    
    return absolute_gap, relative_gap

def compute_comparative_metrics(
    results: Dict[str, Dict[str, Any]],
    baseline_algorithm: str = None
) -> Dict[str, Dict[str, float]]:
    """
    Compute comparative metrics for algorithms.
    
    Args:
        results: Dictionary of algorithm results
        baseline_algorithm: Name of the algorithm to use as baseline
        
    Returns:
        Dictionary of comparative metrics
    """
    metrics = {}
    
    # If no baseline is specified, use the best algorithm by objective
    if baseline_algorithm is None or baseline_algorithm not in results:
        baseline_algorithm = max(
            results.items(),
            key=lambda x: x[1]["objective"]
        )[0]
    
    baseline_obj = results[baseline_algorithm]["objective"]
    baseline_time = results[baseline_algorithm]["runtime"]
    
    for alg_name, alg_results in results.items():
        obj = alg_results["objective"]
        time = alg_results["runtime"]
        
        # Compute metrics
        obj_ratio = obj / baseline_obj if baseline_obj != 0 else float('inf')
        time_ratio = time / baseline_time if baseline_time != 0 else float('inf')
        efficiency = obj_ratio / time_ratio if time_ratio != 0 else float('inf')
        
        metrics[alg_name] = {
            "obj_ratio": obj_ratio,
            "time_ratio": time_ratio,
            "efficiency": efficiency
        }
    
    return metrics