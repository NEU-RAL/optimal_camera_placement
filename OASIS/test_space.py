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
    Compute diminishing step size: 2/(t+2) or similar.
    
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
        A: Inequality constraint matrix (converted to CSC format if sparse)
        b: Inequality constraint bounds
        solver_method: Primary solver method for linprog
        fallback_method: Fallback solver method if primary fails
    
    Returns:
        np.ndarray or None: Solution vector if optimization succeeds; otherwise, None
    """
    dim = len(grad)
    
    # Define bounds for all variables between 0 and 1
    bounds = [(0, 1) for _ in range(dim)]
    
    # Convert A to CSC format if it's not None
    if A is not None and sp.issparse(A):
        A = sp.csc_matrix(A)
    
    # Ensure b is a 1D array
    if b.ndim > 1:
        b = b.flatten()
    
    # Try primary solver method
    try:
        res = linprog(
            c=grad, 
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
    inf_mats: List[sp.spmatrix],
    H0: sp.spmatrix,
    max_iterations: int = 1000,
    min_iterations: int = 2,  # Minimum number of iterations to perform
    convergence_tol: float = 2e-2,
    step_size_strategy: StepSizeStrategy = StepSizeStrategy.DIMINISHING,
    verbose: bool = True,
    check_final_stationarity: bool = True,
    step_size_params: Dict[str, Any] = None
) -> Tuple[np.ndarray, float, int, Dict[str, List]]:
    """
    Performs Frank-Wolfe optimization to select matrices that maximize an objective function.
    
    Args:
        obj_func: Objective function to maximize
        obj_grad: Gradient function of the objective
        selection_init: Initial selection vector
        A: Inequality constraint matrix
        b: Inequality constraint bounds
        inf_mats: List of sparse information matrices
        H0: Prior information matrix
        max_iterations: Maximum number of iterations
        convergence_tol: Convergence tolerance
        step_size_strategy: Strategy for computing step size
        verbose: Whether to print detailed output
        check_final_stationarity: Whether to check stationarity of final solution
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
        "x": [],
        "obj_val": [],
        "step_size": [],
        "duality_gap": [],
        "grad_norm": []
    }
    
    # Initialize selection vector as a continuous variable (float)
    selection_cur = selection_init.copy().astype(float)
    
    # Track best objective value for convergence check
    prev_obj_val = obj_func(selection_cur, inf_mats, H0)
    
    # Main optimization loop
    for iteration in range(max_iterations):
        start_time = time.time()
        
        # Compute objective and gradient
        obj_val = obj_func(selection_cur, inf_mats, H0)
        grad = obj_grad(selection_cur, inf_mats, H0)
        
        # Check for NaN or Inf in gradient
        if not np.all(np.isfinite(grad)):
            logger.error("Non-finite values in gradient, terminating")
            break
        
        # Solve the Linear Minimization Oracle (LMO)
        s = solve_lmo(grad, A, b)
        
        if s is None:
            logger.error(f"LMO failed to find a feasible solution at iteration {iteration}.")
            break
        
        # Compute direction and duality gap
        d = s - selection_cur
        duality_gap = -grad @ d
        grad_norm = np.linalg.norm(grad)
        
        # Save iteration data
        fw_log["iter"].append(iteration)
        # fw_log["x"].append(selection_cur.tolist())
        fw_log["obj_val"].append(obj_val)
        fw_log["duality_gap"].append(duality_gap)
        fw_log["grad_norm"].append(grad_norm)
        
        # Check convergence criteria - only after minimum iterations to avoid premature convergence
        if iteration >= min_iterations:
            if duality_gap < convergence_tol * max(1.0, abs(obj_val)):
                logger.info(f"Converged at iteration {iteration}: duality gap minimized")
                break
                
            if grad_norm < convergence_tol:
                logger.info(f"Converged at iteration {iteration}: gradient norm minimized")
                break
                
            # if abs(obj_val - prev_obj_val) < convergence_tol * max(1.0, abs(obj_val)):
            #     logger.info(f"Converged at iteration {iteration}: objective value stabilized")
            #     break
        
        prev_obj_val = obj_val
        
        # Compute step size based on selected strategy
        if step_size_strategy == StepSizeStrategy.FIXED:
            alpha = step_size_params.get("fixed_alpha", 0.1)
        elif step_size_strategy == StepSizeStrategy.DIMINISHING:
            alpha = step_size_diminishing(iteration, step_size_params.get("gamma", 2.0))
        elif step_size_strategy == StepSizeStrategy.BACKTRACKING:
            alpha = step_size_backtracking(
                obj_func, obj_grad, selection_cur, d,
                step_size_params.get("c", 0.0001),
                step_size_params.get("beta", 0.5),
                step_size_params.get("max_iters", 20),
                inf_mats, H0
            )
        elif step_size_strategy == StepSizeStrategy.LIPSCHITZ:
            alpha = step_size_lipschitz(selection_cur, duality_gap, d, inf_mats, H0)
        else:
            alpha = 2.0 / (iteration + 2)  # Default to classic Frank-Wolfe
        
        fw_log["step_size"].append(alpha)
        
        # Update the selection vector
        selection_cur = selection_cur + alpha * d
        selection_cur = np.clip(selection_cur, 0, 1)  # Project back to [0,1]^n
        
        # Log iteration results
        iteration_time = time.time() - start_time
        if verbose:
            logger.info(
                f"Iteration {iteration}: obj={obj_val:.6f}, "
                f"step={alpha:.6f}, gap={duality_gap:.6e}, "
                f"grad_norm={grad_norm:.6e}, time={iteration_time:.3f}s"
            )
    
    # Make sure we've done at least a few iterations
    if iteration < 10:
        logger.warning(f"Algorithm terminated after only {iteration} iterations - check for implementation issues")
    
    # Get final objective value
    final_obj_val = obj_func(selection_cur, inf_mats, H0)
    
    # Make the stationarity check optional and handle potential failures
    if check_final_stationarity:
        try:
            # Compute the combined FIM at the final point
            combined_fim = H0.copy()
            for xi, Hi in zip(selection_cur, inf_mats):
                combined_fim += xi * Hi
            
            # Compute eigendecomposition to find multiplicity of smallest eigenvalue
            eig_vals, min_eig_vecs = eigsh(
                combined_fim, 
                k=min(3, combined_fim.shape[0]-1), 
                which='SA'
            )
            
            # Check for repeated eigenvalues (within tolerance)
            tol = 1e-8
            unique_eigs = [eig_vals[0]]
            unique_indices = [0]
            
            for i in range(1, len(eig_vals)):
                if abs(eig_vals[i] - eig_vals[0]) > tol:
                    break
                unique_eigs.append(eig_vals[i])
                unique_indices.append(i)
            
            # Get eigenvectors corresponding to repeated eigenvalues
            r = len(unique_indices)
            if r == 1:
                uniq_eig_vecs = min_eig_vecs[:, 0:1]
            else:
                uniq_eig_vecs = min_eig_vecs[:, unique_indices]
            
            # Compute M_primes for stationarity check
            M_primes = []
            for i in inf_mats:
                if r == 1:
                    M_p = -uniq_eig_vecs.T @ i @ uniq_eig_vecs
                else:
                    M_p = -uniq_eig_vecs.T @ i @ uniq_eig_vecs
                M_primes.append(M_p)
            
            # Find active constraints
            active_inds = np.isclose(np.abs(A @ selection_cur - b.flatten()), 0, atol=1e-4)
            active_constraints = A[active_inds] if np.any(active_inds) else None
            
            # Perform stationarity check
            stationarity_result = check_stationarity(r, M_primes, active_constraints, verbose)
            logger.info(f"Stationarity check result: {stationarity_result}")
        except Exception as e:
            logger.warning(f"Stationarity check failed: {str(e)}")
            logger.warning("Continuing without stationarity check")
    
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
    inf_mats: list,
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
    inf_mats: List[sp.spmatrix],
    H0: sp.spmatrix,
    A: np.ndarray,
    b: np.ndarray,
    time_limit: int = 600,
    mip_gap: float = 0.0,
    verbose: bool = False,
    cont_solution: np.ndarray = None
) -> Tuple[np.ndarray, float, Dict[str, Any]]:
    """
    Solves matrix selection problem using branch and cut with Gurobi.
    
    Args:
        inf_mats: List of information matrices to select from
        H0: Prior information matrix
        A: Inequality constraint matrix
        b: Inequality constraint bounds
        time_limit: Time limit in seconds (default: 600)
        mip_gap: MIP gap for termination (default: 0.01)
        verbose: Whether to print detailed output
        cont_solution: Continuous solution to use for initialization (optional)
    
    Returns:
        Tuple containing:
        - Selected binary vector
        - Objective value
        - Solution statistics
    """
    if not GUROBI_AVAILABLE:
        logger.error("Gurobi is not available. Cannot perform branch and cut optimization.")
        return None, float('-inf'), {"status": "Gurobi not available"}

    n = len(inf_mats)  # Number of matrices to select from
    m = H0.shape[0]    # Dimension of matrices
    
    # Create a Gurobi model
    with gp.Env(empty=True) as env:
        env.setParam('OutputFlag', 1 if verbose else 0)
        env.start()
        
        with gp.Model("Matrix_Selection", env=env) as model:
            # Set parameters
            # model.setParam('TimeLimit', time_limit)
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
                model.addConstr(expr <= b[i][0], f"constraint_{i}")

            # Add auxiliary variable for minimum eigenvalue
            t = model.addVar(lb=-GRB.INFINITY, name="t")
            
            # Set the objective to maximize t
            model.setObjective(t, GRB.MAXIMIZE)
            
            # Add initial cut from the zero solution
            if verbose:
                logger.info("Adding initial cut from zero vector")
                
            # Compute eigenvalue and eigenvector for the base matrix H0
            min_eig_val0, min_eig_vec0 = eigsh(H0.tocsc(), k=1, tol=1e-3, which='SA')
            min_eig_val0 = min_eig_val0[0]
            
            # Compute the gradient at this point
            grad0 = np.zeros(n)
            for j in range(n):
                grad0[j] = float(min_eig_vec0.T @ inf_mats[j] @ min_eig_vec0)
            
            # Add the cut: t <= min_eig_val0 + sum_j grad0[j] * x[j]
            cut_expr0 = gp.LinExpr()
            for j in range(n):
                cut_expr0.add(x[j], grad0[j])
            model.addConstr(t <= min_eig_val0 + cut_expr0, "initial_cut_zero")
            
            # If we have a continuous solution, use it to generate an upper bound
            if cont_solution is not None:
                if verbose:
                    logger.info("Using continuous solution for initial bound")
                
                # Compute objective value of continuous solution
                combined_fim_cont = H0.copy()
                for j in range(n):
                    combined_fim_cont += cont_solution[j] * inf_mats[j]
                
                min_eig_val_cont, min_eig_vec_cont = eigsh(combined_fim_cont.tocsc(), k=1, tol=1e-3, which='SA')
                cont_obj_val = min_eig_val_cont[0]
                
                # Set an upper bound for t based on this value
                t.ub = cont_obj_val
                
                # Add a cut based on the continuous solution
                grad_cont = np.zeros(n)
                for j in range(n):
                    grad_cont[j] = float(min_eig_vec_cont.T @ inf_mats[j] @ min_eig_vec_cont)
                
                # Add cut: t <= cont_obj_val + sum_j grad_cont[j] * (x[j] - cont_solution[j])
                rhs_cont = cont_obj_val - np.dot(grad_cont, cont_solution)
                cut_expr_cont = gp.LinExpr()
                for j in range(n):
                    cut_expr_cont.add(x[j], grad_cont[j])
                
                model.addConstr(t <= rhs_cont + cut_expr_cont, "initial_cut_cont")
            
            # Callback function to add cutting planes
            def eigenvalue_callback(model, where):
                if where == GRB.Callback.MIPSOL:
                    # Get current integer solution
                    x_vals = model.cbGetSolution([x[j] for j in range(n)])
                    t_val = model.cbGetSolution(t)
                    
                    # Compute the actual minimum eigenvalue at this solution
                    combined_fim = H0.copy()
                    for j in range(n):
                        if x_vals[j] > 0.5:  # if the binary variable is set to 1
                            combined_fim += inf_mats[j]
                    
                    # Calculate minimum eigenvalue and eigenvector
                    min_eig_val, min_eig_vec = eigsh(combined_fim.tocsc(), k=1, tol=1e-3,which='SA')
                    actual_min_eig = min_eig_val[0]
                    
                    # If the current t is greater than the actual eigenvalue (with some tolerance),
                    # add a cutting plane
                    if t_val > actual_min_eig + 1e-6:
                        # Compute the gradient at this point
                        grad = np.zeros(n)
                        for j in range(n):
                            grad[j] = float(min_eig_vec.T @ inf_mats[j] @ min_eig_vec)
                        
                        # Add cut: t <= actual_min_eig + sum_j grad[j] * (x[j] - x_vals[j])
                        # Simplifies to: t <= actual_min_eig - dot(grad, x_vals) + sum_j grad[j] * x[j]
                        rhs = actual_min_eig - np.dot(grad, x_vals)
                        
                        cut_expr = gp.LinExpr()
                        for j in range(n):
                            cut_expr.add(x[j], grad[j])
                        
                        model.cbLazy(t <= rhs + cut_expr)
                        
                        if verbose:
                            violation = t_val - actual_min_eig
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
                    
                    # Compute the relaxed minimum eigenvalue
                    combined_fim = H0.copy()
                    for j in range(n):
                        combined_fim += x_vals[j] * inf_mats[j]
                    
                    # Calculate minimum eigenvalue and eigenvector
                    min_eig_val, min_eig_vec = eigsh(combined_fim.tocsc(), k=1, tol=1e-3, which='SA')
                    actual_min_eig = min_eig_val[0]
                    
                    # Check if we need to add a cut
                    if t_val > actual_min_eig + 1e-6:
                        # Compute the gradient
                        grad = np.zeros(n)
                        for j in range(n):
                            grad[j] = float(min_eig_vec.T @ inf_mats[j] @ min_eig_vec)
                        
                        # Add the cut
                        rhs = actual_min_eig - np.dot(grad, x_vals)
                        
                        cut_expr = gp.LinExpr()
                        for j in range(n):
                            cut_expr.add(x[j], grad[j])
                        
                        model.cbCut(t <= rhs + cut_expr)
                        
                        if verbose:
                            violation = t_val - actual_min_eig
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
                combined_fim = H0.copy()
                for j in range(n):
                    if x_solution[j] > 0.5:
                        combined_fim += inf_mats[j]
                
                min_eig_val, _ = eigsh(combined_fim.tocsc(), k=1, tol=1e-3, which='SA', ncv=40)
                obj_val = min_eig_val[0]
                
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

# ======================================================================
# Greedy 0/1 Selection Implementation
# ======================================================================

def greedy_01_selection(
    inf_mats: List[sp.spmatrix],
    H0: sp.spmatrix,
    A: np.ndarray,
    b: np.ndarray,
    obj_func: Callable = None,
    verbose: bool = False,
    timeout: Optional[float] = None
) -> Tuple[np.ndarray, float, Dict[str, Any]]:
    """
    Perform greedy 0/1 selection of matrices to maximize the objective function.
    Simplified implementation without parallel evaluation or early termination.
    
    Args:
        inf_mats: List of information matrices to select from
        H0: Prior information matrix
        A: Inequality constraint matrix 
        b: Inequality constraint bounds
        obj_func: Objective function to maximize (if None, uses min_eigenvalue_objective)
        verbose: Whether to print detailed output
        timeout: Maximum execution time in seconds (None means no limit)
        
    Returns:
        Tuple containing:
        - Selected binary vector
        - Objective value
        - Dictionary with statistics and status information
    """
    if obj_func is None:
        obj_func = min_eigenvalue_objective
        
    n = len(inf_mats)
    start_time = time.time()
    
    # Statistics collection
    stats = {
        "iterations": 0,
        "obj_evaluations": 0,
        "constraint_checks": 0,
        "timed_out": False,
        "runtime": 0.0,
        "obj_history": []
    }
    
    # Initialize with empty selection
    current_selection = np.zeros(n, dtype=int)
    
    # Precompute constraint values for empty selection
    constraint_values = np.zeros(A.shape[0])
    
    # Compute initial objective
    current_obj = obj_func(current_selection, inf_mats, H0)
    stats["obj_evaluations"] += 1
    stats["obj_history"].append(current_obj)
    
    if verbose:
        logger.info(f"Starting greedy selection with initial objective: {current_obj:.6f}")
        if timeout is not None:
            logger.info(f"Timeout set to {timeout:.1f} seconds")
    
    # Cache for matrix combinations and their effects
    cached_objectives = {}
    cached_constraints = {}
    
    # Precompute constraint impact of adding each matrix
    constraint_impact = np.zeros((n, A.shape[0]))
    for i in range(n):
        constraint_impact[i] = A[:, i]
    
    # Set of eligible indices (matrices that can potentially be added)
    eligible_indices = set(range(n))
    
    # Main greedy selection loop
    while eligible_indices:
        # Check timeout
        current_time = time.time()
        if timeout is not None and (current_time - start_time) > timeout:
            if verbose:
                logger.warning(f"Greedy selection timed out after {current_time - start_time:.1f} seconds")
            stats["timed_out"] = True
            break
            
        stats["iterations"] += 1
        
        # Collect candidate indices to evaluate
        candidates = list(eligible_indices)
        best_obj_delta = 0
        best_idx = -1
        
        # Evaluate each candidate sequentially
        for idx in candidates:
            # Check if we've cached this evaluation
            selection_key = tuple(np.where(current_selection == 1)[0]) + (idx,)
            if selection_key in cached_objectives:
                obj_delta = cached_objectives[selection_key] - current_obj
            else:
                # Check constraints
                constraints_satisfied = True
                for j in range(A.shape[0]):
                    new_constraint_value = constraint_values[j] + constraint_impact[idx][j]
                    stats["constraint_checks"] += 1
                    if new_constraint_value > b[j][0]:
                        constraints_satisfied = False
                        break
                
                if not constraints_satisfied:
                    # This candidate violates constraints - remove from eligible set
                    eligible_indices.discard(idx)
                    continue
                
                # Set single bit without copying the array
                current_selection[idx] = 1
                
                # Evaluate objective with this matrix added
                test_obj = obj_func(current_selection, inf_mats, H0)
                stats["obj_evaluations"] += 1
                
                # Revert the change
                current_selection[idx] = 0
                
                # Cache the result
                cached_objectives[selection_key] = test_obj
                obj_delta = test_obj - current_obj
            
            if obj_delta > best_obj_delta:
                best_obj_delta = obj_delta
                best_idx = idx
        
        # Check if we found an improvement
        if best_idx >= 0 and best_obj_delta > 0:
            # Update selection
            current_selection[best_idx] = 1
            eligible_indices.discard(best_idx)
            
            # Update objective and constraint values
            current_obj += best_obj_delta
            stats["obj_history"].append(current_obj)
            
            # Update constraint values incrementally
            for j in range(A.shape[0]):
                constraint_values[j] += constraint_impact[best_idx][j]
                
                # If this constraint is now tight, we can eliminate candidates that would violate it
                if b[j][0] - constraint_values[j] < 1e-6:
                    indices_to_remove = []
                    for idx in eligible_indices:
                        if constraint_impact[idx][j] > 0:
                            indices_to_remove.append(idx)
                    
                    for idx in indices_to_remove:
                        eligible_indices.discard(idx)
            
            if verbose:
                logger.info(f"Added matrix {best_idx}, new objective: {current_obj:.6f}")
                elapsed = time.time() - start_time
                if timeout is not None:
                    logger.info(f"Time elapsed: {elapsed:.1f}s / {timeout:.1f}s ({elapsed/timeout*100:.1f}%)")
        else:
            # No improvement possible
            break
    
    # Final statistics
    stats["runtime"] = time.time() - start_time
    stats["selected_count"] = np.sum(current_selection)
    stats["final_constraints"] = constraint_values.tolist()
    
    if verbose:
        logger.info(f"Greedy selection complete. Final selection: {current_selection}")
        logger.info(f"Final objective value: {current_obj:.6f}")
        
        if stats["timed_out"]:
            logger.warning("Note: Solution is incomplete due to timeout")
        
        # Check constraints satisfaction
        for i in range(A.shape[0]):
            logger.info(f"Constraint {i}: {constraint_values[i]:.4f} / {b[i][0]:.4f}")
        
        # Report total runtime
        logger.info(f"Total runtime: {stats['runtime']:.2f} seconds")
        logger.info(f"Objective evaluations: {stats['obj_evaluations']}")
        logger.info(f"Constraint checks: {stats['constraint_checks']}")
        logger.info(f"Iterations: {stats['iterations']}")
        
        if timeout is not None:
            logger.info(f"Timeout limit: {timeout:.2f} seconds")
            logger.info(f"Used {stats['runtime']/timeout*100:.1f}% of available time")
    
    return current_selection, current_obj, stats

def greedy_algorithm_2(
    ground_set: List[int],
    f: Callable[[List[int]], float],
    A: np.ndarray,
    b: np.ndarray,
    verbose: bool = True,
    timeout: Optional[float] = None
) -> Tuple[List[int], float, Dict[str, Any]]:
    """
    Implementation of Algorithm 2 from the paper "Maximization of nonsubmodular functions 
    under multiple constraints with applications" (Ye et al., 2023).
    
    This algorithm greedily maximizes a monotone nondecreasing set function f(A) 
    subject to multiple linear constraints of the form A·x ≤ b.
    
    Args:
        ground_set: List of indices representing the ground set S
        f: Objective function to maximize (takes a list of selected elements)
        A: Constraint matrix where each row represents a constraint
        b: Constraint bounds vector
        verbose: Whether to print progress information
        timeout: Maximum execution time in seconds (None means no limit)
        
    Returns:
        Tuple containing:
        - Selected elements (subset of ground_set)
        - Objective value f(A_g)
        - Dictionary with statistics and status information
    """
    start_time = time.time()
    n = len(ground_set)  # Number of elements in ground set
    m = A.shape[0]       # Number of constraints
    
    # Statistics collection
    stats = {
        "iterations": 0,
        "obj_evaluations": 0,
        "timed_out": False,
        "runtime": 0.0,
        "obj_history": []
    }
    
    # Initialize with empty selection
    A_g = []    # Current greedy solution
    W = ground_set.copy()  # Set of elements not yet considered
    
    # Compute initial objective value
    current_obj = f(A_g)
    stats["obj_evaluations"] += 1
    stats["obj_history"].append(current_obj)
    
    # Track constraint values
    constraint_values = np.zeros(m)
    
    # Precompute constraint impact for each element
    constraint_impact = A  # Each A[i,j] represents impact of element j on constraint i
    
    if verbose:
        print(f"Starting greedy selection with initial objective: {current_obj:.6f}")
    
    # Define the feasible set based on constraints
    def is_feasible(new_element):
        """Check if adding new_element to A_g satisfies all constraints"""
        for i in range(m):
            new_constraint_value = constraint_values[i] + constraint_impact[i, new_element]
            if new_constraint_value > b[i]:
                return False
        return True
    
    # Main greedy selection loop
    while W:
        # Check timeout
        if timeout is not None and (time.time() - start_time) > timeout:
            stats["timed_out"] = True
            if verbose:
                print(f"Greedy selection timed out after {time.time() - start_time:.1f} seconds")
            break
            
        stats["iterations"] += 1
        
        # Find the best element and constraint pair
        best_ratio = -float('inf')
        best_element = None
        best_i = None
        
        for v in W:
            # Calculate marginal gain in objective function
            A_g_plus_v = A_g + [v]
            new_obj = f(A_g_plus_v)
            stats["obj_evaluations"] += 1
            delta_f = new_obj - current_obj
            
            # Skip if no improvement in objective
            if delta_f <= 0:
                continue
                
            # Find the best ratio across all constraints where delta_h_i > 0
            for i in range(m):
                delta_h_i = constraint_impact[i, v]
                
                # Skip if element has no impact on this constraint
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
        if is_feasible(best_element):
            # Add element to solution
            A_g.append(best_element)
            W.remove(best_element)
            
            # Update objective
            current_obj = f(A_g)
            stats["obj_evaluations"] += 1
            stats["obj_history"].append(current_obj)
            
            # Update constraint values
            for i in range(m):
                constraint_values[i] += constraint_impact[i, best_element]
            
            if verbose:
                print(f"Added element {best_element} (best ratio from constraint {best_i})")
                print(f"New objective: {current_obj:.6f}")
        else:
            # Element violates constraints - remove from consideration
            W.remove(best_element)
            if verbose:
                print(f"Element {best_element} violates constraints, removed from consideration")
    
    # Compute final statistics
    stats["runtime"] = time.time() - start_time
    stats["selected_count"] = len(A_g)
    stats["final_constraints"] = constraint_values.tolist()  # Convert to list for JSON
    
    if verbose:
        print(f"Greedy selection complete:")
        print(f"Final selection: {A_g}")
        print(f"Final objective value: {current_obj:.6f}")
        print(f"Runtime: {stats['runtime']:.2f} seconds")
        print(f"Objective evaluations: {stats['obj_evaluations']}")
        
        # Report constraint satisfaction - FIX HERE
        for i in range(m):
            # Convert numpy values to Python floats before formatting
            constraint_val = float(constraint_values[i])
            bound_val = float(b[i][0]) if isinstance(b[i], np.ndarray) else float(b[i])
            utilization = constraint_val / bound_val * 100 if bound_val != 0 else 0
            print(f"Constraint {i}: {constraint_val:.4f} / {bound_val:.4f} ({utilization:.1f}%)")
    
    return A_g, current_obj, stats


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

def round_solution(
    cont_sol: np.ndarray,
    inf_mats: List[sp.spmatrix],
    H0: sp.spmatrix,
    A: np.ndarray,
    b: np.ndarray,
    obj_func: Callable = min_eigenvalue_objective,
    num_samples: int = 100,
    verbose: bool = False
) -> Tuple[np.ndarray, float]:
    """
    Round a continuous solution to obtain a feasible binary solution.
    
    Args:
        cont_sol: Continuous solution vector (treated as probabilities)
        inf_mats: List of information matrices
        H0: Prior information matrix
        A: Inequality constraint matrix
        b: Inequality constraint bounds
        obj_func: Objective function to evaluate solutions
        num_samples: Number of randomized rounding samples to try
        verbose: Whether to print detailed output
    
    Returns:
        Tuple containing:
        - Best binary solution found
        - Corresponding objective value
    """
    n = len(cont_sol)
    
    # Initialize with no solution
    best_binary = None
    best_obj = float('-inf')
    
    # Count feasible solutions
    num_feasible = 0
    
    if verbose:
        logger.info(f"Starting randomized rounding with {num_samples} samples...")
        # Compute variance information for constraints
        logger.info("Variance information for constraints:")
        compute_variance_info(A, cont_sol)
    
    # Try randomized rounding using continuous solution values as probabilities
    for i in range(num_samples):
        binary_sol = np.zeros(n, dtype=int)
        for j in range(n):
            # Each variable has probability cont_sol[j] of being 1
            binary_sol[j] = 1 if np.random.random() < cont_sol[j] else 0
        
        # Check constraints
        feasible = True
        for k in range(A.shape[0]):
            if np.dot(A[k], binary_sol) > b[k][0]:
                feasible = False
                break
        
        if feasible:
            num_feasible += 1
            obj_val = obj_func(binary_sol, inf_mats, H0)
            if obj_val > best_obj:
                best_obj = obj_val
                best_binary = binary_sol.copy()
                
                if verbose and (i+1) % (num_samples // 10) == 0:
                    logger.info(f"Sample {i+1}: Found better solution with objective {obj_val:.6f}")
    
    # Report statistics if in verbose mode
    if verbose:
        feasible_percentage = (num_feasible / num_samples) * 100
        logger.info(f"Rounding complete: {num_feasible} feasible solutions found ({feasible_percentage:.2f}%)")
        if best_binary is not None:
            logger.info(f"Best objective value: {best_obj:.6f}")
    
    # If no feasible solution was found, try a fallback approach
    if best_binary is None:
        if verbose:
            logger.warning("No feasible solution found with randomized rounding. Trying fallback approach.")
        
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
                if np.dot(A[i], binary_sol) > b[i][0]:
                    feasible = False
                    break
            
            # If not feasible, revert
            if not feasible:
                binary_sol[idx] = 0
        
        # Check if this solution is valid
        obj_val = obj_func(binary_sol, inf_mats, H0)
        if obj_val > best_obj:
            best_obj = obj_val
            best_binary = binary_sol.copy()
            
            if verbose:
                logger.info(f"Fallback approach found solution with objective {obj_val:.6f}")
    
    if best_binary is None:
        # If still no solution, return an empty selection as last resort
        return np.zeros(n, dtype=int), 0.0
    
    return best_binary, best_obj

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
            print(f"Attempt {sample+1}/{num_samples}")
        
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
        
        # Calculate objective value (here we use sum as a simple measure,
        # but you might want to use your actual objective function)
        obj_value = omega.sum()
        
        # Keep track of the best solution found
        if obj_value > best_objective:
            best_objective = obj_value
            best_omega = omega.copy()
        
        if verbose and num_samples > 1:
            print(f"  Solution quality: {obj_value}")
    
    # Final solution
    omega = best_omega
    
    if verbose:
        # Check final feasibility
        violations = []
        for j in range(m):
            usage = A[j] @ omega
            if usage > b[j] + 1e-9:
                violations.append((j, usage, b[j]))
        
        print(f"Final solution selects {omega.sum()} elements")
        
        if violations:
            print("WARNING: Solution has constraint violations:")
            for j, usage, capacity in violations:
                print(f"  Constraint {j}: {usage:.4f} > {capacity:.4f}")
        else:
            print("Solution is feasible for all constraints")
    
    return omega

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
        print(f"KNAPSACK-CR for constraint used {usage:.4f}/{capacity:.4f} capacity")
    
    return tau

def round_solution_with_cr(
    cont_sol: np.ndarray,
    inf_mats: List[sp.spmatrix],
    H0: sp.spmatrix,
    A: np.ndarray,
    b: np.ndarray,
    obj_func: Callable,
    num_samples: int = 10,
    verbose: bool = True
) -> Tuple[np.ndarray, float]:
    """
    Round a continuous solution using Contention Resolution (CR).
    
    Args:
        cont_sol: Continuous solution in [0,1]^n
        inf_mats: List of information matrices
        H0: Prior information matrix
        A: Constraint matrix
        b: Constraint bounds
        obj_func: Objective function
        num_samples: Number of rounding attempts
        verbose: Whether to print progress
    
    Returns:
        Tuple of (rounded binary solution, objective value)
    """
    if verbose:
        print("Rounding solution using Contention Resolution method...")
    
    # Perform contention resolution rounding
    binary_sol = algorithm_3_contention_resolution_rounding(
        x_star=cont_sol,
        A=A,
        b=b,
        num_samples=num_samples,
        verbose=verbose
    )
    
    # Calculate objective value of rounded solution
    obj_val = obj_func(binary_sol, inf_mats, H0)
    
    if verbose:
        print(f"Rounded solution objective: {obj_val:.6f}")
        print(f"Selected {np.sum(binary_sol)} elements")
    
    return binary_sol, obj_val

def generate_test_problem_from_algorithm4(
    n: int = 100,
    m: int = 10,
    cardinality: int = 10,
    seed: int = 42,
    gamma: float = 0.01,
    r_factor: float = 1.25,
    Wmax: float = 5.0
) -> tuple:
    """
    Generate a test problem for matrix selection using a graph partitioning approach.
    
    Args:
        n: Number of matrices to choose from
        m: Size of each matrix (and also the number of vertices in the graph)
        cardinality: Maximum number of matrices to select
        seed: Random seed for reproducibility
        gamma: Not used in this implementation, kept for API compatibility
        r_factor: Multiplier for the connectivity threshold
        Wmax: Maximum edge weight
    
    Returns:
        Tuple: (n, m, inf_mats, H0, A_ineq, b_ineq, selection_init, weights, costs)
    """
    np.random.seed(seed)
    
    # Calculate r based on the number of vertices to ensure connectivity
    r = r_factor * np.sqrt(np.log(m) / (np.pi * m))
    
    logger.info(f"Generating {n} test matrices using graph partitioning approach...")
    logger.info(f"Parameters: m={m}, r={r:.4f}, Wmax={Wmax}")
    
    # Generate a connected geometric random graph
    vertices = np.random.uniform(0, 1, size=(m, 2))
    G = nx.random_geometric_graph(m, r, pos={i: vertices[i] for i in range(m)})
    
    # Make sure the graph is connected
    attempts = 0
    while not nx.is_connected(G) and attempts < 10:
        r *= 1.1
        G = nx.random_geometric_graph(m, r, pos={i: vertices[i] for i in range(m)})
        attempts += 1
    
    if not nx.is_connected(G):
        logger.warning("Could not generate a connected graph. Adding minimal edges to connect components.")
        # Add minimal edges to make the graph connected
        components = list(nx.connected_components(G))
        for i in range(1, len(components)):
            # Add an edge between a random node in component 0 and component i
            u = np.random.choice(list(components[0]))
            v = np.random.choice(list(components[i]))
            G.add_edge(u, v)
    
    # Assign random weights to edges
    for u, v in G.edges():
        G[u][v]['weight'] = np.random.uniform(1, Wmax)
    
    # Get all edges and partition them into n subsets
    all_edges = list(G.edges())
    np.random.shuffle(all_edges)
    
    # Handle case where n > number of edges
    if len(all_edges) < n:
        logger.warning(f"Number of edges ({len(all_edges)}) < number of matrices ({n}). Some matrices will be duplicated.")
        # Duplicate edges to reach n
        all_edges = all_edges * (n // len(all_edges) + 1)
        all_edges = all_edges[:n]
    
    edge_partitions = np.array_split(all_edges, n)
    
    # Generate matrices with different non-zero min eigenvalues
    inf_mats = []
    for i in range(n):
        # Create subgraph from this partition
        subgraph = nx.Graph()
        subgraph.add_nodes_from(range(m))  # Add all nodes
        
        # Add the edges from this partition
        for u, v in edge_partitions[i]:
            subgraph.add_edge(u, v, weight=G[u][v]['weight'])
        
        # Generate the Laplacian
        L = nx.laplacian_matrix(subgraph).astype(float)
        
        # Use a random offset to create different minimum eigenvalues
        # This ensures all matrices have different positive min eigenvalues
        min_eig_offset = np.random.uniform(1, 50)  # Random value between 0.05 and 1.0
        
        A = L + min_eig_offset * sp.eye(m)
        
        # Make sure it's symmetric
        A = (A + A.T) / 2
        inf_mats.append(A)
        
        if (i+1) % 10 == 0:
            logger.info(f"Generated {i+1}/{n} matrices")
    
    # Generate initial prior matrix H0 (identity matrix)
    H0 = sp.eye(m) 
    
    # Define constraints - keep these the same as original
    # Cardinality Constraint: sum(x) <= cardinality
    # A_cardinality = np.ones((1, n))
    # b_cardinality = np.array([cardinality])
    
    # Total Weight Constraint: sum(w_i * x_i) <= max_weight
    weights = np.random.uniform(1, 5, size=n)
    A_weight = weights.reshape(1, -1)
    max_weight = 10
    b_weight = np.array([max_weight])
    
    # Total Cost Constraint: sum(c_i * x_i) <= max_cost
    costs = np.random.uniform(10, 20, size=n)
    A_cost = costs.reshape(1, -1)
    max_cost = 100
    b_cost = np.array([max_cost])
    
    # Stack all constraints together
    A_ineq = np.vstack([A_weight, A_cost])
    b_ineq = np.vstack([b_weight, b_cost])
    
    # Initial selection vector (randomly initialized)
    selection_init = np.zeros(n)
    
    logger.info("Test problem generation complete.")
    
    return n, m, inf_mats, H0, A_ineq, b_ineq, selection_init, weights, costs

def run_single_experiment(
    n: int, 
    m: int, 
    seed: int, 
    algorithms: List[str] = ["fw", "rounding", "greedy", "gurobi"],
    time_limit: Optional[float] = None, 
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Runs one single 'experiment' for the pair (n, m),
    storing selection vectors and variance info for each method.
    
    Parameters:
    -----------
    n : int
        Problem dimension n
    m : int
        Problem dimension m
    seed : int
        Random seed for reproducibility
    algorithms : List[str]
        List of algorithms to run. Options include:
        - "fw": Continuous Frank-Wolfe
        - "rounding": Randomized Rounding (requires "fw")
        - "greedy": Greedy Algorithm
        - "gurobi": Gurobi Branch and Cut
    time_limit : Optional[float]
        Time limit in seconds
    verbose : bool
        Whether to print progress information
    
    Returns:
    --------
    Dict[str, Any]
        Dictionary with results for each algorithm
    """
    np.random.seed(seed)
    
    # -- 1) Generate problem --
    _, _, inf_mats, H0, A_ineq, b_ineq, selection_init, _, _ = generate_test_problem_from_algorithm4(
        n=n, 
        m=m, 
        cardinality=int(0.1 * n), 
        seed=seed
    )
    
    run_dict = {}
    fw_x = None  # Store FW solution for use by other algorithms
    
    # -- 2) Continuous Frank–Wolfe --
    if "fw" in algorithms:
        reset_min_eig_gradient_cache()
        start_fw = time.time()
        fw_x, fw_obj, fw_iters, fw_log = frank_wolfe_optimization(
            obj_func=min_eigenvalue_objective,
            obj_grad=min_eigenvalue_gradient,
            selection_init=selection_init,
            A=A_ineq,
            b=b_ineq,
            inf_mats=inf_mats,
            H0=H0,
            max_iterations=1000,
            min_iterations=2,
            convergence_tol=2e-2,
            verbose=verbose,
            check_final_stationarity=False
        )
        end_fw = time.time()
        
        # Compute & print variance info, store as JSON
        fw_variance_info = compute_variance_info(A_ineq, fw_x)
        
        run_dict["continuous_relaxation"] = {
            "obj": float(fw_obj),
            "time": end_fw - start_fw,
            "iterations": fw_iters,
            "selection": fw_x.tolist(),
            "variance_info": fw_variance_info  # store the list of dicts
        }
    
    # -- 3) Randomized Rounding --
    if "rounding" in algorithms:
        if fw_x is None and "fw" not in algorithms:
            if verbose:
                print("Warning: Randomized rounding requested but Frank-Wolfe not run.")
                print("Using initial selection as input to rounding.")
            continuous_solution = selection_init
        else:
            continuous_solution = fw_x
            
        start_rr = time.time()
        rr_x, rr_obj = round_solution(
            cont_sol=continuous_solution,
            inf_mats=inf_mats,
            H0=H0,
            A=A_ineq,
            b=b_ineq,
            obj_func=min_eigenvalue_objective,
            num_samples=100,
            verbose=verbose
        )
        end_rr = time.time()
        
        rr_variance_info = compute_variance_info(A_ineq, rr_x)
        
        run_dict["randomized_rounding"] = {
            "obj": float(rr_obj),
            "time": end_rr - start_rr,
            "selection": rr_x.tolist(),
            "variance_info": rr_variance_info
        }
    
    # -- 4) Greedy Algorithm --
    if "greedy" in algorithms:
        start_greedy = time.time()
        
        # Define objective function for greedy algorithm
        def objective_function(selected_indices):
            if not selected_indices:
                return min_eigenvalue_objective(np.zeros(n), inf_mats, H0)
            
            selection = np.zeros(n, dtype=int)
            for idx in selected_indices:
                selection[idx] = 1
            return min_eigenvalue_objective(selection, inf_mats, H0)
        
        ground_set = list(range(n))
        greedy_selection, greedy_obj, greedy_stats = greedy_algorithm_2(
            ground_set=ground_set,
            f=objective_function,
            A=A_ineq,
            b=b_ineq,
            verbose=verbose,
            timeout=time_limit
        )
        
        # Convert greedy selection to binary vector
        greedy_x = np.zeros(n, dtype=int)
        for idx in greedy_selection:
            greedy_x[idx] = 1
        
        end_greedy = time.time()
        
        greedy_variance_info = compute_variance_info(A_ineq, greedy_x)
        
        run_dict["greedy"] = {
            "obj": float(greedy_obj),
            "time": end_greedy - start_greedy,
            "iterations": greedy_stats["iterations"],
            "obj_evaluations": greedy_stats["obj_evaluations"],
            "selection": greedy_x.tolist(),
            "variance_info": greedy_variance_info
        }
    
    # -- 5) Gurobi Branch & Cut --
    if "gurobi" in algorithms:
        try:
            start_bc = time.time()
            gurobi_x, gurobi_obj, gurobi_stats = branch_and_cut_gurobi(
                inf_mats=inf_mats,
                H0=H0,
                A=A_ineq,
                b=b_ineq,
                time_limit=time_limit,
                verbose=verbose,
                cont_solution=fw_x if fw_x is not None else None
            )
            end_bc = time.time()
            
            if gurobi_x is None:
                run_dict["gurobi"] = {
                    "obj": float('-inf'),
                    "time": end_bc - start_bc,
                    "stats": {"status": "GUROBI_FAILED"},
                    "selection": [],
                    "variance_info": []
                }
            else:
                gurobi_variance_info = compute_variance_info(A_ineq, gurobi_x)
                
                # Convert stats
                gurobi_stats_serial = {
                    k: float(v) if isinstance(v, (int, float, np.float64)) else v
                    for k, v in gurobi_stats.items()
                }
                run_dict["gurobi"] = {
                    "obj": float(gurobi_obj),
                    "time": end_bc - start_bc,
                    "stats": gurobi_stats_serial,
                    "selection": gurobi_x.tolist(),
                    "variance_info": gurobi_variance_info
                }
        except Exception as e:
            if verbose:
                print(f"Error running Gurobi: {str(e)}")
            run_dict["gurobi"] = {
                "obj": float('-inf'),
                "time": 0.0,
                "stats": {"status": f"ERROR: {str(e)}"},
                "selection": [],
                "variance_info": []
            }
    
    return run_dict

def aggregate_stats(runs_data: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Given a list of run dictionaries, compute aggregated statistics for each algorithm.
    
    Parameters:
    -----------
    runs_data : List[Dict[str, Any]]
        List of dictionaries with results from each run
    
    Returns:
    --------
    Dict[str, Any]
        Aggregated statistics
    """
    # Check which algorithms were run
    available_algorithms = set()
    for run_dict in runs_data:
        available_algorithms.update(run_dict.keys())
    
    # Map of algorithm keys to their display names in stats
    algorithm_keys = {
        "continuous_relaxation": "continuous_relaxation",
        "randomized_rounding": "randomized_rounding",
        "greedy": "greedy",
        "gurobi": "gurobi"
    }
    
    stats_block = {
        "num_runs": len(runs_data),
    }
    
    # For each available algorithm, compute statistics
    for algo_key in available_algorithms:
        # Skip if not a recognized algorithm
        if algo_key not in algorithm_keys:
            continue
            
        # Initialize collectors for metrics
        objs = []
        times = []
        iterations = []
        
        # Collect data across all runs
        for run_dict in runs_data:
            if algo_key in run_dict:
                algo_data = run_dict[algo_key]
                objs.append(algo_data["obj"])
                times.append(algo_data["time"])
                if "iterations" in algo_data:
                    iterations.append(algo_data["iterations"])
        
        # Skip if no data (algorithm wasn't run in any iteration)
        if not objs:
            continue
            
        def stats_1d(values):
            d = {}
            arr = [float(v) for v in values]
            d["mean"] = float(np.mean(arr))
            d["std"] = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0
            d["min"] = float(np.min(arr))
            d["max"] = float(np.max(arr))
            return d
        
        # Calculate statistics
        algo_stats = {
            "obj_mean": stats_1d(objs)["mean"],
            "obj_std":  stats_1d(objs)["std"],
            "obj_min":  stats_1d(objs)["min"],
            "obj_max":  stats_1d(objs)["max"],
            "time_mean": stats_1d(times)["mean"],
            "time_std":  stats_1d(times)["std"],
        }
        
        # Add iteration stats if available
        if iterations:
            algo_stats.update({
                "iterations_mean": stats_1d(iterations)["mean"],
                "iterations_std":  stats_1d(iterations)["std"],
            })
        
        # Store statistics
        stats_block[algorithm_keys[algo_key]] = algo_stats
    
    return stats_block

def run_single_experiment(
    n: int, 
    m: int, 
    seed: int, 
    algorithms: List[str] = ["fw", "rounding", "cr", "greedy", "gurobi"],
    time_limit: Optional[float] = None, 
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Runs one single 'experiment' for the pair (n, m),
    storing selection vectors and variance info for each method.
    
    Parameters:
    -----------
    n : int
        Problem dimension n
    m : int
        Problem dimension m
    seed : int
        Random seed for reproducibility
    algorithms : List[str]
        List of algorithms to run. Options include:
        - "fw": Continuous Frank-Wolfe
        - "rounding": Randomized Rounding (requires "fw")
        - "cr": Contention Resolution Rounding (requires "fw")
        - "greedy": Greedy Algorithm
        - "gurobi": Gurobi Branch and Cut
    time_limit : Optional[float]
        Time limit in seconds
    verbose : bool
        Whether to print progress information
    
    Returns:
    --------
    Dict[str, Any]
        Dictionary with results for each algorithm
    """
    np.random.seed(seed)
    
    # -- 1) Generate problem --
    _, _, inf_mats, H0, A_ineq, b_ineq, selection_init, _, _ = generate_test_problem_from_algorithm4(
        n=n, 
        m=m, 
        cardinality=int(0.1 * n), 
        seed=seed
    )
    
    run_dict = {}
    fw_x = None  # Store FW solution for use by other algorithms
    
    # -- 2) Continuous Frank–Wolfe --
    if "fw" in algorithms:
        reset_min_eig_gradient_cache()
        start_fw = time.time()
        fw_x, fw_obj, fw_iters, fw_log = frank_wolfe_optimization(
            obj_func=min_eigenvalue_objective,
            obj_grad=min_eigenvalue_gradient,
            selection_init=selection_init,
            A=A_ineq,
            b=b_ineq,
            inf_mats=inf_mats,
            H0=H0,
            max_iterations=1000,
            min_iterations=2,
            convergence_tol=2e-2,
            verbose=verbose,
            check_final_stationarity=False
        )
        end_fw = time.time()
        
        # Compute & print variance info, store as JSON
        fw_variance_info = compute_variance_info(A_ineq, fw_x)
        
        run_dict["continuous_relaxation"] = {
            "obj": float(fw_obj),
            "time": end_fw - start_fw,
            "iterations": fw_iters,
            "selection": fw_x.tolist(),
            "variance_info": fw_variance_info  # store the list of dicts
        }
    
    # -- 3) Randomized Rounding --
    if "rounding" in algorithms:
        if fw_x is None and "fw" not in algorithms:
            if verbose:
                print("Warning: Randomized rounding requested but Frank-Wolfe not run.")
                print("Using initial selection as input to rounding.")
            continuous_solution = selection_init
        else:
            continuous_solution = fw_x
            
        start_rr = time.time()
        rr_x, rr_obj = round_solution(
            cont_sol=continuous_solution,
            inf_mats=inf_mats,
            H0=H0,
            A=A_ineq,
            b=b_ineq,
            obj_func=min_eigenvalue_objective,
            num_samples=100,
            verbose=verbose
        )
        end_rr = time.time()
        
        rr_variance_info = compute_variance_info(A_ineq, rr_x)
        
        run_dict["randomized_rounding"] = {
            "obj": float(rr_obj),
            "time": end_rr - start_rr,
            "selection": rr_x.tolist(),
            "variance_info": rr_variance_info
        }
        
    # -- 3b) Contention Resolution Rounding --
    if "cr" in algorithms:
        if fw_x is None and "fw" not in algorithms:
            if verbose:
                print("Warning: Contention resolution requested but Frank-Wolfe not run.")
                print("Using initial selection as input to contention resolution.")
            continuous_solution = selection_init
        else:
            continuous_solution = fw_x
            
        start_cr = time.time()
        cr_x, cr_obj = round_solution_with_cr(
            cont_sol=continuous_solution,
            inf_mats=inf_mats,
            H0=H0,
            A=A_ineq,
            b=b_ineq,
            obj_func=min_eigenvalue_objective,
            num_samples=10,  # Number of CR rounding attempts
            verbose=verbose
        )
        end_cr = time.time()
        
        cr_variance_info = compute_variance_info(A_ineq, cr_x)
        
        run_dict["contention_resolution"] = {
            "obj": float(cr_obj),
            "time": end_cr - start_cr,
            "selection": cr_x.tolist(),
            "variance_info": cr_variance_info
        }
    
    # -- 4) Greedy Algorithm --
    if "greedy" in algorithms:
        start_greedy = time.time()
        
        # Define objective function for greedy algorithm
        def objective_function(selected_indices):
            if not selected_indices:
                return min_eigenvalue_objective(np.zeros(n), inf_mats, H0)
            
            selection = np.zeros(n, dtype=int)
            for idx in selected_indices:
                selection[idx] = 1
            return min_eigenvalue_objective(selection, inf_mats, H0)
        
        ground_set = list(range(n))
        greedy_selection, greedy_obj, greedy_stats = greedy_algorithm_2(
            ground_set=ground_set,
            f=objective_function,
            A=A_ineq,
            b=b_ineq,
            verbose=verbose,
            timeout=time_limit
        )
        
        # Convert greedy selection to binary vector
        greedy_x = np.zeros(n, dtype=int)
        for idx in greedy_selection:
            greedy_x[idx] = 1
        
        end_greedy = time.time()
        
        greedy_variance_info = compute_variance_info(A_ineq, greedy_x)
        
        run_dict["greedy"] = {
            "obj": float(greedy_obj),
            "time": end_greedy - start_greedy,
            "iterations": greedy_stats["iterations"],
            "obj_evaluations": greedy_stats["obj_evaluations"],
            "selection": greedy_x.tolist(),
            "variance_info": greedy_variance_info
        }
    
    # -- 5) Gurobi Branch & Cut --
    if "gurobi" in algorithms:
        try:
            start_bc = time.time()
            gurobi_x, gurobi_obj, gurobi_stats = branch_and_cut_gurobi(
                inf_mats=inf_mats,
                H0=H0,
                A=A_ineq,
                b=b_ineq,
                time_limit=3600,
                verbose=verbose,
                cont_solution=fw_x if fw_x is not None else None
            )
            end_bc = time.time()
            
            if gurobi_x is None:
                run_dict["gurobi"] = {
                    "obj": float('-inf'),
                    "time": end_bc - start_bc,
                    "stats": {"status": "GUROBI_FAILED"},
                    "selection": [],
                    "variance_info": []
                }
            else:
                gurobi_variance_info = compute_variance_info(A_ineq, gurobi_x)
                
                # Convert stats
                gurobi_stats_serial = {
                    k: float(v) if isinstance(v, (int, float, np.float64)) else v
                    for k, v in gurobi_stats.items()
                }
                run_dict["gurobi"] = {
                    "obj": float(gurobi_obj),
                    "time": end_bc - start_bc,
                    "stats": gurobi_stats_serial,
                    "selection": gurobi_x.tolist(),
                    "variance_info": gurobi_variance_info
                }
        except Exception as e:
            if verbose:
                print(f"Error running Gurobi: {str(e)}")
            run_dict["gurobi"] = {
                "obj": float('-inf'),
                "time": 0.0,
                "stats": {"status": f"ERROR: {str(e)}"},
                "selection": [],
                "variance_info": []
            }
    
    return run_dict

def aggregate_stats(runs_data: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Given a list of run dictionaries, compute aggregated statistics for each algorithm.
    
    Parameters:
    -----------
    runs_data : List[Dict[str, Any]]
        List of dictionaries with results from each run
    
    Returns:
    --------
    Dict[str, Any]
        Aggregated statistics
    """
    # Check which algorithms were run
    available_algorithms = set()
    for run_dict in runs_data:
        available_algorithms.update(run_dict.keys())
    
    # Map of algorithm keys to their display names in stats
    algorithm_keys = {
        "continuous_relaxation": "continuous_relaxation",
        "randomized_rounding": "randomized_rounding",
        "contention_resolution": "contention_resolution",
        "greedy": "greedy",
        "gurobi": "gurobi"
    }
    
    stats_block = {
        "num_runs": len(runs_data),
    }
    
    # For each available algorithm, compute statistics
    for algo_key in available_algorithms:
        # Skip if not a recognized algorithm
        if algo_key not in algorithm_keys:
            continue
            
        # Initialize collectors for metrics
        objs = []
        times = []
        iterations = []
        
        # Collect data across all runs
        for run_dict in runs_data:
            if algo_key in run_dict:
                algo_data = run_dict[algo_key]
                objs.append(algo_data["obj"])
                times.append(algo_data["time"])
                if "iterations" in algo_data:
                    iterations.append(algo_data["iterations"])
        
        # Skip if no data (algorithm wasn't run in any iteration)
        if not objs:
            continue
            
        def stats_1d(values):
            d = {}
            arr = [float(v) for v in values]
            d["mean"] = float(np.mean(arr))
            d["std"] = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0
            d["min"] = float(np.min(arr))
            d["max"] = float(np.max(arr))
            return d
        
        # Calculate statistics
        algo_stats = {
            "obj_mean": stats_1d(objs)["mean"],
            "obj_std":  stats_1d(objs)["std"],
            "obj_min":  stats_1d(objs)["min"],
            "obj_max":  stats_1d(objs)["max"],
            "time_mean": stats_1d(times)["mean"],
            "time_std":  stats_1d(times)["std"],
        }
        
        # Add iteration stats if available
        if iterations:
            algo_stats.update({
                "iterations_mean": stats_1d(iterations)["mean"],
                "iterations_std":  stats_1d(iterations)["std"],
            })
        
        # Store statistics
        stats_block[algorithm_keys[algo_key]] = algo_stats
    
    return stats_block

def run_experiments_for_n_m_pairs(
    n_values: List[int],
    m_values: List[int],
    algorithms: List[str] = ["fw", "rounding", "cr", "greedy", "gurobi"],
    num_runs: int = 10,
    time_limit: Optional[float] = None,
    verbose: bool = True,
    output_format: str = "json",
    output_dir: str = "./"
):
    """
    For each (n, m) in the Cartesian product of n_values x m_values,
    run experiments with selected algorithms.
    
    Parameters:
    -----------
    n_values : List[int]
        List of n values to test
    m_values : List[int]
        List of m values to test
    algorithms : List[str]
        List of algorithms to run. Options include:
        - "fw": Continuous Frank-Wolfe
        - "rounding": Randomized Rounding
        - "cr": Contention Resolution Rounding
        - "greedy": Greedy Algorithm
        - "gurobi": Gurobi Branch and Cut
    num_runs : int
        Number of runs for each (n, m) pair
    time_limit : Optional[float]
        Time limit for algorithms that support it
    verbose : bool
        Whether to print progress information
    output_format : str
        Output format. Options:
        - "json": JSON files (one per n,m pair)
        - "pickle": Pickle files (one per n,m pair)
        - "hdf5": HDF5 file (all results in one file)
    output_dir : str
        Directory to save output files
    """
    import os
    import json
    import pickle
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # For HDF5 format, we need to import h5py
    h5_file = None
    if output_format == "hdf5":
        try:
            import h5py
            h5_file = h5py.File(os.path.join(output_dir, f"results_{time.strftime('%Y%m%d_%H%M%S')}.h5"), "w")
        except ImportError:
            print("h5py not available. Falling back to JSON format.")
            output_format = "json"
    
    # Algorithm display names for summary table
    algo_display = {
        "fw": "FW",
        "rounding": "RR",
        "cr": "CR",
        "greedy": "GRD", 
        "gurobi": "GRB"
    }
    
    # Algorithm keys in result dictionary
    algo_keys = {
        "fw": "continuous_relaxation",
        "rounding": "randomized_rounding",
        "cr": "contention_resolution",
        "greedy": "greedy",
        "gurobi": "gurobi"
    }
    
    summary_results = [] 
    for n in n_values:
        for m in m_values:
            if verbose:
                print(f"\n=== Running experiments for n={n}, m={m} ===")
                
            runs_data = []
            
            # Run multiple experiments with different seeds
            for run_idx in range(num_runs):
                seed = 1000 * (run_idx + 1)  # or any seed you like
                if verbose:
                    print(f"\n=== (n={n}, m={m}) - Run {run_idx} with seed={seed} ===")
                
                run_result = run_single_experiment(
                    n=n,
                    m=m,
                    seed=seed,
                    algorithms=algorithms,
                    time_limit=time_limit,
                    verbose=verbose
                )
                runs_data.append(run_result)
            
            # Aggregate stats
            stats_block = aggregate_stats(runs_data)
            
            # Build final dictionary for JSON
            runs_out = {}
            for i, rd in enumerate(runs_data):
                runs_out[f"run_{i}"] = rd
            
            output_dict = {
                "params": {
                    "n": n,
                    "m": m,
                    "algorithms": algorithms,
                    "num_runs": num_runs,
                    "time_limit": time_limit
                },
                "runs": runs_out,
                "stats": {
                    "n": n,
                    "m": m,
                    **stats_block  # merges the aggregated stats
                }
            }
            
            # Save results based on the selected format
            if output_format == "json":
                filename = os.path.join(output_dir, f"results_n{n}_m{m}.json")
                with open(filename, "w") as f:
                    json.dump(output_dict, f, indent=2)
                if verbose:
                    print(f"Saved results to {filename}")
            
            elif output_format == "pickle":
                filename = os.path.join(output_dir, f"results_n{n}_m{m}.pkl")
                with open(filename, "wb") as f:
                    pickle.dump(output_dict, f)
                if verbose:
                    print(f"Saved results to {filename}")
            
            elif output_format == "hdf5" and h5_file is not None:
                # Create a group for this (n,m) pair
                group_name = f"n{n}_m{m}"
                group = h5_file.create_group(group_name)
                
                # Store parameters
                param_group = group.create_group("params")
                param_group.attrs["n"] = n
                param_group.attrs["m"] = m
                param_group.attrs["num_runs"] = num_runs
                if time_limit is not None:
                    param_group.attrs["time_limit"] = time_limit
                
                # Store algorithms as a dataset
                param_group.create_dataset("algorithms", data=np.array(algorithms, dtype="S10"))
                
                # Store runs
                runs_group = group.create_group("runs")
                for i, run_data in enumerate(runs_data):
                    run_group = runs_group.create_group(f"run_{i}")
                    
                    # Store data for each algorithm that was run
                    for algo_name, algo_results in run_data.items():
                        algo_group = run_group.create_group(algo_name)
                        
                        # Store scalar values directly as attributes
                        for key, value in algo_results.items():
                            if key not in ["selection", "variance_info", "stats"]:
                                algo_group.attrs[key] = value
                        
                        # Store selection vector as dataset
                        if "selection" in algo_results and algo_results["selection"]:
                            algo_group.create_dataset("selection", data=np.array(algo_results["selection"]))
                        
                        # Store variance info as a group
                        if "variance_info" in algo_results and algo_results["variance_info"]:
                            var_group = algo_group.create_group("variance_info")
                            for i, var_item in enumerate(algo_results["variance_info"]):
                                item_group = var_group.create_group(f"item_{i}")
                                for k, v in var_item.items():
                                    item_group.attrs[k] = v
                        
                        # Store stats as attributes in a subgroup
                        if "stats" in algo_results and algo_results["stats"]:
                            stats_group = algo_group.create_group("stats")
                            for k, v in algo_results["stats"].items():
                                if isinstance(v, (int, float, str, bool)) or v is None:
                                    stats_group.attrs[k] = v if v is not None else "None"
                
                # Store aggregated statistics
                stats_group = group.create_group("stats")
                stats_group.attrs["n"] = n
                stats_group.attrs["m"] = m
                stats_group.attrs["num_runs"] = stats_block["num_runs"]
                
                for algo_name, algo_stats in stats_block.items():
                    if algo_name != "num_runs":
                        algo_stats_group = stats_group.create_group(algo_name)
                        for k, v in algo_stats.items():
                            algo_stats_group.attrs[k] = v
                
                if verbose:
                    print(f"Saved results for n={n}, m={m} to HDF5 file")
            
            # Build summary entry for this (n,m) pair
            summary_entry = {
                "n": n,
                "m": m,
            }
            
            # Add metrics for each algorithm that was run
            for algo in algorithms:
                algo_key = algo_keys.get(algo)
                if algo_key in stats_block:
                    algo_disp = algo_display.get(algo, algo[:3].upper())
                    summary_entry[f"{algo_disp}_obj"] = stats_block[algo_key]["obj_mean"]
                    summary_entry[f"{algo_disp}_time"] = stats_block[algo_key]["time_mean"]
            
            summary_results.append(summary_entry)
    
    # Close HDF5 file if it was created
    if h5_file is not None:
        h5_file.close()
        
    # --- Print summary table for all (n,m) pairs ---
    print("\n\nFINAL SUMMARY TABLE")
    print("-" * 160)
    
    # Build header based on which algorithms were run
    header = f"{'n':>6}  {'m':>6}"
    for algo in algorithms:
        algo_disp = algo_display.get(algo, algo[:3].upper())
        header += f" |  {algo_disp+'_obj':>10}  {algo_disp+'_time':>10}"
    
    print(header)
    print("-" * 160)
    
    # Print each row
    for row in summary_results:
        row_str = f"{row['n']:6d}  {row['m']:6d}"
        
        for algo in algorithms:
            algo_disp = algo_display.get(algo, algo[:3].upper())
            obj_key = f"{algo_disp}_obj"
            time_key = f"{algo_disp}_time"
            
            if obj_key in row and time_key in row:
                row_str += f" |  {row[obj_key]:10.4f}  {row[time_key]:10.2f}"
            else:
                row_str += f" |  {'N/A':>10}  {'N/A':>10}"
        
        print(row_str)
    
    print("-" * 80)

def main():
    import argparse
    import sys
    
    parser = argparse.ArgumentParser(description='Run optimization experiments with multiple algorithms.')
    
    # Problem parameters
    parser.add_argument('--n', type=int, nargs='+', default=[100], 
                        help='List of n values (problem dimension)')
    parser.add_argument('--m', type=int, nargs='+', default=[100], 
                        help='List of m values (constraint dimension)')
    
    # Algorithm selection
    parser.add_argument('--algorithms', type=str, nargs='+', default=['fw', 'rounding', 'cr', 'greedy'], 
                        choices=['fw', 'rounding', 'cr', 'greedy', 'gurobi', 'all'],
                        help='Algorithms to run (use "all" for all algorithms)')
    
    # Experiment parameters
    parser.add_argument('--runs', type=int, default=1, 
                        help='Number of runs per (n,m) pair')
    parser.add_argument('--time-limit', type=float, default=None, 
                        help='Time limit for algorithms in seconds')
    parser.add_argument('--verbose', action='store_true', 
                        help='Enable verbose output')
    
    # Output options
    parser.add_argument('--output-format', type=str, default='json', 
                        choices=['json', 'pickle', 'hdf5'],
                        help='Output file format')
    parser.add_argument('--output-dir', type=str, default='./', 
                        help='Directory to save results')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Process 'all' algorithms choice
    if 'all' in args.algorithms:
        algorithms = ['fw', 'rounding', 'cr', 'greedy', 'gurobi']
    else:
        algorithms = args.algorithms
    
    # Check for dependency between rounding algorithms and fw
    if ('rounding' in algorithms or 'cr' in algorithms) and 'fw' not in algorithms:
        print("Warning: Rounding methods (randomized or contention resolution) typically need Frank-Wolfe results.")
        print("Would you like to add Frank-Wolfe to the algorithms? (y/n)")
        response = input().lower()
        if response == 'y' or response == 'yes':
            algorithms = ['fw'] + algorithms
        else:
            print("Continuing with rounding without Frank-Wolfe.")
            print("Initial selection will be used for rounding.")
    
    # Run experiments
    run_experiments_for_n_m_pairs(
        n_values=args.n,
        m_values=args.m,
        algorithms=algorithms,
        num_runs=args.runs,
        time_limit=args.time_limit,
        verbose=args.verbose,
        output_format=args.output_format,
        output_dir=args.output_dir
    )

if __name__ == "__main__":
    main()