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
    inf_mats: List[sp.spmatrix],
    H0: sp.spmatrix,
    A: np.ndarray,
    b: np.ndarray,
    time_limit: int = 600,
    mip_gap: float = 0.1,
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
    verbose: bool = False,
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
    stats["final_constraints"] = constraint_values.tolist()
    
    if verbose:
        print(f"Greedy selection complete:")
        print(f"Final selection: {A_g}")
        print(f"Final objective value: {current_obj:.6f}")
        print(f"Runtime: {stats['runtime']:.2f} seconds")
        print(f"Objective evaluations: {stats['obj_evaluations']}")
        
        # Report constraint satisfaction
        for i in range(m):
            utilization = constraint_values[i] / b[i] * 100 if b[i] != 0 else 0
            print(f"Constraint {i}: {constraint_values[i]:.4f} / {b[i]:.4f} ({utilization:.1f}%)")
    
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

def algorithm_3_continuous_multi_knapsack(
    A: np.ndarray, 
    b: np.ndarray, 
    y: np.ndarray, 
    verbose: bool = False
) -> np.ndarray:
    """
    Multi-constraint version of Algorithm 3 (continuous contention resolution).
    
    Given:
      - A: 2D array of shape (m, n), representing m knapsack constraints
      - b: 1D array of shape (m,), the capacity bounds for each of the m constraints
      - y: Fractional solution in [0,1]^n (possibly infeasible)
    
    Returns:
      - w: A new solution in [0,1]^n that satisfies A w <= b
    
    Procedure (high-level):
      1) Sort items in descending order of y_i.
      2) Initialize leftover capacities leftover[j] = b[j] for each constraint j.
      3) For each item i in sorted order:
           - fraction_i = y[i]
           - For each constraint j in 0..m-1:
               if A[j, i] > 0:
                   fraction_i = min(fraction_i, leftover[j] / A[j, i])
           - w[i] = fraction_i
           - Update leftover[j] -= A[j, i] * fraction_i for each j
      4) By construction, A w <= b, and w[i] <= y[i].
    """
    m, n = A.shape
    if n != len(y):
        raise ValueError(f"Dimension mismatch: A is {m}x{n}, but y has length {len(y)}.")
    if m != len(b):
        raise ValueError(f"Dimension mismatch: A is {m}x{n}, but b has length {len(b)}.")
    
    # 1) Sort items by descending y_i
    sorted_indices = np.argsort(-y)
    
    # 2) Initialize leftover capacity for each of the m constraints
    leftover = b.copy()
    
    # 3) Allocate w
    w = np.zeros_like(y)
    
    num_partial = 0  # Track how many items get partially cut
    for idx in sorted_indices:
        # Start by trying to keep the full fraction y[idx]
        fraction_i = y[idx]
        
        # For each constraint j, see how much fraction we can afford
        for j in range(m):
            a_ji = A[j, idx]
            if a_ji > 1e-14:  # Only matters if item i uses resource j
                # fraction_i must not exceed leftover[j] / a_ji
                max_feasible_fraction = leftover[j] / a_ji
                if fraction_i > max_feasible_fraction:
                    fraction_i = max_feasible_fraction
        
        # fraction_i is how much we actually keep
        if fraction_i < y[idx] - 1e-14:
            num_partial += 1
        
        w[idx] = max(0.0, fraction_i)  # Ensure nonnegative rounding
        
        # Update leftover capacities
        for j in range(m):
            a_ji = A[j, idx]
            if a_ji > 1e-14:
                used = a_ji * w[idx]
                leftover[j] = max(leftover[j] - used, 0.0)
    
    if verbose:
        # Check final usage
        usage = A @ w
        infeas_count = np.sum(usage > b + 1e-9)
        print(f"[Algorithm 3 - Multi] Final usage: {usage} vs capacity {b}")
        print(f"[Algorithm 3 - Multi] #Constraints violated: {infeas_count}")
        print(f"[Algorithm 3 - Multi] #Items partially cut: {num_partial}")
    
    return w


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
    max_weight = 100
    b_weight = np.array([max_weight])
    
    # Total Cost Constraint: sum(c_i * x_i) <= max_cost
    costs = np.random.uniform(10, 20, size=n)
    A_cost = costs.reshape(1, -1)
    max_cost = 1000
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
    time_limit: Optional[float] = None, 
    verbose: bool = False
) -> Dict[str, Any]:
    """
    Runs one single 'experiment' for the pair (n, m),
    storing selection vectors and variance info for each method.
    """
    np.random.seed(seed)
    
    # -- 1) Generate problem --
    _, _, inf_mats, H0, A_ineq, b_ineq, selection_init, _, _ = generate_test_problem_from_algorithm4(
        n=n, 
        m=m, 
        cardinality=int(0.1 * n), 
        seed=seed
    )
    
    reset_min_eig_gradient_cache()
    # -- 2) Continuous Frank–Wolfe --
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
    
    continuous_results = {
        "obj": float(fw_obj),
        "time": end_fw - start_fw,
        "iterations": fw_iters,
        "selection": fw_x.tolist(),
        "variance_info": fw_variance_info  # store the list of dicts
    }
    
    # -- 3) Randomized Rounding --
    start_rr = time.time()
    rr_x, rr_obj = round_solution(
        cont_sol=fw_x,
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
    
    rounding_results = {
        "obj": float(rr_obj),
        "time": end_rr - start_rr,
        "selection": rr_x.tolist(),
        "variance_info": rr_variance_info
    }
    
    # -- 4) Gurobi Branch & Cut --
    start_bc = time.time()
    gurobi_x, gurobi_obj, gurobi_stats = branch_and_cut_gurobi(
        inf_mats=inf_mats,
        H0=H0,
        A=A_ineq,
        b=b_ineq,
        time_limit=time_limit,
        verbose=True,
        cont_solution=None
    )
    end_bc = time.time()
    
    if gurobi_x is None:
        gurobi_results = {
            "obj": float('-inf'),
            "time": end_bc - start_bc,
            "stats": {"status": "GUROBI_FAILED_OR_UNAVAILABLE"},
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
        gurobi_results = {
            "obj": float(gurobi_obj),
            "time": end_bc - start_bc,
            "stats": gurobi_stats_serial,
             "selection": gurobi_x.tolist(),
            "variance_info": gurobi_variance_info
        }
    
    # -- 5) Compile results for JSON --
    run_dict = {
        "continuous_relaxation": continuous_results,
        "randomized_rounding": rounding_results,
        "gurobi": gurobi_results
    }
    
    return run_dict

def aggregate_stats(runs_data: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Given a list of run dictionaries (each one has
       "continuous_relaxation" -> {"obj":..., "time":..., "iterations":...}
       "randomized_rounding"   -> {"obj":..., "time":...}
       "gurobi"                -> {"obj":..., "time":..., "stats":...}
    ), compute aggregated statistics: mean, std, min, max for "obj" and mean, std for "time".
    
    Returns a dict with the "stats" block you requested.
    """
    continuous_objs = []
    continuous_times = []
    continuous_iters = []
    
    rounding_objs = []
    rounding_times = []
    
    gurobi_objs = []
    gurobi_times = []
    
    # Populate
    for run_dict in runs_data:
        # Continuous
        c_rel = run_dict["continuous_relaxation"]
        continuous_objs.append(c_rel["obj"])
        continuous_times.append(c_rel["time"])
        continuous_iters.append(c_rel["iterations"])
        
        # Randomized rounding
        rr = run_dict["randomized_rounding"]
        rounding_objs.append(rr["obj"])
        rounding_times.append(rr["time"])
        
        # Gurobi
        grb = run_dict["gurobi"]
        gurobi_objs.append(grb["obj"])
        gurobi_times.append(grb["time"])
    
    def stats_1d(values):
        d = {}
        arr = [float(v) for v in values]
        d["mean"] = float(np.mean(arr))
        d["std"] = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0
        d["min"] = float(np.min(arr))
        d["max"] = float(np.max(arr))
        return d
    
    # Build stats block
    stats_block = {
        "num_runs": len(runs_data),
        
        "continuous_relaxation": {
            "obj_mean": stats_1d(continuous_objs)["mean"],
            "obj_std":  stats_1d(continuous_objs)["std"],
            "obj_min":  stats_1d(continuous_objs)["min"],
            "obj_max":  stats_1d(continuous_objs)["max"],
            "time_mean": stats_1d(continuous_times)["mean"],
            "time_std":  stats_1d(continuous_times)["std"],
            # optionally store iteration stats as well
            "iterations_mean": stats_1d(continuous_iters)["mean"],
            "iterations_std":  stats_1d(continuous_iters)["std"],
        },
        
        "randomized_rounding": {
            "obj_mean": stats_1d(rounding_objs)["mean"],
            "obj_std":  stats_1d(rounding_objs)["std"],
            "obj_min":  stats_1d(rounding_objs)["min"],
            "obj_max":  stats_1d(rounding_objs)["max"],
            "time_mean": stats_1d(rounding_times)["mean"],
            "time_std":  stats_1d(rounding_times)["std"]
        },
        
        "gurobi": {
            "obj_mean": stats_1d(gurobi_objs)["mean"],
            "obj_std":  stats_1d(gurobi_objs)["std"],
            "obj_min":  stats_1d(gurobi_objs)["min"],
            "obj_max":  stats_1d(gurobi_objs)["max"],
            "time_mean": stats_1d(gurobi_times)["mean"],
            "time_std":  stats_1d(gurobi_times)["std"]
        }
    }
    
    return stats_block



def run_experiments_for_n_m_pairs(
    n_values: List[int],
    m_values: List[int],
    num_runs: int = 10,
    time_limit: Optional[float] = None,
    verbose: bool = False
):
    """
    For each (n, m) in the Cartesian product of n_values x m_values,
    1) create a separate JSON file named f"results_n{n}_m{m}.json"
    2) do 'num_runs' runs
    3) aggregate stats
    4) write final JSON with structure:
        {
          "runs": { "run_0": {...}, "run_1": {...}, ... },
          "stats": {
              "n": n,
              "m": m,
              "num_runs": num_runs,
              "continuous_relaxation": {...},
              "randomized_rounding":   {...},
              "gurobi":                {...}
          }
        }
    5) Finally, print a summary table for all pairs.
    """
    
    summary_results = [] 
    for n in n_values:
        for m in m_values:
            runs_data = []
            
            # We might vary the seed each run, e.g., 0..4
            for run_idx in range(num_runs):
                seed = 1000 * (run_idx + 1)  # or any seed you like
                print(f"\n=== (n={n}, m={m}) - Run {run_idx} with seed={seed} ===")
                
                run_result = run_single_experiment(
                    n=n,
                    m=m,
                    seed=seed,
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
                "runs": runs_out,
                "stats": {
                    "n": n,
                    "m": m,
                    **stats_block  # merges the aggregated stats
                }
            }
            
            # Write to JSON file
            filename = f"results_n{n}_m{m}.json"
            with open(filename, "w") as f:
                json.dump(output_dict, f, indent=2)
            
            print(f"Saved results for (n={n}, m={m}) to {filename}")
            
            # For summary table, just pick out a few items (e.g. mean objective/time)
            summary_results.append({
                "n": n,
                "m": m,
                # continuous relaxation
                "FW_obj_mean": stats_block["continuous_relaxation"]["obj_mean"],
                "FW_time_mean": stats_block["continuous_relaxation"]["time_mean"],
                # randomized rounding
                "RR_obj_mean": stats_block["randomized_rounding"]["obj_mean"],
                "RR_time_mean": stats_block["randomized_rounding"]["time_mean"],
                # gurobi
                "GRB_obj_mean": stats_block["gurobi"]["obj_mean"],
                "GRB_time_mean": stats_block["gurobi"]["time_mean"],
            })
    
    # --- Print summary table for all (n,m) pairs ---
    print("\n\nFINAL SUMMARY TABLE")
    print("-------------------------------------------------------------------------------------------")
    header = (
        f"{'n':>6}  {'m':>6}  |"
        f"{'FW_obj_mean':>12}  {'FW_time_mean':>12}  |"
        f"{'RR_obj_mean':>12}  {'RR_time_mean':>12}  |"
        f"{'GRB_obj_mean':>12}  {'GRB_time_mean':>12}"
    )
    print(header)
    print("-------------------------------------------------------------------------------------------")
    for row in summary_results:
        print(
            f"{row['n']:6d}  {row['m']:6d}  |"
            f"{row['FW_obj_mean']:12.4f}  {row['FW_time_mean']:12.2f}  |"
            f"{row['RR_obj_mean']:12.4f}  {row['RR_time_mean']:12.2f}  |"
            f"{row['GRB_obj_mean']:12.4f}  {row['GRB_time_mean']:12.2f}"
        )
    print("-------------------------------------------------------------------------------------------")

def main():
    logging.basicConfig(level=logging.INFO)

    n_values = [15000, 20000, 25000, 30000]
    m_values = [10000]
    
    run_experiments_for_n_m_pairs(
        n_values=n_values,
        m_values=m_values,
        num_runs=1,
        time_limit=None,   # pass a time limit if desired, e.g. 600 for 10 minutes
        verbose=False
    )

if __name__ == "__main__":
    main()