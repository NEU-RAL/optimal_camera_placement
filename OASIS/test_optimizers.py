import numpy as np
import scipy.sparse as sp
import scipy.optimize as opt
from scipy.sparse.linalg import eigsh
import time
import logging
import multiprocessing
from joblib import Parallel, delayed

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("frank_wolfe_profiler")

# ======================================================================
# Objective Functions and Gradients
# ======================================================================
def min_eigenvalue_objective(
    x: np.ndarray, 
    inf_mats: list[sp.spmatrix], 
    H0: sp.spmatrix
) -> float:
    """
    Compute the minimum eigenvalue of the sum of selected matrices.
    
    Args:
        x: Selection vector
        inf_mats: list of information matrices
        H0: Prior information matrix
        
    Returns:
        float: Minimum eigenvalue
    """
    combined_fim = H0.copy()
    for xi, Hi in zip(x, inf_mats):
        combined_fim += xi * Hi

    # Add small regularization for numerical stability
    combined_fim += 1e-10 * sp.eye(combined_fim.shape[0])
    
    # Ensure combined_fim is in sparse CSC format
    if sp.issparse(combined_fim):
        combined_fim = combined_fim.tocsc()
    else:
        combined_fim = sp.csc_matrix(combined_fim)
    
    try:
        min_eig_val, _ = eigsh(combined_fim, k=1, which='SA')
        return min_eig_val[0]
    except Exception as e:
        logger.error(f"Error computing eigenvalues: {str(e)}")
        return -1e10

def compute_gradient_component(i, min_eig_vec, Hi):
    """Compute a single component of the gradient vector."""
    if sp.issparse(Hi):
        Hi = Hi.tocsc()
        result = float(min_eig_vec.T @ Hi @ min_eig_vec)
    else:
        result = float(min_eig_vec.T @ Hi @ min_eig_vec)
    return i, result

def compute_gradient_batch(indices, min_eig_vec, inf_mats_batch):
    """Compute gradient components for a batch of matrices."""
    results = []
    for idx, Hi in zip(indices, inf_mats_batch):
        if sp.issparse(Hi):
            Hi = Hi.tocsc()
            result = float(min_eig_vec.T @ Hi @ min_eig_vec)
        else:
            result = float(min_eig_vec.T @ Hi @ min_eig_vec)
        results.append((idx, result))
    return results

def vectorized_gradient_single_matrix(min_eig_vec, Hi):
    """Vectorized computation for a single matrix."""
    if sp.issparse(Hi):
        Hi = Hi.tocsc()
    return float(min_eig_vec.T @ Hi @ min_eig_vec)

def min_eigenvalue_gradient(
    x: np.ndarray, 
    inf_mats: list[sp.spmatrix], 
    H0: sp.spmatrix,
    method="serial",
    n_jobs=None,
    batch_size=None
) -> np.ndarray:
    """
    Compute the gradient of the minimum eigenvalue objective.
    
    Args:
        x: Selection vector
        inf_mats: list of information matrices
        H0: Prior information matrix
        method: Computation method ("serial", "parallel", "vectorized", "parallel_batch")
        n_jobs: Number of parallel jobs (None means use all available cores)
        batch_size: Size of batches for parallel_batch method
        
    Returns:
        np.ndarray: Gradient vector (these are directional derivatives for maximization)
    """
    combined_fim = H0.copy()
    for xi, Hi in zip(x, inf_mats):
        combined_fim += xi * Hi

    # Ensure combined_fim is in sparse CSC format
    if sp.issparse(combined_fim):
        combined_fim = combined_fim.tocsc()
    else:
        combined_fim = sp.csc_matrix(combined_fim)
    
    n = combined_fim.shape[0]
    
    try:
        k = min(5, n - 1)
        eig_vals, eig_vecs = eigsh(combined_fim, k=k, which='SA', maxiter=2000, tol=1e-6)
        min_eig_idx = np.argmin(eig_vals)
        min_eig_vec = eig_vecs[:, min_eig_idx]
        
        grad = np.zeros(len(inf_mats))
        
        if method == "vectorized":
            # Vectorized implementation (if all matrices are dense)
            all_dense = True
            for Hi in inf_mats:
                if sp.issparse(Hi):
                    all_dense = False
                    break
            
            if all_dense:
                v = min_eig_vec.reshape(-1, 1)
                for i, Hi in enumerate(inf_mats):
                    grad[i] = float(v.T @ Hi @ v)
            else:
                for i, Hi in enumerate(inf_mats):
                    if sp.issparse(Hi):
                        Hi_csc = Hi.tocsc()
                        temp = Hi_csc @ min_eig_vec
                        grad[i] = float(min_eig_vec.T @ temp)
                    else:
                        grad[i] = float(min_eig_vec.T @ Hi @ min_eig_vec)
                        
        elif method == "parallel":
            if n_jobs is None:
                n_jobs = max(1, multiprocessing.cpu_count() - 1)
            
            results = Parallel(n_jobs=n_jobs)(
                delayed(compute_gradient_component)(i, min_eig_vec, Hi) 
                for i, Hi in enumerate(inf_mats)
            )
            
            for i, val in results:
                grad[i] = val
                
        elif method == "parallel_batch":
            if n_jobs is None:
                n_jobs = max(1, multiprocessing.cpu_count() - 1)
                
            if batch_size is None:
                batch_size = max(1, len(inf_mats) // n_jobs)
                
            batches = []
            for i in range(0, len(inf_mats), batch_size):
                end = min(i + batch_size, len(inf_mats))
                indices = list(range(i, end))
                matrices = [inf_mats[j] for j in indices]
                batches.append((indices, matrices))
                
            all_results = Parallel(n_jobs=n_jobs)(
                delayed(compute_gradient_batch)(indices, min_eig_vec, matrices) 
                for indices, matrices in batches
            )
            
            for batch_results in all_results:
                for i, val in batch_results:
                    grad[i] = val
                    
        else:  # Default serial implementation
            for i, Hi in enumerate(inf_mats):
                if sp.issparse(Hi):
                    Hi_csc = Hi.tocsc()
                    temp = Hi_csc @ min_eig_vec
                    grad[i] = float(min_eig_vec.T @ temp)
                else:
                    grad[i] = float(min_eig_vec.T @ Hi @ min_eig_vec)
        
        return grad
        
    except Exception as e:
        logger.error(f"Gradient computation failed: {str(e)}. Using fallback method.")
        try:
            dense_fim = combined_fim.toarray()
            eig_vals, eig_vecs = np.linalg.eigh(dense_fim)
            min_eig_idx = np.argmin(eig_vals)
            min_eig_vec = eig_vecs[:, min_eig_idx]
            
            grad = np.zeros(len(inf_mats))
            
            if method == "vectorized":
                for i, Hi in enumerate(inf_mats):
                    if sp.issparse(Hi):
                        Hi = Hi.toarray()
                    grad[i] = float(min_eig_vec.T @ Hi @ min_eig_vec)
            elif method == "parallel" or method == "parallel_batch":
                if n_jobs is None:
                    n_jobs = max(1, multiprocessing.cpu_count() - 1)
                
                results = Parallel(n_jobs=n_jobs)(
                    delayed(compute_gradient_component)(i, min_eig_vec, Hi.toarray() if sp.issparse(Hi) else Hi) 
                    for i, Hi in enumerate(inf_mats)
                )
                
                for i, val in results:
                    grad[i] = val
            else:  # Serial
                for i, Hi in enumerate(inf_mats):
                    if sp.issparse(Hi):
                        Hi = Hi.toarray()
                    grad[i] = float(min_eig_vec.T @ Hi @ min_eig_vec)
            
            return grad
            
        except Exception as fallback_error:
            logger.error(f"Fallback gradient computation failed: {str(fallback_error)}. Returning zero gradient.")
            return np.zeros(len(inf_mats))

# ======================================================================
# LMO Solver
# ======================================================================
def solve_lmo(grad, A, b):
    """Solve the LMO using scipy.optimize.linprog."""
    n = len(grad)
    bounds = [(0, 1) for _ in range(n)]
    try:
        res = opt.linprog(c=-grad, A_ub=A, b_ub=b, bounds=bounds, method='highs')
        if res.success:
            return res.x
        else:
            logger.warning(f"LMO solver failed: {res.message}")
            return None
    except Exception as e:
        logger.error(f"LMO solver exception: {e}")
        return None

# ======================================================================
# Step-Size Strategy
# ======================================================================
def compute_step_size_diminishing(iteration):
    """Diminishing step size: 2/(iteration+2)."""
    return 2.0 / (iteration + 2)

# ======================================================================
# Frank-Wolfe Optimization with Convergence Criteria and Timing Tracking
# ======================================================================
def frank_wolfe_optimization(
    obj_func, obj_grad, selection_init, A, b, inf_mats, H0, 
    max_iterations=10, convergence_tol=1e-4,
    gradient_method="serial", n_jobs=None, batch_size=None
):
    """
    Frank-Wolfe optimization with manual timing tracking and convergence check.
    
    Args:
        obj_func: Objective function
        obj_grad: Gradient function to use
        selection_init: Initial selection vector
        A: Constraint matrix
        b: Constraint bounds
        inf_mats: List of information matrices
        H0: Prior information matrix
        max_iterations: Maximum iterations to run
        convergence_tol: Convergence tolerance (FW gap)
        gradient_method: Method for gradient computation ("serial", "parallel", "vectorized", "parallel_batch")
        n_jobs: Number of parallel jobs for gradient computation
        batch_size: Batch size for parallel_batch method
        
    Returns:
        A tuple of the final selection vector, a dictionary of timing info (including iterations),
        and the final objective value.
    """
    times = {"gradient": 0, "lmo": 0, "step_size": 0, "update": 0, "iteration_total": 0}
    iteration_count = 0
    
    selection_cur = selection_init.copy().astype(float)
    cur_obj = obj_func(selection_cur, inf_mats, H0)
    logger.info(f"Initial objective value: {cur_obj:.6f}")
    
    for iteration in range(max_iterations):
        iter_start = time.perf_counter()
        logger.info(f"Iteration {iteration+1}")
        
        t0 = time.perf_counter()
        grad = obj_grad(selection_cur, inf_mats, H0, method=gradient_method, n_jobs=n_jobs, batch_size=batch_size)
        t1 = time.perf_counter()
        times["gradient"] += (t1 - t0)
        
        t0 = time.perf_counter()
        s = solve_lmo(grad, A, b)
        t1 = time.perf_counter()
        times["lmo"] += (t1 - t0)
        
        if s is None:
            logger.error("LMO failed to return a solution.")
            break
        
        gap = np.dot(grad, s - selection_cur)
        logger.info(f"FW gap: {gap:.6f}")
        
        if 0 <= gap < convergence_tol:
            logger.info(f"Convergence achieved at iteration {iteration+1} with FW gap {gap:.6f}")
            iteration_count = iteration + 1
            break
        
        t0 = time.perf_counter()
        step_size = compute_step_size_diminishing(iteration)
        t1 = time.perf_counter()
        times["step_size"] += (t1 - t0)
        
        logger.info(f"Step size: {step_size:.6f}")
        
        t0 = time.perf_counter()
        selection_new = selection_cur + step_size * (s - selection_cur)
        selection_new = np.clip(selection_new, 0, 1)
        t1 = time.perf_counter()
        times["update"] += (t1 - t0)
        
        new_obj = obj_func(selection_new, inf_mats, H0)
        logger.info(f"Objective: {cur_obj:.6f} -> {new_obj:.6f}, Change: {new_obj - cur_obj:.6f}")
        
        selection_cur = selection_new
        cur_obj = new_obj
        
        iter_end = time.perf_counter()
        times["iteration_total"] += (iter_end - iter_start)
        
        iteration_count = iteration + 1
    
    times["iterations"] = iteration_count
    final_obj = obj_func(selection_cur, inf_mats, H0)
    return selection_cur, times, final_obj

# ======================================================================
# Example Problem Setup (using sparse matrices)
# ======================================================================
def generate_test_problem(n=1000, m=10, seed=42, dense=False):
    """Generate a test problem for benchmarking with a given seed.
       This version forces the use of sparse matrices.
    """
    np.random.seed(seed)
    inf_mats = []
    for _ in range(n):
        # Generate a sparse symmetric matrix
        A_matrix = sp.rand(m, m, density=0.5, format="csc")
        A_matrix = (A_matrix + A_matrix.T) / 2  
        A_matrix += m * sp.eye(m, format="csc")
        inf_mats.append(A_matrix)
    
    H0 = m * sp.eye(m, format="csc")
    A_ineq = np.ones((1, n))
    b_ineq = np.array([round(0.1 * n)])
    selection_init = np.zeros(n)
    return n, m, inf_mats, H0, A_ineq, b_ineq, selection_init

# ======================================================================
# Run Timing Experiment and Side-by-Side Comparison over Multiple Runs
# ======================================================================
def run_timing_experiment(num_runs=1, gradient_method="serial", n_jobs=None, batch_size=None, dense=False):
    total_times = {"gradient": 0, "lmo": 0, "step_size": 0, "update": 0, "iteration_total": 0}
    total_iterations = 0
    total_objective = 0
    
    n = 1000  # Number of matrices
    m = 1000    # Matrix dimension
    
    for run in range(num_runs):
        seed = 42 + run
        print(f"\n--- Run {run+1} with seed {seed} ---")
        n, m, inf_mats, H0, A_ineq, b_ineq, selection_init = generate_test_problem(n=n, m=m, seed=seed, dense=dense)
        _, times, final_obj = frank_wolfe_optimization(
            min_eigenvalue_objective, min_eigenvalue_gradient, selection_init, A_ineq, b_ineq, 
            inf_mats, H0, max_iterations=10, convergence_tol=1e-4,
            gradient_method=gradient_method,
            n_jobs=n_jobs,
            batch_size=batch_size
        )
        for key in total_times.keys():
            total_times[key] += times.get(key, 0)
        total_iterations += times.get("iterations", 0)
        total_objective += final_obj
    
    avg_times = {k: total_times[k] / num_runs for k in total_times}
    avg_iterations = total_iterations / num_runs
    avg_objective = total_objective / num_runs
    return avg_times, avg_iterations, avg_objective

def run_full_experiment(n_jobs=None):
    methods = [
        "serial",          # Standard serial computation
        "vectorized",      # Vectorized computation
        "parallel",        # Standard parallel computation
        "parallel_batch"   # Batch parallel computation
    ]
    
    print("\n=== RUNNING EXPERIMENTS WITH SPARSE MATRICES ===")
    sparse_results = {}
    for method in methods:
        batch_size = 50 if method == "parallel_batch" else None
        print(f"\nRunning experiment with {method} gradient computation")
        avg_times, avg_iter, avg_obj = run_timing_experiment(
            num_runs=1,
            gradient_method=method,
            n_jobs=n_jobs,
            batch_size=batch_size,
            dense=False  # Use sparse matrices
        )
        sparse_results[method] = (avg_times, avg_iter, avg_obj)
    
    print("\nSparse Matrix Results:")
    print("-" * 120)
    print("{:<15} {:>10} {:>10} {:>10} {:>15} {:>10} {:>15}".format(
        "Method", "Gradient", "LMO", "StepSize", "Total/Iter", "Iterations", "Objective"
    ))
    print("-" * 120)
    for method, (times, iter_avg, obj_avg) in sparse_results.items():
        print("{:<15} {:>10.6f} {:>10.6f} {:>10.6f} {:>15.6f} {:>10.2f} {:>15.6f}".format(
            method, times["gradient"], times["lmo"], times["step_size"], times["iteration_total"], iter_avg, obj_avg
        ))
    print("-" * 120)
    
    if "serial" in sparse_results:
        serial_time = sparse_results["serial"][0]["gradient"]
        print("\nGradient Computation Speedup (Sparse Matrices, relative to serial):")
        print("-" * 70)
        for method, (times, _, _) in sparse_results.items():
            if method != "serial":
                speedup = serial_time / times["gradient"] if times["gradient"] > 0 else 0
                print(f"{method:15s}: {times['gradient']:10.6f} seconds ({speedup:6.2f}x speedup)")
        print("-" * 70)
    
    return sparse_results

if __name__ == "__main__":
    n_jobs = max(1, multiprocessing.cpu_count() - 1)
    print(f"Running with {n_jobs} parallel jobs for gradient computation")
    run_full_experiment(n_jobs=n_jobs)
