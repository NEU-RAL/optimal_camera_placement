import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh, lobpcg
import time
import matplotlib.pyplot as plt
import logging
import os
import json
from datetime import datetime
from typing import List, Tuple, Dict, Any

# Create a NumPy-compatible JSON encoder
class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder for NumPy types"""
    def default(self, obj):
        if isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        return super(NumpyEncoder, self).default(obj)

"""
Test script for minimum eigenvalue maximization problem.

This script defines the objective function and gradient for maximizing
the minimum eigenvalue of a weighted sum of matrices, then solves the
optimization problem using various methods.
"""
from optimization_methods import (
    frank_wolfe_optimization,
    branch_and_cut_gurobi,
    greedy_algorithm_2,
    randomized_rounding,
    algorithm_3_contention_resolution_rounding,
    round_solution_with_cr,
    compute_variance_info,
    compute_suboptimality_bound
)
from graph_generation import generate_test_matrices, generate_test_problem_constraints

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("test_min_eigenvalue")

# ======================================================================
# Objective Function and Gradient Definitions
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
        min_eig_val, _ = eigsh(combined_fim.tocsc(), k=1, which='SA', tol=1e-2, ncv=40)
        return float(min_eig_val[0])
    except Exception as e:
        logger.error(f"Error computing eigenvalues: {str(e)}")
        # Return a very negative value to indicate failure
        return -1e10

def min_eigenvalue_gradient(
    x: np.ndarray, 
    inf_mats: list,
    H0: sp.spmatrix,
) -> np.ndarray:
    """
    Compute the gradient of the minimum eigenvalue objective.
    
    Args:
        x: Selection vector
        inf_mats: List of information matrices
        H0: Prior information matrix
        
    Returns:
        np.ndarray: Gradient vector
    """
    # Cache for matrix in CSC format (if needed for repeat evaluations)
    if not hasattr(min_eigenvalue_gradient, "cached_inf_mats"):
        min_eigenvalue_gradient.cached_inf_mats = [
            Hi.tocsc() if not sp.isspmatrix_csc(Hi) else Hi
            for Hi in inf_mats
        ]
        
    # Compute the combined matrix
    combined_fim = H0.copy()
    for xi, Hi in zip(x, inf_mats):
        combined_fim += xi * Hi
    
    # Compute eigenvector corresponding to minimum eigenvalue
    min_eig_val, min_eig_vec = eigsh(combined_fim.tocsc(), k=1, which='SA', tol=1e-4)
    min_eig_vec = min_eig_vec.flatten()
    
    # Compute gradient components
    grad = np.zeros(len(inf_mats))
    for i, Hi in enumerate(inf_mats):
        temp = Hi.dot(min_eig_vec)
        grad[i] = float(min_eig_vec.dot(temp))
    
    return grad

def reset_min_eig_gradient_cache():
    """
    Clears the cached attributes in min_eigenvalue_gradient
    """
    cache_attrs = [
        "cached_inf_mats",
    ]
    for attr in cache_attrs:
        if hasattr(min_eigenvalue_gradient, attr):
            delattr(min_eigenvalue_gradient, attr)

# ======================================================================
# Experiment Utilities
# ======================================================================

def run_experiments(
    n: int = 100,
    m: int = 50,
    k: int = 20,
    algorithms: List[str] = ["fw", "greedy", "rounding", "cr"],
    time_limit: int = 300,
    seed: int = 42,
    output_dir: str = None
) -> Dict[str, Any]:
    """
    Run optimization experiments with multiple algorithms.
    
    Args:
        n: Number of matrices to choose from
        m: Size of each matrix
        k: Cardinality constraint (number of matrices to select)
        algorithms: List of algorithms to run
        time_limit: Time limit in seconds for exact methods
        seed: Random seed
        output_dir: Directory to save results and plots
        
    Returns:
        Dict with experiment results
    """
    logger.info(f"Generating problem instance n={n}, m={m}, k={k}")
    
    # Generate test matrices
    inf_mats, H0, weights = generate_test_matrices(n=n, m=m, seed=seed)
    
    # Generate constraints
    A, b = generate_test_problem_constraints(n, weights, k)
    
    # Initialize results
    results = {
        "problem": {
            "n": n,
            "m": m,
            "k": k,
            "seed": seed
        },
        "algorithms": {}
    }
    
    # Initial solution (all zeros)
    selection_init = np.zeros(n)
    
    # ======================================================================
    # 1. Frank-Wolfe Algorithm
    # ======================================================================
    if "fw" in algorithms:
        logger.info("Running Frank-Wolfe algorithm...")
        start_time = time.time()
        
        reset_min_eig_gradient_cache()
        fw_solution, fw_obj, fw_iters, fw_log = frank_wolfe_optimization(
            obj_func=min_eigenvalue_objective,
            obj_grad=min_eigenvalue_gradient,
            selection_init=selection_init,
            A=A,
            b=b,
            max_iterations=1000,
            convergence_tol=1e-3,
            step_size_strategy="diminishing",
            verbose=True,
            args=(inf_mats, H0)
        )
        
        fw_time = time.time() - start_time
        
        results["algorithms"]["frank_wolfe"] = {
            "solution": fw_solution.tolist(),
            "objective": float(fw_obj),
            "iterations": fw_iters,
            "runtime": fw_time,
            "log": fw_log
        }
        
        logger.info(f"Frank-Wolfe complete: obj={fw_obj:.6f}, time={fw_time:.2f}s")
    
    # ======================================================================
    # 2. Greedy Algorithm
    # ======================================================================
    if "greedy" in algorithms:
        logger.info("Running Greedy algorithm...")
        start_time = time.time()
        
        greedy_solution, greedy_obj, greedy_stats = greedy_algorithm_2(
            obj_func=min_eigenvalue_objective,
            A=A,
            b=b,
            n=n,
            verbose=True,
            timeout=time_limit,
            args=(inf_mats, H0)
        )
        
        greedy_time = time.time() - start_time
        
        results["algorithms"]["greedy"] = {
            "solution": greedy_solution.tolist(),
            "objective": float(greedy_obj),
            "runtime": greedy_time,
            "stats": {
                "iterations": greedy_stats["iterations"],
                "obj_evaluations": greedy_stats["obj_evaluations"],
                "selected_count": greedy_stats["selected_count"]
            }
        }
        
        logger.info(f"Greedy complete: obj={greedy_obj:.6f}, time={greedy_time:.2f}s")
    
    # ======================================================================
    # 3. Randomized Rounding
    # ======================================================================
    if "rounding" in algorithms and "fw" in algorithms:
        logger.info("Running Randomized Rounding...")
        start_time = time.time()
        
        rr_solution, rr_obj, rr_stats = randomized_rounding(
            cont_sol=fw_solution,
            obj_func=min_eigenvalue_objective,
            A=A,
            b=b,
            num_samples=100,
            verbose=True,
            args=(inf_mats, H0)
        )
        
        rr_time = time.time() - start_time
        
        # Calculate suboptimality gap
        abs_gap, rel_gap = compute_suboptimality_bound(rr_obj, fw_obj)
        
        results["algorithms"]["randomized_rounding"] = {
            "solution": rr_solution.tolist(),
            "objective": float(rr_obj),
            "runtime": rr_time,
            "abs_gap": float(abs_gap),
            "rel_gap": float(rel_gap),
            "stats": {
                "num_samples": rr_stats["num_samples"],
                "num_feasible": rr_stats["num_feasible"],
                "feasibility_rate": rr_stats["feasibility_rate"]
            }
        }
        
        logger.info(f"Randomized Rounding complete: obj={rr_obj:.6f}, time={rr_time:.2f}s")
        logger.info(f"Suboptimality gap: absolute={abs_gap:.6f}, relative={rel_gap:.2%}")
    
    # ======================================================================
    # 4. Contention Resolution Rounding
    # ======================================================================
    if "cr" in algorithms and "fw" in algorithms:
        logger.info("Running Contention Resolution Rounding...")
        start_time = time.time()
        
        cr_solution, cr_obj, cr_stats = round_solution_with_cr(
            cont_sol=fw_solution,
            obj_func=min_eigenvalue_objective,
            A=A,
            b=b,
            num_samples=10,
            verbose=True,
            args=(inf_mats, H0)
        )
        
        cr_time = time.time() - start_time
        
        # Calculate suboptimality gap
        abs_gap, rel_gap = compute_suboptimality_bound(cr_obj, fw_obj)
        
        results["algorithms"]["contention_resolution"] = {
            "solution": cr_solution.tolist(),
            "objective": float(cr_obj),
            "runtime": cr_time,
            "abs_gap": float(abs_gap),
            "rel_gap": float(rel_gap),
            "stats": {
                "num_samples": cr_stats["num_samples"],
                "runtime": cr_stats["runtime"]
            }
        }
        
        logger.info(f"Contention Resolution complete: obj={cr_obj:.6f}, time={cr_time:.2f}s")
        logger.info(f"Suboptimality gap: absolute={abs_gap:.6f}, relative={rel_gap:.2%}")
    
    # ======================================================================
    # 5. Branch and Cut (if Gurobi is available)
    # ======================================================================
    if "branch_and_cut" in algorithms:
        try:
            import gurobipy
            logger.info("Running Branch and Cut...")
            start_time = time.time()
            
            bc_solution, bc_obj, bc_stats = branch_and_cut_gurobi(
                obj_func=min_eigenvalue_objective,
                obj_grad=min_eigenvalue_gradient,
                A=A,
                b=b,
                n=n,
                time_limit=time_limit,
                verbose=True,
                cont_solution=fw_solution if "fw" in algorithms else None,
                args=(inf_mats, H0)
            )
            
            bc_time = time.time() - start_time
            
            results["algorithms"]["branch_and_cut"] = {
                "solution": bc_solution.tolist() if bc_solution is not None else None,
                "objective": float(bc_obj),
                "runtime": bc_time,
                "stats": {
                    "status": bc_stats.get("status", "UNKNOWN"),
                    "mip_gap": bc_stats.get("mip_gap", None),
                    "num_nodes": bc_stats.get("num_nodes", None)
                }
            }
            
            logger.info(f"Branch and Cut complete: obj={bc_obj:.6f}, time={bc_time:.2f}s")
        except ImportError:
            logger.warning("Gurobi not available, skipping Branch and Cut")
    
    # ======================================================================
    # Compare Results
    # ======================================================================
    logger.info("\nResults Summary:")
    
    # Create a summary table
    summary = []
    for alg_name, alg_results in results["algorithms"].items():
        summary.append({
            "algorithm": alg_name,
            "objective": alg_results["objective"],
            "runtime": alg_results["runtime"]
        })
    
    # Sort by objective value (descending)
    summary.sort(key=lambda x: x["objective"], reverse=True)
    
    # Print summary
    logger.info(f"{'Algorithm':<25} {'Objective':<15} {'Runtime (s)':<15}")
    logger.info("-" * 55)
    for item in summary:
        logger.info(f"{item['algorithm']:<25} {item['objective']:<15.6f} {item['runtime']:<15.2f}")
    
    # Save results if output directory is specified
    if output_dir is not None:
        # Create a timestamp for folder name (to the minute)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        
        # Create a folder with timestamp
        result_dir = os.path.join(output_dir, f"experiment_{timestamp}")
        os.makedirs(result_dir, exist_ok=True)
        
        # Save results to JSON
        results_file = os.path.join(result_dir, "results.json")
        
        # Create a serializable copy of results (excluding sparse matrices)
        serializable_results = results.copy()
        
        # Save to JSON with NumPy encoder
        with open(results_file, 'w') as f:
            json.dump(serializable_results, f, indent=2, cls=NumpyEncoder)
        
        # Save parameter summary
        params_file = os.path.join(result_dir, "parameters.txt")
        with open(params_file, 'w') as f:
            f.write(f"Experiment Parameters:\n")
            f.write(f"n (number of matrices): {n}\n")
            f.write(f"m (matrix dimension): {m}\n")
            f.write(f"k (cardinality constraint): {k}\n")
            f.write(f"Algorithms used: {', '.join(algorithms)}\n")
            f.write(f"Seed: {seed}\n")
            f.write(f"Time limit: {time_limit} seconds\n")
            f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        # Save plots
        plot_results(results, os.path.join(result_dir, "summary_plots.png"))
        
        logger.info(f"Results saved to directory: {result_dir}")
    
    return results

def plot_results(results: Dict[str, Any], output_file: str = None):
    """
    Plot optimization results.
    
    Args:
        results: Results dictionary from run_experiments
        output_file: Optional file path to save the plot
    """
    plt.figure(figsize=(12, 10))
    
    # Plot 1: Objective values comparison
    plt.subplot(2, 2, 1)
    algs = []
    objs = []
    for alg_name, alg_results in results["algorithms"].items():
        algs.append(alg_name)
        objs.append(alg_results["objective"])
    
    plt.bar(algs, objs)
    plt.title("Objective Value Comparison")
    plt.ylabel("Minimum Eigenvalue")
    plt.xticks(rotation=45)
    
    # Plot 2: Runtime comparison
    plt.subplot(2, 2, 2)
    algs = []
    times = []
    for alg_name, alg_results in results["algorithms"].items():
        algs.append(alg_name)
        times.append(alg_results["runtime"])
    
    plt.bar(algs, times)
    plt.title("Runtime Comparison")
    plt.ylabel("Time (seconds)")
    plt.xticks(rotation=45)
    
    # Plot 3: Frank-Wolfe convergence
    if "frank_wolfe" in results["algorithms"]:
        plt.subplot(2, 2, 3)
        fw_log = results["algorithms"]["frank_wolfe"]["log"]
        plt.plot(fw_log["iter"], fw_log["obj_val"])
        plt.title("Frank-Wolfe Convergence")
        plt.xlabel("Iteration")
        plt.ylabel("Objective Value")
        plt.grid(True)
        
        # Plot 4: Frank-Wolfe duality gap
        plt.subplot(2, 2, 4)
        plt.semilogy(fw_log["iter"], fw_log["duality_gap"])
        plt.title("Frank-Wolfe Duality Gap")
        plt.xlabel("Iteration")
        plt.ylabel("Duality Gap (log scale)")
        plt.grid(True)
    
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file)
    else:
        plt.show()

# ======================================================================
# Main function
# ======================================================================

def main():
    """
    Main function to run the experiments.
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='Run min eigenvalue maximization experiments')
    parser.add_argument('--n', type=int, default=10, help='Number of matrices to choose from')
    parser.add_argument('--m', type=int, default=100, help='Size of each matrix')
    parser.add_argument('--k', type=int, default=3, help='Cardinality constraint')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--time-limit', type=int, default=300, help='Time limit in seconds')
    parser.add_argument('--algorithms', nargs='+', default=['fw', 'greedy', 'rounding', 'cr', 'branch_and_cut'],
                        choices=['fw', 'greedy', 'rounding', 'cr', 'branch_and_cut', 'all'],
                        help='Algorithms to run')
    parser.add_argument('--output', type=str, default='./results', help='Output directory for results and plots')
    
    args = parser.parse_args()
    
    # Process 'all' algorithms choice
    if 'all' in args.algorithms:
        algorithms = ['fw', 'greedy', 'rounding', 'cr', 'branch_and_cut']
    else:
        algorithms = args.algorithms
    
    # Create output directory if it doesn't exist
    if args.output and not os.path.exists(args.output):
        os.makedirs(args.output)
    
    # Run experiments
    results = run_experiments(
        n=args.n,
        m=args.m,
        k=args.k,
        algorithms=algorithms,
        time_limit=args.time_limit,
        seed=args.seed,
        output_dir=args.output
    )

if __name__ == "__main__":
    main()