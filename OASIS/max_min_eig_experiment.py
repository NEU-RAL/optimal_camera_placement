import os
import json
import time
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh, lobpcg, spilu
import matplotlib.pyplot as plt
import logging
import argparse
from datetime import datetime
from typing import List, Tuple, Dict, Any, NamedTuple
import traceback

# Import your modules
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
from objectives import min_eig_gradient, min_eig_objective
from fim_utils import precompute_fim_stack, combined_fim

# NumPy-compatible JSON encoder
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

def prepare_cache_from_quadruplets(quadruplets, prior_quads, matrix_size):
    """
    Prepare a cache dictionary from quadruplet lists for use with min_eig functions.
    """
    # Extract components from quadruplets
    rows = []
    cols = []
    data0 = []
    owner = []
    
    # Process regular quadruplets
    for q in quadruplets:
        rows.append(q.row)
        cols.append(q.col)
        data0.append(q.value)
        owner.append(q.matrix_idx)
    
    # Convert to arrays
    rows = np.array(rows)
    cols = np.array(cols)
    data0 = np.array(data0)
    owner = np.array(owner)
    
    # Create pattern matrix
    pattern = sp.csc_matrix((np.ones_like(data0), (rows, cols)), shape=(matrix_size, matrix_size))
    
    # Create H0 from prior quadruplets
    H0_rows = [q.row for q in prior_quads]
    H0_cols = [q.col for q in prior_quads]
    H0_data = [q.value for q in prior_quads]
    H0 = sp.csc_matrix((H0_data, (H0_rows, H0_cols)), shape=(matrix_size, matrix_size))
    
    # Create cache dictionary
    cache = {
        "pattern": pattern,
        "data0": data0,
        "owner": owner,
        "rows": rows,
        "cols": cols,
        "H0": H0,
        "shape": (matrix_size, matrix_size)
    }
    
    return cache

def run_experiments(
    n: int,
    m: int,
    k: int,
    algorithms: List[str],
    time_limit: int,
    seed: int,
    output_dir: str,
    use_quadruplets: bool = True,
    existing_fw_solution = None,
    existing_fw_obj = None
) -> Dict[str, Any]:
    """
    Run optimization experiments with multiple algorithms.
    """
    logger = logging.getLogger("experiment")
    logger.info(f"Generating problem instance n={n}, m={m}, k={k}, seed={seed}")
    
    # Generate test matrices
    quadruplets, prior_quads, weights = generate_test_matrices(n=n, m=m, seed=seed)
    cache = prepare_cache_from_quadruplets(quadruplets, prior_quads, m)
    
    # Generate constraints
    A, b = generate_test_problem_constraints(n, weights, k)
    
    # Initialize results
    results = {
        "problem": {
            "n": n,
            "m": m,
            "k": k,
            "seed": seed,
            "representation": "quadruplets" if use_quadruplets else "sparse_matrices"
        },
        "algorithms": {}
    }
    
    # Initial solution (all zeros)
    selection_init = np.zeros(n)
    
    # Wrapper functions to use the cache
    def obj_func_wrapper(x, *args):
        if not isinstance(x, np.ndarray):
            x = np.array(x)
        return min_eig_objective(x, cache)
    
    def grad_func_wrapper(x, *args):
        if not isinstance(x, np.ndarray):
            x = np.array(x)
        return min_eig_gradient(x, cache)
    
    # Use existing FW solution if provided
    fw_solution = None
    fw_obj = None
    
    # ======================================================================
    # 1. Frank-Wolfe Algorithm
    # ======================================================================
    if "fw" in algorithms:
        logger.info("Running Frank-Wolfe algorithm...")
        start_time = time.time()
        
        fw_solution, fw_obj, fw_iters, fw_log = frank_wolfe_optimization(
            obj_func=obj_func_wrapper,
            obj_grad=grad_func_wrapper,
            selection_init=selection_init,
            A=A,
            b=b,
            max_iterations=1000,
            convergence_tol=1e-3,
            verbose=True,
            args=()  # No extra args needed with the cache
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
    elif existing_fw_solution is not None and existing_fw_obj is not None:
        # Use provided FW solution
        fw_solution = existing_fw_solution
        fw_obj = existing_fw_obj
    
    # ======================================================================
    # 2. Greedy Algorithm
    # ======================================================================
    if "greedy" in algorithms:
        logger.info("Running Greedy algorithm...")
        start_time = time.time()
        
        greedy_solution, greedy_obj, greedy_stats = greedy_algorithm_2(
            obj_func=obj_func_wrapper,
            A=A,
            b=b,
            n=n,
            verbose=True,
            timeout=time_limit,
            args=()  # No extra args needed
        )
        
        greedy_time = time.time() - start_time
        
        results["algorithms"]["greedy"] = {
            "solution": greedy_solution.tolist(),
            "objective": float(greedy_obj),
            "runtime": greedy_time,
            "stats": {
                "iterations": greedy_stats["iterations"],
                "obj_evaluations": greedy_stats["obj_evaluations"],
                "selected_count": greedy_stats["selected_count"],
                "time_per_iter": greedy_stats.get("time_per_iter", [])
            }
        }
        
        logger.info(f"Greedy complete: obj={greedy_obj:.6f}, time={greedy_time:.2f}s")
    
    # ======================================================================
    # 3. Randomized Rounding
    # ======================================================================
    if "rounding" in algorithms and fw_solution is not None:
        logger.info("Running Randomized Rounding...")
        start_time = time.time()
        
        rr_solution, rr_obj, rr_stats = randomized_rounding(
            cont_sol=fw_solution,
            obj_func=obj_func_wrapper,
            A=A,
            b=b,
            num_samples=100,
            verbose=True,
            args=()  # No extra args needed
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
    if "cr" in algorithms and fw_solution is not None:
        logger.info("Running Contention Resolution Rounding...")
        start_time = time.time()
        
        cr_solution, cr_obj, cr_stats = round_solution_with_cr(
            cont_sol=fw_solution,
            obj_func=obj_func_wrapper,
            A=A,
            b=b,
            num_samples=10,
            verbose=True,
            args=()  # No extra args needed
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
                obj_func=obj_func_wrapper,
                obj_grad=grad_func_wrapper,
                A=A,
                b=b,
                n=n,
                time_limit=time_limit,
                verbose=True,
                cont_solution=fw_solution if fw_solution is not None else None,
                args=()  # No extra args needed
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
        except Exception as e:
            logger.error(f"Error running Branch and Cut: {e}")
            traceback.print_exc()

    # Generate summary table
    generate_summary_table(results, logger)
    
    # Save results to file if output directory is specified
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        results_file = os.path.join(output_dir, "results.json")
        with open(results_file, 'w') as f:
            json.dump(results, f, cls=NumpyEncoder, indent=2)
        logger.info(f"Results saved to {results_file}")
    
    return results

def generate_summary_table(results, logger):
    """Generate and log a summary table of results"""
    logger.info("=" * 80)
    logger.info("Summary of Results:")
    logger.info("=" * 80)
    
    # Create format string for the table
    header_format = "{:<20} {:<12} {:<12} {:<12} {:<12}"
    row_format = "{:<20} {:<12.6f} {:<12.2f} {:<12.2%} {:<12}"
    
    # Print header
    logger.info(header_format.format("Algorithm", "Objective", "Runtime (s)", "Gap to FW", "Selected"))
    logger.info("-" * 80)
    
    # Get Frank-Wolfe objective for comparison if available
    fw_obj = results["algorithms"].get("frank_wolfe", {}).get("objective", None)
    
    # Print each algorithm's results
    for alg_name, alg_results in results["algorithms"].items():
        obj = alg_results["objective"]
        runtime = alg_results["runtime"]
        
        # Calculate gap to FW (if FW was run)
        if fw_obj is not None and alg_name != "frank_wolfe":
            rel_gap = (fw_obj - obj) / fw_obj if fw_obj > 0 else float('inf')
        else:
            rel_gap = 0.0
        
        # Get number of selected elements if available
        if "solution" in alg_results:
            if isinstance(alg_results["solution"], list):
                selected = sum(1 for x in alg_results["solution"] if x > 0.5)
            else:
                selected = sum(1 for x in alg_results["solution"] if x > 0.5)
        else:
            selected = "N/A"
            
        logger.info(row_format.format(alg_name, obj, runtime, rel_gap, selected))
    
    logger.info("=" * 80)
    
    # Add the summary to the results dictionary
    summary_table = []
    for alg_name, alg_results in results["algorithms"].items():
        obj = alg_results["objective"]
        runtime = alg_results["runtime"]
        
        if fw_obj is not None and alg_name != "frank_wolfe":
            rel_gap = (fw_obj - obj) / fw_obj if fw_obj > 0 else float('inf')
        else:
            rel_gap = 0.0
            
        if "solution" in alg_results:
            if isinstance(alg_results["solution"], list):
                selected = sum(1 for x in alg_results["solution"] if x > 0.5)
            else:
                selected = sum(1 for x in alg_results["solution"] if x > 0.5)
        else:
            selected = None
            
        summary_table.append({
            "algorithm": alg_name,
            "objective": obj,
            "runtime": runtime,
            "gap_to_fw": rel_gap,
            "selected": selected
        })
    
    results["summary"] = summary_table

def run_experiment_series():
    """Run a series of experiments with varying parameters"""
    # Parameter ranges
    m_values = [5000, 10000, 25000, 50000]
    n_values = [100, 500, 1000, 5000, 10000, 25000, 50000]
    k_fractions = [0.1, 0.2, 0.3, 0.4, 0.5]
    seeds = [42, 13, 71, 999, 123]
    
    # All algorithms to run
    all_algorithms = ['fw', 'greedy', 'rounding', 'cr', 'branch_and_cut']
    
    # Create a timestamped base directory for this run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_output_dir = f"results/{timestamp}_experiment_series"
    os.makedirs(base_output_dir, exist_ok=True)
    
    # Set up logging for the experiment series
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(f"{base_output_dir}/experiment_log.txt"),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger("experiment_series")
    
    # Results storage
    all_results = []
    
    # Loop structure: m -> n -> k -> seed
    for m in m_values:
        for n in n_values:
            # Skip if n > m as it doesn't make sense to have more matrices than matrix size
            if n > m:
                logger.info(f"Skipping configuration m={m}, n={n} as n > m")
                continue
                
            for k_frac in k_fractions:
                # Calculate actual k value as percentage of n
                k = max(1, int(n * k_frac))
                
                for seed in seeds:
                    # Create configuration identifier
                    config_id = f"m{m}_n{n}_k{k}_seed{seed}"
                    logger.info(f"\n\n{'='*80}")
                    logger.info(f"EXPERIMENT: m={m}, n={n}, k={k} ({k_frac*100:.0f}% of n), seed={seed}")
                    logger.info(f"{'='*80}\n")
                    
                    # Create directory for this specific configuration
                    config_dir = f"{base_output_dir}/{config_id}"
                    os.makedirs(config_dir, exist_ok=True)
                    
                    try:
                        # Run FW first to determine timings for other algorithms
                        logger.info("Running Frank-Wolfe to determine timings for other algorithms...")
                        fw_result = run_experiments(
                            n=n,
                            m=m,
                            k=k,
                            algorithms=['fw'],
                            time_limit=3600,  # 1 hour max for FW
                            seed=seed,
                            output_dir=os.path.join(config_dir, "fw"),
                            use_quadruplets=True
                        )
                        
                        # Extract FW runtime to set time limits for other algorithms
                        fw_runtime = fw_result['algorithms']['frank_wolfe']['runtime']
                        logger.info(f"FW runtime: {fw_runtime:.2f} seconds")
                        
                        # Extract FW solution and objective
                        fw_solution = np.array(fw_result['algorithms']['frank_wolfe']['solution'])
                        fw_obj = fw_result['algorithms']['frank_wolfe']['objective']
                        
                        # Set time limits for greedy and branch_and_cut (1000x FW with max 1 hour)
                        greedy_time_limit = min(3600, max(300, int(fw_runtime * 1000)))
                        branch_time_limit = min(3600, max(300, int(fw_runtime * 1000)))
                        
                        logger.info(f"Setting time limits: greedy={greedy_time_limit}s, branch_and_cut={branch_time_limit}s")
                        
                        # Run greedy algorithm
                        logger.info("Running Greedy algorithm...")
                        greedy_result = run_experiments(
                            n=n,
                            m=m,
                            k=k,
                            algorithms=['greedy'],
                            time_limit=greedy_time_limit,
                            seed=seed,
                            output_dir=os.path.join(config_dir, "greedy"),
                            use_quadruplets=True,
                            existing_fw_solution=fw_solution,
                            existing_fw_obj=fw_obj
                        )
                        
                        # Run rounding algorithms
                        logger.info("Running Rounding algorithms...")
                        rounding_result = run_experiments(
                            n=n,
                            m=m,
                            k=k,
                            algorithms=['rounding', 'cr'],
                            time_limit=300,  # These are usually quick
                            seed=seed,
                            output_dir=os.path.join(config_dir, "rounding"),
                            use_quadruplets=True,
                            existing_fw_solution=fw_solution,
                            existing_fw_obj=fw_obj
                        )
                        
                        # Run branch_and_cut
                        logger.info("Running Branch and Cut...")
                        bc_result = run_experiments(
                            n=n,
                            m=m,
                            k=k,
                            algorithms=['branch_and_cut'],
                            time_limit=branch_time_limit,
                            seed=seed,
                            output_dir=os.path.join(config_dir, "branch_and_cut"),
                            use_quadruplets=True,
                            existing_fw_solution=fw_solution,
                            existing_fw_obj=fw_obj
                        )
                        
                        # Combine results from all algorithm runs
                        combined_result = {
                            'problem': fw_result['problem'],
                            'algorithms': {
                                **fw_result.get('algorithms', {}),
                                **greedy_result.get('algorithms', {}),
                                **rounding_result.get('algorithms', {}),
                                **bc_result.get('algorithms', {})
                            }
                        }
                        
                        # Generate summary
                        generate_summary_table(combined_result, logger)
                        
                        # Save combined result
                        with open(f"{config_dir}/combined_results.json", 'w') as f:
                            json.dump(combined_result, f, cls=NumpyEncoder, indent=2)
                        
                        # Add this experiment's summary to the overall results
                        experiment_summary = {
                            'm': m,
                            'n': n,
                            'k': k,
                            'k_fraction': k_frac,
                            'seed': seed
                        }
                        
                        for summary in combined_result['summary']:
                            alg = summary['algorithm']
                            experiment_summary[f"{alg}_obj"] = summary['objective']
                            experiment_summary[f"{alg}_time"] = summary['runtime']
                            experiment_summary[f"{alg}_gap"] = summary['gap_to_fw']
                            experiment_summary[f"{alg}_selected"] = summary['selected']
                        
                        all_results.append(experiment_summary)
                        
                        # Save updated consolidated results after each experiment
                        with open(f"{base_output_dir}/all_experiments.json", 'w') as f:
                            json.dump(all_results, f, cls=NumpyEncoder, indent=2)
                        
                    except Exception as e:
                        logger.error(f"ERROR in experiment m={m}, n={n}, k={k}, seed={seed}: {e}", exc_info=True)
                        # Log the error but continue with other experiments
    
    logger.info("Experiment series complete!")
    return all_results


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run optimization experiment series')
    parser.add_argument('--subset', action='store_true', help='Run a smaller subset of experiments for testing')
    args = parser.parse_args()
    
    try:
            results = run_experiment_series()
            print(f"Full experiment series completed successfully!")
    except Exception as e:
        print(f"Error in experiment series: {e}")
        traceback.print_exc()