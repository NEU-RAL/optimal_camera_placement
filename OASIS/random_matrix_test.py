#!/usr/bin/env python
"""
Enhanced Random Matrix Experiment with Multiple Runs per Configuration,
Distribution Analysis, and Performance Profiles.

This script generates test problems using random positive-definite matrices,
builds the optimization problem (with constraints), and runs three methods:
1. Continuous relaxation via Frank-Wolfe 
2. Greedy 0/1 selection (with timeout)
3. Gurobi Branch-and-Cut (optimal solution)

For each (n,m) configuration, it runs multiple instances with different seeds
to capture the distribution of results. It produces comprehensive visualizations
including bar charts, violin plots, and performance profiles.
"""

import logging
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import time
import scipy.sparse as sp
from typing import List, Tuple, Dict, Optional, Any
import pandas as pd
from collections import defaultdict

# Import your core functions from test_space
import test_space

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Custom JSON encoder to handle NumPy arrays
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

def generate_random_pd_matrices(n: int, m: int, density=0.3, min_eigen=0.1, max_eigen=1.0, seed=42) -> List[sp.spmatrix]:
    """
    Generate n random positive definite sparse matrices of size m×m with 
    minimum eigenvalues in the range [min_eigen, max_eigen].
    """
    np.random.seed(seed)
    inf_mats = []
    for i in range(n):
        # Create a random sparse matrix
        A = sp.random(m, m, density=density, format="csc", random_state=seed+i)
        
        # Make it symmetric
        A = (A + A.T) / 2

        # use a simplified approach based on Gershgorin circle theorem
        # The smallest eigenvalue is bounded below by the minimum row sum
        row_sums = np.abs(A).sum(axis=1).A1
        diagonal = A.diagonal()
        min_gershgorin = np.min(diagonal - (row_sums - np.abs(diagonal)))
        
        # Calculate shift to achieve desired minimum eigenvalue
        base_shift = max(0, -min_gershgorin + 1e-6)  # Ensure positive definiteness
        target_min_eigen = np.random.uniform(min_eigen, max_eigen)
        
        # Apply the shift
        A = A + (base_shift + target_min_eigen) * sp.eye(m)
        
        inf_mats.append(A)
    return inf_mats

def generate_random_test_problem(n: int, m: int, seed: int = 42) -> Tuple:
    """
    Build a test problem using n random PD matrices of size m×m.
    
    Returns:
        Tuple: (n, m, inf_mats, H0, A_ineq, b_ineq, selection_init, weights, costs, locations)
    """
    np.random.seed(seed)
    inf_mats = generate_random_pd_matrices(n, m, density=0.3, min_eigen=0.5, max_eigen=1.0, seed=seed)
    H0 = m * sp.eye(m)
    cardinality = round(0.1 * n)
    # A_cardinality = np.ones((1, n))
    # b_cardinality = np.array([cardinality])
    weights = np.random.uniform(1, 5, size=n)
    A_weight = weights.reshape(1, -1)
    max_weight = round(0.08 * sum(weights))
    b_weight = np.array([max_weight])
    costs = np.random.uniform(10, 20, size=n)
    A_cost = costs.reshape(1, -1)
    max_cost = round(0.05 * sum(costs))
    b_cost = np.array([max_cost])
    A_ineq = np.vstack([A_weight, A_cost])
    b_ineq = np.vstack([b_weight, b_cost])
    selection_init = np.random.uniform(0, 1, size=n)
    # Generate random locations (optional, for visualization)
    locations = np.random.uniform(0, 10, size=(n, 2))
    return n, m, inf_mats, H0, A_ineq, b_ineq, selection_init, weights, costs, locations

def create_performance_profiles(results_dict: Dict, method_names: List[str], 
                              reference_method: str = "gurobi", 
                              max_ratio: float = 10.0,
                              plot_title: str = "Performance Profile") -> Dict:
    """
    Create performance profiles for comparing different methods.
    
    Args:
        results_dict: Dictionary containing experiment results
        method_names: List of method names to compare
        reference_method: Method to use as the reference (usually the best one)
        max_ratio: Maximum performance ratio to show in the plot
        plot_title: Title for the performance profile plot
        
    Returns:
        Dictionary with performance profile data
    """
    # Extract results for each method and problem instance
    performance_data = defaultdict(list)
    problem_keys = []
    
    # Collect all valid problems with the reference method result
    for key, result in results_dict.items():
        # Skip problems where the reference method (e.g., gurobi) has no result
        if (reference_method not in result or 
            "obj" not in result[reference_method] or 
            result[reference_method]["obj"] is None):
            continue
            
        # Reference value for this problem
        ref_obj = result[reference_method]["obj"]
        
        # Only include problems with a positive reference objective
        if ref_obj <= 0:
            continue
            
        problem_keys.append(key)
        
        # Calculate performance ratio for each method
        for method in method_names:
            if method == reference_method:
                # Reference method has ratio 1.0 by definition
                performance_data[method].append(1.0)
            else:
                if method not in result or "obj" not in result[method]:
                    # Method doesn't have a result for this problem
                    performance_data[method].append(float('inf'))
                else:
                    obj_value = result[method]["obj"]
                    if obj_value is None or obj_value <= 0:
                        # Handle invalid or negative objectives
                        performance_data[method].append(float('inf'))
                    else:
                        # Performance ratio: reference / method (for maximization problems)
                        # For minimization problems, use: method / reference
                        ratio = ref_obj / obj_value
                        performance_data[method].append(ratio)
    
    # Check if we have any valid problems
    n_problems = len(problem_keys)
    if n_problems == 0:
        logger.warning("No valid problems found for performance profile! Skipping profile creation.")
        return {
            "profile_data": {},
            "tau_values": [],
            "problem_keys": []
        }
    
    # Prepare the x-axis (performance ratio values)
    tau_values = np.logspace(0, np.log10(max_ratio), 100)
    
    # Calculate cumulative distribution for each method
    profile_data = {}
    
    for method in method_names:
        ratios = performance_data[method]
        profile = []
        
        for tau in tau_values:
            # Count problems solved within performance ratio tau
            count = sum(1 for r in ratios if r <= tau)
            profile.append(count / n_problems)
            
        profile_data[method] = profile
    
    # Create the performance profile plot
    plt.figure(figsize=(10, 6))
    for method in method_names:
        plt.semilogx(tau_values, profile_data[method], label=method, linewidth=2)
    
    plt.grid(True, which="both", ls="--")
    plt.xlabel("Performance Ratio τ")
    plt.ylabel("Fraction of Problems P(r ≤ τ)")
    plt.title(plot_title)
    plt.legend()
    plt.xlim(1, max_ratio)
    plt.ylim(0, 1.05)
    
    # Save the figure
    plt.tight_layout()
    os.makedirs("figures", exist_ok=True)
    plt.savefig(f"figures/performance_profile.pdf")
    plt.savefig(f"figures/performance_profile.png", dpi=300)
    plt.close()
    
    return {
        "profile_data": profile_data,
        "tau_values": tau_values.tolist(),
        "problem_keys": problem_keys
    }

def create_violin_plots(results_df: pd.DataFrame, save_dir: str = "figures"):
    """
    Create violin plots to show the distribution of objective values and computation times.
    
    Args:
        results_df: DataFrame containing experiment results
        save_dir: Directory to save the plots
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Create violin plots for objective values
    plt.figure(figsize=(12, 8))
    sns.violinplot(data=results_df, x="n", y="objective", hue="method", split=False, inner="quartile")
    plt.title("Distribution of Objective Values by Problem Size (n)")
    plt.xlabel("Number of Matrices (n)")
    plt.ylabel("Objective Value")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(title="Method")
    plt.tight_layout()
    plt.savefig(f"{save_dir}/objective_distribution_by_n.pdf")
    plt.savefig(f"{save_dir}/objective_distribution_by_n.png", dpi=300)
    plt.close()
    
    # Create violin plots for computation times (log scale)
    plt.figure(figsize=(12, 8))
    sns.violinplot(data=results_df, x="n", y="time", hue="method", split=False, inner="quartile")
    plt.yscale('log')
    plt.title("Distribution of Computation Times by Problem Size (n)")
    plt.xlabel("Number of Matrices (n)")
    plt.ylabel("Computation Time (s, log scale)")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(title="Method")
    plt.tight_layout()
    plt.savefig(f"{save_dir}/time_distribution_by_n.pdf")
    plt.savefig(f"{save_dir}/time_distribution_by_n.png", dpi=300)
    plt.close()
    
    # Create violin plots grouped by matrix size (m)
    plt.figure(figsize=(12, 8))
    sns.violinplot(data=results_df, x="m", y="objective", hue="method", split=False, inner="quartile")
    plt.title("Distribution of Objective Values by Matrix Size (m)")
    plt.xlabel("Matrix Size (m)")
    plt.ylabel("Objective Value")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(title="Method")
    plt.tight_layout()
    plt.savefig(f"{save_dir}/objective_distribution_by_m.pdf")
    plt.savefig(f"{save_dir}/objective_distribution_by_m.png", dpi=300)
    plt.close()
    
    # Create relative performance violin plots 
    # (normalize by optimal/reference solution for each instance)
    reference_df = results_df[results_df['method'] == 'Exact (B&C)'].copy()
    ref_dict = {(row['n'], row['m'], row['run']): row['objective'] 
               for _, row in reference_df.iterrows() if not np.isnan(row['objective'])}
    
    # Create a new column with relative performance
    relative_df = results_df.copy()
    relative_df['relative_perf'] = float('nan')
    
    for idx, row in relative_df.iterrows():
        key = (row['n'], row['m'], row['run'])
        if key in ref_dict and ref_dict[key] > 0:
            relative_df.at[idx, 'relative_perf'] = row['objective'] / ref_dict[key]
    
    # Filter out rows without reference solution and non-finite values
    relative_df = relative_df.dropna(subset=['relative_perf'])
    relative_df = relative_df[relative_df['relative_perf'] < float('inf')]
    
    # Create violin plot of relative performance
    plt.figure(figsize=(12, 8))
    sns.violinplot(data=relative_df, x="method", y="relative_perf", inner="quartile")
    plt.axhline(y=1.0, color='r', linestyle='--', alpha=0.7)
    plt.title("Relative Performance Distribution (Compared to Exact Solution)")
    plt.xlabel("Method")
    plt.ylabel("Relative Objective (Method / Exact)")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(f"{save_dir}/relative_performance_distribution.pdf")
    plt.savefig(f"{save_dir}/relative_performance_distribution.png", dpi=300)
    plt.close()

def plot_bar_charts_by_n_m(results_df: pd.DataFrame, save_dir: str = "figures"):
    """
    Create bar charts showing mean and error bars for different n and m values.
    
    Args:
        results_df: DataFrame containing experiment results
        save_dir: Directory to save the plots
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Group by n, m, and method to get statistics
    stats_df = results_df.groupby(['n', 'm', 'method']).agg({
        'objective': ['mean', 'std', 'min', 'max'],
        'time': ['mean', 'std', 'min', 'max']
    }).reset_index()
    
    # For each fixed m, create a plot of n vs. objective and time
    for m in results_df['m'].unique():
        m_df = stats_df[stats_df['m'] == m]
        
        # Create plot for objective values
        plt.figure(figsize=(12, 8))
        
        # Set positions for grouped bars
        n_values = sorted(m_df['n'].unique())
        x = np.arange(len(n_values))
        width = 0.2  # Width of bars
        
        # Plot each method
        methods = sorted(m_df['method'].unique())
        method_colors = {'Continuous Relaxation': 'blue', 
                        'Randomized Rounding': 'orange',
                        'Greedy 0/1': 'green', 
                        'Exact (B&C)': 'red'}
        
        # Plot in two groups: relaxation methods and integer methods
        relaxation_methods = ['Continuous Relaxation']
        integer_methods = ['Randomized Rounding', 'Greedy 0/1', 'Exact (B&C)']
        
        # First subplot for relaxation vs. exact
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12), sharex=True)
        
        # Plot relaxation vs. exact in first subplot
        for i, method in enumerate(relaxation_methods + ['Exact (B&C)']):
            if method in method_colors and method in m_df['method'].values:
                method_df = m_df[m_df['method'] == method]
                
                # Ensure all n values exist for this method
                plot_data = []
                plot_error = []
                
                for n in n_values:
                    n_data = method_df[method_df['n'] == n]
                    if len(n_data) > 0:
                        plot_data.append(n_data['objective']['mean'].values[0])
                        plot_error.append(n_data['objective']['std'].values[0])
                    else:
                        plot_data.append(0)
                        plot_error.append(0)
                        
                pos = x - width/2 if method == 'Continuous Relaxation' else x + width/2
                ax1.bar(pos, plot_data, width, label=method, color=method_colors[method], 
                       yerr=plot_error, capsize=5)
        
        ax1.set_ylabel('Objective Value')
        ax1.set_title(f'Relaxation vs. Exact: Objective Values (m = {m})')
        ax1.legend()
        ax1.grid(True, linestyle='--', alpha=0.7)
        
        # Plot integer methods in second subplot
        for i, method in enumerate(integer_methods):
            if method in method_colors and method in m_df['method'].values:
                method_df = m_df[m_df['method'] == method]
                
                # Ensure all n values exist for this method
                plot_data = []
                plot_error = []
                
                for n in n_values:
                    n_data = method_df[method_df['n'] == n]
                    if len(n_data) > 0:
                        plot_data.append(n_data['objective']['mean'].values[0])
                        plot_error.append(n_data['objective']['std'].values[0])
                    else:
                        plot_data.append(0)
                        plot_error.append(0)
                
                # Adjust bar positions
                pos = x + (i - 1) * width
                ax2.bar(pos, plot_data, width, label=method, color=method_colors[method], 
                       yerr=plot_error, capsize=5)
        
        ax2.set_ylabel('Objective Value')
        ax2.set_xlabel('Number of Matrices (n)')
        ax2.set_title(f'Integer Methods: Objective Values (m = {m})')
        ax2.set_xticks(x)
        ax2.set_xticklabels(n_values)
        ax2.legend()
        ax2.grid(True, linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        plt.savefig(f"{save_dir}/bar_objective_m_{m}.pdf")
        plt.savefig(f"{save_dir}/bar_objective_m_{m}.png", dpi=300)
        plt.close()
        
        # Create plot for computation times (log scale)
        fig, ax = plt.subplots(figsize=(12, 8))
        
        for i, method in enumerate(methods):
            if method in method_colors and method in m_df['method'].values:
                method_df = m_df[m_df['method'] == method]
                
                # Ensure all n values exist for this method
                plot_data = []
                plot_error = []
                
                for n in n_values:
                    n_data = method_df[method_df['n'] == n]
                    if len(n_data) > 0:
                        plot_data.append(n_data['time']['mean'].values[0])
                        plot_error.append(n_data['time']['std'].values[0])
                    else:
                        plot_data.append(0)
                        plot_error.append(0)
                
                # Adjust bar positions
                pos = x + (i - len(methods)/2 + 0.5) * width
                ax.bar(pos, plot_data, width, label=method, color=method_colors[method], 
                      yerr=plot_error, capsize=5)
        
        ax.set_ylabel('Computation Time (s)')
        ax.set_xlabel('Number of Matrices (n)')
        ax.set_title(f'Computation Times (m = {m})')
        ax.set_xticks(x)
        ax.set_xticklabels(n_values)
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.set_yscale('log')
        
        plt.tight_layout()
        plt.savefig(f"{save_dir}/bar_time_m_{m}.pdf")
        plt.savefig(f"{save_dir}/bar_time_m_{m}.png", dpi=300)
        plt.close()
    
    # Similar plots for fixed n values (just changing the grouping)
    for n in results_df['n'].unique():
        n_df = stats_df[stats_df['n'] == n]
        
        # Create plot for objective values
        plt.figure(figsize=(12, 8))
        
        # Set positions for grouped bars
        m_values = sorted(n_df['m'].unique())
        x = np.arange(len(m_values))
        width = 0.2  # Width of bars
        
        # Plot each method
        methods = sorted(n_df['method'].unique())
        method_colors = {'Continuous Relaxation': 'blue', 
                        'Randomized Rounding': 'orange',
                        'Greedy 0/1': 'green', 
                        'Exact (B&C)': 'red'}
        
        # Plot in two groups: relaxation methods and integer methods
        relaxation_methods = ['Continuous Relaxation']
        integer_methods = ['Randomized Rounding', 'Greedy 0/1', 'Exact (B&C)']
        
        # First subplot for relaxation vs. exact
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12), sharex=True)
        
        # Plot relaxation vs. exact in first subplot
        for i, method in enumerate(relaxation_methods + ['Exact (B&C)']):
            if method in method_colors and method in n_df['method'].values:
                method_df = n_df[n_df['method'] == method]
                
                # Ensure all m values exist for this method
                plot_data = []
                plot_error = []
                
                for m in m_values:
                    m_data = method_df[method_df['m'] == m]
                    if len(m_data) > 0:
                        plot_data.append(m_data['objective']['mean'].values[0])
                        plot_error.append(m_data['objective']['std'].values[0])
                    else:
                        plot_data.append(0)
                        plot_error.append(0)
                        
                pos = x - width/2 if method == 'Continuous Relaxation' else x + width/2
                ax1.bar(pos, plot_data, width, label=method, color=method_colors[method], 
                       yerr=plot_error, capsize=5)
        
        ax1.set_ylabel('Objective Value')
        ax1.set_title(f'Relaxation vs. Exact: Objective Values (n = {n})')
        ax1.legend()
        ax1.grid(True, linestyle='--', alpha=0.7)
        
        # Plot integer methods in second subplot
        for i, method in enumerate(integer_methods):
            if method in method_colors and method in n_df['method'].values:
                method_df = n_df[n_df['method'] == method]
                
                # Ensure all m values exist for this method
                plot_data = []
                plot_error = []
                
                for m in m_values:
                    m_data = method_df[method_df['m'] == m]
                    if len(m_data) > 0:
                        plot_data.append(m_data['objective']['mean'].values[0])
                        plot_error.append(m_data['objective']['std'].values[0])
                    else:
                        plot_data.append(0)
                        plot_error.append(0)
                
                # Adjust bar positions
                pos = x + (i - 1) * width
                ax2.bar(pos, plot_data, width, label=method, color=method_colors[method], 
                       yerr=plot_error, capsize=5)
        
        ax2.set_ylabel('Objective Value')
        ax2.set_xlabel('Matrix Size (m)')
        ax2.set_title(f'Integer Methods: Objective Values (n = {n})')
        ax2.set_xticks(x)
        ax2.set_xticklabels(m_values)
        ax2.legend()
        ax2.grid(True, linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        plt.savefig(f"{save_dir}/bar_objective_n_{n}.pdf")
        plt.savefig(f"{save_dir}/bar_objective_n_{n}.png", dpi=300)
        plt.close()
        
        # Create plot for computation times (log scale)
        fig, ax = plt.subplots(figsize=(12, 8))
        
        for i, method in enumerate(methods):
            if method in method_colors and method in n_df['method'].values:
                method_df = n_df[n_df['method'] == method]
                
                # Ensure all m values exist for this method
                plot_data = []
                plot_error = []
                
                for m in m_values:
                    m_data = method_df[method_df['m'] == m]
                    if len(m_data) > 0:
                        plot_data.append(m_data['time']['mean'].values[0])
                        plot_error.append(m_data['time']['std'].values[0])
                    else:
                        plot_data.append(0)
                        plot_error.append(0)
                
                # Adjust bar positions
                pos = x + (i - len(methods)/2 + 0.5) * width
                ax.bar(pos, plot_data, width, label=method, color=method_colors[method], 
                      yerr=plot_error, capsize=5)
        
        ax.set_ylabel('Computation Time (s)')
        ax.set_xlabel('Matrix Size (m)')
        ax.set_title(f'Computation Times (n = {n})')
        ax.set_xticks(x)
        ax.set_xticklabels(m_values)
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.set_yscale('log')
        
        plt.tight_layout()
        plt.savefig(f"{save_dir}/bar_time_n_{n}.pdf")
        plt.savefig(f"{save_dir}/bar_time_n_{n}.png", dpi=300)
        plt.close()

def run_experiment_with_multiple_seeds(
    n: int, 
    m: int, 
    num_runs: int = 5, 
    base_seed: int = 42,
    timeout_multiplier: int = 10
) -> Dict[str, Any]:
    """
    Run experiment for a fixed (n, m) combination with multiple seeds.
    
    Args:
        n: Number of matrices
        m: Matrix size
        num_runs: Number of runs with different seeds
        base_seed: Base seed value
        timeout_multiplier: Multiplier for greedy timeout
        
    Returns:
        Dictionary with results for all runs
    """
    all_run_results = {}
    
    logger.info(f"Running {num_runs} experiments for n = {n}, m = {m}")
    
    for run in range(num_runs):
        seed = base_seed + run
        logger.info(f"Run {run+1}/{num_runs} with seed {seed}")
        
        # Generate test problem
        n_val, m_val, inf_mats, H0, A_ineq, b_ineq, selection_init, weights, costs, locations = \
            generate_random_test_problem(n=n, m=m, seed=seed)
        
        # Method 1: Continuous Relaxation via Frank-Wolfe
        start_time = time.time()
        fw_result, fw_obj, fw_iters, fw_log = test_space.frank_wolfe_optimization(
            test_space.min_eigenvalue_objective,
            test_space.min_eigenvalue_gradient,
            selection_init,
            A_ineq,
            b_ineq,
            inf_mats,
            H0,
            max_iterations=1000,
            min_iterations=3,
            step_size_strategy=test_space.StepSizeStrategy.DIMINISHING,
            verbose=True,
            check_final_stationarity=False
        )
        fw_time = time.time() - start_time
        
        # Round the Frank-Wolfe solution to a binary solution
        start_round_time = time.time()
        fw_rounded_result, fw_bin_obj = test_space.round_solution(
            fw_result,
            inf_mats,
            H0,
            A_ineq,
            b_ineq,
            obj_func=test_space.min_eigenvalue_objective,
            num_samples=100,
            verbose=True
        )
        fw_round_time = time.time() - start_round_time
        fw_total_time = fw_time + fw_round_time
        
        # Method 2: Greedy 0/1 Selection with timeout based on FW time
        greedy_timeout = fw_time * timeout_multiplier
        
        start_time = time.time()
        greedy_result, greedy_obj, greedy_timed_out = test_space.greedy_01_selection(
            inf_mats,
            H0,
            A_ineq,
            b_ineq,
            obj_func=test_space.min_eigenvalue_objective,
            verbose=True,
            timeout=None
        )
        greedy_time = time.time() - start_time
        
        # Method 3: Gurobi Branch-and-Cut (if available)
        if test_space.GUROBI_AVAILABLE:
            start_time = time.time()
            gurobi_result, gurobi_obj, gurobi_stats = test_space.branch_and_cut_gurobi(
                inf_mats,
                H0,
                A_ineq,
                b_ineq,
                verbose=True
            )
            gurobi_time = time.time() - start_time
        else:
            gurobi_result = None
            gurobi_obj = None
            gurobi_stats = {"status": "Gurobi not available"}
            gurobi_time = None
        
        # Store results for this run
        run_results = {
            "continuous_relaxation": {
                "obj": fw_obj,
                "time": fw_time,
                "iterations": fw_iters
            },
            "randomized_rounding": {
                "obj": fw_bin_obj,
                "time": fw_round_time,
                "total_time": fw_total_time
            },
            "greedy": {
                "obj": greedy_obj,
                "time": greedy_time,
                "timed_out": greedy_timed_out,
                "timeout_limit": greedy_timeout
            },
            "gurobi": {
                "obj": gurobi_obj,
                "time": gurobi_time,
                "stats": gurobi_stats
            }
        }
        
        all_run_results[f"run_{run}"] = run_results
        
        # Log progress
        logger.info(f"Completed run {run+1}/{num_runs}")
        logger.info(f"  Continuous relaxation: obj={fw_obj:.6f}, time={fw_time:.2f}s")
        logger.info(f"  Randomized rounding: obj={fw_bin_obj:.6f}, time={fw_total_time:.2f}s")
        logger.info(f"  Greedy: obj={greedy_obj:.6f}, time={greedy_time:.2f}s" + 
                  (" (timed out)" if greedy_timed_out else ""))
        if gurobi_obj is not None:
            logger.info(f"  Gurobi: obj={gurobi_obj:.6f}, time={gurobi_time:.2f}s")
    
    # Compute statistics across all runs
    stats = {
        "n": n,
        "m": m,
        "num_runs": num_runs,
        "continuous_relaxation": {
            "obj_mean": np.mean([all_run_results[f"run_{run}"]["continuous_relaxation"]["obj"] 
                               for run in range(num_runs)]),
            "obj_std": np.std([all_run_results[f"run_{run}"]["continuous_relaxation"]["obj"] 
                              for run in range(num_runs)]),
            "obj_min": np.min([all_run_results[f"run_{run}"]["continuous_relaxation"]["obj"] 
                              for run in range(num_runs)]),
            "obj_max": np.max([all_run_results[f"run_{run}"]["continuous_relaxation"]["obj"] 
                              for run in range(num_runs)]),
            "time_mean": np.mean([all_run_results[f"run_{run}"]["continuous_relaxation"]["time"] 
                                for run in range(num_runs)]),
            "time_std": np.std([all_run_results[f"run_{run}"]["continuous_relaxation"]["time"] 
                               for run in range(num_runs)])
        },
        "randomized_rounding": {
            "obj_mean": np.mean([all_run_results[f"run_{run}"]["randomized_rounding"]["obj"] 
                               for run in range(num_runs)]),
            "obj_std": np.std([all_run_results[f"run_{run}"]["randomized_rounding"]["obj"] 
                              for run in range(num_runs)]),
            "obj_min": np.min([all_run_results[f"run_{run}"]["randomized_rounding"]["obj"] 
                              for run in range(num_runs)]),
            "obj_max": np.max([all_run_results[f"run_{run}"]["randomized_rounding"]["obj"] 
                              for run in range(num_runs)]),
            "time_mean": np.mean([all_run_results[f"run_{run}"]["randomized_rounding"]["total_time"] 
                                for run in range(num_runs)]),
            "time_std": np.std([all_run_results[f"run_{run}"]["randomized_rounding"]["total_time"] 
                               for run in range(num_runs)])
        },
        "greedy": {
            "obj_mean": np.mean([all_run_results[f"run_{run}"]["greedy"]["obj"] 
                               for run in range(num_runs)]),
            "obj_std": np.std([all_run_results[f"run_{run}"]["greedy"]["obj"] 
                              for run in range(num_runs)]),
            "obj_min": np.min([all_run_results[f"run_{run}"]["greedy"]["obj"] 
                              for run in range(num_runs)]),
            "obj_max": np.max([all_run_results[f"run_{run}"]["greedy"]["obj"] 
                              for run in range(num_runs)]),
            "time_mean": np.mean([all_run_results[f"run_{run}"]["greedy"]["time"] 
                                for run in range(num_runs)]),
            "time_std": np.std([all_run_results[f"run_{run}"]["greedy"]["time"] 
                               for run in range(num_runs)]),
            "timed_out_count": sum([all_run_results[f"run_{run}"]["greedy"]["timed_out"] 
                                  for run in range(num_runs)])
        }
    }
    
    # Add Gurobi stats if available
    if test_space.GUROBI_AVAILABLE:
        stats["gurobi"] = {
            "obj_mean": np.mean([all_run_results[f"run_{run}"]["gurobi"]["obj"] 
                               for run in range(num_runs) 
                               if all_run_results[f"run_{run}"]["gurobi"]["obj"] is not None]),
            "obj_std": np.std([all_run_results[f"run_{run}"]["gurobi"]["obj"] 
                              for run in range(num_runs)
                              if all_run_results[f"run_{run}"]["gurobi"]["obj"] is not None]),
            "obj_min": np.min([all_run_results[f"run_{run}"]["gurobi"]["obj"] 
                              for run in range(num_runs)
                              if all_run_results[f"run_{run}"]["gurobi"]["obj"] is not None]),
            "obj_max": np.max([all_run_results[f"run_{run}"]["gurobi"]["obj"] 
                              for run in range(num_runs)
                              if all_run_results[f"run_{run}"]["gurobi"]["obj"] is not None]),
            "time_mean": np.mean([all_run_results[f"run_{run}"]["gurobi"]["time"] 
                                for run in range(num_runs)
                                if all_run_results[f"run_{run}"]["gurobi"]["time"] is not None]),
            "time_std": np.std([all_run_results[f"run_{run}"]["gurobi"]["time"] 
                               for run in range(num_runs)
                               if all_run_results[f"run_{run}"]["gurobi"]["time"] is not None])
        }
    
    # Return both individual runs and aggregated statistics
    return {
        "runs": all_run_results,
        "stats": stats
    }

def create_summary_table(results_df, output_file):
    """
    Create a comprehensive summary table from the results DataFrame.
    
    Args:
        results_df: DataFrame containing all results
        output_file: Path to save the summary table
    """
    # Avoid multi-index issues by using a simpler approach
    summary_data = []
    
    # Define method order for sorting
    method_order = {'Continuous Relaxation': 0, 'Randomized Rounding': 1, 'Greedy 0/1': 2, 'Exact (B&C)': 3}
    
    # Process each n, m, method combination separately
    for n in sorted(results_df['n'].unique()):
        for m in sorted(results_df['m'].unique()):
            # Get reference values from exact solution if available
            exact_rows = results_df[(results_df['n'] == n) & 
                                   (results_df['m'] == m) & 
                                   (results_df['method'] == 'Exact (B&C)')]
            exact_obj_mean = None
            if len(exact_rows) > 0:
                exact_obj_mean = exact_rows['objective'].mean()
            
            # Process each method
            for method in sorted(results_df['method'].unique(), 
                                key=lambda x: method_order.get(x, 99)):
                # Get data for this configuration
                method_data = results_df[(results_df['n'] == n) & 
                                        (results_df['m'] == m) & 
                                        (results_df['method'] == method)]
                
                if len(method_data) == 0:
                    continue  # Skip if no data for this combination
                
                # Calculate statistics
                obj_mean = method_data['objective'].mean()
                obj_std = method_data['objective'].std()
                obj_min = method_data['objective'].min()
                obj_max = method_data['objective'].max()
                time_mean = method_data['time'].mean()
                time_std = method_data['time'].std()
                timeouts = method_data['timed_out'].sum()
                
                # Calculate relative performance if possible
                rel_perf = None
                if method != 'Exact (B&C)' and exact_obj_mean is not None and exact_obj_mean > 0:
                    rel_perf = obj_mean / exact_obj_mean
                
                # Add to summary data
                summary_data.append({
                    'n': n,
                    'm': m,
                    'method': method,
                    'obj_mean': obj_mean,
                    'obj_std': obj_std,
                    'obj_min': obj_min,
                    'obj_max': obj_max,
                    'time_mean': time_mean,
                    'time_std': time_std,
                    'timeouts': timeouts,
                    'rel_perf': rel_perf
                })
    
    # Write the summary table
    with open(output_file, 'w') as f:
        f.write("SUMMARY OF EXPERIMENTAL RESULTS\n")
        f.write("==============================\n\n")
        
        f.write("Table 1: Objective Values and Computation Times by Problem Size\n")
        f.write("-" * 120 + "\n")
        header = (
            f"{'n':<4}{'m':<4}{'Method':<22}"
            f"{'Obj Mean':<10}{'Obj Std':<10}{'Obj Min':<10}{'Obj Max':<10}"
            f"{'Time Mean':<10}{'Time Std':<10}{'Timeouts':<10}{'Rel. Perf.':<10}"
        )
        f.write(header + "\n")
        f.write("-" * 120 + "\n")
        
        # Write each row
        for row in summary_data:
            n = row['n']
            m = row['m']
            method = row['method']
            obj_mean = row['obj_mean']
            obj_std = row['obj_std']
            obj_min = row['obj_min']
            obj_max = row['obj_max']
            time_mean = row['time_mean']
            time_std = row['time_std']
            timeouts = row['timeouts']
            rel_perf = row['rel_perf']
            
            table_row = (
                f"{n:<4}{m:<4}{method:<22}"
                f"{obj_mean:<10.4f}{obj_std:<10.4f}{obj_min:<10.4f}{obj_max:<10.4f}"
                f"{time_mean:<10.2f}{time_std:<10.2f}{timeouts:<10d}"
            )
            
            # Add relative performance if available
            if rel_perf is not None:
                table_row += f"{rel_perf:<10.4f}"
            else:
                table_row += f"{'N/A':<10}"
            
            f.write(table_row + "\n")
            
        f.write("-" * 120 + "\n\n")
        
        # Add notes
        f.write("Notes:\n")
        f.write("- Obj Mean: Mean objective value across all runs\n")
        f.write("- Obj Std: Standard deviation of objective values\n")
        f.write("- Time Mean: Mean computation time in seconds\n")
        f.write("- Time Std: Standard deviation of computation times\n")
        f.write("- Timeouts: Number of runs where the greedy algorithm timed out\n")
        f.write("- Rel. Perf.: Relative performance compared to exact solution (Obj Mean / Exact Obj Mean)\n")
        f.write("- 'Continuous Relaxation' is the solution to the continuous relaxation of the problem\n")
        f.write("- 'Randomized Rounding' is the randomized rounding of the continuous relaxation\n")
        f.write("- 'Greedy 0/1' is the greedy binary solution\n")
        f.write("- 'Exact (B&C)' is the optimal solution found via branch-and-cut\n")

def main():
    """
    Main function to run the full experiment with multiple combinations
    of n and m values, using multiple runs per configuration.
    """
    # Define experiment parameters
    n_values = [100, 150, 500, 1000, 1500, 5000, 10000]  # Can increase for the actual paper
    m_values = [100, 500, 1000, 1500, 5000, 10000]  # Can increase for the actual paper
    num_runs_per_config = 1    # Use 5-10 runs per configuration for robust statistics
    base_seed = 42
    timeout_multiplier = 10
    
    # Dictionary to hold all experiment results
    all_results = {}
    
    # Create directories
    os.makedirs("figures", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    
    # Collect all raw data for analysis
    all_data_rows = []
    
    # Process all configurations
    for m in m_values:
        for n in n_values:
            logger.info(f"Starting experiments for n = {n}, m = {m}")
            
            # Run multiple experiments with different seeds
            config_results = run_experiment_with_multiple_seeds(
                n=n, 
                m=m, 
                num_runs=num_runs_per_config, 
                base_seed=base_seed,
                timeout_multiplier=timeout_multiplier
            )
            
            # Store results
            key = f"n_{n}_m_{m}"
            all_results[key] = config_results
            
            # Save results incrementally
            with open(f"results/results_n_{n}_m_{m}.json", "w") as f:
                json.dump(config_results, f, indent=4, cls=NumpyEncoder)
            
            # Collect data for DataFrame
            stats = config_results["stats"]
            for run in range(num_runs_per_config):
                run_data = config_results["runs"][f"run_{run}"]
                
                # Add continuous relaxation data
                all_data_rows.append({
                    "n": n,
                    "m": m,
                    "run": run,
                    "method": "Continuous Relaxation",
                    "objective": run_data["continuous_relaxation"]["obj"],
                    "time": run_data["continuous_relaxation"]["time"],
                    "timed_out": False
                })
                
                # Add randomized rounding data
                all_data_rows.append({
                    "n": n,
                    "m": m,
                    "run": run,
                    "method": "Randomized Rounding",
                    "objective": run_data["randomized_rounding"]["obj"],
                    "time": run_data["randomized_rounding"]["total_time"],
                    "timed_out": False
                })
                
                # Add greedy data
                all_data_rows.append({
                    "n": n,
                    "m": m,
                    "run": run,
                    "method": "Greedy 0/1",
                    "objective": run_data["greedy"]["obj"],
                    "time": run_data["greedy"]["time"],
                    "timed_out": run_data["greedy"]["timed_out"]
                })
                
                # Add Gurobi data if available
                if "gurobi" in run_data and run_data["gurobi"]["obj"] is not None:
                    all_data_rows.append({
                        "n": n,
                        "m": m,
                        "run": run,
                        "method": "Exact (B&C)",
                        "objective": run_data["gurobi"]["obj"],
                        "time": run_data["gurobi"]["time"],
                        "timed_out": False
                    })
            
            # Print summary for this configuration
            print(f"\nSummary for n = {n}, m = {m} ({num_runs_per_config} runs):")
            print(f"Continuous Relaxation: {stats['continuous_relaxation']['obj_mean']:.4f} ± {stats['continuous_relaxation']['obj_std']:.4f}, time: {stats['continuous_relaxation']['time_mean']:.2f}s")
            print(f"Randomized Rounding:   {stats['randomized_rounding']['obj_mean']:.4f} ± {stats['randomized_rounding']['obj_std']:.4f}, time: {stats['randomized_rounding']['time_mean']:.2f}s")
            print(f"Greedy 0/1:            {stats['greedy']['obj_mean']:.4f} ± {stats['greedy']['obj_std']:.4f}, time: {stats['greedy']['time_mean']:.2f}s, timeouts: {stats['greedy']['timed_out_count']}/{num_runs_per_config}")
            if "gurobi" in stats:
                print(f"Exact (B&C):          {stats['gurobi']['obj_mean']:.4f} ± {stats['gurobi']['obj_std']:.4f}, time: {stats['gurobi']['time_mean']:.2f}s")
    
    # Create a DataFrame from all collected data
    results_df = pd.DataFrame(all_data_rows)
    
    # Save the complete DataFrame
    results_df.to_csv("results/all_results.csv", index=False)
    
    # Generate comprehensive visualizations
    logger.info("Generating visualizations...")
    
    # Create violin plots
    create_violin_plots(results_df, save_dir="figures")
    
    # Create bar charts
    plot_bar_charts_by_n_m(results_df, save_dir="figures")
    
    # Create performance profiles for integer methods
    integer_methods = ["Randomized Rounding", "Greedy 0/1", "Exact (B&C)"]
    
    # Convert DataFrame back to dictionary format for performance profiles
    profile_results = {}
    for _, row in results_df.iterrows():
        key = f"n_{row['n']}_m_{row['m']}_run_{row['run']}"
        if key not in profile_results:
            profile_results[key] = {}
        
        method_key = row['method'].lower().replace(" ", "_").replace("(", "").replace(")", "")
        profile_results[key][method_key] = {
            "obj": row['objective'],
            "time": row['time']
        }
    
    # Create performance profile
    profile_data = create_performance_profiles(
        profile_results,
        method_names=["randomized_rounding", "greedy_01", "exact_bc"],
        reference_method="exact_bc",
        plot_title="Performance Profile for Integer Methods"
    )
    
    # Save profile data
    with open("results/performance_profile_data.json", "w") as f:
        json.dump(profile_data, f, indent=4, cls=NumpyEncoder)
    
    # Create summary table
    create_summary_table(results_df, "results/summary_table.txt")
    
    logger.info("Experiment completed successfully!")

# Execute the main function when the script is run directly
if __name__ == "__main__":
    main()