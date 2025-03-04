#!/usr/bin/env python
"""
Random Matrix Experiment with Multiple n and m Values,
Incremental JSON Saving, Summary Table, and Separate Graphs.

This script generates test problems using random positive-definite matrices
(via generate_random_pd_matrices), builds the optimization problem (with constraints),
and runs three methods:
  - Frank–Wolfe optimization,
  - Greedy 0/1 selection, and
  - Gurobi Branch-and-Cut (if available).

For each (n, m) combination, it records the objective and computation time.
After all experiments, it prints a summary table and produces separate line plots:
  - For each fixed m, a graph of computation time vs. n.
  - For each fixed n, a graph of computation time vs. m.
"""

import logging
import json
import numpy as np
import matplotlib.pyplot as plt
import os
import time
import scipy.sparse as sp
from typing import List, Tuple

# Import your core functions from test_space (assumed to be in the same folder)
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
    np.random.seed(seed)
    inf_mats = generate_random_pd_matrices(n, m, density=0.3, min_eigen=0.5, max_eigen=1.0, seed=seed)
    H0 = m * sp.eye(m)
    weights = np.random.uniform(1, 5, size=n)
    A_weight = weights.reshape(1, -1)
    max_weight = round(0.08 * sum(weights))
    b_weight = np.array([max_weight])
    
    costs = np.random.uniform(10, 20, size=n)
    A_cost = costs.reshape(1, -1)
    max_cost = round(0.05 * sum(costs))
    b_cost = np.array([max_cost])
    
    # Original A_ineq, b_ineq from weight and cost constraints
    A_ineq = np.vstack([A_weight, A_cost])
    b_ineq = np.vstack([b_weight, b_cost])
    
    # Example "location" assignments: each sensor i has a location in {0, 1, ..., L-1}
    L = 30  # number of possible locations
    locations = np.random.randint(low=0, high=L, size=n)
    
    # Build the "no two sensors per location" constraints
    unique_locs = np.unique(locations)
    A_no_two = np.zeros((len(unique_locs), n))
    b_no_two = np.ones((len(unique_locs), 1))

    for row_idx, loc in enumerate(unique_locs):
        sensor_indices = np.where(locations == loc)[0]
        A_no_two[row_idx, sensor_indices] = 1

    # Append them to A_ineq, b_ineq
    A_ineq = np.vstack([A_ineq, A_no_two])
    b_ineq = np.vstack([b_ineq, b_no_two])
    
    selection_init = np.random.uniform(0, 1, size=n)
    
    return (n, m, inf_mats, H0, A_ineq, b_ineq, selection_init, weights, costs, locations)

# Define experiment parameters: arrays for n and m values
n_values = [50, 75, 100]
m_values = [50, 100, 150]
seed = 42

# Dictionary to hold all experiment results
all_results = {}

# Run experiments over all (n, m) combinations and record results for all three methods
for n in n_values:
    for m in m_values:
        logger.info(f"Running experiment for n = {n}, m = {m}")
        n_val, m_val, inf_mats, H0, A_ineq, b_ineq, selection_init, weights, costs, locations = \
            generate_random_test_problem(n=n, m=m, seed=seed)
        
        # Method 1: Frank–Wolfe Optimization
        start_time = time.time()
        fw_result, fw_obj, fw_iters, fw_log = test_space.frank_wolfe_optimization(
            test_space.min_eigenvalue_objective,
            test_space.min_eigenvalue_gradient,
            selection_init,
            A_ineq,
            b_ineq,
            inf_mats,
            H0,
            max_iterations=100,
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
        
        # Method 2: Greedy 0/1 Selection
        start_time = time.time()
        greedy_result, greedy_obj = test_space.greedy_01_selection(
            inf_mats,
            H0,
            A_ineq,
            b_ineq,
            obj_func=test_space.min_eigenvalue_objective,
            verbose=True
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
        
        exp_results = {
            "n": n,
            "m": m,
            "frank_wolfe": {
                "obj": fw_obj,
                "obj_binary": fw_bin_obj,
                "time": fw_time,
                "rounding_time": fw_round_time,
                "total_time": fw_total_time,
                "iterations": fw_iters,
                "selection": fw_result,
                "selection_binary": fw_rounded_result,
                "log": fw_log
            },
            "greedy": {
                "obj": greedy_obj,
                "time": greedy_time,
                "selection": greedy_result
            },
            "gurobi": {
                "obj": gurobi_obj,
                "time": gurobi_time,
                "selection": gurobi_result,
                "stats": gurobi_stats
            }
        }
        key = f"n_{n}_m_{m}"
        all_results[key] = exp_results
        
        # Save results incrementally
        with open("random_matrix_experiment_results.json", "w") as f:
            json.dump(all_results, f, indent=4, cls=NumpyEncoder)
        logger.info(f"Results saved for n = {n}, m = {m}")

# ---------------------- SUMMARY TABLE -------------------------
print("\nSUMMARY TABLE")
print("-" * 120)
header = (
    f"{'n':<5}{'m':<5}"
    f"{'FW_obj':<12}{'FW_bin_obj':<12}{'FW_time(s)':<12}"
    f"{'Greedy_obj':<12}{'Greedy_time(s)':<15}"
    f"{'Gurobi_obj':<12}{'Gurobi_time(s)':<15}"
)
print(header)
print("-" * 120)
for key in sorted(all_results.keys()):
    res = all_results[key]
    n_val = res["n"]
    m_val = res["m"]
    fw_obj = res["frank_wolfe"]["obj"]
    fw_bin_obj = res["frank_wolfe"]["obj_binary"]
    fw_time = res["frank_wolfe"]["time"]  # Using total time including rounding
    greedy_obj = res["greedy"]["obj"]
    greedy_time = res["greedy"]["time"]
    if res["gurobi"]["time"] is not None:
        gurobi_obj = res["gurobi"]["obj"]
        gurobi_time = res["gurobi"]["time"]
    else:
        gurobi_obj = "N/A"
        gurobi_time = "N/A"
    row = (
        f"{n_val:<5}{m_val:<5}"
        f"{fw_obj:<12.4f}{fw_bin_obj:<12.4f}{fw_time:<12.4f}"
        f"{greedy_obj:<12.4f}{greedy_time:<15.4f}"
        f"{gurobi_obj!s:<12}{gurobi_time!s:<15}"
    )
    print(row)
print("-" * 120)



# --------------------------------------------------
# Results Visualization
# --------------------------------------------------
# --------------------------------------------------
# Global style adjustments
# --------------------------------------------------
plt.rcParams.update({
    "font.size": 12,
    "axes.labelsize": 12,
    "axes.titlesize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "lines.linewidth": 2,
    "lines.markersize": 6
})

# Optional: Define some hatch patterns for bar plots.
# Adjust as you prefer.
hatches = ['//', '\\\\', 'xx', '..']

# --------------------------------------------------
# 1. For each fixed m, produce stacked subplots: 
#    (Top) Times vs n; (Bottom) Objectives vs n
# --------------------------------------------------
for m in m_values:
    n_list = []
    fw_times, greedy_times, gurobi_times = [], [], []
    fw_objs, fw_bin_objs, greedy_objs, gurobi_objs = [], [], [], []
    
    # Gather data
    for n in n_values:
        key = f"n_{n}_m_{m}"
        if key in all_results:
            n_list.append(n)
            res = all_results[key]
            
            # Times
            fw_times.append(res["frank_wolfe"]["time"])  
            greedy_times.append(res["greedy"]["time"])
            gurobi_times.append(res["gurobi"]["time"] if res["gurobi"]["time"] is not None else np.nan)
            
            # Objectives
            fw_objs.append(res["frank_wolfe"]["obj"])
            fw_bin_objs.append(res["frank_wolfe"]["obj_binary"])
            greedy_objs.append(res["greedy"]["obj"])
            gurobi_objs.append(res["gurobi"]["obj"] if res["gurobi"]["obj"] is not None else np.nan)
    
    if len(n_list) == 0:
        continue
    
    # --------------------------------------------------
    # Create figure and subplots
    # --------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 10), sharex=False)
    fig.subplots_adjust(hspace=0.3)  # Increase vertical space if needed
    
    # --------------------------------------------------
    # Top subplot: line chart of times
    # --------------------------------------------------
    ax1.plot(n_list, fw_times, marker='o', markerfacecolor='white', label="Frank-Wolfe")
    ax1.plot(n_list, greedy_times, marker='s', markerfacecolor='white', label="Greedy 0/1")
    ax1.plot(n_list, gurobi_times, marker='^', markerfacecolor='white', label="Gurobi B&C")
    
    ax1.set_xlabel("Number of Matrices (n)")
    ax1.set_ylabel("Computation Time (s)")
    ax1.set_title(f"Computation Time vs. n (m = {m})")
    ax1.legend(loc='best')
    ax1.grid(True, linestyle=":", linewidth=0.6)
    
    # --------------------------------------------------
    # Bottom subplot: bar chart of objectives
    # --------------------------------------------------
    x = np.arange(len(n_list))
    width = 0.2
    
    rects1 = ax2.bar(
        x - 1.5*width, fw_objs, width, 
        label="Frank-Wolfe (cont.)",
        hatch=hatches[0], edgecolor='black'
    )
    rects2 = ax2.bar(
        x - 0.5*width, fw_bin_objs, width, 
        label="Frank-Wolfe (binary)",
        hatch=hatches[1], edgecolor='black'
    )
    rects3 = ax2.bar(
        x + 0.5*width, greedy_objs, width, 
        label="Greedy 0/1",
        hatch=hatches[2], edgecolor='black'
    )
    rects4 = ax2.bar(
        x + 1.5*width, gurobi_objs, width, 
        label="Gurobi B&C",
        hatch=hatches[3], edgecolor='black'
    )
    
    ax2.set_xlabel("Number of Matrices (n)")
    ax2.set_ylabel("Objective Value")
    ax2.set_title(f"Objective Values vs. n (m = {m})")
    ax2.set_xticks(x)
    ax2.set_xticklabels(n_list)
    ax2.legend(loc='best')
    ax2.grid(True, linestyle=":", linewidth=0.6)

    fig.tight_layout()
    plt.savefig(f"times_and_objectives_vs_n_m_{m}.png", dpi=300)
    plt.close(fig)

# --------------------------------------------------
# 2. For each fixed n, produce stacked subplots:
#    (Top) Times vs m; (Bottom) Objectives vs m
# --------------------------------------------------
for n in n_values:
    m_list = []
    fw_times, greedy_times, gurobi_times = [], [], []
    fw_objs, fw_bin_objs, greedy_objs, gurobi_objs = [], [], [], []
    
    # Gather data
    for m in m_values:
        key = f"n_{n}_m_{m}"
        if key in all_results:
            m_list.append(m)
            res = all_results[key]
            
            # Times
            fw_times.append(res["frank_wolfe"]["time"])
            greedy_times.append(res["greedy"]["time"])
            gurobi_times.append(res["gurobi"]["time"] if res["gurobi"]["time"] is not None else np.nan)
            
            # Objectives
            fw_objs.append(res["frank_wolfe"]["obj"])
            fw_bin_objs.append(res["frank_wolfe"]["obj_binary"])
            greedy_objs.append(res["greedy"]["obj"])
            gurobi_objs.append(res["gurobi"]["obj"] if res["gurobi"]["obj"] is not None else np.nan)
    
    if len(m_list) == 0:
        continue

    # --------------------------------------------------
    # Create figure and subplots
    # --------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 10), sharex=False)
    fig.subplots_adjust(hspace=0.3)
    
    # --------------------------------------------------
    # Top subplot: line chart of times
    # --------------------------------------------------
    ax1.plot(m_list, fw_times, marker='o', markerfacecolor='white', label="Frank-Wolfe (rounding)")
    ax1.plot(m_list, greedy_times, marker='s', markerfacecolor='white', label="Greedy 0/1")
    ax1.plot(m_list, gurobi_times, marker='^', markerfacecolor='white', label="Gurobi B&C")
    
    ax1.set_xlabel("Matrix Size (m)")
    ax1.set_ylabel("Computation Time (s)")
    ax1.set_title(f"Computation Time vs. m (n = {n})")
    ax1.legend(loc='best')
    ax1.grid(True, linestyle=":", linewidth=0.6)
    
    # --------------------------------------------------
    # Bottom subplot: bar chart of objectives
    # --------------------------------------------------
    x = np.arange(len(m_list))
    width = 0.2
    
    rects1 = ax2.bar(
        x - 1.5*width, fw_objs, width, 
        label="Frank-Wolfe (cont.)",
        hatch=hatches[0], edgecolor='black'
    )
    rects2 = ax2.bar(
        x - 0.5*width, fw_bin_objs, width, 
        label="Frank-Wolfe (binary)",
        hatch=hatches[1], edgecolor='black'
    )
    rects3 = ax2.bar(
        x + 0.5*width, greedy_objs, width, 
        label="Greedy 0/1",
        hatch=hatches[2], edgecolor='black'
    )
    rects4 = ax2.bar(
        x + 1.5*width, gurobi_objs, width, 
        label="Gurobi B&C",
        hatch=hatches[3], edgecolor='black'
    )
    
    ax2.set_xlabel("Matrix Size (m)")
    ax2.set_ylabel("Objective Value")
    ax2.set_title(f"Objective Values vs. m (n = {n})")
    ax2.set_xticks(x)
    ax2.set_xticklabels(m_list)
    ax2.legend(loc='best')
    ax2.grid(True, linestyle=":", linewidth=0.6)

    fig.tight_layout()
    plt.savefig(f"times_and_objectives_vs_m_n_{n}.png", dpi=300)
    plt.close(fig)
