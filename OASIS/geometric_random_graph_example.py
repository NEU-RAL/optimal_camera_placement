#!/usr/bin/env python
"""
Geometric Random Graph Experiment with Multiple n and m Values,
Incremental JSON Saving, and Separate Graphs for Each Fixed m and n.

For each fixed m, this script produces a line graph showing computation time vs n,
and for each fixed n, a graph showing computation time vs m.
"""

import logging
import json
import numpy as np
import matplotlib.pyplot as plt
import os

# Import your core functions from test_space (assumed to be in the same folder)
import optimization_methods

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

# Define arrays for n (number of matrices) and m (matrix sizes)
n_values = [100]
m_values = [100]
gamma_value = 1.0 

# Dictionary to hold all experiment results
all_results = {}

# Run experiments over all (n, m) combinations
for n in n_values:
    for m in m_values:
        logger.info(f"Running experiment for n = {n}, m = {m}")
        results = optimization_methods.run_algorithm4_example(verbose=True, n=n, m=m, gamma=gamma_value)
        key = f"n_{n}_m_{m}"
        all_results[key] = results
        # Save results incrementally
        with open("geometric_experiment_results.json", "w") as f:
            json.dump(all_results, f, indent=4, cls=NumpyEncoder)
        logger.info(f"Results saved for n = {n}, m = {m}")

# Visualization: Produce separate graphs for each fixed m (speed vs n)
for m in m_values:
    x_n = []
    times_by_method = {"Frank-Wolfe": [], "Greedy": [], "Gurobi": []}
    for n in n_values:
        key = f"n_{n}_m_{m}"
        if key in all_results:
            x_n.append(n)
            res = all_results[key]
            # For Frank-Wolfe
            if "frank_wolfe" in res:
                times_by_method["Frank-Wolfe"].append(res["frank_wolfe"]["time"])
            else:
                times_by_method["Frank-Wolfe"].append(np.nan)
            # For Greedy
            if "greedy" in res:
                times_by_method["Greedy"].append(res["greedy"]["time"])
            else:
                times_by_method["Greedy"].append(np.nan)
            # For Gurobi (if available)
            if "gurobi" in res and "time" in res["gurobi"]:
                times_by_method["Gurobi"].append(res["gurobi"]["time"])
            else:
                times_by_method["Gurobi"].append(np.nan)
    plt.figure(figsize=(8, 6))
    for method, times in times_by_method.items():
        plt.plot(x_n, times, marker='o', label=method)
    plt.xlabel("Number of Matrices (n)")
    plt.ylabel("Computation Time (s)")
    plt.title(f"Computation Time vs n (for m = {m})")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.savefig(f"speed_vs_n_m_{m}.png", dpi=300)
    plt.show()

# Visualization: Produce separate graphs for each fixed n (speed vs m)
for n in n_values:
    x_m = []
    times_by_method = {"Frank-Wolfe": [], "Greedy": [], "Gurobi": []}
    for m in m_values:
        key = f"n_{n}_m_{m}"
        if key in all_results:
            x_m.append(m)
            res = all_results[key]
            # For Frank-Wolfe
            if "frank_wolfe" in res:
                times_by_method["Frank-Wolfe"].append(res["frank_wolfe"]["time"])
            else:
                times_by_method["Frank-Wolfe"].append(np.nan)
            # For Greedy
            if "greedy" in res:
                times_by_method["Greedy"].append(res["greedy"]["time"])
            else:
                times_by_method["Greedy"].append(np.nan)
            # For Gurobi (if available)
            if "gurobi" in res and "time" in res["gurobi"]:
                times_by_method["Gurobi"].append(res["gurobi"]["time"])
            else:
                times_by_method["Gurobi"].append(np.nan)
    plt.figure(figsize=(8, 6))
    for method, times in times_by_method.items():
        plt.plot(x_m, times, marker='o', label=method)
    plt.xlabel("Matrix Size (m)")
    plt.ylabel("Computation Time (s)")
    plt.title(f"Computation Time vs m (for n = {n})")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.savefig(f"speed_vs_m_n_{n}.png", dpi=300)
    plt.show()
