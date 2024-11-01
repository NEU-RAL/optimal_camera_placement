import numpy as np
from scipy.optimize import linprog, minimize
import time
from optimizations import greedy_selection, frank_wolfe_optimization, scipy_minimize
from enum import Enum

# Define Metric Enum for selection
class Metric(Enum):
    LOGDET = 1
    MIN_EIG = 2
    MSE = 3

# Synthetic Data Generation Function
def generate_random_fim(num_matrices, matrix_size):
    """
    Generates a list of random symmetric positive-definite matrices.
    Args:
        num_matrices (int): Number of matrices to generate.
        matrix_size (int): Size of each square matrix.
    Returns:
        List[np.ndarray]: List of positive-definite matrices.
    """
    inf_mats = []
    for _ in range(num_matrices):
        A = np.random.rand(matrix_size, matrix_size)
        inf_mat = A @ A.T  # Ensures the matrix is symmetric positive-definite
        inf_mats.append(inf_mat)
    return inf_mats

# Parameters for testing
num_matrices = 10  # Number of sensor candidates
matrix_size = 12   # Size of each FIM matrix (depends on num_poses and num_points)
num_poses = 10
num_runs = 1
k = 5               # Number of sensors to select
n_iters = 50        # Number of iterations for Frank-Wolfe optimization

# Generate synthetic data
inf_mats = generate_random_fim(num_matrices, matrix_size)
H0 = np.eye(matrix_size) * 0.1  # Prior matrix as a small regularization term
selection_init = np.ones(num_matrices) / num_matrices  # Initial selection vector

# Constraint setup for Frank-Wolfe
A = np.ones((1, num_matrices))
b = np.array([k])  # Ensure that only `k` sensors are selected

# Helper function to time each optimization method
def time_function(func, *args, **kwargs):
    start_time = time.time()
    result = func(*args, **kwargs)
    end_time = time.time()
    print(f"{func.__name__} execution time: {end_time - start_time:.4f} seconds")
    return result

# Run Greedy Selection
print("\nRunning Greedy Selection")
best_selection_indices, best_score, avail_cand = time_function(
    greedy_selection,
    inf_mat=np.array(inf_mats),
    prior=H0,
    Nc=k,
    metric=Metric.LOGDET,
    num_runs=num_runs
)
print("Greedy Selection Results:", best_selection_indices, best_score)

# Run Frank-Wolfe Optimization
print("\nRunning Frank-Wolfe Optimization")
final_solution, selection_cur, min_eig_val_rounded, min_eig_val_unrounded, iter_count = time_function(
    frank_wolfe_optimization,
    inf_mats=inf_mats,
    prior=H0,
    n_iters=n_iters,
    selection_init=selection_init,
    k=k,
    num_poses=num_poses,
    num_runs=num_runs,
    A=A,
    b=b
)
print("Frank-Wolfe Optimization Results:", final_solution, min_eig_val_rounded, min_eig_val_unrounded)

# Run Scipy Minimize Optimization
print("\nRunning Scipy Minimize Optimization")
rounded_sol, continuous_sol, min_eig_val_rounded, min_eig_val_unr = time_function(
    scipy_minimize,
    inf_mats=inf_mats,
    H0=H0,
    selection_init=selection_init,
    k=k,
    num_poses=num_poses
)
print("Scipy Minimize Results:", rounded_sol, min_eig_val_rounded, min_eig_val_unr)

# Verification of Constraints and Consistency of Results
print("\nVerification of Results")
# Ensure that the number of selected sensors does not exceed `k`
assert np.sum(rounded_sol) <= k, "Scipy Minimize solution violates the 'at most k' constraint."
assert np.sum(final_solution) <= k, "Frank-Wolfe solution violates the 'at most k' constraint."
assert len(best_selection_indices) <= k, "Greedy Selection solution violates the 'at most k' constraint."

# Compare scores across methods to see if they are consistent
print("Best Score from Greedy Selection:", best_score)
print("Minimum Eigenvalue (Rounded) from Frank-Wolfe:", min_eig_val_rounded)
print("Minimum Eigenvalue (Rounded) from Scipy Minimize:", min_eig_val_rounded)
print("Minimum Eigenvalue (Continuous) from Scipy Minimize:", min_eig_val_unr)
