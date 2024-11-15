import numpy as np
from scipy.optimize import linprog, minimize
import time
from optimizations import (
    greedy_selection,
    frank_wolfe_optimization,
    scipy_minimize,
    roundsolution,
    roundsolution_breakties,
    roundsolution_madow,
    scipy_minimize_lse,
    branch_and_bound_with_cuts)
from enum import Enum

# Define Metric Enum for selection
class Metric(Enum):
    LOGDET = 1
    MIN_EIG = 2
    MSE = 3

def generate_random_fim(num_matrices, matrix_size):
    """
    Generates a list of matrices where half are identity matrices and the other half
    are random symmetric positive-definite matrices with eigenvalues greater than 10.
    
    Args:
        num_matrices (int): Number of matrices to generate.
        matrix_size (int): Size of each square matrix.
    
    Returns:
        List[np.ndarray]: List of matrices as specified.
    """
    np.random.seed(40)
    inf_mats = []
    half_num = num_matrices // 2

    for _ in range(num_matrices - 5):
        inf_mats.append(np.eye(matrix_size))
    

    target_min_eigenvalue = 5  # Initial minimum eigenvalue target

    # for _ in range(half_num, num_matrices):
    #     # Generate a random symmetric positive-definite matrix
    #     A = np.random.rand(matrix_size, matrix_size)
    #     inf_mat = A @ A.T
        
    #     # Scale the matrix so that its smallest eigenvalue meets the target
    #     min_eigenvalue = np.linalg.eigvalsh(inf_mat).min()
    #     if min_eigenvalue < target_min_eigenvalue:
    #         inf_mat *= (target_min_eigenvalue / min_eigenvalue + 1e-6)  # Add a small epsilon for stability
        
    A = np.random.rand(matrix_size, matrix_size)
    inf_mat = A @ A.T
    min_eigenvalue = np.linalg.eigvalsh(inf_mat).min()
    if min_eigenvalue < target_min_eigenvalue:
        inf_mat *= (target_min_eigenvalue / min_eigenvalue + 1e-6)
    inf_mats.append(inf_mat)
    inf_mats.append(inf_mat)
    inf_mats.append(inf_mat)
    inf_mats.append(inf_mat)
    inf_mats.append(inf_mat)        
        # Increase the target for the next matrix
        # target_min_eigenvalue += 2

    for i, mat in enumerate(inf_mats):
        # Compute the eigenvalues of the current matrix
        eigvals = np.linalg.eigvalsh(mat)  # Use eigvalsh for symmetric or Hermitian matrices
        
        # Print the minimum eigenvalue
        print(f"Minimum eigenvalue of matrix {i}: {eigvals.min()}")
    return inf_mats


# Define pose and measurement dimensions
pose_dim = 6  # Each pose has 6 variables
measurement_dim = 30  # Total measurement dimension

num_poses = 6
num_matrices = 10  # Number of sensor candidates

# Calculate matrix size
matrix_size = measurement_dim + num_poses * pose_dim

# Update your selection vector and information matrices accordingly
selection_init = np.zeros(num_matrices)
num_runs = 1
k = 3              # Number of sensors to select
n_iters = 10000

# Generate synthetic data
inf_mats = generate_random_fim(num_matrices, matrix_size)
H0 = np.eye(matrix_size)
selection_init = np.ones(num_matrices) / num_matrices 

# Constraint setup for Frank-Wolfe
A = np.ones((1, num_matrices))
b = np.array([k])

# Helper function to time each optimization method
def time_function(func, *args, **kwargs):
    start_time = time.time()
    result = func(*args, **kwargs)
    end_time = time.time()
    print(f"{func.__name__} execution time: {end_time - start_time:.4f} seconds")
    return result

# Run Greedy Selection
print("\nRunning Greedy Selection")
selection_vec, best_score, avail_cand = time_function(
    greedy_selection,
    inf_mats=np.array(inf_mats),
    prior=H0,
    Nc=k,
    metric=Metric.MIN_EIG,
    num_runs=num_runs
)
print("Greedy Selection Results:", selection_vec, best_score)

# Run Frank-Wolfe Optimization
print("\nRunning Frank-Wolfe Optimization")
final_solution, min_eig_val_rounded, i = time_function(
    frank_wolfe_optimization,
    inf_mats=inf_mats,
    prior=H0,
    n_iters=n_iters,
    selection_init=selection_init,
    k=k,
    num_poses=num_poses,
    A=A,
    b=b
)
print("Frank-Wolfe Optimization Results:", final_solution, min_eig_val_rounded)

print("\n #########################################################################")

# Run Scipy Optimization
print("\nRunning Scipy Minimize Optimization")
continuous_sol_scipy, min_eig_val_scipy = time_function(
    scipy_minimize,
    inf_mats=inf_mats,
    H0=H0,
    selection_init=selection_init,
    k=k,
    num_poses=num_poses,
    A=A,
    b=b
)

print("Scipy Minimize Results:", continuous_sol_scipy, min_eig_val_scipy)

# Basic Rounding
basic_rounding_solution = roundsolution(continuous_sol_scipy, k)
print("\nBasic Rounding Solution:", basic_rounding_solution)

# Rounding with Tie-Breaking (using smallest eigenvalue)
rounding_tie_break_solution = roundsolution_breakties(continuous_sol_scipy, k, inf_mats, H0)
print("\nRounding with Tie-Breaking Solution:", rounding_tie_break_solution)

# Madow's Rounding (Probabilistic Rounding)
madow_rounding_solution = roundsolution_madow(continuous_sol_scipy, k)
print("\nMadow's Probabilistic Rounding Solution:", madow_rounding_solution)

print("\nRunning Scipy Optimization with Smoothing (LSE)")
selection_scipy_lse, approx_min_eig_val_scipy_lse = time_function(
    scipy_minimize_lse,
    inf_mats=inf_mats,
    H0=H0,
    selection_init=selection_init,
    num_poses=num_poses,
    A=A,
    b=b
)
print("Scipy Optimization with Smoothing Results:", selection_scipy_lse, approx_min_eig_val_scipy_lse)

# **Basic Rounding**
basic_rounding_solution_scipy_lse = roundsolution(selection_scipy_lse, k)
print("\nBasic Rounding Solution (Scipy LSE):", basic_rounding_solution_scipy_lse)

# **Rounding with Tie-Breaking (using smallest eigenvalue)**
rounding_tie_break_solution_scipy_lse = roundsolution_breakties(selection_scipy_lse, k, inf_mats, H0)
print("\nRounding with Tie-Breaking Solution (Scipy LSE):", rounding_tie_break_solution_scipy_lse)

# **Madow's Rounding (Probabilistic Rounding)**
madow_rounding_solution_scipy_lse = roundsolution_madow(selection_scipy_lse, k)
print("\nMadow's Probabilistic Rounding Solution (Scipy LSE):", madow_rounding_solution_scipy_lse)