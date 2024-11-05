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
    nsopy_optimize,
    branch_and_bound_with_cuts)
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
        inf_mat = A @ A.T
        inf_mats.append(inf_mat)
    return inf_mats

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
    inf_mats = []
    half_num = num_matrices // 2

    # First half as identity matrices
    for _ in range(half_num):
        inf_mats.append(np.eye(matrix_size))
    
    # Second half with eigenvalues greater than 10
    for _ in range(half_num, num_matrices):
        # Generate a random symmetric positive-definite matrix
        A = np.random.rand(matrix_size, matrix_size)
        inf_mat = A @ A.T
        
        # Scale the matrix so that its smallest eigenvalue is greater than 10
        min_eigenvalue = np.linalg.eigvalsh(inf_mat).min()
        if min_eigenvalue <= 10:
            inf_mat *= (10 / min_eigenvalue + 1e-6)  # Add a small epsilon for stability
        
        inf_mats.append(inf_mat)
    
    return inf_mats

# Parameters for testing
num_matrices = 10  # Number of sensor candidates
num_poses = 10
num_runs = 1
k = 5               # Number of sensors to select
n_iters = 10000       # Number of iterations for Frank-Wolfe optimization
matrix_size = 60

# Generate synthetic data
inf_mats = generate_random_fim(num_matrices, matrix_size)
H0 = np.eye(matrix_size) # Prior matrix as a small regularization term
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
    # num_runs=num_runs,
    A=A,
    b=b
)
print("Frank-Wolfe Optimization Results:", final_solution, min_eig_val_rounded)

print("\n #########################################################################")

# Run Frank-Wolfe Optimization with Greedy
print("\nRunning Frank-Wolfe Optimization with Greedy Solution")
final_solution, min_eig_val_rounded, i = time_function(
    frank_wolfe_optimization,
    inf_mats=inf_mats,
    prior=H0,
    n_iters=n_iters,
    selection_init=selection_vec,
    k=k,
    num_poses=num_poses,
    # num_runs=num_runs,
    A=A,
    b=b
)
print("Greedy-Frank-Wolfe Optimization Results:", final_solution, min_eig_val_rounded)

# Basic Rounding
basic_rounding_solution = roundsolution(final_solution, k)
print("\nBasic Rounding Solution:", basic_rounding_solution)

# Rounding with Tie-Breaking (using smallest eigenvalue)
rounding_tie_break_solution = roundsolution_breakties(final_solution, k, inf_mats, H0)
print("\nRounding with Tie-Breaking Solution:", rounding_tie_break_solution)

# Madow's Rounding (Probabilistic Rounding)
madow_rounding_solution = roundsolution_madow(final_solution, k)
print("\nMadow's Probabilistic Rounding Solution:", madow_rounding_solution)

# Branch-and-Bound with Cuts
branch_and_bound_solution, branch_and_bound_score = branch_and_bound_with_cuts(
    inf_mats=inf_mats,
    H0=H0,
    num_poses=num_poses,
    k=k,
    relaxed_solution=final_solution
)
print("\nBranch-and-Bound with Cuts Solution:", branch_and_bound_solution)
print("\nBranch-and-Bound with Cuts Score:", branch_and_bound_score)

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

# Branch-and-Bound with Cuts
branch_and_bound_solution, branch_and_bound_score = branch_and_bound_with_cuts(
    inf_mats=inf_mats,
    H0=H0,
    num_poses=num_poses,
    k=k,
    relaxed_solution=continuous_sol_scipy
)
print("\nBranch-and-Bound with Cuts Solution:", branch_and_bound_solution)
print("\nBranch-and-Bound with Cuts Score:", branch_and_bound_score)

print("\nRunning Nsopy Optimization")
selection_nsopy, min_eig_val_nsopy = time_function(
    nsopy_optimize,
    inf_mats=inf_mats,
    H0=H0,
    selection_init=selection_init,
    # k=k,
    num_poses=num_poses,
    A=A,
    b=b,
    # n_iters=n_iters
)
print("Nsopy Optimization Results:", selection_nsopy, min_eig_val_nsopy)

# **Basic Rounding**
basic_rounding_solution_nsopy = roundsolution(selection_nsopy, k)
print("\nBasic Rounding Solution (Nsopy):", basic_rounding_solution_nsopy)

# **Rounding with Tie-Breaking (using smallest eigenvalue)**
rounding_tie_break_solution_nsopy = roundsolution_breakties(selection_nsopy, k, inf_mats, H0)
print("\nRounding with Tie-Breaking Solution (Nsopy):", rounding_tie_break_solution_nsopy)

# **Madow's Rounding (Probabilistic Rounding)**
madow_rounding_solution_nsopy = roundsolution_madow(selection_nsopy, k)
print("\nMadow's Probabilistic Rounding Solution (Nsopy):", madow_rounding_solution_nsopy)

# **Branch-and-Bound with Cuts**
branch_and_bound_solution_nsopy, branch_and_bound_score_nsopy = branch_and_bound_with_cuts(
    inf_mats=inf_mats,
    H0=H0,
    num_poses=num_poses,
    k=k,
    relaxed_solution=selection_nsopy
)
print("\nBranch-and-Bound with Cuts Solution (Nsopy):", branch_and_bound_solution_nsopy)
print("\nBranch-and-Bound with Cuts Score (Nsopy):", branch_and_bound_score_nsopy)
