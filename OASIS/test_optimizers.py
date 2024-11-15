import numpy as np
import time
import yaml
from enum import Enum
from optimizations import (
    greedy_selection,
    frank_wolfe_optimization,
    scipy_minimize,
    scipy_minimize_lse,
    roundsolution,
    roundsolution_breakties,
    roundsolution_madow
)

# Define Metric Enum for selection
class Metric(Enum):
    LOGDET = 1
    MIN_EIG = 2
    MSE = 3

def generate_random_fim(num_matrices, matrix_size):
    """
    Generates a list of completely random symmetric positive-definite matrices
    using random eigenvalues and orthogonal matrices.
    """
    np.random.seed(40)
    inf_mats = []

    for i in range(num_matrices):
        # Generate random positive eigenvalues
        eigenvalues = np.random.uniform(low=2, high=100, size=matrix_size)
        # Generate a random orthogonal matrix
        Q, _ = np.linalg.qr(np.random.randn(matrix_size, matrix_size))
        # Construct the positive-definite matrix
        inf_mat = Q @ np.diag(eigenvalues) @ Q.T
        inf_mats.append(inf_mat)

        # Calculate and print the minimum eigenvalue
        min_eig_val = np.min(eigenvalues)
        print(f"Matrix {i + 1}: Minimum eigenvalue = {min_eig_val:.4f}")

    return inf_mats

# Helper function to time each optimization method
def time_function(func, *args, **kwargs):
    start_time = time.time()
    result = func(*args, **kwargs)
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"{func.__name__} execution time: {execution_time:.4f} seconds")
    return result, execution_time

# Main testing function
def run_tests():
    # Define pose and measurement dimensions
    pose_dim = 6  # Each pose has 6 variables
    measurement_dim = 30  # Total measurement dimension

    num_poses = 6

    # Define the list of num_matrices to test
    num_matrices_list = [10, 100, 1000]

    # Initialize a list to store the results
    results = []

    # Iterate over the values of num_matrices
    for num_matrices in num_matrices_list:
        # Determine k_values based on num_matrices
        if num_matrices == 10:
            k_values = [5, 3]  # Test with k = 5 and k = 3 when num_matrices == 10
        else:
            k_values = [5]      # Test only with k = 5 for other num_matrices

        # Calculate matrix size
        matrix_size = measurement_dim + num_poses * pose_dim

        # Generate synthetic data
        inf_mats = generate_random_fim(num_matrices, matrix_size)
        H0 = np.eye(matrix_size)
        selection_init = np.ones(num_matrices, dtype=np.float16) / num_matrices

        # Iterate over k_values
        for k in k_values:
            # Constraint setup for optimization methods
            A = np.ones((1, num_matrices))
            b = np.array([k])

            print(f"\nRunning tests for num_matrices = {num_matrices}, k = {k}")

            # Dictionary to store results for this test case
            test_case_result = {
                'num_matrices': num_matrices,
                'k': k,
                'results': {}
            }

            # Run Greedy Selection
            print("\nRunning Greedy Selection")
            (selection_vec, best_score, avail_cand), exec_time = time_function(
                greedy_selection,
                inf_mats=np.array(inf_mats),
                prior=H0,
                Nc=k,
                metric=Metric.MIN_EIG,
                num_runs=1,
                num_poses=num_poses
            )
            if num_matrices == 10:
                print("Greedy Selection Results (selection vector):", selection_vec)
                selection_vector = selection_vec.tolist()
            else:
                selection_vector = None
            print("Greedy Selection Best Score:", best_score)

            # Store results
            test_case_result['results']['Greedy Selection'] = {
                'execution_time': exec_time,
                'best_score': best_score,
                'selection_vector': selection_vector
            }

            print("\n" + "#" * 70)

            # Run Scipy Minimize Optimization
            print("\nRunning Scipy Minimize Optimization")
            (continuous_sol_scipy, min_eig_val_scipy), exec_time = time_function(
                scipy_minimize,
                inf_mats=inf_mats,
                H0=H0,
                selection_init=selection_init,
                k=k,
                num_poses=num_poses,
                A=A,
                b=b
            )
            if num_matrices == 10:
                print("Scipy Minimize Results (selection vector):", continuous_sol_scipy)
                print("Scipy Minimize Results (K - max):", roundsolution(continuous_sol_scipy, k))
                print("Scipy Minimize Results (Breakties):", roundsolution_breakties(continuous_sol_scipy, k, inf_mats, H0))
                print("Scipy Minimize Results (Madow):", roundsolution_madow(continuous_sol_scipy, k))
                selection_vector = continuous_sol_scipy.tolist()
            else:
                selection_vector = None
            print("Scipy Minimize Best Score:", min_eig_val_scipy)

            # Store results
            test_case_result['results']['Scipy Minimize'] = {
                'execution_time': exec_time,
                'best_score': min_eig_val_scipy,
                'selection_vector': selection_vector
            }

            print("\n" + "#" * 70)

            # Run Scipy Optimization with Smoothing (LSE)
            print("\nRunning Scipy Optimization with Smoothing (LSE)")
            (selection_scipy_lse, approx_min_eig_val_scipy_lse), exec_time = time_function(
                scipy_minimize_lse,
                inf_mats=inf_mats,
                H0=H0,
                selection_init=selection_init,
                num_poses=num_poses,
                A=A,
                b=b
            )
            if num_matrices == 10:
                print("Scipy Optimization with Smoothing Results (selection vector):", selection_scipy_lse)
                print("Scipy Minimize LSE Results (K - max):", roundsolution(selection_scipy_lse, k))
                print("Scipy Minimize LSE Results (Breakties):", roundsolution_breakties(selection_scipy_lse, k, inf_mats, H0))
                print("Scipy Minimize LSE Results (Madow):", roundsolution_madow(selection_scipy_lse, k))
                selection_vector = selection_scipy_lse.tolist()
            else:
                selection_vector = None
            print("Scipy Optimization with Smoothing Best Score:", approx_min_eig_val_scipy_lse)

            # Store results
            test_case_result['results']['Scipy Optimization with LSE'] = {
                'execution_time': exec_time,
                'best_score': approx_min_eig_val_scipy_lse,
                'selection_vector': selection_vector
            }

            print("\n" + "#" * 70)

            # Run Frank-Wolfe Optimization only for num_matrices = 10 and 100
            if num_matrices in [10, 100]:
                n_iters = 10000 if num_matrices == 10 else 1000
                print("\nRunning Frank-Wolfe Optimization")
                (final_solution, min_eig_val_rounded, i), exec_time = time_function(
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
                if num_matrices == 10:
                    print("Frank-Wolfe Optimization Results (selection vector):", final_solution)
                    print("Frank-Wolfe Optimization Results (K - max):", roundsolution(final_solution, k))
                    print("Frank-Wolfe Optimization Results (Breakties):", roundsolution_breakties(final_solution, k, inf_mats, H0))
                    print("Frank-Wolfe Optimization Results (Madow):", roundsolution_madow(final_solution, k))
                    selection_vector = final_solution.tolist()
                else:
                    selection_vector = None
                print("Frank-Wolfe Optimization Best Score:", min_eig_val_rounded)

                # Store results
                test_case_result['results']['Frank-Wolfe Optimization'] = {
                    'execution_time': exec_time,
                    'best_score': min_eig_val_rounded,
                    'selection_vector': selection_vector
                }

                print("\n" + "#" * 70)
            else:
                # print("Frank-Wolfe Optimization skipped for num_matrices =", num_matrices)
                # Indicate that Frank-Wolfe was skipped
                test_case_result['results']['Frank-Wolfe Optimization'] = {
                    'skipped': True
                }

            # Append the test case result to the results list
            results.append(test_case_result)

    print("\nTesting complete.")

if __name__ == "__main__":
    run_tests()