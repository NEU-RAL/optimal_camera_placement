import numpy as np
import time
import yaml
from enum import Enum
from scipy.sparse import random as sparse_random
from scipy.sparse import csr_matrix, diags
from scipy.stats import uniform
from optimizations import (
    greedy_selection,
    frank_wolfe_optimization,
    scipy_minimize,
    scipy_minimize_lse,
    roundsolution,
    roundsolution_breakties,
    roundsolution_madow
)
import FIM as fim
import pandas as pd

# Define Metric Enum for selection
class Metric(Enum):
    LOGDET = 1
    MIN_EIG = 2
    MSE = 3

def generate_random_fim(num_matrices, matrix_size, density=0.1, min_eigenvalue=2.0, random_state=40):
    """
    Generates a list of random sparse symmetric positive-definite matrices.

    Each matrix is constructed as A = B^T B + cI, where:
    - B is a random sparse matrix with a specified density.
    - c is a constant added to the diagonal to ensure positive-definiteness.

    Args:
        num_matrices (int): Number of matrices to generate.
        matrix_size (int): Size of the square matrices (number of rows/columns).
        density (float, optional): Density of the random sparse matrix B (fraction of non-zero elements). Default is 0.1.
        min_eigenvalue (float, optional): Minimum eigenvalue for positive-definiteness. Default is 2.0.
        random_state (int, optional): Seed for reproducibility. Default is 40.

    Returns:
        List[scipy.sparse.csr_matrix]: List of sparse symmetric positive-definite matrices in CSR format.
    """
    np.random.seed(random_state)
    inf_mats = []

    for i in range(num_matrices):
        # Generate a random sparse matrix B with standard normal distributed non-zero entries
        B = sparse_random(matrix_size, matrix_size, density=density, format='csr', 
                         data_rvs=np.random.randn)

        # Compute A = B^T B to ensure positive semi-definiteness
        A = B.transpose().dot(B)

        # Add c*I to make it positive definite
        # Here, c is set to min_eigenvalue to ensure all eigenvalues >= min_eigenvalue
        c = min_eigenvalue
        A += diags([c] * matrix_size, format='csr')

        inf_mats.append(A)

        # Optional: Verify the minimum eigenvalue (for debugging purposes)
        # For large matrices, computing eigenvalues can be expensive
        # Uncomment the following lines if you want to verify for small matrices
        # if matrix_size <= 100:
        #     eigvals = np.linalg.eigvalsh(A.toarray())
        #     min_eig_val = np.min(eigvals)
        #     print(f"Matrix {i + 1}: Minimum eigenvalue = {min_eig_val:.4f}")

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
    measurement_dim = 10  # Total measurement dimension

    num_poses_list = [6]

    # Define the list of num_matrices to test
    density = 0.05        # 5% non-zero elements
    min_eigenvalue = 5.0  # Minimum eigenvalue

    num_matrices_list = [10]
    # k_values
    k_values = [2]

    # Initialize a list to store the results
    results = []

    results_n_k = np.zeros((len(num_poses_list), len(num_matrices_list), len(k_values), 4, 6))

    # Iterate over the values of num_matrices
    for p_index, num_poses in enumerate(num_poses_list):
        #iterate over different matrix sizes
        for n_index,num_matrices in enumerate(num_matrices_list):
            # Calculate matrix size
            matrix_size = measurement_dim + num_poses * pose_dim
            # Generate synthetic data
            inf_mats = generate_random_fim(num_matrices, matrix_size, density, min_eigenvalue)
            H0 = diags([min_eigenvalue] * matrix_size, format='csr')
            selection_init = np.ones(num_matrices, dtype=np.float16) / num_matrices

            # Iterate over k_values
            for k_index, k in enumerate(k_values):
                # Constraint setup for optimization methods
                A = csr_matrix(np.ones((1, num_matrices)))  # Create A as a sparse matrix
                b = np.array([k])

                print(f"\nRunning tests for matrix_size = {matrix_size},  num_matrices = {num_matrices}, k = {k}")

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

                print("Greedy Selection Results (selection vector):", np.nonzero(selection_vec))
                selection_vector = selection_vec.tolist()
                print("Greedy Selection Best Score:", best_score)

                # Store results
                test_case_result['results']['Greedy Selection'] = {
                    'execution_time': exec_time,
                    'best_score': best_score,
                    'selection_vector': selection_vector
                }
                results_n_k[p_index, n_index, k_index, 0, 0]= best_score
                results_n_k[p_index, n_index, k_index, 0, 1]= best_score
                results_n_k[p_index, n_index, k_index, 0, 2] = best_score
                results_n_k[p_index, n_index, k_index, 0, 3] = best_score
                results_n_k[p_index, n_index, k_index, 0, 4]= exec_time
                # results_n_k[n_index, k_index, 0, 5]= 0

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

                print("Scipy Minimize Results (selection vector):", continuous_sol_scipy)
                print("Scipy Minimize Best Score unrounded:", min_eig_val_scipy)
                k_max_sol = roundsolution(continuous_sol_scipy, k)
                print("\nScipy Minimize Results (K - max):",np.nonzero(k_max_sol) )
                k_max_score = fim.find_min_eig_pair(inf_mats, np.array(k_max_sol), H0, num_poses)[0]
                print("Scipy Minimize (K - max) score:",k_max_score )
                break_ties_sol = roundsolution_breakties(continuous_sol_scipy, k, inf_mats, H0)
                print("\nScipy Minimize Results (Breakties):", np.nonzero(break_ties_sol))
                break_ties_score = fim.find_min_eig_pair(inf_mats, np.array(break_ties_sol), H0, num_poses)[0]
                print("Scipy Minimize Breakties score:", break_ties_score)
                madow_sol = roundsolution_madow(continuous_sol_scipy, k)
                print("\nScipy Minimize Results (Madow):",np.nonzero(madow_sol ))
                madow_score = fim.find_min_eig_pair(inf_mats, np.array(madow_sol), H0, num_poses)[0]
                print("Scipy Minimize madow score:",madow_score)
                selection_vector = continuous_sol_scipy.tolist()

                # Store results
                test_case_result['results']['Scipy Minimize'] = {
                    'execution_time': exec_time,
                    'best_score': min_eig_val_scipy,
                    'selection_vector': selection_vector
                }
                results_n_k[p_index, n_index, k_index, 1, 0] = min_eig_val_scipy
                results_n_k[p_index, n_index, k_index, 1, 1] = k_max_score
                results_n_k[p_index, n_index, k_index, 1, 2] = break_ties_score
                results_n_k[p_index, n_index, k_index, 1, 3] = madow_score
                results_n_k[p_index, n_index, k_index, 1, 4]= exec_time
                # results_n_k[n_index, k_index, 1, 5] = 0

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

                print("Scipy Optimization with Smoothing Results (selection vector):", selection_scipy_lse)
                print("Scipy Optimization with Smoothing Best Score:", approx_min_eig_val_scipy_lse)
                k_max_sol_lse = roundsolution(selection_scipy_lse, k)
                print("\nScipy Minimize LSE Results (K - max):", np.nonzero(k_max_sol_lse))
                k_max_score_lse = fim.find_min_eig_pair(inf_mats, np.array(k_max_sol_lse), H0, num_poses)[0]
                print("Scipy Minimize LSE (K - max) score:", k_max_score_lse)
                breakties_sol_lse = roundsolution_breakties(selection_scipy_lse, k, inf_mats, H0)
                print("\nScipy Minimize LSE Results (Breakties):", np.nonzero(breakties_sol_lse))
                break_ties_score_lse = fim.find_min_eig_pair(inf_mats, np.array(breakties_sol_lse), H0, num_poses)[0]
                print("Scipy Minimize LSE Breakties score:", break_ties_score_lse)
                madow_sol_lse = roundsolution_madow(selection_scipy_lse, k)
                print("\nScipy Minimize LSE Results (Madow):", np.nonzero((madow_sol_lse)))
                madow_score_lse = fim.find_min_eig_pair(inf_mats, np.array(madow_sol_lse), H0, num_poses)[0]
                print("Scipy Minimize LSE Madow score:", madow_score_lse)
                selection_vector = selection_scipy_lse.tolist()


                # Store results
                test_case_result['results']['Scipy Optimization with LSE'] = {
                    'execution_time': exec_time,
                    'best_score': approx_min_eig_val_scipy_lse,
                    'selection_vector': selection_vector
                }
                results_n_k[p_index, n_index, k_index, 2, 0] = approx_min_eig_val_scipy_lse
                results_n_k[p_index, n_index, k_index, 2, 1] = k_max_score_lse
                results_n_k[p_index, n_index, k_index, 2, 2] = break_ties_score_lse
                results_n_k[p_index, n_index, k_index, 2, 3] = madow_score_lse
                results_n_k[p_index, n_index, k_index, 2, 4] = exec_time
                # results_n_k[n_index, k_index, 2, 5] = 0

                print("\n" + "#" * 70)

                # Run Frank-Wolfe Optimization only for num_matrices = 10 and 100
                # if num_matrices in [10, 100, 300]:
                #     n_iters = 10000 if num_matrices == 10 else 1000
                #     print("\nRunning Frank-Wolfe Optimization")
                #     (final_solution, min_eig_val_unrounded, i), exec_time = time_function(
                #         frank_wolfe_optimization,
                #         inf_mats=inf_mats,
                #         prior=H0,
                #         n_iters=n_iters,
                #         selection_init=selection_init,
                #         k=k,
                #         num_poses=num_poses,
                #         A=A,
                #         b=b
                #     )
                #
                #     print("Frank-Wolfe Optimization Results (selection vector):", final_solution)
                #     print("Frank-Wolfe Optimization Best Score:", min_eig_val_unrounded)
                #     k_max_sol_fw = roundsolution(final_solution, k)
                #     print("Frank-Wolfe Optimization Results (K - max):", np.nonzero(k_max_sol_fw))
                #     k_max_score_fw = fim.find_min_eig_pair(inf_mats, np.array(k_max_sol_fw), H0, num_poses)[0]
                #     print("Frank-Wolfe  (K - max) score:", k_max_score_fw)
                #     breakties_sol_fw = roundsolution_breakties(final_solution, k, inf_mats, H0)
                #     print("Frank-Wolfe Optimization Results (Breakties):", np.nonzero(breakties_sol_fw))
                #     break_ties_score_fw = fim.find_min_eig_pair(inf_mats, np.array(breakties_sol_fw), H0, num_poses)[0]
                #     print("Frank-WolfeBreakties score:", break_ties_score_fw)
                #     madow_sol_fw = roundsolution_madow(final_solution, k)
                #     print("Frank-Wolfe Optimization Results (Madow):", np.nonzero(madow_sol_fw))
                #     madow_score_fw = fim.find_min_eig_pair(inf_mats, np.array(madow_sol_fw), H0, num_poses)[0]
                #     print("Frank-Wolfe Madow score:", madow_score_fw)
                #     selection_vector = final_solution.tolist()
                #
                #
                #     # Store results
                #     test_case_result['results']['Frank-Wolfe Optimization'] = {
                #         'execution_time': exec_time,
                #         'best_score': min_eig_val_unrounded,
                #         'selection_vector': selection_vector
                #     }
                #     results_n_k[p_index, n_index, k_index, 3, 0] = min_eig_val_unrounded
                #     results_n_k[p_index, n_index, k_index, 3, 1] = k_max_score_fw
                #     results_n_k[p_index, n_index, k_index, 3, 2] = break_ties_score_fw
                #     results_n_k[p_index, n_index, k_index, 3, 3] = madow_score_fw
                #     results_n_k[p_index, n_index, k_index, 3, 4] = exec_time
                #     # results_n_k[n_index, k_index, 2, 5] = 0
                #
                #     print("\n" + "#" * 70)
                # else:
                #     # print("Frank-Wolfe Optimization skipped for num_matrices =", num_matrices)
                #     # Indicate that Frank-Wolfe was skipped
                #     test_case_result['results']['Frank-Wolfe Optimization'] = {
                #         'skipped': True
                #     }
                #
                # # Append the test case result to the results list
                # results.append(test_case_result)
        #save the numpy array for specific number of poses (matrix size)
    np.save("../results/results_p{}_n{}_k{}.pkl".format(len(num_poses_list), len(num_matrices_list), len(k_values)), results_n_k)

if __name__ == "__main__":
    run_tests()