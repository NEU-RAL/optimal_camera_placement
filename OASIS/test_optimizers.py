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
    scipy_minimize_lse_d,
    # roundsolution,               # Removed rounding functions
    # roundsolution_breakties,
    # roundsolution_madow,
    gurobi_branch_and_cut
)
import FIM as fim
import pandas as pd
import os
import datetime
import utilities

# Define Metric Enum for selection
class Metric(Enum):
    LOGDET = 1
    MIN_EIG = 2
    MSE = 3

def generate_random_fim(num_matrices, matrix_size, density=0.1, min_eigenvalue=5.0, random_state=40):
    """
    Generates a list of random sparse symmetric positive-definite matrices.
    """
    np.random.seed(random_state)
    inf_mats = []

    for i in range(num_matrices):
        # Generate a random sparse matrix B with standard normal distributed non-zero entries
        B = sparse_random(matrix_size, matrix_size, density=density, format='csr', 
                          data_rvs=np.random.randn)
        # Compute A = B^T + B to ensure positive semi-definiteness
        #A = B.transpose().dot(B)
        A = B.transpose() + B

        # Add c*I to make it positive definite
        c = min_eigenvalue
        A += diags([c] * matrix_size, format='csr')
        assert utilities.check_symmetric(A)
        #print(A.getnnz()/np.prod(A.shape))
        inf_mats.append(A)

    return inf_mats

def generate_fim_with_identity(num_matrices, matrix_size, k, density=0.1, min_eigenvalue=2.0, random_state=40):
    """
    Generates a list of random sparse symmetric positive-definite matrices with most being identity matrices.
    """
    np.random.seed(random_state)
    inf_mats = []

    # Generate `num_matrices - k` identity matrices
    identity_matrix = diags([min_eigenvalue] * matrix_size, format='csr')
    for _ in range(num_matrices - k):
        inf_mats.append(identity_matrix)

    # Generate `k` random symmetric positive-definite matrices
    for _ in range(k):
        # Generate a random sparse matrix B with standard normal distributed non-zero entries
        B = sparse_random(matrix_size, matrix_size, density=density, format='csr', 
                          data_rvs=np.random.randn)

        # Compute A = B^T + B to ensure positive semi-definiteness
        # A = B.transpose().dot(B)
        A = B.transpose() + B

        # Add c*I to make it positive definite
        c = min_eigenvalue
        A += diags([c] * matrix_size, format='csr')
        assert utilities.check_symmetric(A)
        # print(A.getnnz()/np.prod(A.shape))

        inf_mats.append(A)

    # Shuffle the matrices to ensure the random ones are not clustered at the end
    np.random.shuffle(inf_mats)

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
    # Define pose dimension and base measurement dimension
    pose_dim = 6  # Each pose has 6 variables

    # Define test parameters
    density = 0.1        # 10% non-zero elements
    min_eigenvalue = 5.0  # Minimum eigenvalue

    # Test configurations
    k_values = [5, 10]
    num_matrices_list = [10, 50]
    num_poses_list = [15]  # Chosen to achieve matrix size of 100, 500, 1000

    # Initialize a results array
    # Dimensions: (num_poses, num_matrices, k_values, optimizers, metrics)
    # Metrics indices: 0 - best_score, 1 - exec_time
    results_n_k = np.zeros(
        (len(num_poses_list), len(num_matrices_list), len(k_values), 4, 2)
    )

    # Mapping of optimizer names to their corresponding indices in the results_n_k array
    optimizer_indices = {
        "Greedy Selection": 0,
        "Scipy Minimize LSE": 1,
        "Scipy Minimize LSE dense": 2,
        "Frank-Wolfe": 3,
        # "Gurobi Branch and Cut": 3  # Uncomment if using Gurobi
    }

    # Initialize a list to store all test case results
    all_test_case_results = []

    # Define directories for saving results
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    results_dir = os.path.join("results", timestamp)
    os.makedirs(results_dir, exist_ok=True)

    # Directory for intermediate results
    intermediate_dir = os.path.join(results_dir, "intermediate_results")
    os.makedirs(intermediate_dir, exist_ok=True)

    # Directory for selection vectors
    selection_vectors_dir = os.path.join(results_dir, "selection_vectors")
    os.makedirs(selection_vectors_dir, exist_ok=True)

    # Define batch size for periodic saving
    save_interval = 100  # Save every 100 test cases
    test_case_counter = 0  # Counter for test cases

    # Iterate over configurations
    for p_index, num_poses in enumerate(num_poses_list):
        for n_index, num_matrices in enumerate(num_matrices_list):
            for k_index, k in enumerate(k_values):
                # Calculate matrix size
                measurement_dim = 10
                matrix_size = measurement_dim + num_poses * pose_dim

                # Check feasibility of the configuration
                if k > num_matrices:
                    print(f"Skipping test for k={k}, num_matrices={num_matrices} (k > num_matrices)")
                    continue

                # Constraint setup for optimization methods
                A_constraint = csr_matrix(np.ones((1, num_matrices)))  # Create A as a sparse matrix
                b_constraint = np.array([k])

                # Generate synthetic data
                inf_mats = generate_random_fim(
                    num_matrices=num_matrices,
                    matrix_size=matrix_size,
                    density=density,
                    min_eigenvalue=min_eigenvalue
                )
                H0 = diags([min_eigenvalue] * matrix_size, format='csr')
                selection_init = np.ones(num_matrices, dtype=np.float16) / num_matrices

                # Initialize a dictionary to store results for the current test case
                test_case_result = {
                    'configuration': {
                        'num_poses': num_poses,
                        'num_matrices': num_matrices,
                        'k': k,
                        'matrix_size': matrix_size,
                        'density': density,
                        'min_eigenvalue': min_eigenvalue
                    },
                    'results': {}
                }

                # Test each optimizer
                for optimizer_name, optimizer_func, requires_constraints, needs_rounding in [
                    ("Greedy Selection", greedy_selection, False, False),
                    ("Scipy Minimize LSE", scipy_minimize_lse, True, False),
                    ("Scipy Minimize LSE dense", scipy_minimize_lse_d, True, False),
                    ("Frank-Wolfe", frank_wolfe_optimization, True, False),
                    # ("Gurobi Branch and Cut", gurobi_branch_and_cut, True, False)  # Uncomment if Gurobi is used
                ]:
                    try:
                        print(f"\nRunning {optimizer_name} for Matrix Size: {matrix_size}, k: {k}, Decision Variables: {num_matrices}")
                        
                        # Pass only required arguments based on whether constraints are needed
                        if requires_constraints:
                            result, exec_time = time_function(
                                optimizer_func,
                                inf_mats=inf_mats,
                                H0=H0,
                                selection_init=selection_init,
                                num_poses=num_poses,
                                A=A_constraint,
                                b=b_constraint
                            )
                        else:
                            result, exec_time = time_function(
                                optimizer_func,
                                inf_mats=inf_mats,
                                prior=H0,
                                Nc=k,
                                metric=Metric.MIN_EIG,
                                num_runs=1,
                                num_poses=num_poses
                            )

                        # Process results
                        if optimizer_name == "Greedy Selection":
                            selection_vec, best_score, _ = result

                            # Print results
                            print(f"{optimizer_name} Results (selection vector indices):", np.nonzero(selection_vec)[0])
                            print(f"{optimizer_name} Best Score:", best_score)

                            # Save selection vector to a separate file
                            selection_filename = f"p{p_index}_n{n_index}_k{k_index}_{optimizer_name.replace(' ', '_')}.npy"
                            selection_path = os.path.join(selection_vectors_dir, selection_filename)
                            np.save(selection_path, selection_vec)

                            # Store in test_case_result
                            test_case_result['results'][optimizer_name] = {
                                'execution_time': exec_time,
                                'best_score': float(best_score),
                                'selection_vector_path': selection_path
                            }

                            # Store in results_n_k
                            optimizer_idx = optimizer_indices[optimizer_name]
                            results_n_k[p_index, n_index, k_index, optimizer_idx, 0] = float(best_score)
                            results_n_k[p_index, n_index, k_index, optimizer_idx, 1] = exec_time

                        else:
                            selection_vec = result[0]
                            best_score = result[1]

                            # Print results
                            print(f"{optimizer_name} Results (selection vector):", selection_vec)
                            print(f"{optimizer_name} Best Score:", best_score)

                            # Save the selection vector
                            selection_filename = f"p{p_index}_n{n_index}_k{k_index}_{optimizer_name.replace(' ', '_')}.npy"
                            selection_path = os.path.join(selection_vectors_dir, selection_filename)
                            np.save(selection_path, selection_vec)

                            # Store in test_case_result
                            test_case_result['results'][optimizer_name] = {
                                'execution_time': exec_time,
                                'best_score': float(best_score),
                                'selection_vector_path': selection_path
                            }

                            # Store in results_n_k
                            optimizer_idx = optimizer_indices[optimizer_name]
                            results_n_k[p_index, n_index, k_index, optimizer_idx, 0] = float(best_score)
                            results_n_k[p_index, n_index, k_index, optimizer_idx, 1] = exec_time

                    except Exception as e:
                        print(f"{optimizer_name} failed: {e}")
                        # Optionally, store the exception information
                        test_case_result['results'][optimizer_name] = {
                            'error': str(e)
                        }
                        # Assign np.nan to relevant entries in results_n_k
                        optimizer_idx = optimizer_indices.get(optimizer_name)
                        if optimizer_idx is not None:
                            results_n_k[p_index, n_index, k_index, optimizer_idx, :] = np.nan

                # Append the current test case results to the list
                all_test_case_results.append(test_case_result)
                test_case_counter += 1

                # Check if it's time to save intermediate results
                if test_case_counter % save_interval == 0:
                    print(f"\nSaving intermediate results after {test_case_counter} test cases...")
                    # Define filenames with batch number
                    batch_number = test_case_counter // save_interval
                    intermediate_results_path = os.path.join(intermediate_dir, f"results_batch_{batch_number}.yaml")
                    intermediate_results_n_k_path = os.path.join(intermediate_dir, f"results_n_k_batch_{batch_number}.npy")
                    
                    # Save the accumulated test case results
                    with open(intermediate_results_path, "w") as f:
                        yaml.dump(all_test_case_results, f)
                    
                    # Save the current state of results_n_k
                    np.save(intermediate_results_n_k_path, results_n_k)
                    
                    # Reset the accumulated results
                    all_test_case_results = []
                    results_n_k.fill(0)  # Reset the NumPy array

    # After all test cases are processed, save any remaining results
    if all_test_case_results:
        print(f"\nSaving final batch of {test_case_counter % save_interval} test cases...")
        batch_number = (test_case_counter // save_interval) + 1
        intermediate_results_path = os.path.join(intermediate_dir, f"results_batch_{batch_number}.yaml")
        intermediate_results_n_k_path = os.path.join(intermediate_dir, f"results_n_k_batch_{batch_number}.npy")
        
        # Save the accumulated test case results
        with open(intermediate_results_path, "w") as f:
            yaml.dump(all_test_case_results, f)
        
        # Save the current state of results_n_k
        np.save(intermediate_results_n_k_path, results_n_k)

    # Save all test case results as a YAML file for easier readability
    with open(os.path.join(results_dir, "all_test_case_results.yaml"), "w") as f:
        yaml.dump(all_test_case_results, f)

    # Save experiment configuration
    exp_config = {
        "pose_dim": pose_dim,
        "num_poses_list": num_poses_list,
        "num_matrices_list": num_matrices_list,
        "k_values": k_values,
        "algos": ["greedy", "scipy-lse", "frank-wolfe"],  # Still working on Gurobi
        "metrics": ["best_score", "exec_time"]
    }
    with open(os.path.join(results_dir, "exp_config.yaml"), "w") as f:
        yaml.dump(exp_config, f)


if __name__ == "__main__":
    run_tests()
