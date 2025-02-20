import numpy as np
import time
import yaml
from enum import Enum
from scipy.sparse import random as sparse_random
from scipy.sparse import csr_matrix, diags
from scipy.stats import uniform

from OASIS import utilities
from OASIS.optimizations import (
    greedy_selection,
    frank_wolfe_optimization,
    min_eig_obj,
    min_eig_grad, min_eig_grad_noschur
)

import os
import datetime

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

def generate_fim_simple_identity(num_matrices, matrix_size, min_eigenvalue=2.0):
    """
    Generates a set of matrices with increasing min eigen values
    """

    inf_mats = []

    # Generate `num_matrices - k` identity matrices
    identity_matrix = diags([min_eigenvalue] * matrix_size, format='csr')
    for i in range(num_matrices):
        inf_mats.append(identity_matrix)
        identity_matrix = identity_matrix * 1.2
        #print(identity_matrix.todense())


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
if __name__ == "__main__":
    # Define pose dimension and base measurement dimension
    pose_dim = 6  # Each pose has 6 variables

    # Define test parameters
    density = 0.1  # 10% non-zero elements
    min_eigenvalue = 5.0  # Minimum eigenvalue

    # Test configurations
    k_values = [3]
    num_matrices_list = [ 10]
    num_poses_list = [15]  # Chosen to achieve matrix size of 100, 500, 1000

    # Initialize a results array
    # Dimensions: (num_poses, num_matrices, k_values, optimizers, metrics)
    # Metrics indices: 0 - best_score, 1 - exec_time

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
                print(f"\nRunning Frank-wolfe for Matrix Size: {matrix_size}, k: {k}, Decision Variables: {num_matrices}")
                # Generate synthetic data
                inf_mats = generate_random_fim(
                    num_matrices=num_matrices,
                    matrix_size=matrix_size,
                    density=density,
                    min_eigenvalue=min_eigenvalue
                )
                # inf_mats = generate_fim_with_identity(
                #     num_matrices=num_matrices,
                #     matrix_size=matrix_size,
                #     k=k,
                #     density=density,
                #     min_eigenvalue=min_eigenvalue
                # )
                # inf_mats=generate_fim_simple_identity(num_matrices, matrix_size)
                H0 = diags([min_eigenvalue] * matrix_size, format='csr')
                selection_init = np.ones(num_matrices, dtype=np.float16) / num_matrices

                selection_cur, min_eig_val, iteration, fw_log = frank_wolfe_optimization(
                    min_eig_obj,
                    min_eig_grad_noschur,
                    selection_init,
                    A_constraint,
                    b_constraint,
                    inf_mats,
                    H0,
                    num_poses
                )

                print(f"Results (selection vector):", selection_cur)
                print(f"Best Score:", min_eig_val)
