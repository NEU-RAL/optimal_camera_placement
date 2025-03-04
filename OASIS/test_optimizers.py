import numpy as np
import time
import os
import datetime
import matplotlib.pyplot as plt
from enum import Enum
from scipy.sparse import csr_matrix, diags, random

# Import your optimization functions (assumed to be defined in optimizations.py)
from optimizations import (
    greedy_selection,
    cvxpy_minimize_lse,
    multiple_randomized_rounds,
    compute_variance_info,
    evaluate_solution
)
import utilities  # assumed to provide utilities.check_symmetric

# Define Metric Enum for selection (if used in greedy_selection)
class Metric(Enum):
    LOGDET = 1
    MIN_EIG = 2
    MSE = 3

def generate_fim(num_matrices, matrix_size, generator_type="random", density=0.1,
                 min_eigenvalue=2.0, random_state=40, **kwargs):
    """
    Generates a list of random sparse symmetric positive-definite matrices
    according to the chosen generator type.

    Parameters
    ----------
    num_matrices : int
        Total number of matrices to generate.
    matrix_size : int
        Size of each matrix.
    generator_type : str, optional
        Type of generation. Options:
         - "incremental": Uses sparse random matrices with min eigenvalues increasing by a fixed increment.
         - "random": Generates matrices with completely random eigenvalues, chosen uniformly from [low, high].
         - "two_group": Generates matrices in two groups: a fraction have min eigenvalue = group1_min and the rest have min eigenvalue = group2_min.
    density : float, optional
        Density used for sparse generation (only for "incremental").
    min_eigenvalue : float, optional
        Starting minimum eigenvalue (for "incremental").
    random_state : int, optional
        Seed for reproducibility.
    **kwargs : dict
        Additional parameters for the "random" and "two_group" generators.

    Returns
    -------
    inf_mats : list of sparse matrices
        A shuffled list of generated matrices (in CSR format).
    """
    np.random.seed(random_state)
    inf_mats = []

    if generator_type == "incremental":
        current_min_eigenvalue = min_eigenvalue
        for i in range(num_matrices):
            B = random(matrix_size, matrix_size, density=density, format='csr', data_rvs=np.random.randn)
            A = B.transpose().dot(B)
            A += diags([current_min_eigenvalue] * matrix_size, format='csr')
            assert utilities.check_symmetric(A), "Matrix not symmetric"
            inf_mats.append(A)
            current_min_eigenvalue += 2

    elif generator_type == "random":
        low = kwargs.get("low", 0.1)
        high = kwargs.get("high", 20.0)
        from scipy.sparse import csr_matrix
        for i in range(num_matrices):
            Q, _ = np.linalg.qr(np.random.randn(matrix_size, matrix_size))
            eigvals = np.random.uniform(low, high, size=matrix_size)
            D = np.diag(eigvals)
            A_dense = Q @ D @ Q.T
            A_dense = (A_dense + A_dense.T) / 2
            A = csr_matrix(A_dense)
            inf_mats.append(A)

    elif generator_type == "two_group":
        fraction_group1 = kwargs.get("fraction_group1", 0.5)
        group1_min = kwargs.get("group1_min", 1.0)
        group2_min = kwargs.get("group2_min", 10.0)
        upper_bound1 = kwargs.get("upper_bound1", 20.0)
        upper_bound2 = kwargs.get("upper_bound2", 30.0)
        num_group1 = int(np.round(num_matrices * fraction_group1))
        num_group2 = num_matrices - num_group1

        def generate_matrix_with_min_eig(matrix_size, min_eig, upper_bound):
            Q, _ = np.linalg.qr(np.random.randn(matrix_size, matrix_size))
            eigvals = np.random.uniform(min_eig, upper_bound, size=matrix_size)
            eigvals[0] = min_eig
            eigvals = np.sort(eigvals)
            D = np.diag(eigvals)
            A_dense = Q @ D @ Q.T
            A_dense = (A_dense + A_dense.T) / 2
            from scipy.sparse import csr_matrix
            return csr_matrix(A_dense)

        for i in range(num_group1):
            A_mat = generate_matrix_with_min_eig(matrix_size, group1_min, upper_bound1)
            inf_mats.append(A_mat)
        for i in range(num_group2):
            A_mat = generate_matrix_with_min_eig(matrix_size, group2_min, upper_bound2)
            inf_mats.append(A_mat)
    else:
        raise ValueError("Unknown generator type: {}".format(generator_type))

    # Shuffle the list reproducibly
    perm = np.random.permutation(len(inf_mats))
    inf_mats = [inf_mats[i] for i in perm]
    return inf_mats

def time_function(func, *args, **kwargs):
    import time
    start_time = time.time()
    result = func(*args, **kwargs)
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"{func.__name__} execution time: {execution_time:.4f} seconds")
    return result, execution_time

def convert_numpy(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    return obj

def run_budget_tests():
    """
    For a dummy "at most K cameras" constraint (i.e. sum(x) <= K),
    we run tests for increasing values of K (as percentages).
    For each K:
      1) Solve the CVXPY problem -> fractional solution + objective
      2) compute_variance_info to get (Expectation, Variance, ratio)
      3) multiple_randomized_rounds from the fractional solution
      4) run greedy_selection
      5) store + plot bar charts
         where x-axis label is "Exp:xx.x\nVar:yy.y"
    """
    # Configuration
    pose_dim = 6
    density = 0.1
    min_eigenvalue = 2
    num_matrices = 100
    num_poses = 10
    k_percentages = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.45, 0.5, 0.55, 0.65, 0.70]
    
    # Arrays to store results (for plotting)
    exps = []       # expectation from variance_info
    vars_ = []      # variance from variance_info
    frac_objs = []  # fractional objective
    random_objs = []  
    greedy_objs = []
    
    for perc in k_percentages:
        k = int(round(num_matrices * perc))
        print("\n====================================")
        print(f"Testing with K = {k} (K = {perc*100:.0f}% of {num_matrices})")
        print("====================================\n")
        
        # Single-row constraint: sum(x) <= k
        A_constraint = csr_matrix(np.ones((1, num_matrices)))
        b_constraint = np.array([k])
        
        # Matrix size
        measurement_dim = num_poses
        matrix_size = measurement_dim + num_poses * pose_dim
        
        # Generate the candidate FIMs
        inf_mats = generate_fim(num_matrices=num_matrices,
                                matrix_size=matrix_size,
                                generator_type="two_group",
                                density=density,
                                min_eigenvalue=min_eigenvalue,
                                random_state=42)
        
        # Prior matrix
        H0 = diags([min_eigenvalue]*matrix_size, format='csr')
        selection_init = np.ones(num_matrices, dtype=np.float16)/num_matrices
        
        # 1) Solve CVXPY problem
        (frac_solution, fractional_obj), _ = time_function(
            cvxpy_minimize_lse,
            inf_mats=inf_mats,
            H0=H0,
            selection_init=selection_init,
            num_poses=num_poses,
            A=A_constraint,
            b=b_constraint
        )
        print("Fractional solution:", frac_solution)
        print("Fractional objective:", fractional_obj)
        frac_objs.append(fractional_obj)
        
        # 2) compute_variance_info
        #   This returns a list of (E, Var, ratio) for each row of A.
        #   Here, p=1, so we only have variance_info[0].
        variance_data = compute_variance_info(A_constraint.toarray(), frac_solution)
        E_val, Var_val, ratio_val = variance_data[0]  # single row
        exps.append(E_val)
        vars_.append(Var_val)
        
        # 3) multiple_randomized_rounds from fractional solution
        def objective_func(binary_vec):
            return evaluate_solution(inf_mats, H0, binary_vec)
        best_rand_y, best_rand_obj, feas_rate = multiple_randomized_rounds(
            frac_solution, A_constraint, b_constraint, objective_func, n_samples=1000
        )
        print("Randomized best obj:", best_rand_obj, "(feas rate:", feas_rate, ")")
        random_objs.append(best_rand_obj)
        
        # 4) run greedy_selection
        from optimizations import greedy_selection
        g_vec, g_obj, _ = greedy_selection(
            inf_mats=inf_mats,
            prior=H0,
            Nc=k,
            metric=Metric.MIN_EIG,
            num_runs=1,
            num_poses=num_poses
        )
        print("Greedy objective:", g_obj)
        greedy_objs.append(g_obj)
        
    # Now plot results:
    x_positions = np.arange(len(exps))
    width = 0.25
    
    # Build the x-axis labels from exps, vars_
    x_labels = [f"Exp:{exps[i]:.1f}\nVar:{vars_[i]:.1f}" for i in range(len(exps))]
    
    fig, ax = plt.subplots(1,2, figsize=(14,6))

    # Plot objectives
    pos_frac = x_positions - width
    pos_rand = x_positions
    pos_greedy = x_positions + width

    ax[0].bar(pos_frac, frac_objs, width, label="Fractional", color='skyblue')
    ax[0].bar(pos_rand, random_objs, width, label="Random Round", color='lightgreen')
    ax[0].bar(pos_greedy, greedy_objs, width, label="Greedy", color='salmon')
    ax[0].set_xticks(x_positions)
    ax[0].set_xticklabels(x_labels)
    ax[0].legend()
    ax[0].set_xlabel("Expectation/Variance")
    ax[0].set_ylabel("Objective Score")
    ax[0].set_title("Objective Scores")

    # Plot gap percentages
    frac_array = np.array(frac_objs)
    rand_gaps = 100*(frac_array - np.array(random_objs))/frac_array
    greed_gaps = 100*(frac_array - np.array(greedy_objs))/frac_array

    ax[1].bar(pos_rand, rand_gaps, width, label="Random Gap (%)", color='lightgreen')
    ax[1].bar(pos_greedy, greed_gaps, width, label="Greedy Gap (%)", color='salmon')
    ax[1].set_xticks(x_positions)
    ax[1].set_xticklabels(x_labels)
    ax[1].legend()
    ax[1].set_xlabel("Expectation/Variance")
    ax[1].set_ylabel("Gap (%)")
    ax[1].set_title("Optimality Gaps")
    
    plt.tight_layout()
    plt.show()

if __name__=="__main__":
    run_budget_tests()

# import numpy as np
# import time
# import os
# import datetime
# import matplotlib.pyplot as plt
# from enum import Enum
# from scipy.sparse import csr_matrix, diags, random as sparse_random

# # Import your optimization functions (assumed to be defined in optimizations.py)
# from optimizations import (
#     cvxpy_minimize_lse,
#     multiple_randomized_rounds,
#     compute_variance_info,
#     evaluate_solution
# )
# import utilities  # assumed to provide utilities.check_symmetric

# # Define Metric Enum for selection (if used elsewhere)
# class Metric(Enum):
#     LOGDET = 1
#     MIN_EIG = 2
#     MSE = 3

# def generate_fim(num_matrices, matrix_size, generator_type="random", density=0.1,
#                  min_eigenvalue=2.0, random_state=40, **kwargs):
#     """
#     Generates a list of random sparse symmetric positive-definite matrices
#     according to the chosen generator type.
#     """
#     np.random.seed(random_state)
#     inf_mats = []

#     if generator_type == "incremental":
#         current_min_eigenvalue = min_eigenvalue
#         for i in range(num_matrices):
#             B = sparse_random(matrix_size, matrix_size, density=density, format='csr', data_rvs=np.random.randn)
#             A = B.transpose().dot(B)
#             A += diags([current_min_eigenvalue] * matrix_size, format='csr')
#             assert utilities.check_symmetric(A), "Matrix not symmetric"
#             inf_mats.append(A)
#             current_min_eigenvalue += 2

#     elif generator_type == "random":
#         low = kwargs.get("low", 0.1)
#         high = kwargs.get("high", 20.0)
#         from scipy.sparse import csr_matrix
#         for i in range(num_matrices):
#             Q, _ = np.linalg.qr(np.random.randn(matrix_size, matrix_size))
#             eigvals = np.random.uniform(low, high, size=matrix_size)
#             D = np.diag(eigvals)
#             A_dense = Q @ D @ Q.T
#             A_dense = (A_dense + A_dense.T) / 2
#             A = csr_matrix(A_dense)
#             inf_mats.append(A)

#     elif generator_type == "two_group":
#         fraction_group1 = kwargs.get("fraction_group1", 0.5)
#         group1_min = kwargs.get("group1_min", 1.0)
#         group2_min = kwargs.get("group2_min", 10.0)
#         upper_bound1 = kwargs.get("upper_bound1", 20.0)
#         upper_bound2 = kwargs.get("upper_bound2", 30.0)
#         num_group1 = int(np.round(num_matrices * fraction_group1))
#         num_group2 = num_matrices - num_group1

#         def generate_matrix_with_min_eig(matrix_size, min_eig, upper_bound):
#             Q, _ = np.linalg.qr(np.random.randn(matrix_size, matrix_size))
#             eigvals = np.random.uniform(min_eig, upper_bound, size=matrix_size)
#             eigvals[0] = min_eig
#             eigvals = np.sort(eigvals)
#             D = np.diag(eigvals)
#             A_dense = Q @ D @ Q.T
#             A_dense = (A_dense + A_dense.T) / 2
#             from scipy.sparse import csr_matrix
#             return csr_matrix(A_dense)

#         for i in range(num_group1):
#             A_mat = generate_matrix_with_min_eig(matrix_size, group1_min, upper_bound1)
#             inf_mats.append(A_mat)
#         for i in range(num_group2):
#             A_mat = generate_matrix_with_min_eig(matrix_size, group2_min, upper_bound2)
#             inf_mats.append(A_mat)
#     else:
#         raise ValueError("Unknown generator type: {}".format(generator_type))

#     # Shuffle the list reproducibly
#     perm = np.random.permutation(len(inf_mats))
#     inf_mats = [inf_mats[i] for i in perm]
#     return inf_mats

# def time_function(func, *args, **kwargs):
#     start_time = time.time()
#     result = func(*args, **kwargs)
#     end_time = time.time()
#     execution_time = end_time - start_time
#     print(f"{func.__name__} execution time: {execution_time:.4f} seconds")
#     return result, execution_time

# def convert_numpy(obj):
#     if isinstance(obj, np.ndarray):
#         return obj.tolist()
#     elif isinstance(obj, (np.float32, np.float64)):
#         return float(obj)
#     elif isinstance(obj, (np.int32, np.int64)):
#         return int(obj)
#     return obj

# ###############################################################################
# # Iterative Selection Procedure (for cardinality constraint only)
# ###############################################################################
# def iterative_selection(inf_mats, H0, num_poses, final_target):
#     """
#     Implements an iterative selection scheme:
#       - Start with the full candidate set (all indices).
#       - In the first iteration, set k_temp to 60% of the current candidate set size.
#       - Solve the CVXPY problem (with constraint sum(x) <= k_temp) on the current candidate set,
#         then perform randomized rounding.
#       - Prune: remove candidates that are rounded to 0.
#       - In subsequent iterations, set k_temp to 50% of the current candidate set size.
#       - Repeat until the candidate set size is approximately 2 * final_target.
    
#     Parameters:
#       inf_mats : list
#          List of candidate FIM matrices.
#       H0 : sparse matrix
#          The prior matrix.
#       num_poses : int
#          Number of poses (passed to CVXPY optimization).
#       final_target : int
#          Final desired number of selections (e.g., 10% of original candidates).
         
#     Returns:
#       candidate_set : list
#          The final list of candidate indices.
#       final_frac_solution : numpy array
#          The final fractional solution on the reduced candidate set.
#       final_binary_solution : numpy array
#          The final binary solution (from randomized rounding) on the reduced candidate set.
#     """
#     candidate_set = list(range(len(inf_mats)))
#     iteration = 0
#     current_frac_solution = None

#     while len(candidate_set) > 2 * final_target:
#         current_size = len(candidate_set)
#         if iteration == 0:
#             k_temp = int(np.ceil(0.6 * current_size))
#         else:
#             k_temp = int(np.ceil(0.5 * current_size))
#         print(f"Iteration {iteration}: Candidate set size = {current_size}, using k_temp = {k_temp}")
        
#         # Build the "at most k" constraint for the current candidate set:
#         A_curr = csr_matrix(np.ones((1, current_size)))
#         b_curr = np.array([k_temp])
        
#         # Current selection initialization.
#         selection_init = np.ones(current_size, dtype=np.float16) / current_size
        
#         # Extract candidate FIMs.
#         current_inf_mats = [inf_mats[i] for i in candidate_set]
        
#         # Solve CVXPY on the current candidate set.
#         (frac_sol_current, obj_current), cvx_time_current = time_function(
#             cvxpy_minimize_lse,
#             inf_mats=current_inf_mats,
#             H0=H0,
#             selection_init=selection_init,
#             num_poses=num_poses,
#             A=A_curr,
#             b=b_curr
#         )
#         print("Fractional solution for current candidate set:", frac_sol_current)
        
#         # Use randomized rounding on the current fractional solution.
#         def obj_func(binary_vec):
#             return evaluate_solution(current_inf_mats, H0, binary_vec)
#         best_rand, rand_obj, feas_rate = multiple_randomized_rounds(
#             frac_sol_current, A_curr, b_curr, obj_func, n_samples=500
#         )
#         print("Randomized rounding (current candidate set):", best_rand)
        
#         # Prune: keep only indices where the binary solution equals 1.
#         new_candidate_set = [candidate_set[i] for i in range(len(best_rand)) if best_rand[i] == 1]
#         print(f"Pruning: candidate set size reduced from {current_size} to {len(new_candidate_set)}")
#         candidate_set = new_candidate_set
#         current_frac_solution = frac_sol_current  # update last fractional solution (if needed)
#         iteration += 1
        
#     # Final optimization on the reduced candidate set.
#     current_size = len(candidate_set)
#     print(f"Final candidate set size: {current_size}")
#     A_final = csr_matrix(np.ones((1, current_size)))
#     b_final = np.array([int(round(0.5 * current_size))])  # final constraint: ~50% of remaining
#     selection_init = np.ones(current_size, dtype=np.float16) / current_size
#     current_inf_mats = [inf_mats[i] for i in candidate_set]
#     (final_frac_solution, final_obj), _ = time_function(
#         cvxpy_minimize_lse,
#         inf_mats=current_inf_mats,
#         H0=H0,
#         selection_init=selection_init,
#         num_poses=num_poses,
#         A=A_final,
#         b=b_final
#     )
#     def final_obj_func(binary_vec):
#         return evaluate_solution(current_inf_mats, H0, binary_vec)
#     final_binary_solution, final_rand_obj, final_feas_rate = multiple_randomized_rounds(
#         final_frac_solution, A_final, b_final, final_obj_func, n_samples=500
#     )
#     print("Final binary solution from iterative selection:", final_binary_solution)
#     print("Final objective (iterative method):", evaluate_solution(current_inf_mats, H0, final_binary_solution))
#     return candidate_set, final_frac_solution, final_binary_solution

# ###############################################################################
# # Run Iterative Selection, Full Candidate Rounding, and Compare
# ###############################################################################
# def run_budget_tests_iterative():
#     """
#     Implements the iterative selection scheme for a pure cardinality constraint.
#     Final desired selection (final_target) is set to 10% of the original candidate set.
    
#     After obtaining the final solution from the iterative method, we also:
#       - Compute a "global" fractional solution on the full candidate set with the original
#         desired cardinality constraint.
#       - Apply randomized rounding to the full candidate set fractional solution.
#       - Compute the optimality gaps (percentage differences relative to the full fractional solution).
    
#     The final optimality gap for our iterative method is:
#          Gap_iter (%) = 100 * (f_frac_full - f_iterative) / f_frac_full.
#     Similarly, for the full candidate rounding we compute:
#          Gap_full (%) = 100 * (f_frac_full - f_full_random) / f_frac_full.
#     """
#     # Configuration
#     pose_dim = 6
#     density = 0.1
#     min_eigenvalue = 2
#     num_matrices = 100  # Original candidate set size
#     num_poses = 10

#     # Final desired selection: 10% of original candidates.
#     final_target = int(round(0.10 * num_matrices))
#     print(f"Final target (K_final) = {final_target}")

#     # Matrix size calculation.
#     matrix_size = num_poses + num_poses * pose_dim

#     # Generate candidate FIMs.
#     inf_mats = generate_fim(num_matrices=num_matrices, matrix_size=matrix_size, 
#                             generator_type="random", density=density, min_eigenvalue=min_eigenvalue, random_state=42)
    
#     # Create H0 as the prior matrix.
#     H0 = diags([min_eigenvalue] * matrix_size, format='csr')

#     # --- Run Iterative Selection ---
#     final_candidate_set, final_frac_solution, final_binary_solution = iterative_selection(
#         inf_mats, H0, num_poses, final_target
#     )
#     print("\nFinal selected candidate indices from iterative selection:", final_candidate_set)
#     f_iterative = evaluate_solution([inf_mats[i] for i in final_candidate_set], H0, final_binary_solution)
#     print("Final objective (iterative method):", f_iterative)

#     # --- Global (Full Candidate Set) Fractional Solution ---
#     A_orig = csr_matrix(np.ones((1, num_matrices)))
#     b_orig = np.array([final_target])
#     selection_init_full = np.ones(num_matrices, dtype=np.float16) / num_matrices
#     (frac_solution_full, frac_obj_full), _ = time_function(
#         cvxpy_minimize_lse,
#         inf_mats=inf_mats,
#         H0=H0,
#         selection_init=selection_init_full,
#         num_poses=num_poses,
#         A=A_orig,
#         b=b_orig
#     )
#     print("Fractional solution on full candidate set:", frac_solution_full)
#     print("Fractional objective (full candidate set):", frac_obj_full)
    
#     # --- Global Randomized Rounding on Full Candidate Set ---
#     def full_obj_func(binary_vec):
#         return evaluate_solution(inf_mats, H0, binary_vec)
#     full_random_solution, full_random_obj, full_feas_rate = multiple_randomized_rounds(
#         frac_solution_full, A_orig, b_orig, full_obj_func, n_samples=500
#     )
#     print("Full candidate set randomized rounding solution:", full_random_solution)
#     print("Full candidate set randomized rounding objective:", full_random_obj)
#     print(f"Feasibility rate (full candidate set): {full_feas_rate*100:.2f}%")
    
#     # --- Compute Optimality Gaps ---
#     gap_iter = 100 * (frac_obj_full - f_iterative) / frac_obj_full
#     gap_full = 100 * (frac_obj_full - full_random_obj) / frac_obj_full
#     print("\nFinal Optimality Gaps:")
#     print(f"  Iterative Method Gap: {gap_iter:.2f}%")
#     print(f"  Full Candidate Rounding Gap: {gap_full:.2f}%")
    
#     # --- Plot Comparison ---
#     x_labels = ["Full Fractional"]
#     x = np.arange(len(x_labels))
#     width = 0.3

#     fig, ax = plt.subplots(1, 2, figsize=(12,5))

#     # Plot objectives.
#     ax[0].bar(x - width/2, [frac_obj_full], width, label="Fractional (Full)", color='skyblue')
#     ax[0].bar(x, [full_random_obj], width, label="Full Randomized", color='lightgreen')
#     ax[0].bar(x + width/2, [f_iterative], width, label="Iterative Method", color='salmon')
#     ax[0].set_ylabel("Objective Score")
#     ax[0].set_title("Objective Scores (Full vs. Iterative)")
#     ax[0].set_xticks(x)
#     ax[0].set_xticklabels(x_labels)
#     ax[0].legend()
#     ax[0].grid(True, axis='y')

#     # Plot gap percentages.
#     ax[1].bar(x - width/2, [100 * (frac_obj_full - full_random_obj) / frac_obj_full], width,
#               label="Full Randomized Gap (%)", color='lightgreen')
#     ax[1].bar(x + width/2, [100 * (frac_obj_full - f_iterative) / frac_obj_full], width,
#               label="Iterative Method Gap (%)", color='salmon')
#     ax[1].set_ylabel("Gap (%)")
#     ax[1].set_title("Optimality Gaps (Relative to Full Fractional)")
#     ax[1].set_xticks(x)
#     ax[1].set_xticklabels(x_labels)
#     ax[1].legend()
#     ax[1].grid(True, axis='y')

#     plt.tight_layout()
#     plt.show()
    
#     return final_candidate_set, final_frac_solution, final_binary_solution, gap_iter, gap_full

# if __name__ == "__main__":
#     run_budget_tests_iterative()

