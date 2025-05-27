from OASIS import utilities
from OASIS import visualize
from OASIS import methods
from gtsam import Point3, Cal3_S2
import numpy as np
import math
from DataGenerator import sim_data_utils as sdu


def compute_opt_config_sum_inf():
    K = Cal3_S2(50.0, 50.0, 0.0, 50.0, 50.0)

    # Example constraints (maximum 6 candidates)
    N = 6
    A = np.ones((1, N))  # Sum of candidates <= 6
    b = np.array([6])  # Maximum number of selections allowed

    radius = 5.0
    height = 5.0
    up = Point3(0, 0, 1)
    target = Point3(0, 0, height)
    trajs = []
    best_configs = []
    best_configs_brute = []
    num_cands = N  # Candidates (e.g., cameras)

    pose_rots, pose_trans = utilities.generate_candidate_poses((0, 2 * math.pi), 4, (-math.pi / 2, math.pi / 2), 4)

    # Random robot trajectory
    rott = np.eye(3) 
    points, poses_circle = sdu.create_room_world(20, 12, K)
    best_config_circle, cost_circle, _, _ = methods.greedy_selection(points, poses_circle, K, pose_rots, pose_trans, num_cands, methods.Metric.min_eig)
    
    trajs.append(poses_circle.copy())
    best_configs.append(best_config_circle.copy())
    print("Best config (Greedy, circle):", best_config_circle)
    print(f"The logdet for circular trajectory (greedy): {cost_circle:.2f}")
    visualize.show_camconfigs(best_config_circle)

    # Now run the General Frank-Wolfe method with constraints
    print("Running Frank-Wolfe Optimization with Constraints...")
    
    selection_init = np.ones(N) / N  # Uniform initial selection
    inf_mats = np.random.rand(N, N, N)  # Placeholder for actual information matrices
    H0 = np.eye(N)  # Identity matrix as prior

    best_config_fw, selection_fw, min_eig_val, min_eig_val_unrounded, iterations = methods.gen_frank_wolfe(
        inf_mats=inf_mats,  # Placeholder information matrices
        H0=H0,  # Prior information matrix
        n_iters=600,  # Number of iterations
        selection_init=selection_init,  # Initial selection
        k=num_cands,  # Number of candidates to select
        num_poses=N,  # Number of robot poses
        A=A,  # Constraint matrix (sum <= 6)
        b=b   # Upper bound for constraints
    )

    # Output results for Frank-Wolfe
    print("Best config (Frank-Wolfe):", best_config_fw)
    print(f"Minimum eigenvalue (rounded solution): {min_eig_val:.2f}")
    print(f"Minimum eigenvalue (unrounded solution): {min_eig_val_unrounded:.2f}")
    print(f"Total iterations: {iterations}")
    
    # Visualization of the Frank-Wolfe optimized camera configuration
    visualize.show_camconfigs(best_config_fw)

if __name__ == '__main__':
    compute_opt_config_sum_inf()
