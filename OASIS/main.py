import numpy as np
from simulation import generate_random_landmarks, generate_trajectory, generate_candidate_extrinsics, generate_measurements
from utils import construct_candidate_inf_mats
from visualization import visualize_3d
import gtsam
from optimizations import greedy_selection, Metric, scipy_minimize_lse

def main():
    # Step 1: Simulate scenario
    num_points = 30
    num_poses = 10
    points = generate_random_landmarks(num_points, cube_size=20.0)
    poses, K = generate_trajectory(num_poses=num_poses, radius=15.0, height=5.0)
    num_candidates = 15
    extr_cand = generate_candidate_extrinsics(num_candidates=num_candidates, min_baseline=2.0, fov_angle_deg=30.0)
    intrinsics = [K]*num_candidates

    measurements = generate_measurements(poses, points, extr_cand, intrinsics, noise_sigma=1.0)

    # Step 2: Construct candidate information matrices
    inf_mats, H0 = construct_candidate_inf_mats(measurements, points, poses, intrinsics, extr_cand)

    # Step 3: Run an optimization method from optimization.py
    # For example, run greedy_selection to pick Nc sensors.
    Nc = 5
    # We must pass num_poses for Schur complement dimensioning
    selection_vector, best_score, _ = greedy_selection(inf_mats, H0, Nc, metric=Metric.MIN_EIG, num_runs=1, num_poses=num_poses)

    print("Greedy selection vector:", selection_vector)
    print("Greedy best score:", best_score)

    # A simple constraint: sum of selection <= Nc
    A_constraint = np.ones((1, len(inf_mats)))
    b_constraint = np.array([Nc])
    selection_init = np.zeros(len(inf_mats))
    selection, fw_score = scipy_minimize_lse(inf_mats, H0, selection_init, num_poses, A_constraint, b_constraint)  

    print("Selection LSE:", selection)
    print("Score:", fw_score)

    # Step 4: Visualization (optional)
    visualize_3d(poses, points)

if __name__ == "__main__":
    main()