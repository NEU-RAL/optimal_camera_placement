import numpy as np
import gtsam
from gtsam import symbol_shorthand, DoglegOptimizer, LevenbergMarquardtOptimizer, \
    GenericProjectionFactorCal3_S2, noiseModel, PriorFactorPose3, PriorFactorPoint3, Values
from gtsam import Point3
import math
from scipy.sparse import csr_matrix, diags

L = symbol_shorthand.L
X = symbol_shorthand.X

def check_symmetric(a, rtol=1e-5, atol=1e-5):
    return np.allclose(a, a.T, rtol=rtol, atol=atol)

def build_factor_graph(measurements, poses, points, intrinsics, extrinsics, selected_inds=[]):
    graph = gtsam.NonlinearFactorGraph()
    measurement_noise = gtsam.noiseModel.Isotropic.Sigma(2, 1.0)

    if len(selected_inds) == 0:
        selected_inds = range(len(extrinsics))

    for i, pose in enumerate(poses):
        for j, pt in enumerate(points):
            for k in selected_inds:
                meas = measurements[i,k,j]
                if (meas[0] == 0 and meas[1] ==0):
                    continue
                measured = meas.reshape((2,1))
                K = intrinsics[k] if isinstance(intrinsics, list) else intrinsics
                factor = gtsam.GenericProjectionFactorCal3_S2(
                    measured, measurement_noise, X(i), L(j),
                    K, False, False, extrinsics[k]
                )
                graph.push_back(factor)

    gt_vals = gtsam.Values()
    pose_mask = np.zeros(len(poses))
    points_mask = np.zeros(len(points))

    for i, p in enumerate(poses):
        gt_vals.insert(X(i), p)
        pose_mask[i] = 1

    for j, pt in enumerate(points):
        gt_vals.insert(L(j), pt)
        points_mask[j] = 1

    large_sigma = 1e6
    point_noise = gtsam.noiseModel.Isotropic.Sigma(3, large_sigma)
    for j, pt in enumerate(points):
        graph.push_back(gtsam.PriorFactorPoint3(L(j), pt, point_noise))

    return graph, gt_vals, pose_mask, points_mask


def compute_info_matrix(graph, values):
    """
    Compute information matrix (FIM) by linearizing the factor graph at given values.
    """
    lin_graph = graph.linearize(values)
    # hessian
    A = lin_graph.hessian()[0]
    return A

def extract_full_matrix(A, num_poses, num_points):
    """
    GTSAM arranges variables in some order. We assume
    all points come first (3 per point), then poses (6 per pose).
    This might vary depending on the ordering.
    Here we assume a standard ordering: L(0..num_points-1), X(0..num_poses-1).
    """
    # Just return as-is for simplicity. A is typically in the correct order if inserted consistently.
    # In a more elaborate script, you'd reorder rows/cols to ensure correct block layout.
    return A

def compute_schur_complement(H_full, num_poses):
    """
    Compute Schur complement of the pose variables.
    Assuming block structure: [L; X], with L first then X.
    """
    total_size = H_full.shape[0]
    pose_dim = 6
    measurement_dim = total_size - num_poses*pose_dim
    Hll = H_full[:measurement_dim,:measurement_dim]
    Hlx = H_full[:measurement_dim, measurement_dim:]
    Hxx = H_full[measurement_dim:, measurement_dim:]

    # Use pseudo-inverse or solve
    try:
        Hll_inv = np.linalg.pinv(Hll)
    except:
        Hll_inv = np.linalg.pinv(Hll.toarray()) # if sparse

    H_schur = Hxx - Hlx.T @ Hll_inv @ Hlx
    return H_schur

def compute_min_eig(H_schur):
    eigvals = np.linalg.eigvalsh(H_schur)
    return eigvals[0]

def compute_rmse(result, poses):
    """
    Compute RMSE between optimized poses and ground truth.
    result: GTSAM Values after optimization
    poses: ground truth
    """
    error = 0.0
    count = 0
    for i in range(len(poses)):
        pose_est = result.atPose3(X(i))
        diff = pose_est.translation() - poses[i].translation()
        error += np.dot(diff, diff)
        count += 1
    rmse = math.sqrt(error / count)
    return rmse

def add_pose_prior(graph, values, pose_idx=0, prior_std=0.1):
    """
    Add a small prior on the first pose to fix gauge.
    """
    noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([prior_std]*6))
    graph.push_back(PriorFactorPose3(X(pose_idx), values.atPose3(X(pose_idx)), noise))

def add_point_priors(graph, values, points_mask, prior_std=0.5):
    """
    (Optional) Add weak priors on points if needed for stability.
    """
    noise = gtsam.noiseModel.Isotropic.Sigma(3, prior_std)
    for i, v in enumerate(points_mask):
        if v == 1:
            graph.push_back(PriorFactorPoint3(L(i), values.atPoint3(L(i)), noise))

def construct_candidate_inf_mats(measurements, points, poses, intrinsics, extrinsics):
    """
    Constructs information matrices for each candidate extrinsic configuration individually.
    That is, for each candidate camera, we build a factor graph considering measurements from only that camera,
    linearize and get its contribution to the FIM.

    Returns:
        inf_mats: List of CSR matrices, one per candidate.
        H0: A prior information matrix (csr_matrix) for stabilization.
    """
    num_poses = len(poses)
    num_points = len(points)
    inf_mats = []
    
    # Create a weak prior for stability (assuming all poses and points are variables)
    # Pose dimension = 6 * num_poses, Point dimension = 3 * num_points
    total_dim = num_poses * 6 + num_points * 3
    # A small prior on the last pose for gauge fixing:
    H0 = diags([1e-3]*total_dim, 0, shape=(total_dim,total_dim), format='csr')

    # If you want a different structure for H0, adjust here.

    # For each candidate extrinsic
    for c_idx, ext in enumerate(extrinsics):
        # Build graph for only that candidate camera
        graph_c, gt_vals_c, pose_mask_c, points_mask_c = build_factor_graph(
            measurements, poses, points, intrinsics, extrinsics, selected_inds=[c_idx]
        )
        # Add a small prior to fix gauge
        # Just add a prior on the first pose:
        add_pose_prior(graph_c, gt_vals_c, pose_idx=0, prior_std=0.1)

        # Create initial estimate (just use GT for simplicity)
        initial_c = gtsam.Values()
        for i,p in enumerate(poses):
            initial_c.insert(X(i), p)
        for j,pt in enumerate(points):
            initial_c.insert(L(j), pt)

        # Optimize
        params = gtsam.LevenbergMarquardtParams()
        lin_graph = graph_c.linearize(gt_vals_c)  # linearize around ground truth or a known good initial solution
        FIM = lin_graph.hessian()[0]  # Directly obtain Hessian
        fim_c = 0.5*(FIM + FIM.T)

        # Convert to CSR if needed, ensure symmetry
        # Usually fim_c is a numpy array from hessian(), we can convert to csr:
        fim_c_csr = csr_matrix(fim_c)
        # Ensure symmetric
        fim_c_csr = 0.5*(fim_c_csr + fim_c_csr.T)

        inf_mats.append(fim_c_csr)

    return inf_mats, H0