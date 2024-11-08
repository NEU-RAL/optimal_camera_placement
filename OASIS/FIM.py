import utilities
from gtsam import Point3, Cal3_S2, PinholeCameraCal3_S2
import numpy as np
import gtsam
from gtsam import (
    DoglegOptimizer, LevenbergMarquardtOptimizer,
    GenericProjectionFactorCal3_S2, NonlinearFactorGraph,
    PriorFactorPoint3, PriorFactorPose3, Values, ImuFactor,
    noiseModel
)

# RangeFactor,
from numpy import linalg as la

L = gtsam.symbol_shorthand.L
X = gtsam.symbol_shorthand.X

# class CustomRangeBearingFactor(noiseModel.NoiseModelFactor):
#     def __init__(self, key1, key2, measurement, noise_model):
#         """
#         Initializes the radar factor with range and bearing measurements.
        
#         Args:
#             key1 (int): Key for the sensor pose in the factor graph.
#             key2 (int): Key for the target point (landmark) in the factor graph.
#             measurement (array-like): Measured [range, bearing] to the target.
#             noise_model (gtsam.noiseModel): Noise model for the measurement.
#         """
#         super().__init__(noise_model, key1, key2)
#         self.measurement = measurement  # Expected format: [range, bearing]

#     def error(self, values):
#         """
#         Computes the error between the predicted and measured range and bearing.
        
#         Args:
#             values (Values): The current values (estimates) in the factor graph.

#         Returns:
#             np.ndarray: The error vector, which is [range_error, bearing_error].
#         """
#         # Retrieve the sensor pose and target point (landmark) from the graph's values
#         pose = values.atPose3(self.keys[0])
#         point = values.atPoint3(self.keys[1])

#         # Calculate the relative position from the sensor pose to the target point
#         relative_position = pose.transform_to(point)

#         # Calculate predicted range and bearing
#         range_predicted = np.linalg.norm(relative_position)
#         bearing_predicted = np.arctan2(relative_position.y(), relative_position.x())

#         # Calculate the error as the difference between predicted and measured values
#         range_error = range_predicted - self.measurement[0]
#         bearing_error = bearing_predicted - self.measurement[1]

#         return np.array([range_error, bearing_error])


def build_graph(sensor_data, poses, points, extrinsics, sensor_types, remove_ill_posed=False):
    """
    Generalized function to construct a factor graph based on various sensor types.
    
    Args:
        sensor_data (dict): Dictionary containing measurements for each sensor type.
        poses (list): List of poses where each sensor is located.
        points (list): List of 3D points for landmarks.
        extrinsics (list): Extrinsic parameters for each sensor.
        sensor_types (dict): Maps sensor IDs to sensor types ('GPS', 'IMU', 'LiDAR', 'Radar').
        remove_ill_posed (bool): If True, remove poorly constrained measurements.

    Returns:
        tuple: Factor graph, ground truth values, pose mask, and points mask.
    """
    # Initialize the factor graph and ground truth values container
    graph = NonlinearFactorGraph()
    gt_vals = Values()

    # Define noise models for each sensor type
    noise_models = {
        'GPS': noiseModel.Isotropic.Sigma(3, 0.5),       # Example noise for GPS position
        'IMU': noiseModel.Diagonal.Sigmas(np.array([0.1, 0.1, 0.1, 0.01, 0.01, 0.01])),  # IMU noise
        'LiDAR': noiseModel.Isotropic.Sigma(1, 0.1),     # Range noise for LiDAR
        'Radar': noiseModel.Diagonal.Sigmas([0.2, 0.1])  # Range and bearing noise for Radar
    }

    # Loop over each sensor, add corresponding factors
    for sensor_id, sensor_type in sensor_types.items():
        if sensor_type == 'GPS':
            # Add GPS factors: absolute positioning for each pose
            for i, pose in enumerate(poses):
                gps_position = sensor_data[sensor_id]['positions'][i]
                factor = PriorFactorPoint3(L(i), gps_position, noise_models['GPS'])
                graph.add(factor)

        # elif sensor_type == 'IMU':
        #     # Add IMU factors: integrated motion constraints between consecutive poses
        #     imu_data = sensor_data[sensor_id]
        #     for i in range(1, len(poses)):
        #         prev_pose = poses[i - 1]
        #         curr_pose = poses[i]
        #         factor = ImuFactor(X(i - 1), X(i), imu_data[i - 1], noise_models['IMU'])
        #         graph.add(factor)

        # elif sensor_type == 'LiDAR':
        #     # Add LiDAR range factors between poses and points
        #     for i, pose in enumerate(poses):
        #         for j, point in enumerate(points):
        #             range_measurement = sensor_data[sensor_id]['ranges'][i, j]
        #             if range_measurement > 0:
        #                 factor = RangeFactor(X(i), L(j), range_measurement, noise_models['LiDAR'])
        #                 graph.add(factor)

        # elif sensor_type == 'Radar':
        #     # Add Radar factors: range and bearing between poses and points
        #     for i, pose in enumerate(poses):
        #         for j, point in enumerate(points):
        #             range_bearing = sensor_data[sensor_id]['range_bearing'][i, j]
        #             if range_bearing[0] > 0:  # Check if range is valid
        #                 factor = CustomRangeBearingFactor(X(i), L(j), range_bearing, noise_models['Radar'])
        #                 graph.add(factor)

    # Fill in ground truth values (optional based on setup)
    pose_mask = np.ones(len(poses))
    points_mask = np.ones(len(points))
    
    return graph, gt_vals, pose_mask, points_mask

def compute_CRLB(vals, graph):
    """
    Computes the Cramer-Rao Lower Bound (CRLB) by linearizing the graph.
    Args:
        vals (Values): Ground truth values for optimization.
        graph (NonlinearFactorGraph): Factor graph for the problem.

    Returns:
        tuple: Hessian matrix (representing the Fisher Information Matrix) and optionally, covariance matrix.
    """
    # Linearize the factor graph based on the current state estimates in 'vals'
    lin_graph = graph.linearize(vals)
    hess = lin_graph.hessian()[0]  # Hessian of the linearized factor graph, representing the FIM

    # Attempt to compute the covariance matrix, if Hessian is invertible
    cov = None
    try:
        cov = np.linalg.inv(hess)
    except np.linalg.LinAlgError:
        print("Hessian is singular; could not compute covariance.")

    return hess, cov

def compute_schur_fim(fim, num_poses):
    """
    Computes the Schur complement of the FIM for reduced dimensionality.
    Args:
        fim (np.ndarray): Full Fisher Information Matrix (FIM).
        num_poses (int): Number of poses in the graph.

    Returns:
        np.ndarray: Schur complement of the FIM with respect to the pose variables.
    """
    # Extract blocks of the FIM matrix for poses and landmarks
    Hxx = fim[-num_poses * 6:, -num_poses * 6:]    # Pose-to-pose information
    Hll = fim[0: -num_poses * 6, 0: -num_poses * 6]  # Landmark-to-landmark information
    Hlx = fim[0: -num_poses * 6, -num_poses * 6:]  # Landmark-to-pose cross information

    Hxx_schur = Hxx - Hlx.T @ np.linalg.pinv(Hll) @ Hlx
    Hxx_schur = (Hxx_schur + Hxx_schur.T) / 2

    return Hxx_schur

def build_hfull(sensor_data, points, poses, sensor_types, extrinsics, selection=[]):
    """
    Constructs the full FIM by building a factor graph and computing the CRLB.
    Args:
        sensor_data (dict): Measurements from each sensor type.
        points (list): List of 3D points.
        poses (list): List of poses (positions of each sensor).
        sensor_types (dict): Mapping of sensor IDs to sensor types (e.g., 'GPS', 'IMU', 'LiDAR', 'Radar').
        extrinsics (list): List of extrinsic parameters for candidate sensors.
        selection (list): Sensor indices to consider in the graph.

    Returns:
        tuple: Full information matrix, graph, ground truth values, pose mask, and points mask.
    """
    # Build the factor graph for the selected sensors
    graph, gtvals, poses_mask, points_mask = build_graph(sensor_data, poses, points, extrinsics, sensor_types, selection)
    fim, crlb = compute_CRLB(gtvals, graph)

    # Initialize the full FIM with appropriate dimensions
    num_poses, num_points = len(poses), len(points)
    h_full = np.zeros((num_poses * 6 + num_points * 3, num_poses * 6 + num_points * 3), dtype=np.float64)

    # Populate pose and landmark (point) blocks in the full FIM
    pose_inds, fim_idx = [], 0
    for val in gtvals.keys():
        sym = gtsam.Symbol(val)
        if sym.chr() == ord('x'):
            pose_idx = sym.index()
            pose_inds.append(pose_idx)
            h_full_pose_idx_start = num_points * 3 + pose_idx * 6
            fim_pose_idx_start = num_points * 3 + fim_idx * 6
            h_full[h_full_pose_idx_start:h_full_pose_idx_start + 6, h_full_pose_idx_start:h_full_pose_idx_start + 6] = fim[fim_pose_idx_start:fim_pose_idx_start + 6, fim_pose_idx_start:fim_pose_idx_start + 6]
            fim_idx += 1

    # Populate landmark blocks
    fim_idx = 0
    for val in gtvals.keys():
        sym = gtsam.Symbol(val)
        if sym.chr() == ord('l'):
            point_idx = sym.index()
            h_full[point_idx * 3:(point_idx + 1) * 3, point_idx * 3:(point_idx + 1) * 3] = fim[fim_idx * 3:(fim_idx + 1) * 3, fim_idx * 3:(fim_idx + 1) * 3]
            for p_idx_fim, pose_idx in enumerate(pose_inds):
                h_full_pose_idx_start = num_points * 3 + pose_idx * 6
                fim_pose_idx_start = num_points * 3 + p_idx_fim * 6
                h_full[point_idx * 3:(point_idx + 1) * 3, h_full_pose_idx_start:h_full_pose_idx_start + 6] = fim[fim_idx * 3:(fim_idx + 1) * 3, fim_pose_idx_start:fim_pose_idx_start + 6]
                h_full[h_full_pose_idx_start:h_full_pose_idx_start + 6, point_idx * 3:(point_idx + 1) * 3] = fim[fim_pose_idx_start:fim_pose_idx_start + 6, fim_idx * 3:(fim_idx + 1) * 3]
            fim_idx += 1

    assert utilities.check_symmetric(h_full)
    return h_full, graph, gtvals, poses_mask, points_mask

def construct_candidate_inf_mats(sensor_data, sensor_types, extrinsics, points, poses):
    """
    Constructs FIMs for each candidate sensor configuration.
    Args:
        sensor_data (dict): Simulated measurements from each sensor type.
        sensor_types (dict): Mapping of sensor IDs to sensor types.
        extrinsics (list): Candidate extrinsic parameters for each sensor.
        points (list): 3D points.
        poses (list): Sensor poses.

    Returns:
        tuple: Array of information matrices for each candidate and factor count debug list.
    """
    num_poses, num_points = len(poses), len(points)
    inf_mat_size = (num_poses * 6 + num_points * 3, num_poses * 6 + num_points * 3)
    inf_mats = np.zeros((0, *inf_mat_size))
    h_sum = np.zeros(inf_mat_size)
    debug_num_facs = []

    for j, extr in enumerate(extrinsics):
        # Build FIM for this candidate sensor configuration
        h_sensor, graph, gtvals, poses_mask, points_mask = build_hfull(sensor_data, points, poses, sensor_types, extrinsics, selection=[j])
        h_sum += h_sensor
        inf_mats = np.append(inf_mats, h_sensor[None], axis=0)
        debug_num_facs.append(graph.nrFactors())

    print("Number of candidates:", len(extrinsics))
    return inf_mats, debug_num_facs

def compute_info_metric(poses, points, sensor_data, sensor_types, extrinsics, selection, h_prior):
    """
    Computes the observability metric for a given sensor selection.
    Args:
        poses (list): List of sensor poses.
        points (list): List of 3D points.
        sensor_data (dict): Measurements for each sensor type.
        sensor_types (dict): Mapping of sensor IDs to sensor types.
        extrinsics (list): Extrinsic parameters for each candidate sensor.
        selection (list): Indices of selected sensors.
        h_prior (np.ndarray): Prior information matrix.

    Returns:
        float: Smallest eigenvalue of the Schur complement (observability metric).
    """
    # Build FIM using the selected sensor configuration
    h_full, graph, gtvals, poses_mask, points_mask = build_hfull(sensor_data, points, poses, sensor_types, extrinsics, selection)
    h_full += h_prior

    # Compute the Schur complement of the FIM
    fim = compute_schur_fim(h_full, len(poses))
    least_fim_eig = np.linalg.eigvalsh(fim)[0]

    return least_fim_eig

def combine_inf_mats(inf_mats, x):
    """
    Combines the information matrices weighted by the vector x.

    Args:
        inf_mats (List[np.ndarray]): List of information matrices.
        x (np.ndarray): Weight vector.

    Returns:
        np.ndarray: Combined information matrix.
    """
    combined_inf_mat = np.zeros_like(inf_mats[0])
    for xi, Hi in zip(x, inf_mats):
        combined_inf_mat += xi * Hi
    return combined_inf_mat



# def compute_CRLB(vals, graph):
#     """
#     Computes the Cramer-Rao Lower Bound (CRLB) for estimation.
#     Args:
#         vals (Values): Ground truth values for optimization.
#         graph (NonlinearFactorGraph): Factor graph for the problem.

#     Returns:
#         tuple: Hessian matrix (as Fisher Information Matrix) and covariance (if computable).
#     """
#     lin_graph = graph.linearize(vals)
#     hess = lin_graph.hessian()[0]  # Compute the Hessian, representing the FIM
#     cov = None
#     # Uncomment below if you want to compute covariance, but ensure the matrix is invertible
#     # try:
#     #     cov = np.linalg.inv(hess)
#     # except Exception:
#     #     print("Exception in inverse: info mat is singular")
#     #     return hess, None
#     return hess, cov

# def compute_schur_fim(fim, num_poses):
#     """
#     Computes the Schur complement of the FIM for reduced dimensionality.
#     Args:
#         fim (np.ndarray): Full FIM.
#         num_poses (int): Number of poses in the problem.

#     Returns:
#         np.ndarray: Schur complement of FIM with respect to pose variables.
#     """
#     # Separate FIM into submatrices for poses and landmarks
#     Hxx = fim[-num_poses * 6:, -num_poses * 6:]
#     Hll = fim[0: -num_poses * 6, 0: -num_poses * 6]
#     Hlx = fim[0: -num_poses * 6, -num_poses * 6:]
    
#     # Calculate the Schur complement
#     Hxx_schur = Hxx - Hlx.T @ np.linalg.pinv(Hll) @ Hlx
#     return Hxx_schur

# def build_graph(measurements, poses, points, intrinsics, extrinsics, inds=[], rm_ill_posed=False):
#     """
#     Constructs a factor graph based on camera measurements, poses, points, and intrinsics.
#     Args:
#         measurements (np.ndarray): Simulated measurements from each camera.
#         poses (list): List of camera poses.
#         points (list): List of 3D points.
#         intrinsics (list): List of camera intrinsics.
#         extrinsics (list): List of camera extrinsic parameters.
#         inds (list): List of camera indices to use. Defaults to all.
#         rm_ill_posed (bool): If True, removes points observed by fewer than 2 cameras.

#     Returns:
#         tuple: Factor graph, ground truth values, pose mask, and points mask.
#     """
#     # Define noise model for measurements
#     measurement_noise = gtsam.noiseModel.Isotropic.Sigma(2, 1.0)  # One-pixel noise model

#     # Initialize factor graph
#     graph = NonlinearFactorGraph()
#     dict = {}  # Maps landmarks to factors in the graph
#     dict_lm_poses = {}  # Maps landmarks to the set of poses observing them

#     if len(inds) == 0:
#         inds = range(len(extrinsics))  # Use all extrinsics if none specified

#     # Add measurement factors for each camera and point
#     for i, pose in enumerate(poses):
#         for j, point in enumerate(points):
#             for k in inds:
#                 comp_pose = extrinsics[k]
#                 pose_wc = pose.compose(comp_pose)  # Compute camera pose w.r.t world
#                 camera = PinholeCameraCal3_S2(pose_wc, intrinsics[k])
#                 measurement = measurements[i, k, j]
                
#                 # Skip if measurement is zero (indicating no observation)
#                 if measurement[0] == 0 and measurement[1] == 0:
#                     continue
                
#                 # Create projection factor and add it to the graph
#                 factor = GenericProjectionFactorCal3_S2(
#                     measurement, measurement_noise, X(i), L(j), intrinsics[k], comp_pose)
#                 graph.push_back(factor)
                
#                 # Track factors and observing poses for each landmark
#                 dict.setdefault(j, []).append(graph.nrFactors() - 1)
#                 dict_lm_poses.setdefault(j, set()).add(i)

#     # Identify ill-posed points and mark factors for removal if rm_ill_posed is True
#     rm_indices, rm_lm_indices = [], []
#     uniq_poses = set()
#     for k, v in dict_lm_poses.items():
#         uniq_poses.update(v)
#         if len(v) < 2:
#             rm_indices.extend(dict[k])
#             rm_lm_indices.append(k)
#     if rm_ill_posed:
#         for i in rm_indices:
#             graph.remove(i)

#     # Insert ground truth values for poses and points
#     gt_vals = Values()
#     pose_mask = np.zeros(len(poses))
#     points_mask = np.zeros(len(points))
#     for k in graph.keyVector():
#         sym = gtsam.Symbol(k)
#         if sym.chr() == ord('x'):
#             gt_vals.insert(X(sym.index()), poses[sym.index()])
#             pose_mask[sym.index()] = 1
#         elif sym.chr() == ord('l'):
#             gt_vals.insert(L(sym.index()), points[sym.index()])
#             points_mask[sym.index()] = 1

#     return graph, gt_vals, pose_mask, points_mask

# def build_hfull(measurements, points, poses, intrinsics, extr_cand, ind=[]):
#     """
#     Constructs the full FIM by building a factor graph and computing the CRLB.
#     Args:
#         measurements (np.ndarray): Simulated measurements from each camera.
#         points (list): List of 3D points.
#         poses (list): List of camera poses.
#         intrinsics (list): List of camera intrinsics.
#         extr_cand (list): List of extrinsic parameters for candidate cameras.
#         ind (list): Camera indices to consider.

#     Returns:
#         tuple: Full information matrix, graph, ground truth values, pose mask, and points mask.
#     """
#     # Build graph and compute CRLB to get the full FIM
#     graph, gtvals, poses_mask, points_mask = build_graph(measurements, poses, points, intrinsics, extr_cand, ind)
#     fim, crlb = compute_CRLB(gtvals, graph)

#     # Initialize the full FIM matrix with appropriate dimensions
#     num_poses, num_points = len(poses), len(points)
#     h_full = np.zeros((num_poses * 6 + num_points * 3, num_poses * 6 + num_points * 3), dtype=np.float64)

#     # Map Hessian elements to the full FIM matrix for poses and points
#     pose_inds, fim_idx = [], 0
#     tot_num_points = len(points)
#     for val in gtvals.keys():
#         sym = gtsam.Symbol(val)
#         if sym.chr() == ord('x'):
#             h_full_pose_idx = sym.index()
#             pose_inds.append(h_full_pose_idx)
#             h_full_pose_idx_start = tot_num_points * 3 + h_full_pose_idx * 6
#             fim_pose_idx_start = num_points * 3 + fim_idx * 6
#             h_full[h_full_pose_idx_start: h_full_pose_idx_start + 6, h_full_pose_idx_start: h_full_pose_idx_start + 6] = fim[fim_pose_idx_start: fim_pose_idx_start + 6, fim_pose_idx_start: fim_pose_idx_start + 6]
#             fim_idx += 1

#     # Populate the landmark (point) blocks in the full FIM
#     fim_idx = 0
#     for val in gtvals.keys():
#         sym = gtsam.Symbol(val)
#         if sym.chr() == ord('l'):
#             idx = sym.index()
#             h_full[idx * 3: (idx + 1) * 3, idx * 3: (idx + 1) * 3] = fim[fim_idx * 3: (fim_idx + 1) * 3, fim_idx * 3: (fim_idx + 1) * 3]
#             for p_idx_fim, pose_idx in enumerate(pose_inds):
#                 h_full_pose_idx_start = tot_num_points * 3 + pose_idx * 6
#                 fim_pose_idx_start = num_points * 3 + p_idx_fim * 6
#                 h_full[idx * 3: (idx + 1) * 3, h_full_pose_idx_start: h_full_pose_idx_start + 6] = fim[fim_idx * 3: (fim_idx + 1) * 3, fim_pose_idx_start: fim_pose_idx_start + 6]
#                 h_full[h_full_pose_idx_start: h_full_pose_idx_start + 6, idx * 3: (idx + 1) * 3] = fim[fim_pose_idx_start: fim_pose_idx_start + 6, fim_idx * 3: (fim_idx + 1) * 3]
#             fim_idx += 1

#     assert utilities.check_symmetric(h_full)
#     return h_full, graph, gtvals, poses_mask, points_mask

# def construct_candidate_inf_mats(measurements, intrinsics, extr_cand, points, poses):
#     """
#     Constructs FIMs for each candidate sensor configuration.
#     Args:
#         measurements (np.ndarray): Simulated measurements.
#         intrinsics (list): Camera intrinsics.
#         extr_cand (list): Candidate extrinsic parameters.
#         points (list): 3D points.
#         poses (list): Camera poses.

#     Returns:
#         tuple: Array of information matrices for each candidate and factor count debug list.
#     """
#     num_poses, num_points = len(poses), len(points)
#     inf_mat_size = (num_poses * 6 + num_points * 3, num_poses * 6 + num_points * 3)
#     inf_mats = np.zeros((0, *inf_mat_size))
#     h_sum = np.zeros(inf_mat_size)
#     debug_num_facs = []

#     for j, cand in enumerate(extr_cand):
#         h_cam, graph, gtvals, poses_mask, points_mask = build_hfull(measurements, points, poses, intrinsics, extr_cand, ind=[j])
#         h_sum += h_cam
#         inf_mats = np.append(inf_mats, h_cam[None], axis=0)
#         debug_num_facs.append(graph.nrFactors())

#     print("Number of candidates:", len(extr_cand))
#     return inf_mats, debug_num_facs

# def compute_info_metric(poses, points, meas, intrinsics, cands, selection, h_prior):
#     """
#     Computes the observability metric for a given sensor selection.
#     Args:
#         poses (list): List of poses.
#         points (list): List of 3D points.
#         meas (np.ndarray): Measurements.
#         intrinsics (list): Camera intrinsics.
#         cands (list): Extrinsic candidates.
#         selection (list): Indices of selected cameras.
#         h_prior (np.ndarray): Prior information matrix.

#     Returns:
#         float: Smallest eigenvalue of the Schur complement (observability metric).
#     """
#     h_full, graph, gtvals, poses_mask, points_mask = build_hfull(meas, points, poses, intrinsics, cands, selection)
#     h_full += h_prior
#     fim = compute_schur_fim(h_full, len(poses))
#     least_fim_eig = np.linalg.eigvalsh(fim)[0]
#     return least_fim_eig

def find_min_eig_pair(inf_mats, selection, H0, num_poses):
    """
    Finds the minimum eigenvalue and eigenvector for the selected sensors.
    Args:
        inf_mats (np.ndarray): Array of information matrices for each candidate.
        selection (np.ndarray): Binary vector indicating selected sensors.
        H0 (np.ndarray): Prior information matrix.
        num_poses (int): Number of poses.

    Returns:
        tuple: Smallest eigenvalue, eigenvector, and final FIM.
    """
    inds = np.where(selection > 1e-10)[0]
    final_inf_mat = sum(selection[i] * inf_mats[i] for i in inds) + H0
    H_schur = compute_schur_fim(final_inf_mat, num_poses)
    eigvals = np.linalg.eigvalsh(H_schur)
    # print("Min eig of the Schur complement:", eigvals.min())

    assert utilities.check_symmetric(H_schur)
    eigvals, eigvecs = la.eigh(H_schur)
    return eigvals[0], eigvecs[:, 0], final_inf_mat