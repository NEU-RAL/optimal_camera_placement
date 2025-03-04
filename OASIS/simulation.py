import numpy as np
import math
import random
from gtsam import Cal3_S2, Point3, Pose3, Rot3, PinholeCameraCal3_S2
import gtsam
from typing import List

def generate_random_landmarks(num_points=100, cube_size=100.0):
    """
    Generate random 3D landmark points within a large cube of size cube_size.
    A larger cube ensures a wide spatial distribution of landmarks.
    """
    x = -(cube_size/2) + np.random.rand(num_points) * cube_size
    y = -(cube_size/2) + np.random.rand(num_points) * cube_size
    z = -(cube_size/2) + np.random.rand(num_points) * cube_size
    points = [Point3(x[i], y[i], z[i]) for i in range(num_points)]
    return points

def generate_trajectory(num_poses=10, radius=10.0, height=0.0):
    """
    Generate a circular trajectory of camera poses around the origin.
    A circle of radius 10 ensures cameras have a good view of the landmark set.
    """
    angles = np.linspace(0, 2 * np.pi, num_poses, endpoint=False)
    poses = []
    target = Point3(0,0,0)
    up = Point3(0,0,1)
    K = Cal3_S2(300,300,0,320,240)
    for theta in angles:
        position = Point3(radius * math.cos(theta), radius * math.sin(theta), height)
        camera = PinholeCameraCal3_S2.Lookat(position, target, up, K)
        poses.append(camera.pose())
    return poses, K

def generate_candidate_extrinsics(num_candidates=20, min_baseline=2.0, fov_angle_deg=30.0):
    """
    Generate a set of candidate camera extrinsics (relative poses) in a more controlled manner:
    - Distribute yaw angles evenly around 360 degrees.
    - Set a small pitch range to ensure they aren't all looking straight ahead.
    - Ensure a minimum baseline along a ring in xy-plane to avoid degenerate baselines.
    """
    extrinsics = []
    fov_angle = math.radians(fov_angle_deg)

    # Evenly distribute yaw angles
    yaw_angles = np.linspace(0, 2*math.pi, num_candidates, endpoint=False)

    # Use a small pitch range around 0 for variety
    # For simplicity, let's pick a slight range of pitches, e.g. between -fov_angle/4 and fov_angle/4
    pitch_range = fov_angle / 4
    pitches = np.linspace(-pitch_range, pitch_range, num_candidates)

    # Place translation on a small ring around the camera (instead of random)
    # This ensures a stable baseline. Let's place all extrinsics at distance min_baseline from the origin
    # but vary their angle to ensure a spread.
    trans_angles = np.linspace(0, 2*math.pi, num_candidates, endpoint=False)

    for i in range(num_candidates):
        yaw = yaw_angles[i]
        pitch = pitches[i]

        Rz = Rot3.Yaw(yaw)
        Rx = Rot3.Rx(pitch)
        rot = Rz.compose(Rx)

        # Place translation on a ring of radius = min_baseline
        tx = min_baseline * math.cos(trans_angles[i])
        ty = min_baseline * math.sin(trans_angles[i])
        tz = 0.0

        extrinsics.append(Pose3(rot, Point3(tx, ty, tz)))

    return extrinsics

def generate_measurements(poses, points, extrinsics, intrinsics, noise_sigma=1.0):
    """
    Project points into candidate cameras defined by extrinsics from each pose.
    Adds small Gaussian noise and checks if within image bounds.
    """
    measurement_noise = noise_sigma
    measurements = np.zeros((len(poses), len(extrinsics), len(points), 2))
    for i, base_pose in enumerate(poses):
        for k, ext in enumerate(extrinsics):
            # Compose to get world-camera pose
            pose_wc = base_pose.compose(ext)
            # Handle intrinsics (could be a single K or a list)
            if isinstance(intrinsics, list):
                K = intrinsics[k]
            else:
                K = intrinsics
            camera = PinholeCameraCal3_S2(pose_wc, K)
            for j, pt in enumerate(points):
                try:
                    uv = camera.project(pt)
                    uv_noisy = uv + measurement_noise * np.random.randn(2)
                    # Check image bounds
                    if 0 < uv_noisy[0] < 640 and 0 < uv_noisy[1] < 480:
                        measurements[i, k, j, :] = uv_noisy
                except:
                    pass
    return measurements