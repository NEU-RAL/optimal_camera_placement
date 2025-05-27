import utilities
import os


processed_dir = "/home/auv/software/optimal_camera_placement/herw-rw-experiment-mwe/data/processed"
opti_poses_file = os.path.join(processed_dir, "time_to_opti_poses.csv")
tag_data_file = os.path.join(processed_dir, "time_tag_poses.csv")

intrinsics, T_c4_c, poses, points,measurements = utilities.read_april_tag_data(opti_poses_file, tag_data_file, processed_dir)