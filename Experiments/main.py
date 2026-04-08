from OASIS import utilities
from OASIS import visualize
from OASIS import methods
from gtsam import Point3, Cal3_S2,PinholeCameraCal3_S2
import numpy as np
import math
from DataGenerator import sim_data_utils as sdu

'''
Experiment #3: Maximize the SUM of log det (Inf Mat) over a set of random trajectories to estimate a opt camera placement. 
'''
def compute_opt_config_sum_inf():
    K = Cal3_S2(50.0, 50.0, 0.0, 50.0, 50.0)

    radius = 5.0
    height = 5.0
    up = Point3(0, 0, 1)
    target = Point3(0, 0, height)
    trajs = []
    best_configs = []
    best_configs_brute = []
    num_cands = 2

    pose_rots, pose_trans = utilities.generate_candidate_poses((0, 2 * math.pi), 4, (-math.pi / 2, math.pi / 2), 4)
    #random
    poses = utilities.create_random_robot_traj(rott, np.array([0.0, -5.0, 5.0]), 20)
    best_config_circle, cost_circle, _, _ = methods.greedy_selection(points, poses_circle, K, pose_rots, pose_trans, num_cands, methods.Metric.min_eig)
    trajs.append(poses.copy())
    best_configs.append(best_config.copy())
    print("best config random: ")
    print(best_config)
    print("The logdet for random traj: {:.2f} ".format(cost_rand))


if __name__ == '__main__':
    '''
    RUN Experiments
    '''
    #Setup the parameters
    K = Cal3_S2(50.0, 50.0, 0.0, 50.0, 50.0)

    radius = 5.0
    height = 5.0
    up = Point3(0, 0, 1)
    target = Point3(0, 0, height)
    trajs = []
    best_configs = []
    best_configs_brute = []
    num_cands = 6

    pose_rots, pose_trans = sdu.generate_candidate_poses((0, 2 * math.pi), 4, (-math.pi / 2, math.pi / 2), 4)
    # circle
    points, poses_circle = sdu.create_room_world(20, 12, K)
    best_config_circle, cost_circle, _, _ = methods.greedy_selection(points, poses_circle, K, pose_rots, pose_trans, num_cands, methods.Metric.min_eig)
    # best_config_brute_cirle, cost_brute_circle = brute_force_selection_stereo(points, poses_circle, K, num_cands)
    trajs.append(poses_circle.copy())
    best_configs.append(best_config_circle.copy())
    # best_configs_brute.append(best_config_brute_cirle.copy())
    print("best config circle: ")
    print(best_config_circle)
    print("The logdet for circular traj greedy: {:.2f} ".format(cost_circle))
    visualize.show_camconfigs(best_config_circle)

    # sideward
    position = Point3(0, 0, height)
    target = Point3(0.0, 3.0, height)
    camera = PinholeCameraCal3_S2.Lookat(position, target, up, K)
    rott = camera.pose().rotation().matrix()
    poses_side = sdu.create_forward_side_robot_traj(rott, np.array([-5.0, 0.0, 5.0]), 20, False)
    best_config_side, cost_side = methods.greedy_selection(points, poses_side, K, pose_rots, pose_trans, num_cands)
    # best_config_brute_side, cost_brute_side = brute_force_selection_stereo(points, poses_side, K, num_cands)
    trajs.append(poses_side.copy())
    best_configs.append(best_config_side.copy())
    # best_configs_brute.append(best_config_brute_side.copy())
    print("best config side: ")
    print(best_config_side)
    print("The logdet for side traj: {:.2f} ".format(cost_side))
    visualize.show_camconfigs(best_config_side)

    # #forward
    poses = sdu.create_forward_side_robot_traj(rott, np.array([0.0, -5.0, 5.0]), 20)
    best_config, cost_forward = methods.greedy_selection(points, poses, K, pose_rots, pose_trans, num_cands)
    #   best_config_brute_forward, cost_brute_forward= brute_force_selection_stereo(points, poses, K, num_cands)
    trajs.append(poses.copy())
    best_configs.append(best_config.copy())
    #   best_configs_brute.append(best_config_brute_forward.copy())
    print("best config forward: ")
    print(best_config)
    print("The logdet for forward traj: {:.2f} ".format(cost_forward))

    # #random
    poses = sdu.create_random_robot_traj(rott, np.array([0.0, -5.0, 5.0]), 20)
    best_config, cost_rand = methods.greedy_selection(points, poses, K, pose_rots, pose_trans, num_cands)
    #   best_config_brute_rand, cost_brute_rand = brute_force_selection_stereo(points, poses, K, num_cands)
    trajs.append(poses.copy())
    best_configs.append(best_config.copy())
    #  best_configs_brute.append(best_config_brute_rand.copy())
    print("best config random: ")
    print(best_config)
    print("The logdet for random traj: {:.2f} ".format(cost_rand))