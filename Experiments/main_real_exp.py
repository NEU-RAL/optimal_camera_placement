from OASIS import utilities
import numpy as np
import os
from ProblemBuilder import FIM as fwolf
import argparse

'''
REal data testing _ experimental 
'''
if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter,
                                     description='runs experiments for different benchmark \
                                                     algorithms for optimal camera placement\n\n')
    parser.add_argument('-i', '--in_dir', help='input directory for data', default=".")
    parser.add_argument('-s', '--select_k', help='number of cameras to select', default=2)
    parser.add_argument('-t', '--traj_type', help='Type of trajectory 1:circle, 2:side, 3:forward, 4:random', default=1)
    parser.add_argument('-o', '--output_dir', help='Output dir for output bag file', default='.')

    args = parser.parse_args()
    processed_dir = args.in_dir
    processed_dir = "/home/auv/software/optimal_camera_placement/herw-rw-experiment-mwe/data/processed"
    opti_poses_file = os.path.join(processed_dir, "time_to_opti_poses.csv")
    tag_data_file = os.path.join(processed_dir, "time_tag_poses.csv")

    intrinsics, T_c4_c, poses, points, measurements, poses_with_noise, points_with_noise = utilities.read_april_tag_data(opti_poses_file, tag_data_file, processed_dir)
    # fwolf.build_hfull(measurements, points, poses, intrinsics, T_c4_c)
    ''' Number of cameras to be selected'''
    select_k = args.select_k
    num_poses = len(poses)
    num_points = len(points)

    h_prior = np.zeros((num_poses * 6 + num_points * 3, num_poses * 6 + num_points * 3))
    h_prior[-6:, -6:] = np.eye(6)* 1000
    # h_prior[-num_poses * 6:, -num_poses * 6:] = np.eye(num_poses * 6)
    # h_prior[0: -num_poses * 6, 0: -num_poses * 6:] = np.eye(num_points * 3)
    # h_prior = h_prior # * 1e-3


    best_configs = []
    best_configs_fw = []

    ''' Construct factor graph as if we have all the 300 cameras. edges going between the poses and the landmarks'''
    ''' write the infomat as a combination of the selection variables.'''
    inf_mats, debug_nr_facs = fwolf.construct_candidate_inf_mats(measurements, intrinsics, T_c4_c, points, poses)

    num_cands = len(T_c4_c)
    selection_init = np.zeros(num_cands)
    # for i in best_selection_indices:
    #     selection_init[i] = 1
    # selection_init[0] = 1
    # selection_init[1] = 1
    # selection_init[2] = 1
    selection_init = np.ones(num_cands)
    #selection_init = selection_init*select_k/num_cands
    ''' build the prior FIM '''
    num_poses = len(poses)
    num_points = len(points)

    ''' call frankewolf iterations'''
    selection_fw, selection_fw_unr, cost_fw, cost_fw_unrounded, num_iters = fwolf.franke_wolfe(inf_mats,h_prior, 1000, selection_init.flatten(), select_k,num_poses)

    print("The Score for traj franke_wolfe with solution. rounded: {:.9f}, unrounded: {:.9f} ".format(cost_fw,
                                                                                                      cost_fw_unrounded))
    print("selection: ")
    print(np.argwhere(selection_fw == 1))

    best_selection_indices_fw = []

    for i in range(selection_fw.shape[0]):
        if selection_fw[i] == 1:
            best_selection_indices_fw.append(i)
    '''
       Compute the RMSE for the best camera placement
       '''
    rmse_g = fwolf.compute_rmse(measurements, poses, points, intrinsics, T_c4_c, best_selection_indices, poses_with_noise,
                                points_with_noise)
    rmse_g_loc = fwolf.compute_rmse(measurements, poses, points, intrinsics, T_c4_c, best_selection_indices_fw,
                                    poses_with_noise, points_with_noise, loc=True)