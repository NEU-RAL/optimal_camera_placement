from OASIS  import utilities
from OASIS  import visualize
from OASIS import methods
from gtsam import Cal3_S2
import numpy as np
import os
import random
from ProblemBuilder import FIM as fim
from DataGenerator import sim_data_utils as sdu
import argparse
from Experiments import exp_utils
import shutil

'''
Generate multiple simulations of a trajectory and landmarks within
a room environment, run all benchmark algorithms for different select_k.
log the results
'''
if __name__ == '__main__':
    ''' construct the 3D world and the trajectory'''
    ''' Sample all the camera configurations. In sim  I have ~300 configs '''
    ''' The goal is to pick the best N among these placeents.'''
    ''' Run greedy first, get a initial baseline.'''
    ''' Use greedy solution as initial value'''

    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter,
                                     description='runs experiments for different benchmark \
                                                 algorithms for optimal camera placement\n\n')

    parser.add_argument('-n', '--num_runs', help='number of runs in the experiment', default=5)
    parser.add_argument('-s', '--select_k', help='number of cameras to select', default=2)
    parser.add_argument('-t', '--traj_type', help='Type of trajectory 1:circle, 2:side, 3:forward, 4:random', default=4)
    parser.add_argument('-o', '--output_dir', help='Output dir for output bag file', default='./results')


    args = parser.parse_args()

    num_points = 25*4
    num_poses =50
    max_select_k = 3
    K = Cal3_S2(100.0, 100.0, 0.0, 50.0, 50.0)
    ''' Number of cameras to be selected'''
    #select_k = args.select_k

    traj = ""
    ''' For a particular type of trajectory'''
    for traj_ind in range(4, 5):
        if traj_ind == 1:
            traj = "circle"
        elif traj_ind == 2:
            traj = "side"
        elif traj_ind == 3:
            traj = "forward"
        elif traj_ind >= 4:
            traj = "random"

        output_file_Path = os.path.join(args.output_dir, traj+"_{}runs".format(args.num_runs))
        if os.path.exists(output_file_Path):
            shutil.rmtree(output_file_Path)
        os.mkdir(output_file_Path)

        # Setup the parameters
        # best_configs = {i: [] for i in range(2, max_select_k)}
        # best_configs_fw = {i: [] for i in range(2, max_select_k)}

        rmse_greedy_slam = {i: [] for i in range(2, max_select_k)}
        rmse_greedy_loc = {i: [] for i in range(2, max_select_k)}
        rmse_greedy_gt_slam = {i: [] for i in range(2, max_select_k)}
        rmse_greedy_gt_loc = {i: [] for i in range(2, max_select_k)}

        rmse_fw_slam = {i: [] for i in range(2, max_select_k)}
        rmse_fw_loc = {i: [] for i in range(2, max_select_k)}
        rmse_fw_gt_slam = {i: [] for i in range(2, max_select_k)}
        rmse_fw_gt_loc = {i: [] for i in range(2, max_select_k)}

        rmse_scipy_slam = {i: [] for i in range(2, max_select_k)}
        rmse_scipy_loc = {i: [] for i in range(2, max_select_k)}
        rmse_scipy_gt_slam = {i: [] for i in range(2, max_select_k)}
        rmse_scipy_gt_loc = {i: [] for i in range(2, max_select_k)}

        times_g = {i: [] for i in range(2, max_select_k)}
        times_fw = {i: [] for i in range(2, max_select_k)}
        times_scipy = {i: [] for i in range(2, max_select_k)}

        iters_fw = {i: [] for i in range(2, max_select_k)}

        greedy_scores = {i: [] for i in range(2, max_select_k)}
        fw_scores = {i: [] for i in range(2, max_select_k)}
        fw_scores_unrounded = {i: [] for i in range(2, max_select_k)}

        scipy_scores = {i: [] for i in range(2, max_select_k)}
        scipy_scores_unrounded = {i: [] for i in range(2, max_select_k)}

        greedy_selected_cands = {i: [] for i in range(2, max_select_k)}
        fw_selected_cands = {i: [] for i in range(2, max_select_k)}
        scipy_selected_cands = {i: [] for i in range(2, max_select_k)}

        fw_solution_unr_list = {i: [] for i in range(2, max_select_k)}
        scipy_solution_unr_list = {i: [] for i in range(2, max_select_k)}


        '''
        EQUAL, RAND and STANDARD
        '''
        equal_scores = {i: [] for i in range(2, max_select_k)}
        stnd_scores = {i: [] for i in range(2, max_select_k)}
        rand_scores = {i: [] for i in range(2, max_select_k)}
        equal_selected_cands = {i: [] for i in range(2, max_select_k)}
        stnd_selected_cands = {i: [] for i in range(2, max_select_k)}
        rand_selected_cands = {i: [] for i in range(2, max_select_k)}

        rmse_equal_slam ={i: [] for i in range(2, max_select_k)}
        rmse_equal_loc  ={i: [] for i in range(2, max_select_k)}
        rmse_stnd_slam = {i: [] for i in range(2, max_select_k)}
        rmse_stnd_loc  = {i: [] for i in range(2, max_select_k)}
        rmse_rand_slam = {i: [] for i in range(2, max_select_k)}
        rmse_rand_loc  = {i: [] for i in range(2, max_select_k)}
        prior_scale = 1e-3
        h_prior = np.zeros((num_poses * 6 + num_points * 3, num_poses * 6 + num_points * 3))
        h_prior[-6:, -6:] = np.eye(6) * (1/prior_scale)
        # h_prior[-num_poses * 6:, -num_poses * 6:] = np.eye(num_poses * 6)
        # h_prior[0: -num_poses * 6, 0: -num_poses * 6:] = np.eye(num_points * 3)
        # h_prior = h_prior #* 1e-3

        all_poses, all_points, all_measurements, all_extr_cand, all_intrinsics, all_poses_with_noise, all_points_with_noise =[],[],[],[],[],[],[]
        all_measurements_equal =[]
        all_measurements_stnd = []
        all_extr_cand_equal =[]
        all_extr_cand_stnd =[]
        all_intrinsics_e =[]
        all_intrinsics_s =[]

        ''' Run a set of experiments and grab data'''
        for exp_ind in range(args.num_runs):
            ''' Generate the simulation data for greedy, franke-wolfe and RAND'''
            poses, points, measurements, extr_cand, intrinsics, poses_with_noise, points_with_noise = sdu.generate_simulation_data(K, traj_ind, num_points, num_poses, False ) #args.traj_type

            ''' write the data to log'''
            exp_utils.write_data_logs(os.path.join(output_file_Path, "data_log_" + traj + "_{}.txt".format(exp_ind)),
                                      poses, points, extr_cand, measurements, poses_with_noise,
                                      points_with_noise)

            ''' For this same data i.e poses and points, generate extr_cands and measurements for EQUAL and STANDARD'''
            ''' the extr_cands for equal and stnd are a list corresponding to the selectKs as shown = [2,2,3,3,3,4,4,4,4,5,5,5,5,5,6,6,6,6,6,6]'''
            measurements_equal, extr_cand_equal, intrinsics_e = sdu.generate_meas_extr_EQUAL(poses, points, K, list(range(2,max_select_k)))
            measurements_stnd, extr_cand_stnd, intrinsics_s = sdu.generate_meas_extr_STANDARD(poses, points, K, list(range(2,max_select_k)))
            ''' write the data to log for EQUAL'''
            exp_utils.write_data_logs(os.path.join(output_file_Path, "data_log_equal_" + traj + "_{}.txt".format(exp_ind)),
                                      poses, points, extr_cand_equal, measurements_equal, poses_with_noise,
                                      points_with_noise)
            ''' write the data to log for STANDARD'''
            exp_utils.write_data_logs(os.path.join(output_file_Path, "data_log_standard_" + traj + "_{}.txt".format(exp_ind)),
                                      poses, points, extr_cand_stnd, measurements_stnd, poses_with_noise,
                                      points_with_noise)
            all_poses.append(poses)
            all_points.append(points)
            all_measurements.append(measurements)
            all_extr_cand.append(extr_cand)
            all_intrinsics.append(intrinsics)
            all_points_with_noise.append(points_with_noise)
            all_poses_with_noise.append(poses_with_noise)

            all_measurements_equal.append(measurements_equal)
            all_measurements_stnd.append(measurements_stnd)
            all_extr_cand_equal.append(extr_cand_equal)
            all_extr_cand_stnd.append(extr_cand_stnd)
            all_intrinsics_e.append(intrinsics_e)
            all_intrinsics_s.append(intrinsics_s)



        ''' Run optimization on different number of selected candidates'''
        for select_k in range(2, max_select_k):
            '''
            Cases of Equal and Random and standard configurations
            '''
            ###EQUAL case candidates to evaluate
            b_i =0
            for f in range((select_k-1), 1, -1):
                b_i = b_i + f

            #selection_equal = extr_cand_equal[b_i: b_i + select_k]
            #selection_stnd = extr_cand_stnd[b_i: b_i + select_k]
            selection_equal = list(range(b_i , b_i + select_k))
            selection_stnd = list(range(b_i , b_i + select_k))
            selection_rand = random.sample(range(len(extr_cand)), select_k)

            score_equal = 0.0
            score_stnd = 0.0
            score_rand = 0.0
            for traj_ind in range(args.num_runs):
                score_equal = score_equal + fim.compute_info_metric(all_poses[traj_ind], all_points[traj_ind], all_measurements_equal[traj_ind], all_intrinsics_e[traj_ind], all_extr_cand_equal[traj_ind], selection_equal, h_prior)
                score_stnd = score_stnd + fim.compute_info_metric(all_poses[traj_ind], all_points[traj_ind], all_measurements_stnd[traj_ind],all_intrinsics_s[traj_ind], all_extr_cand_stnd[traj_ind], selection_stnd, h_prior)
                score_rand = score_rand + fim.compute_info_metric(all_poses[traj_ind], all_points[traj_ind], all_measurements[traj_ind], all_intrinsics[traj_ind], all_extr_cand[traj_ind], selection_rand, h_prior)
            ''' Just duplicate entries for times, scores and candidates for each run. Note that we have only one set of scores, time and selected candidate. but RMSEs exist for each individual trajectrory'''
            for traj_ind in range(args.num_runs):
                equal_scores[select_k].append(score_equal)
                stnd_scores[select_k].append(score_stnd)
                rand_scores[select_k].append(score_rand)
                equal_selected_cands[select_k].append(selection_equal)
                stnd_selected_cands[select_k].append(selection_stnd)
                rand_selected_cands[select_k].append(selection_rand)


            print("RMSE for EQUAL, STANDARD and RAND")
            for traj_ind in range(args.num_runs):
                rmse_equal = exp_utils.compute_rmse(all_measurements_equal[traj_ind], all_poses[traj_ind], all_points[traj_ind], all_intrinsics_e[traj_ind], all_extr_cand_equal[traj_ind], selection_equal, all_poses_with_noise[traj_ind],
                                          all_points_with_noise[traj_ind], prior_scale)
                rmse_equal_l = exp_utils.compute_rmse(all_measurements_equal[traj_ind], all_poses[traj_ind], all_points[traj_ind], all_intrinsics_e[traj_ind],  all_extr_cand_equal[traj_ind], selection_equal,
                                            all_poses_with_noise[traj_ind], all_points_with_noise[traj_ind], prior_scale, loc=True)

                rmse_equal_slam[select_k].append(rmse_equal)
                rmse_equal_loc[select_k].append(rmse_equal_l)

                rmse_stnd = exp_utils.compute_rmse(all_measurements_stnd[traj_ind], all_poses[traj_ind], all_points[traj_ind], all_intrinsics_s[traj_ind], all_extr_cand_stnd[traj_ind], selection_stnd,
                                          all_poses_with_noise[traj_ind], all_points_with_noise[traj_ind], prior_scale)
                rmse_stnd_l = exp_utils.compute_rmse(all_measurements_stnd[traj_ind], all_poses[traj_ind], all_points[traj_ind], all_intrinsics_s[traj_ind], all_extr_cand_stnd[traj_ind], selection_stnd,
                                            all_poses_with_noise[traj_ind], all_points_with_noise[traj_ind],prior_scale, loc=True)

                rmse_stnd_slam[select_k].append(rmse_stnd)
                rmse_stnd_loc[select_k].append(rmse_stnd_l)

                rmse_rand = exp_utils.compute_rmse(all_measurements[traj_ind], all_poses[traj_ind], all_points[traj_ind], all_intrinsics[traj_ind], all_extr_cand[traj_ind], selection_rand,
                                          all_poses_with_noise[traj_ind],all_points_with_noise[traj_ind], prior_scale)
                rmse_rand_l = exp_utils.compute_rmse(all_measurements[traj_ind], all_poses[traj_ind], all_points[traj_ind], all_intrinsics[traj_ind], all_extr_cand[traj_ind], selection_rand,
                                            all_poses_with_noise[traj_ind], all_points_with_noise[traj_ind], prior_scale, loc=True)

                rmse_rand_slam[select_k].append(rmse_rand)
                rmse_rand_loc[select_k].append(rmse_rand_l)


            #score, rmse, selected candidates

            ''' ###########################################################################################################'''
            best_score_g, best_config_g, selected_inds_g, time_greedy, best_score_fw,best_score_fw_unrounded,\
            best_config_fw, selected_inds_fw, solution_fw_unr, time_fw, num_iters_fw = methods.run_single_experiment_exp(all_poses, all_points, all_measurements, all_intrinsics, extr_cand, select_k, h_prior, args.num_runs) #extr_cand, intrinsics, select_k and h_prior are same for all simulations

            ''' Just duplicate entries for times, scores and candidates for each run. Note that we have only one set of scores, time and selected candidate. but RMSEs exist for each individual trajectrory'''
            for traj_ind in range(args.num_runs):
                greedy_scores[select_k].append(best_score_g)
                fw_scores[select_k].append(best_score_fw)
                fw_scores_unrounded[select_k].append(best_score_fw_unrounded)
                # scipy_scores[select_k].append(best_score_scipy)
                # scipy_scores_unrounded[select_k].append(best_score_scipy_unrounded)

                greedy_selected_cands[select_k].append(selected_inds_g)
                fw_selected_cands[select_k].append(selected_inds_fw)
                # scipy_selected_cands[select_k].append(selected_inds_scipy)

                fw_solution_unr_list[select_k].append(solution_fw_unr.tolist())
                # scipy_solution_unr_list[select_k].append(solution_scipy_unr.tolist())

                times_g[select_k].append(time_greedy)
                times_fw[select_k].append(time_fw)
                iters_fw[select_k].append(num_iters_fw)

            '''
               Compute the RMSEs for the best camera placement
               '''
            '''---------------------- '''
            print("RMSE for Greedy-------------------------------------")
            for traj_ind in range(args.num_runs):
                rmse_g = exp_utils.compute_rmse(all_measurements[traj_ind], all_poses[traj_ind], all_points[traj_ind], all_intrinsics[traj_ind], all_extr_cand[traj_ind],selected_inds_g,all_poses_with_noise[traj_ind], all_points_with_noise[traj_ind],prior_scale )
                rmse_g_loc = exp_utils.compute_rmse(all_measurements[traj_ind], all_poses[traj_ind], all_points[traj_ind], all_intrinsics[traj_ind], all_extr_cand[traj_ind], selected_inds_g, all_poses_with_noise[traj_ind], all_points_with_noise[traj_ind], prior_scale,loc=True)
                rmse_g_gt = exp_utils.compute_rmse(all_measurements[traj_ind], all_poses[traj_ind], all_points[traj_ind], all_intrinsics[traj_ind], all_extr_cand[traj_ind], selected_inds_g,  all_poses[traj_ind],all_points[traj_ind],prior_scale)
                rmse_g_gt_loc = exp_utils.compute_rmse(all_measurements[traj_ind], all_poses[traj_ind], all_points[traj_ind], all_intrinsics[traj_ind], all_extr_cand[traj_ind], selected_inds_g, all_poses[traj_ind],all_points[traj_ind],prior_scale, loc=True)
                rmse_greedy_slam[select_k].append(rmse_g)
                rmse_greedy_loc[select_k].append(rmse_g_loc)
                rmse_greedy_gt_slam[select_k].append(rmse_g_gt)
                rmse_greedy_gt_loc[select_k].append(rmse_g_gt_loc)

            print("RMSE for Franke-wolfe------------------------------------------------")
            for traj_ind in range(args.num_runs):
                rmse_fw = exp_utils.compute_rmse(all_measurements[traj_ind], all_poses[traj_ind], all_points[traj_ind], all_intrinsics[traj_ind], all_extr_cand[traj_ind], selected_inds_fw, all_poses_with_noise[traj_ind], all_points_with_noise[traj_ind], prior_scale)
                rmse_fw_l = exp_utils.compute_rmse(all_measurements[traj_ind], all_poses[traj_ind], all_points[traj_ind], all_intrinsics[traj_ind], all_extr_cand[traj_ind], selected_inds_fw, all_poses_with_noise[traj_ind], all_points_with_noise[traj_ind], prior_scale, loc=True)
                rmse_fw_gt = exp_utils.compute_rmse(all_measurements[traj_ind], all_poses[traj_ind], all_points[traj_ind], all_intrinsics[traj_ind], all_extr_cand[traj_ind], selected_inds_fw, all_poses[traj_ind], all_points[traj_ind], prior_scale)
                rmse_fw_gt_l = exp_utils.compute_rmse(all_measurements[traj_ind], all_poses[traj_ind], all_points[traj_ind], all_intrinsics[traj_ind], all_extr_cand[traj_ind], selected_inds_fw, all_poses[traj_ind], all_points[traj_ind], prior_scale, loc=True)
                rmse_fw_slam[select_k].append(rmse_fw)
                rmse_fw_loc[select_k].append(rmse_fw_l)
                rmse_fw_gt_slam[select_k].append(rmse_fw_gt)
                rmse_fw_gt_loc[select_k].append(rmse_fw_gt_l)

            visualize.reset()

        for select_k in range(2, max_select_k):
            rmse_errors_g = np.vstack((times_g[select_k], greedy_scores[select_k], rmse_greedy_slam[select_k], rmse_greedy_loc[select_k],rmse_greedy_gt_slam[select_k], rmse_greedy_gt_loc[select_k])).T
            result_log_arr = np.hstack((np.array(greedy_selected_cands[select_k]), rmse_errors_g))

            rmse_errors_fw = np.vstack((iters_fw[select_k], times_fw[select_k], fw_scores[select_k], fw_scores_unrounded[select_k], rmse_fw_slam[select_k], rmse_fw_loc[select_k], rmse_fw_gt_slam[select_k], rmse_fw_gt_loc[select_k])).T
            result_log_arr = np.hstack((result_log_arr,np.array(fw_selected_cands[select_k]), np.array(fw_solution_unr_list[select_k]), rmse_errors_fw))
            '''
            EQUAL, RAND and STANDARD
            '''
            rmse_errors_equal = np.vstack((equal_scores[select_k],rmse_equal_slam[select_k], rmse_equal_loc[select_k])).T
            result_log_arr = np.hstack((result_log_arr, np.array(equal_selected_cands[select_k]), rmse_errors_equal))

            rmse_errors_stnd = np.vstack((stnd_scores[select_k], rmse_stnd_slam[select_k], rmse_stnd_loc[select_k])).T
            result_log_arr = np.hstack((result_log_arr, np.array(stnd_selected_cands[select_k]), rmse_errors_stnd))

            rmse_errors_rand = np.vstack((rand_scores[select_k], rmse_rand_slam[select_k], rmse_rand_loc[select_k])).T
            result_log_arr = np.hstack((result_log_arr, np.array(rand_selected_cands[select_k]), rmse_errors_rand))


            ''' Header : greedy_selection, time_greedy, greedy_score, rmse_slam_g, rmse_loc_g rmse_slam_gt_g, rmse_loc_gt_g, 
                         fw_selection, fw_unrounded, num_iters, time_fw, fw_score, fw_score_unrounded, rmse_slam, rmse_loc, rmse_slam_gt, rmse_loc_gt,
                         eq_selection, eq_score, rmse_eq_slam, rmse_eq_loc,
                         stnd_selection, stnd_score, rmse_stnd_slam, rmse_stnd_loc
                         rand_selection, rand_score, rmse_rand_slam, rmse_rand_loc'''
            np.savetxt(os.path.join(output_file_Path, "results_"+traj+"{}_.txt".format(select_k)), result_log_arr)

    ''' visualize the final selection with franke wolfe'''
    #visualize.show_camconfig_with_world([best_config_g, best_configs_fw], 3,["greedy", "franke_wolfe"],  K, poses, points)