from enum import Enum
from . import utilities
from . import visualize
import numpy as np
import matplotlib.pyplot as plt
from functools import partial
from typing import List, Optional
import time
import gtsam
from scipy.optimize import linprog, minimize_scalar
from gtsam.utils import plot
from ProblemBuilder import FIM as infmat
from gtsam import (DoglegOptimizer,
                   GenericProjectionFactorCal3_S2,
                   NonlinearFactorGraph, PinholeCameraCal3_S2, PriorFactorPoint3, PriorFactorPose3, Values)
from Experiments import exp_utils
from scipy.optimize import minimize, Bounds
from numpy import linalg as la

L = gtsam.symbol_shorthand.L
X = gtsam.symbol_shorthand.X

class Metric(Enum):
    logdet = 1
    min_eig = 2
    mse = 3

'''
methods to run the algorithms to minimize the expectation over multiple trajectories
'''
def greedy_selection(points, poses, K,pose_rots, pose_trans, Nc, metric= Metric.logdet, loc= False):
    num_rot_samples = len(pose_rots)
    num_trans_samples = len(pose_trans)
    avail_cand = np.ones((num_rot_samples * num_trans_samples, 1))
    last_best_extr = []

    best_extr = []
    best_score= 0.0
    best_graph = None
    best_gtvals= None
    #build the prior FIM
    num_poses = len(poses)
    num_points = len(points)
    h_prior = np.zeros((num_poses*6+num_points*3 , num_poses*6+num_points*3))
    h_prior[-num_poses*6: , -num_poses*6:] = np.eye(num_poses*6)
    h_prior[0 : -num_poses*6, 0: -num_poses*6: ] =  np.eye(num_points*3)
    # For each camera
    for i in range(0, Nc):
        max_inf = 0.0
        last_best_extr = best_extr
        selected_cand = 0
        for j, trans in enumerate(pose_trans):
            for k, rot in enumerate(pose_rots):
                if avail_cand[j*num_rot_samples + k, 0] == 1:
                    extr_cand = last_best_extr.copy()
                    c1 = gtsam.Pose3(gtsam.Rot3(rot), gtsam.Point3(trans[0], trans[1] ,trans[2]))
                    extr_cand.append(c1)
                    points_fil, points_mask = check_and_filter_points(poses, points, extr_cand, K, False)
                    if loc:
                        result, graph = getMLE_multicam_loc(poses, points_fil, extr_cand, K)
                    else:
                        graph, gtvals, poses_mask, points_mask = getMLE_multicam(poses, points, extr_cand, K,points_mask)
                    gt_vals = gtsam.Values()
                    if not loc:
                        gt_vals = gtvals
                    else:
                        for i1, pose in enumerate(poses):
                            gt_vals.insert(X(i1), pose)
                    fim, crlb = infmat.compute_CRLB(gt_vals, graph)
                    #build full info matrix
                    h_full = np.zeros((num_poses*6+num_points*3 , num_poses*6+num_points*3))
                    num_ps = np.count_nonzero(poses_mask)
                    num_lms = np.count_nonzero(points_mask)
                    tot_num_points = len(points)

                    '''
                    Extract all the block diagonal elements corresponding to the poses (bottom right square block)
                    into the full hessian
                    '''
                    pose_inds = []
                    fim_idx = 0
                    for val in gt_vals.keys():
                        sym = gtsam.Symbol(val)
                        if sym.chr() == ord('x'):
                            h_full_pose_idx = sym.index()
                            pose_inds.append(h_full_pose_idx)
                            h_full_pose_idx_start = tot_num_points*3 + h_full_pose_idx * 6
                            fim_pose_idx_start = num_lms * 3 + fim_idx * 6
                            h_full[ h_full_pose_idx_start : h_full_pose_idx_start + 6, h_full_pose_idx_start : h_full_pose_idx_start + 6 ] =  fim[fim_pose_idx_start: fim_pose_idx_start + 6 , fim_pose_idx_start: fim_pose_idx_start + 6]
                            fim_idx = fim_idx + 1

                    fim_idx=0
                    for val in gt_vals.keys():
                        sym = gtsam.Symbol(val)
                        if sym.chr() == ord('l'):
                            idx = sym.index()
                            h_full[idx*3: (idx+1)*3, idx*3: (idx+1)*3] = fim[fim_idx*3: (fim_idx+1)*3, fim_idx*3: (fim_idx+1)*3 ]
                            p_idx_fim =0
                            for pose_idx in pose_inds:
                                h_full_pose_idx_start = tot_num_points*3 + pose_idx*6
                                fim_pose_idx_start = num_lms * 3 + p_idx_fim * 6
                                h_full[idx*3: (idx+1)*3, h_full_pose_idx_start: h_full_pose_idx_start + 6] = fim[fim_idx*3: (fim_idx+1)*3, fim_pose_idx_start: fim_pose_idx_start + 6]
                                h_full[h_full_pose_idx_start: h_full_pose_idx_start + 6 , idx * 3: (idx + 1) * 3] = fim[fim_pose_idx_start: fim_pose_idx_start + 6 , fim_idx * 3: (fim_idx + 1) * 3]
                                p_idx_fim = p_idx_fim +1
                            fim_idx = fim_idx + 1
                    # if fim.shape == h_full.shape:
                    #     print(np.array_equal(fim, h_full))
                    assert (utilities.check_symmetric(h_full))
                    h_full = h_full + h_prior
                    least_fim_eig = 0.0
                    if not loc:
                        fim = infmat.compute_schur_fim(h_full, len(poses))
                        #least_fim_eig = math.log(np.linalg.det(np.eye(fim.shape[0]) + fim))
                        if metric == Metric.logdet:
                            sign, least_fim_eig = np.linalg.slogdet(fim)
                            least_fim_eig = sign * least_fim_eig
                        if metric == Metric.min_eig:
                            assert(utilities.check_symmetric(fim))
                            least_fim_eig = np.linalg.eigvalsh(fim)[0]
                    #least_fim_eig = compute_logdet(fim)

                    if least_fim_eig > max_inf:
                        max_inf = least_fim_eig
                        best_extr = extr_cand
                        selected_cand = j*num_rot_samples + k
                        best_graph = graph
                        best_gtvals = gt_vals
        best_score = max_inf
        avail_cand[selected_cand] = 0
        print("Best Score till now: "+ str(best_score))
        print("Next best Camera is: ")
        print(best_extr[-1])
        print("------------------")

    print("Selected candidates are : ")
    print(np.argwhere(avail_cand.flatten()==0))

    '''
    Compute the RMSE for the best camera placement
    '''
    ''' Give some initial values which are noisy'''
    initial_estimate = Values()
    for val in best_gtvals.keys():
        '''
        If the resl variable is a pose, store the values
        '''
        if gtsam.Symbol(val).chr() == ord('x'):
            pose = best_gtvals.atPose3(val)
            transformed_pose = pose.retract(0.1 * np.random.randn(6, 1))
            initial_estimate.insert(val, transformed_pose)
        elif gtsam.Symbol(val).chr() == ord('l'):
            point = best_gtvals.atPoint3(val)
            transformed_point = point + 0.1 * np.random.randn(3)
            initial_estimate.insert(val, transformed_point)

    # Optimize the graph and print results
    params = gtsam.DoglegParams()
    #params.setVerbosity('ERROR')
    optimizer = DoglegOptimizer(best_graph, initial_estimate, params)
    try:
        result = optimizer.optimize()
    except Exception:
        result = Values()
        pass
    #result.print('Final results:\n')
    rmse = exp_utils.compute_traj_error(result, poses)
    print("The RMSE of the estimated trajectory with best camera placement: "+ str(rmse))

    return best_extr, max_inf, rmse, avail_cand

def greedy_selection_new(measurements, intrinsics, all_cands, points, poses, Nc, h_prior,  metric= Metric.logdet):
    avail_cand = np.ones((len(all_cands), 1))
    # build the prior FIM
    num_poses = len(poses)
    num_points = len(points)
    best_selection_indices = []
    best_config = []
    best_score = 0.0

    # For each camera
    for i in range(0, Nc):
        max_inf = 0.0
        selected_cand = 0
        for j, cand in enumerate(all_cands):
            if avail_cand[j, 0] == 1:
                cur_cands = best_config.copy()
                cur_cands.append(cand)
                cur_selection = best_selection_indices.copy()
                cur_selection.append(j)
                h_full,  graph, gtvals, poses_mask, points_mask = infmat.build_hfull(measurements, points, poses, intrinsics, all_cands, cur_selection)
                h_full = h_full + h_prior
                least_fim_eig = 0.0

                fim = infmat.compute_schur_fim(h_full, len(poses))
                # least_fim_eig = math.log(np.linalg.det(np.eye(fim.shape[0]) + fim))
                if metric == Metric.logdet:
                    sign, least_fim_eig = np.linalg.slogdet(fim)
                    least_fim_eig = sign * least_fim_eig
                    # print(np.linalg.det(fim))
                    # print(least_fim_eig)
                    # print("-------------------------")
                if metric == Metric.min_eig:
                    assert (utilities.check_symmetric(fim))
                    #print( np.linalg.eigvalsh(fim)[0:8])
                    least_fim_eig = np.linalg.eigvalsh(fim)[0]
                # least_fim_eig = compute_logdet(fim)

                if least_fim_eig > max_inf:
                    max_inf = least_fim_eig
                    selected_cand = j

        best_score = max_inf
        best_config.append(all_cands[selected_cand])
        best_selection_indices.append(selected_cand)
        avail_cand[selected_cand] = 0
        print("Best Score till now: " + str(best_score))
        print("Next best Camera is: ")
        print(best_config[-1])
        print("------------------")

    print("Selected candidates are : ")
    print(np.argwhere(avail_cand.flatten() == 0))

    return best_config, best_selection_indices, best_score

def greedy_selection_exp(measurements, intrinsics, all_cands, points, poses, Nc, h_prior, num_runs, metric= Metric.logdet):
    avail_cand = np.ones((len(all_cands), 1))
    best_selection_indices = []
    best_config = []
    best_score = 0.0
    # For each camera
    for i in range(0, Nc):
        max_inf = 0.0
        selected_cand = 0
        for j, cand in enumerate(all_cands):
            if avail_cand[j, 0] == 1:
                cur_cands = best_config.copy()
                cur_cands.append(cand)
                cur_selection = best_selection_indices.copy()
                cur_selection.append(j)
                least_fim_eig = 0.0
                # Here we should build H for all trajectories, find schur and eig values for all trahectories and maximize that sum
                for traj_ind in range(num_runs):
                    h_full,  graph, gtvals, poses_mask, points_mask = infmat.build_hfull(measurements[traj_ind], points[traj_ind], poses[traj_ind], intrinsics[traj_ind], all_cands, cur_selection)
                    h_full = h_full + h_prior
                    fim = infmat.compute_schur_fim(h_full, len(poses[traj_ind]))
                    # least_fim_eig = math.log(np.linalg.det(np.eye(fim.shape[0]) + fim))
                    if metric == Metric.logdet:
                        sign, least_fim_eig = np.linalg.slogdet(fim)
                        least_fim_eig = sign * least_fim_eig
                    if metric == Metric.min_eig:
                        assert (utilities.check_symmetric(fim))
                        #print( np.linalg.eigvalsh(fim)[0:8])
                        least_fim_eig = least_fim_eig + np.linalg.eigvalsh(fim)[0]
                    # least_fim_eig = compute_logdet(fim)

                if least_fim_eig > max_inf:
                    max_inf = least_fim_eig
                    selected_cand = j
                    # best_graph = graph
                    # best_gtvals = gt_vals
        best_score = max_inf
        best_config.append(all_cands[selected_cand])
        best_selection_indices.append(selected_cand)
        avail_cand[selected_cand] = 0
        print("Best Score till now: " + str(best_score))
        print("Next best Camera is: ")
        print(best_config[-1])
        print("------------------")

    print("Selected candidates are : ")
    print(np.argwhere(avail_cand.flatten() == 0))

    return best_config, best_selection_indices, best_score

def gen_frank_wolfe_exp(inf_mats, H0, n_iters, selection_init, k, num_poses, num_runs, A, b):
    """
    Franke-Wolfe optimization method for multiple trajectories with added constraints.

    Arguments:
    inf_mats -- The information matrices for each trajectory.
    H0 -- Prior information matrix.
    n_iters -- Number of iterations.
    selection_init -- Initial sensor selection.
    k -- Number of sensors to select.
    num_poses -- Number of poses.
    num_runs -- Number of runs/trajectories.
    A -- Constraint matrix (for cost, weight, etc.).
    b -- Constraint vector (limits for cost, weight, etc.).

    Returns:
    final_solution -- Final sensor selection after optimization.
    selection_cur -- Current selection at the end of optimization.
    min_eig_val_rounded -- Minimum eigenvalue of the rounded solution.
    min_eig_val_unrounded -- Minimum eigenvalue of the unrounded solution.
    i -- Final iteration count.
    """
    selection_cur = selection_init
    u_i = float("inf")
    prev_min_eig_score = 0

    for i in range(n_iters):
        # Compute gradient
        grad = np.zeros(selection_cur.shape)
        min_eig_val_score = 0.0
        for traj_ind in range(num_runs):
            # Compute the minimum eigenvalue and eigenvector
            min_eig_val, min_eig_vec, final_inf_mat = infmat.find_min_eig_pair(inf_mats[traj_ind], selection_cur, H0, num_poses)
            min_eig_val_score += min_eig_val

            # Compute gradient for each trajectory
            Hxx = final_inf_mat[-num_poses * 6:, -num_poses * 6:]
            Hll = final_inf_mat[0: -num_poses * 6, 0: -num_poses * 6:]
            Hlx = final_inf_mat[0: -num_poses * 6, -num_poses * 6:]
            for ind in range(selection_cur.shape[0]):
                Hc = inf_mats[traj_ind][ind]
                Hxx_c = Hc[-num_poses * 6:, -num_poses * 6:]
                Hll_c = Hc[0: -num_poses * 6, 0: -num_poses * 6:]
                Hlx_c = Hc[0: -num_poses * 6, -num_poses * 6:]
                t0 = Hlx.T
                t1 = np.linalg.pinv(Hll)
                t2 = t0 @ t1
                grad_schur = Hxx_c - (Hlx_c.T @ t1 @ t0.T  - t2 @ Hll_c @ t1 @ t0.T + t2 @ Hlx_c)
                grad[ind] += min_eig_vec.T @ grad_schur @ min_eig_vec

        # Solve the constrained linear subproblem using linprog
        result = linprog(c=-grad, A_ub=A, b_ub=b, bounds=(0, 1), method='highs')

        # The rounded solution from linprog
        rounded_sol = result.x

        # Stopping criteria
        if abs(min_eig_val_score - prev_min_eig_score) < 1e-4:   #1e-8 for prior of 1
            break

        # Step size
        alpha = 1.0 / (i + 3.0)
        print(f"step size: {alpha:.9f}, iter : {i:.9f}, gradient norm : {np.linalg.norm(grad):.9f}, min eig : {min_eig_val:.15f}")

        prev_min_eig_score = min_eig_val_score

        # Update the selection
        selection_cur = selection_cur + alpha * (rounded_sol - selection_cur)

    print("ended the optimization")
    print(f"norm of the gradient : {np.linalg.norm(selection_cur):.6f}")
    print(selection_cur)

    # Final solution rounding
    final_solution = roundsolution(selection_cur, k)
    print(final_solution)

    # Compute the eigenvalues for both rounded and unrounded solutions
    min_eig_val_unrounded = 0.0
    min_eig_val_rounded = 0.0
    for traj_ind in range(num_runs):
        # Unrounded solution eigenvalue
        min_eig_val, _, _ = infmat.find_min_eig_pair(inf_mats[traj_ind], selection_cur, H0, num_poses)
        min_eig_val_unrounded += min_eig_val

        # Rounded solution eigenvalue
        min_eig_val, _, _ = infmat.find_min_eig_pair(inf_mats[traj_ind], final_solution, H0, num_poses)
        min_eig_val_rounded += min_eig_val

    return final_solution, selection_cur, min_eig_val_rounded, min_eig_val_unrounded, i

def frank_wolfe_exp(inf_mats, H0, n_iters, selection_init, k, num_poses, num_runs):
    selection_cur= selection_init
    u_i = float("inf")
    #this is now a sum of mineigs for all trajs
    prev_min_eig_score = 0
    for i in range(n_iters):
        #Compute gradient
        grad = np.zeros(selection_cur.shape)
        min_eig_val_score = 0.0
        for traj_ind in range(num_runs):
            # compute the minimum eigen value and eigen vector
            min_eig_val, min_eig_vec, final_inf_mat = infmat.find_min_eig_pair(inf_mats[traj_ind], selection_cur, H0, num_poses)
            min_eig_val_score = min_eig_val_score + min_eig_val
            ''' required for gradient of schur'''
            Hxx = final_inf_mat[-num_poses * 6:, -num_poses * 6:]
            Hll = final_inf_mat[0: -num_poses * 6, 0: -num_poses * 6:]
            Hlx = final_inf_mat[0: -num_poses * 6, -num_poses * 6:]
            for ind in range(selection_cur.shape[0]):
                #grad[ind] = min_eig_vec.T @ inf_mats[ind] @ min_eig_vec
                #gradient schur
                Hc = inf_mats[traj_ind][ind]
                Hxx_c = Hc[-num_poses * 6:, -num_poses * 6:]
                Hll_c = Hc[0: -num_poses * 6, 0: -num_poses * 6:]
                Hlx_c = Hc[0: -num_poses * 6, -num_poses * 6:]
                t0 = Hlx.T
                t1 = np.linalg.pinv(Hll)
                t2 = t0 @ t1
                grad_schur = Hxx_c - (Hlx_c.T @ t1 @ t0.T  - t2 @ Hll_c @ t1 @ t0.T + t2 @ Hlx_c )
                grad[ind] = grad[ind] + min_eig_vec.T @ grad_schur @ min_eig_vec

        #round the solution and pick top k
        #rounded_sol = roundsolution(grad, k)
        rounded_sol = roundsolution_madow(grad, k)
        #rounded_sol = roundsolution_breakties(grad, k,inf_mats, H0)

        # Compute dual upper bound from linear approximation
        # u_i = min( u_i, min_eig_val + grad @ (rounded_sol - selection_cur))
        #print("dual Gap: {:.9f}".format(u_i - min_eig_val))
        # if u_i - min_eig_val < 1e-5:
        #     break
        if abs(min_eig_val_score - prev_min_eig_score) < 1e-4:   #1e-8 for prior of 1
            break
        # Step size
        alpha = 1.0 / (i + 3.0)  # original 2.0 / (i + 2.0). Was playing around with this.
        print("step size: {:.9f}, iter : {:.9f}, gradient norm : {:.9f}, min eig : {:.15f}".format(alpha, i,
                                                                                                 np.linalg.norm(grad),
                                                                                                 min_eig_val))
        prev_min_eig_score = min_eig_val_score
        # temp_solution = roundsolution_breakties(selection_cur, k, inf_mats, H0)
        # min_eig_val_temp, _, _ = find_min_eig_pair(inf_mats, temp_solution, H0, num_poses)
        # min_eig_val_temp_unr, _, _ = find_min_eig_pair(inf_mats, selection_cur, H0, num_poses)
        # print("temp solution score rounded:{:.9f}, unrounded: {:.9f}".format(min_eig_val_temp,min_eig_val_temp_unr ))


        selection_cur = selection_cur + alpha * (rounded_sol - selection_cur)

    print("ended the optimization")
    #final_solution = roundsolution(selection_cur, k)
    print("norm of the gradient : {:.6f}".format(np.linalg.norm(selection_cur)))
    print(selection_cur)

    #final_solution = roundsolution_breakties(selection_cur, k, inf_mats, H0)
    final_solution = roundsolution(selection_cur, k)
    print(final_solution)
    min_eig_val_unrounded = 0.0
    min_eig_val_rounded = 0.0
    for traj_ind in range(num_runs):
        # compute the minimum eigen value and eigen vector
        min_eig_val, _, _ = infmat.find_min_eig_pair(inf_mats[traj_ind], selection_cur, H0, num_poses)
        min_eig_val_unrounded = min_eig_val_unrounded + min_eig_val
        min_eig_val, _, _ = infmat.find_min_eig_pair(inf_mats[traj_ind], final_solution, H0, num_poses)
        min_eig_val_rounded = min_eig_val_rounded + min_eig_val

    return final_solution, selection_cur, min_eig_val_rounded, min_eig_val_unrounded, i


def run_single_experiment_exp(poses, points, measurements, intrinsics, extr_cand, select_k, h_prior, num_runs):

    ''' Perform greedy selection method using minimum eigen value metric '''
    s_g = time.time()
    best_config_g, best_selection_indices, best_score_g = greedy_selection_exp(
        measurements, intrinsics, extr_cand, points, poses, select_k, h_prior, num_runs, metric=Metric.min_eig)
    e_g = time.time()
    time_greedy = e_g - s_g

    print("The score for traj greedy: {:.9f} ".format(best_score_g))

    ''' Construct factor graph and infomats '''
    s_opt_prep = time.time()
    all_inf_mats = []
    all_debug_nr_facs = []
    for traj_ind in range(num_runs):
        inf_mats, debug_nr_facs = infmat.construct_candidate_inf_mats(measurements[traj_ind], intrinsics[traj_ind], extr_cand, points[traj_ind], poses[traj_ind])
        all_inf_mats.append(inf_mats)
        all_debug_nr_facs.append(debug_nr_facs)
    num_cands = len(extr_cand)
    e_opt_prep = time.time()

    ''' Define A matrix and b vector for constraints (e.g., select exactly/at most k candidates) '''
    A = np.ones((1, num_cands))
    b = np.array([select_k])

    # Generate all pairs of candidate indices
    for pair in combinations(range(num_cands), 2):
        exclusivity_row = np.zeros(num_cands)  # Create a row of zeros
        exclusivity_row[pair[0]] = 1           # Set the first candidate of the pair
        exclusivity_row[pair[1]] = 1           # Set the second candidate of the pair
        A = np.vstack([A, exclusivity_row])    # Add this row to the A matrix
        b = np.append(b, 1) 

    ''' Initial selection (equal weights or based on greedy) '''
    selection_init = np.ones(num_cands)
    selection_init = selection_init * select_k / num_cands
    num_poses = len(poses[0])

    ''' Run Frank-Wolfe with A and b constraints '''
    s_f = time.time()
    print("################# FRANK-WOLFE SELECTION ############################")
    selection_fw, selection_fw_unr, cost_fw, cost_fw_unrounded, num_iters = frank_wolfe_exp(
        all_inf_mats, h_prior, 600, selection_init.flatten(), select_k, num_poses, num_runs)

    e_f = time.time()
    print("The Score for traj franke_wolfe with solution. rounded: {:.9f}, unrounded: {:.9f} ".format(cost_fw, cost_fw_unrounded))
    print("selection: ")
    print(np.argwhere(selection_fw == 1))

    time_fw = (e_opt_prep - s_opt_prep) + (e_f - s_f)

    ''' Return results '''
    best_selection_indices_fw = []
    best_configs_fw = []
    for i in range(selection_fw.shape[0]):
        if selection_fw[i] == 1:
            best_configs_fw.append(extr_cand[i])
            best_selection_indices_fw.append(i)

    return best_score_g, best_config_g, best_selection_indices, time_greedy, cost_fw, cost_fw_unrounded, best_configs_fw,\
           best_selection_indices_fw, selection_fw_unr, time_fw, num_iters

def check_and_filter_points(poses, points, extrinsics, K, toplot=False):
    if toplot:
        plt.ion()
    fig1, ax1 = visualize.initialize_3d_plot(number=1, limits=np.array([[-30, 30], [-30, 30], [-30, 30]]),
                                                    view=[-30, -90])
    dict = {}
    # Define the camera observation noise model
    measurement_noise = gtsam.noiseModel.Isotropic.Sigma(2, 1.0)  # one pixel in u and v
    # Simulated measurements from each camera pose, adding them to the factor graph
    for i, pose in enumerate(poses):
        for point in points:
            if toplot:
                plot.plot_point3_on_axes(ax1, point, 'ro')
        for k, comp_pose in enumerate(extrinsics):

            # compose pose with comp_pose T_wb * T_bc = T_wc, we get the pose of component camera W.R.T world
            pose_wc = pose.compose(comp_pose)
            if toplot:
                plot.plot_pose3_on_axes(ax1, pose_wc, axis_length=1, P=None, scale=1)
            camera = PinholeCameraCal3_S2(pose_wc, K)
            # print(pose)
            # print(comp_pose)
            # print(pose_wc)
            for j, point in enumerate(points):
                try:
                    measurement = camera.project(point)
                    if (measurement[0] > 1 and measurement[0] < (2 * K.px() - 2) and measurement[1] > 1 and measurement[
                        1] < (2 * K.py() - 2)):
                        measurement = measurement + 1.0 * np.random.randn(2)
                        factor = GenericProjectionFactorCal3_S2(
                            measurement, measurement_noise, X(i), L(j), K, comp_pose)
                        if toplot:
                            plot.plot_point3_on_axes(ax1, point, 'b*')
                        if dict.get(j) is None:
                            dict[j] = 1
                        else:
                            dict[j] = dict[j] + 1
                    else:
                        # print("Measurement is out of bounds: ")
                        # print(measurement)
                        pass

                except Exception:
                    pass
                    # print("Exception at Point")
                    # print(point)
        if toplot:
            fig1.canvas.draw()
            fig1.canvas.flush_events()
            ax1.cla()
            #time.sleep(0.2)
            ax1.set_xlabel("X_axis")
            ax1.set_ylabel("Y_axis")
            ax1.set_zlabel("Z_axis")

    #plt.show()
    #This is not needed for localizaruion
    rm_indices = []
    # for k, v in dict.items():
    #     if v < 3:
    #         print("lm index: " + str(k) + ", lm value : " + str(points[k]))
    #         rm_indices.append(k)
    final_points_gtsam = []
    points_mask = np.ones(len(points))
    for i, pt in enumerate(points):
        if i not in rm_indices and i in dict.keys():
            final_points_gtsam.append(pt)
        else:
            points_mask[i] = 0
    #print(len(final_points_gtsam))
    return final_points_gtsam, points_mask

def error_pose(measurement: np.ndarray, K: np.ndarray, body_p_sensor:np.ndarray, point: np.ndarray,  this: gtsam.CustomFactor, values: gtsam.Values, J: Optional[List[np.ndarray]]):
    """
    This is he error function for custom factor for localization only given fixed landmarks
    """
    key = this.keys()[0]
    pos = values.atPose3(key)
    h0 = np.zeros((6, 6), order='F')
    h00 = np.zeros((6, 6), order='F')
    pose_wc = pos.compose(body_p_sensor, h0, h00)
    h0_hat = body_p_sensor.inverse().AdjointMap()
    Dpose = np.zeros((2, 6), order='F')
    Dpoint = np.zeros((2, 3), order='F')
    Dcal = np.zeros((2, 5), order='F')
    camera = PinholeCameraCal3_S2(pose_wc, K)
    measurement_hat = camera.project(point, Dpose, Dpoint, Dcal)
    error = measurement_hat - measurement
    if J is not None:
        J[0] = Dpose @ h0
    return error

def getMLE_multicam_loc(poses, points, extrinsics, K):

    # Define the camera observation noise model
    measurement_noise = gtsam.noiseModel.Isotropic.Sigma(2, 1.0)  # one pixel in u and v

    # Create a factor graph
    graph = NonlinearFactorGraph()

    # Add a prior on pose x1. This indirectly specifies where the origin is.
    # 0.3 rad std on roll,pitch,yaw and 0.1m on x,y,z
    pose_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.3, 0.3, 0.3, 0.1, 0.1, 0.1]))

    dict = {}
    # Simulated measurements from each camera pose, adding them to the factor graph
    for i, pose in enumerate(poses):
        for k, comp_pose in enumerate(extrinsics):
            #compose pose with comp_pose T_wb * T_bc = T_wc, we get the pose of component camera W.R.T world
            pose_wc = pose.compose(comp_pose)
            camera = PinholeCameraCal3_S2(pose_wc, K)
           # print(pose)
            #print(comp_pose)
            #print(pose_wc)
            for j, point in enumerate(points):
                try:
                    measurement = camera.project(point)
                    if(measurement[0] > 1 and measurement[0] < (2*K.px()-2) and measurement[1] > 1 and measurement[1] < (2*K.py()-2)):
                        measurement = measurement + 1.0 * np.random.randn(2)
                        factor = gtsam.CustomFactor(measurement_noise, gtsam.KeyVector([X(i)]), partial(error_pose, measurement, K,comp_pose, point ))
                        graph.add(factor)

                        #print("projection factor x"+str(i)+" - l"+str(j))
                        if dict.get(i) is None:
                            dict[i] = [graph.nrFactors()-1]
                        else:
                            dict[i] = dict[i] + [graph.nrFactors()-1]
                    else:
                        #print("Measurement is out of bounds for L" + str(j))
                        #print(measurement)
                        pass

                except Exception:
                    #print("exception at L" + str(j))
                    pass
                    #print("Exception at Point")
                    #print(point)

    # for k, v in dict.items():
    #     print("Number of factors for pose: " + str(k)+" factors : "+str(len(v)))

    # rm_indices = []
    # rm_lm_indices=[]
    #
    # for k, v in dict.items():
    #     #print("lm index: " + str(k)+" factors : "+str(v))
    #     if len(v) < 2:
    #         print("lm index: " + str(k) + ", lm value : " + str(points[k]))
    #         rm_indices= rm_indices + v
    #         rm_lm_indices.append(k)
    #
    # for i in rm_indices:
    #     graph.remove(i)


    # Create the data structure to hold the initial estimate to the solution
    # Intentionally initialize the variables off from the ground truth
    initial_estimate = Values()
    for i, pose in enumerate(poses):
        transformed_pose = pose.retract(0.1 * np.random.randn(6, 1))
        initial_estimate.insert(X(i), transformed_pose)

    factor = PriorFactorPose3(X(0), poses[0], pose_noise)
    graph.add(factor)
    print("Number of factors in graph" + str(graph.nrFactors()))


    # # Optimize the graph and print results
    # params = gtsam.LevenbergMarquardtParams()
    # optimizer = gtsam.LevenbergMarquardtOptimizer(graph, initial_estimate, params)
    #
    # #params = gtsam.DoglegParams()
    # # params.setVerbosity('DELTA')
    # #optimizer = DoglegOptimizer(graph, initial_estimate, params)
    #
    # # params = gtsam.GaussNewtonParams()
    # # optimizer = gtsam.GaussNewtonOptimizer(graph, initial_estimate, params)
    #
    # # Optimize the factor graph
    # result = optimizer.optimize()
    # #print('Optimizing:')
    # try:
    #     result = optimizer.optimize()
    # except Exception:
    #     result = Values()
    #     pass
    # #result.print("result")
    result = Values()
    return result, graph

def getMLE_multicam(poses, points, extrinsics, K, points_mask):

    # Define the camera observation noise model
    measurement_noise = gtsam.noiseModel.Isotropic.Sigma(2, 1.0)  # one pixel in u and v

    # Create a factor graph
    graph = NonlinearFactorGraph()

    # Add a prior on pose x1. This indirectly specifies where the origin is.
    # 0.3 rad std on roll,pitch,yaw and 0.1m on x,y,z
    pose_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.3, 0.3, 0.3, 0.1, 0.1, 0.1]))
    #factor = PriorFactorPose3(X(0), poses[0], pose_noise)
    #graph.push_back(factor)
    #print("prior factor x0")
    dict = {}
    dict_lm_poses={}
    # Simulated measurements from each camera pose, adding them to the factor graph
    for i, pose in enumerate(poses):
        for j, point in enumerate(points):
            if not points_mask[j]:
                continue
            for k, comp_pose in enumerate(extrinsics):
                # compose pose with comp_pose T_wb * T_bc = T_wc, we get the pose of component camera W.R.T world
                pose_wc = pose.compose(comp_pose)
                camera = PinholeCameraCal3_S2(pose_wc, K)
                try:
                    measurement = camera.project(point)
                    # Check if the point is in the frustum of the camera
                    if (measurement[0] > 1 and measurement[0] < (2 * K.px() - 2) and measurement[1] > 1 and measurement[
                        1] < (2 * K.py() - 2)):
                        measurement = measurement + 1.0 * np.random.randn(2)
                        factor = GenericProjectionFactorCal3_S2(
                            measurement, measurement_noise, X(i), L(j), K, comp_pose)
                        graph.push_back(factor)

                        # print("projection factor x"+str(i)+" - l"+str(j))
                        #if tgl:
                        if dict.get(j) is None:
                            dict[j] = [graph.nrFactors() - 1]
                        else:
                            dict[j] = dict[j] + [graph.nrFactors() - 1]

                        if dict_lm_poses.get(j) is None:
                            dict_lm_poses[j] = {i}
                        else:
                            dict_lm_poses[j].add(i)
                           # tgl = False
                    else:
                        # print("Measurement is out of bounds for L" + str(j))
                        # print(measurement)
                        pass

                except Exception:
                    # print("exception at L" + str(j))
                    pass
                    # print("Exception at Point")
                    # print(point)

    rm_indices = []
    rm_lm_indices=[]
    for k, v in dict_lm_poses.items():
        #print("lm index: " + str(k)+" factors : "+str(v))
        if len(v) < 2:
            #print("lm index: " + str(k) + ", lm value : " + str(points[k]))
            rm_indices = rm_indices + dict[k]
            rm_lm_indices.append(k)
    #print(graph.keys())
    for i in rm_indices:
        graph.remove(i)
    #print(graph.keys())
    #print("Number of factors in graph"+str(graph.nrFactors()))
    ###graph.print('Factor Graph:\n')

    gt_vals = Values()
    # when we dont have enough constraints for poses.
    num_lms = 0
    num_ps = 0
    pose_mask= np.zeros(len(poses))
    for k in graph.keyVector():
        sym = gtsam.Symbol(k)
        # print(sym.chr())
        if sym.chr() == ord('x'):
            num_ps = num_ps + 1
            pose_idx = sym.index()
            gt_vals.insert(X(pose_idx), poses[pose_idx])
            #print('x'+str(pose_idx))
            pose_mask[pose_idx] = 1
        if sym.chr() == ord('l'):
            num_lms = num_lms + 1
            lm_idx = sym.index()
           # print("l" + str(lm_idx))

    # for i, pose in enumerate(poses):
    #     gt_vals.insert(X(i), pose)

    ins_prior = True
    point_noise = gtsam.noiseModel.Isotropic.Sigma(3, 0.1)
    for j, point in enumerate(points):
        if j not in rm_lm_indices and j in dict.keys():
            if ins_prior:
                factor = PriorFactorPoint3(L(j), point, point_noise)
                # graph.push_back(factor)
                ins_prior = False
            gt_vals.insert(L(j), point)
            #print("l"+str(j))
        else:
            points_mask[j] = 0


    #print("num of poses in graph: "+ str(num_ps))
    #print("num of lms in graph: " + str(num_lms))
    if num_ps != len(poses):
        #print("not all poses have observed enough landmarks")
        pass
    #print(graph.keys())
    print("num of factors: "+str(graph.size()))
    print("Number of factors in graph" + str(graph.nrFactors()))
    return graph, gt_vals, pose_mask, points_mask


def roundsolution(selection,k):
    idx = np.argpartition(selection, -k)[-k:]
    rounded_sol = np.zeros(len(selection))
    if k > 0:
        rounded_sol[idx] = 1.0
    return rounded_sol

def roundsolution_breakties(selection,k, all_mats, H0):
    s_rnd = np.round(selection, decimals=5)
    # print(np.argsort(s_rnd))
    # print(s_rnd[np.argsort(s_rnd)])
    all_eigs= []
    for m in all_mats:
        m_p = H0 + m
        assert (utilities.check_symmetric(m_p))
        eigvals, _ = la.eigh(m_p)
        all_eigs.append(eigvals[0])
    all_eigs = np.array(all_eigs)

    zipped_vals = np.array([(s_rnd[i], all_eigs[i]) for i in range(len(s_rnd))], dtype=[('w', 'float'), ('weight', 'float')])
    idx = np.argpartition(zipped_vals, -k, order=['w', 'weight'])[-k:]
    rounded_sol = np.zeros(len(s_rnd))
    if k > 0:
        rounded_sol[idx] = 1.0
    return rounded_sol

def roundsolution_madow(selection, k):
    num = len(selection)
    phi = np.zeros(num)
    rounded_sol = np.zeros(num)
    phi[1:] = np.cumsum(selection[1:])
    u = np.random.rand()

    for i in range(k):
        for j in range(num):
            if (phi[j] <= u+i) and (u+i < phi[j+1]):
                if rounded_sol[j] == 1:
                    continue
                rounded_sol[j] = 1
                break
    print("number of candidate selected after rounding" + str(np.sum(rounded_sol)))
    return rounded_sol


'''
################################################################
Scipy optimization methods
'''
def min_eig_obj_with_jac(x, inf_mats, H0, num_poses):
    # compute the minimum eigen value and eigen vector
    min_eig_val, min_eig_vec, final_inf_mat = find_min_eig_pair(inf_mats, x, H0, num_poses)
    # Compute gradient
    grad = np.zeros(x.shape)
    ''' required for gradient of schur'''
    Hxx = final_inf_mat[-num_poses * 6:, -num_poses * 6:]
    Hll = final_inf_mat[0: -num_poses * 6, 0: -num_poses * 6:]
    Hlx = final_inf_mat[0: -num_poses * 6, -num_poses * 6:]
    for ind in range(x.shape[0]):
        # grad[ind] = min_eig_vec.T @ inf_mats[ind] @ min_eig_vec
        # gradient schur
        Hc = inf_mats[ind]
        Hxx_c = Hc[-num_poses * 6:, -num_poses * 6:]
        Hll_c = Hc[0: -num_poses * 6, 0: -num_poses * 6:]
        Hlx_c = Hc[0: -num_poses * 6, -num_poses * 6:]
        t0 = Hlx.T
        t1 = np.linalg.pinv(Hll)
        t2 = t0 @ t1
        grad_schur = Hxx_c - (Hlx_c.T @ t1 @ t0.T - t2 @ Hll_c @ t1 @ t0.T + t2 @ Hlx_c)
        grad[ind] = min_eig_vec.T @ grad_schur @ min_eig_vec
    return -1.0*min_eig_val, -1.0*grad

def scipy_minimize(inf_mats,H0, selection_init, k,num_poses):
    bounds = tuple([(0,1) for i in range(selection_init.shape[0])])
    cons = (
        { 'type': 'eq', 'fun': lambda x : np.sum(x) - k}
    )
    res = minimize(min_eig_obj_with_jac, selection_init, method='trust-constr', jac=True, args=(inf_mats,H0, num_poses),
                   constraints= cons, bounds=bounds, options={'disp': True})
    print(res.x)
    rounded_sol = roundsolution(res.x, k)
    print(rounded_sol)
    min_eig_val_unr, _, _ = find_min_eig_pair(inf_mats, res.x, H0, num_poses)
    min_eig_val_rounded, _, _ = find_min_eig_pair(inf_mats, rounded_sol, H0, num_poses)
    return rounded_sol,res.x,  min_eig_val_rounded, min_eig_val_unr