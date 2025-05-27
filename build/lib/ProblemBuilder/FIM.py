from OASIS import utilities
from gtsam import Point3, Cal3_S2, PinholeCameraCal3_S2
import numpy as np
import time
import gtsam
from gtsam import (DoglegOptimizer, LevenbergMarquardtOptimizer,
                    GenericProjectionFactorCal3_S2,
                    NonlinearFactorGraph,
                    PriorFactorPoint3, PriorFactorPose3,  Values)
from numpy import linalg as la

from scipy.optimize import minimize, Bounds

L = gtsam.symbol_shorthand.L
X = gtsam.symbol_shorthand.X

def compute_CRLB(vals, graph):
    lin_graph = graph.linearize(vals)
    hess = lin_graph.hessian()[0]
    cov = None
    # try:
    #     cov = np.linalg.inv(hess)
    # except Exception:
    #     print("Exception in inverse: info mat is singular")
    #     return hess, None

    return hess, cov

def compute_schur_fim(fim, num_poses):

    Hxx = fim[ -num_poses*6: , -num_poses*6: ]
    Hll = fim[0 : -num_poses*6, 0: -num_poses*6: ]
    Hlx = fim [0: -num_poses*6 , -num_poses*6 : ]

    Hxx_schur = Hxx - Hlx.T @ np.linalg.pinv(Hll) @ Hlx
    return Hxx_schur

def build_graph(measurements, poses, points, intrinsics, extrinsics, inds=[], rm_ill_posed=False):
    # Define the camera observation noise model
    measurement_noise = gtsam.noiseModel.Isotropic.Sigma(2, 1.0)  # one pixel in u and v

    # Create a factor graph
    graph = NonlinearFactorGraph()

    # Add a prior on pose x1. This indirectly specifies where the origin is.
    # 0.3 rad std on roll,pitch,yaw and 0.1m on x,y,z
    #pose_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.3, 0.3, 0.3, 0.1, 0.1, 0.1]))
    # factor = PriorFactorPose3(X(0), poses[0], pose_noise)
    # graph.push_back(factor)
    # print("prior factor x0")
    dict = {}
    dict_lm_poses = {}

    if len(inds) == 0:
        inds = range(len(extrinsics))
    # Simulated measurements from each camera pose, adding them to the factor graph
    for i, pose in enumerate(poses):
        for j, point in enumerate(points):
            for k in inds:
                comp_pose = extrinsics[k]
                # compose pose with comp_pose T_wb * T_bc = T_wc, we get the pose of component camera W.R.T world
                pose_wc = pose.compose(comp_pose)
                camera = PinholeCameraCal3_S2(pose_wc, intrinsics[k])
                measurement = measurements[i, k, j]
                if measurement[0] == 0 and measurement[1] == 0:
                    continue
                # measurement_true = camera.project(point)
                # if (measurement_true - measurement)[0] > 4.0 or (measurement_true - measurement)[1] > 4.0:
                #     print("big error: {:.6f}, {:.6f} ".format((measurement_true - measurement)[0],
                #                                               (measurement_true - measurement)[1]))
                #     cccc = 2.0
                #assert(np.sum(measurement_true-measurement) < 2.0)
                factor = GenericProjectionFactorCal3_S2(
                    measurement, measurement_noise, X(i), L(j), intrinsics[k], comp_pose)
                graph.push_back(factor)
                if dict.get(j) is None:
                    dict[j] = [graph.nrFactors() - 1]
                else:
                    dict[j] = dict[j] + [graph.nrFactors() - 1]

                if dict_lm_poses.get(j) is None:
                    dict_lm_poses[j] = {i}
                else:
                    dict_lm_poses[j].add(i)
                #
                # if dict_lm_facs.get(j) is None:
                #     dict_lm_facs[j] = [graph.nrFactors() - 1]
                # else:
                #     dict_lm_facs[j] = dict_lm_facs[j] + [graph.nrFactors() - 1]

    # print(graph.keys())
    # print("Number of factors in graph"+str(graph.nrFactors()))
    ###graph.print('Factor Graph:\n')
    rm_indices = []
    rm_lm_indices = []
    uniq_poses = set()
    for k, v in dict_lm_poses.items():
        # print("lm index: " + str(k)+" poses : "+str(v))
        # print(type(v))
        uniq_poses = uniq_poses.union(v)
        if len(v) < 2:
            # print("lm index: " + str(k) + ", lm value : " + str(points[k]))
            rm_indices = rm_indices + dict[k]
            rm_lm_indices.append(k)
    # print(graph.keys())
    # print("total number of unique poses: {}".format(len(uniq_poses)))
    if rm_ill_posed:
        for i in rm_indices:
            graph.remove(i)

    gt_vals = Values()
    # when we dont have enough constraints for poses.
    num_lms = 0
    num_ps = 0
    pose_mask = np.zeros(len(poses))
    points_mask = np.zeros(len(points))
    for k in graph.keyVector():
        sym = gtsam.Symbol(k)
        # print(sym.chr())
        if sym.chr() == ord('x'):
            num_ps = num_ps + 1
            pose_idx = sym.index()
            gt_vals.insert(X(pose_idx), poses[pose_idx])
            # print('x'+str(pose_idx))
            pose_mask[pose_idx] = 1
        if sym.chr() == ord('l'):
            num_lms = num_lms + 1
            lm_idx = sym.index()
            gt_vals.insert(L(lm_idx), points[lm_idx])
            points_mask[lm_idx] = 1
        # print("l" + str(lm_idx))

    if num_ps != len(poses):
        # print("not all poses have observed enough landmarks")
        pass
    # print(graph.keys())
    #print("num of factors: " + str(graph.size()))
    #print("Number of factors in graph" + str(graph.nrFactors()))
    return graph, gt_vals, pose_mask, points_mask

'''
1. build the factor graph with all possible combinations of camera placements
2. get the schur complement.
'''
def build_hfull(measurements, points, poses, intrinsics, extr_cand,ind = []):
    num_poses = len(poses)
    num_points = len(points)

    graph, gtvals, poses_mask, points_mask = build_graph(measurements, poses, points, intrinsics, extr_cand, ind)

    fim, crlb = compute_CRLB(gtvals, graph)
    # build full info matrix
    h_full = np.zeros((num_poses * 6 + num_points * 3, num_poses * 6 + num_points * 3), dtype=np.float64)
    num_ps = np.count_nonzero(poses_mask)
    num_lms = np.count_nonzero(points_mask)
    tot_num_points = len(points)
    '''
    Extract all the block diagonal elements corresponding to the poses (bottom right square block)
    into the full hessian
    '''
    pose_inds = []
    fim_idx = 0
    for val in gtvals.keys():
        sym = gtsam.Symbol(val)
        if sym.chr() == ord('x'):
            h_full_pose_idx = sym.index()
            pose_inds.append(h_full_pose_idx)
            h_full_pose_idx_start = tot_num_points * 3 + h_full_pose_idx * 6
            fim_pose_idx_start = num_lms * 3 + fim_idx * 6
            h_full[h_full_pose_idx_start: h_full_pose_idx_start + 6,
            h_full_pose_idx_start: h_full_pose_idx_start + 6] = fim[fim_pose_idx_start: fim_pose_idx_start + 6,
                                                                fim_pose_idx_start: fim_pose_idx_start + 6]
            fim_idx = fim_idx + 1

    fim_idx = 0
    for val in gtvals.keys():
        sym = gtsam.Symbol(val)
        if sym.chr() == ord('l'):
            idx = sym.index()
            h_full[idx * 3: (idx + 1) * 3, idx * 3: (idx + 1) * 3] = fim[fim_idx * 3: (fim_idx + 1) * 3,
                                                                     fim_idx * 3: (fim_idx + 1) * 3]
            p_idx_fim = 0
            for pose_idx in pose_inds:
                h_full_pose_idx_start = tot_num_points * 3 + pose_idx * 6
                fim_pose_idx_start = num_lms * 3 + p_idx_fim * 6
                h_full[idx * 3: (idx + 1) * 3, h_full_pose_idx_start: h_full_pose_idx_start + 6] = fim[fim_idx * 3: (fim_idx + 1) * 3, fim_pose_idx_start: fim_pose_idx_start + 6]
                h_full[h_full_pose_idx_start: h_full_pose_idx_start + 6, idx * 3: (idx + 1) * 3] = fim[fim_pose_idx_start: fim_pose_idx_start + 6, fim_idx * 3: (fim_idx + 1) * 3]
                p_idx_fim = p_idx_fim + 1
            fim_idx = fim_idx + 1
    # if fim.shape == h_full.shape:
    #     print(np.array_equal(fim, h_full))
    assert (utilities.check_symmetric(h_full))
    return h_full,  graph, gtvals, poses_mask, points_mask

def construct_candidate_inf_mats(measurements, intrinsics, extr_cand, points, poses):
    num_poses = len(poses)
    num_points = len(points)
    inf_mat_size = (num_poses * 6 + num_points * 3, num_poses * 6 + num_points * 3)
    inf_mats = np.zeros((0,inf_mat_size[0], inf_mat_size[1] ))
    h_sum  = np.zeros(inf_mat_size)
    i =0
    debug_num_facs=[]
    for j, cand in enumerate(extr_cand):
        h_cam, graph, gtvals, poses_mask, points_mask = build_hfull(measurements, points, poses, intrinsics, extr_cand, ind=[j])
        h_sum = h_sum + h_cam
        inf_mats = np.append(inf_mats,h_cam[None] , axis=0)
        debug_num_facs.append(graph.nrFactors())
        i = i + 1
    print("Number of candidates : "+ str(i))
    # hfull, _, _ , _, _= build_hfull(measurements, points, poses, intrinsics, extr_cand)
    # hdiff = hfull - h_sum
    # print(np.allclose(hfull, h_sum))
    return inf_mats,debug_num_facs

def find_min_eig_pair(inf_mats,selection, H0, num_poses):
    inds = np.where(selection > 1e-10)[0]
    #print(selection[inds])
    #final_inf_mat = np.sum(inf_mats[inds], axis=0)
    final_inf_mat = np.zeros(inf_mats[0].shape)
    for i in range(len(inds)):
        final_inf_mat = final_inf_mat + selection[inds[i]] * inf_mats[inds[i]]
    # add prior infomat H0
    final_inf_mat = final_inf_mat + H0
    H_schur = compute_schur_fim(final_inf_mat,num_poses )
    assert(utilities.check_symmetric(H_schur))
    #s_t = time.time()
    eigvals, eigvecs = la.eigh(H_schur)
    # e_t = time.time()
    # print("time taken to compute eigen vals and vectors dense: {:.6f}".format(e_t - s_t))
    #print(selection)
    # print("eigen vals")
    # print(eigvals[0:8])
    # print("null space")
    # print(scipy.linalg.null_space(H_schur).shape)
    return eigvals[0], eigvecs[:,0], final_inf_mat

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
    # print(all_eigs[np.argsort(s_rnd)])
    # print("----------------------------")

    zipped_vals = np.array([(s_rnd[i], all_eigs[i]) for i in range(len(s_rnd))], dtype=[('w', 'float'), ('weight', 'float')])
    idx = np.argpartition(zipped_vals, -k, order=['w', 'weight'])[-k:]
    rounded_sol = np.zeros(len(s_rnd))
    if k > 0:
        rounded_sol[idx] = 1.0
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

''' #########################################################'''













