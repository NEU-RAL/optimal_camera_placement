from optimal_camera_placement.ProblemBuilder import FIM as fim
import numpy as np
import gtsam
import math

def compute_info_metric(poses, points, meas, intrinsics, cands, selection, h_prior):
    num_poses = len(poses)
    num_points = len(points)
    h_full, graph, gtvals, poses_mask, points_mask = fim.build_hfull(meas, points, poses, intrinsics, cands, selection)
    h_full = h_full + h_prior
    fim = fim.compute_schur_fim(h_full, len(poses))
    least_fim_eig = np.linalg.eigvalsh(fim)[0]
    return least_fim_eig

def compute_rmse(measurements, poses, points, intrinsics, extr_cand,selected_inds,poses_with_noise, points_with_noise,h_prior_val, loc= False ):
    '''build the factor graph with the configuration '''
    graph_g, gtvals_g, poses_mask_g, points_mask_g = build_graph(measurements, poses, points, intrinsics,  extr_cand,
                                                                 selected_inds, rm_ill_posed = not loc)
    # Add a prior on pose x1. This indirectly specifies where the origin is.
    #same as the prior on the first pose given
    pose_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([h_prior_val,h_prior_val,h_prior_val,h_prior_val,h_prior_val,h_prior_val]))
    for i in range(len(poses)):
        if poses_mask_g[i] == 1:
            factor = PriorFactorPose3(X(i), poses[i], pose_noise)
            graph_g.push_back(factor)
            break
    for i in range(len(points)):
        if not loc:
            break
        if points_mask_g[i] == 1:
            point_noise = gtsam.noiseModel.Isotropic.Sigma(3, 0.1)
            factor = PriorFactorPoint3(L(i), points[i], point_noise)
            graph_g.push_back(factor)
            # if not loc:
            #     break
    # print("prior factor x0")
    ''' Give some initial values which are noisy'''
    initial_estimate_g = Values()
    n_poses_graph = 0
    n_points_graph = 0
    for val in gtvals_g.keys():
        '''
        If the resl variable is a pose, store the values
        '''
        if gtsam.Symbol(val).chr() == ord('x'):
            pose_ind = gtsam.Symbol(val).index()
            transformed_pose = poses_with_noise[pose_ind]
            # pose = gtvals.atPose3(val)
            # transformed_pose = pose.retract(0.1 * np.random.randn(6, 1))
            initial_estimate_g.insert(val, transformed_pose)
            n_poses_graph = n_poses_graph + 1
        elif gtsam.Symbol(val).chr() == ord('l'):
            point_ind = gtsam.Symbol(val).index()
            transformed_point = points_with_noise[point_ind]
            # point = gtvals.atPoint3(val)
            # transformed_point = point + 0.1 * np.random.randn(3)
            initial_estimate_g.insert(val, transformed_point)
            n_points_graph = n_points_graph + 1
    # print(poses_mask_g)
    print("num poses : " + str(n_poses_graph))
    print("num points : " + str(n_points_graph))
    # Optimize the graph and print results
    # params1 = gtsam.DoglegParams()
    # optimizer1 = DoglegOptimizer(graph_g, initial_estimate_g, params1)
    # try:
    #     result1 = optimizer1.optimize()
    # except Exception:
    #     result1 = Values()
    #     pass
    # # result.print('Final results:\n')
    # rmse1 = utilities.compute_traj_error(result1, poses, initial_estimate_g)
    # print("The RMSE of the estimated trajectory with best camera placement dogleg: " + str(rmse1))
    params = gtsam.LevenbergMarquardtParams()
    # params.setVerbosity('ERROR')
    optimizer = LevenbergMarquardtOptimizer(graph_g, initial_estimate_g, params)
    try:
        result = optimizer.optimize()
    except Exception:
        result = Values()
        pass
    # result.print('Final results:\n')
    rmse = utilities.compute_traj_error(result, poses, initial_estimate_g)
    print("The RMSE of the estimated trajectory with best camera placement: " + str(rmse))
    return rmse


def compute_traj_error(result, poses, initial_estimate=[]):

    lm_ind = 0
    exp_pose_ests = np.zeros((len(poses), 6))
    # exp_lm_ests = np.zeros((len(points), 3))
    rmse = 0.0
    num_poses_err = 0
    for res in result.keys():
        '''
        If the resl variable is a pose, store the values
        '''
        if gtsam.Symbol(res).chr() == ord('x'):
            # print(gtsam.Symbol(res))
            est_rot = result.atPose3(res).rotation().rpy()
            est_trans = result.atPose3(res).translation()
            # print(gtsam.Symbol(res).index())
            # print(poses[gtsam.Symbol(res).index()].translation())
            # print(initial_estimate.atPose3(res).translation())
            # print(est_trans)
            rmse = rmse + np.square(
                np.linalg.norm(est_trans - poses[gtsam.Symbol(res).index()].translation()))
            num_poses_err = num_poses_err + 1
    rmse = rmse / len(poses)
    rmse = math.sqrt(rmse)

    return rmse

def write_data_logs(filename, poses, landmarks,extrinsics, measurements, poses_init, landmarks_init):
    # Simulated measurements from each camera pose, adding them to the factor graph
    with open(filename, 'w') as f:
        #write all the extrinsic/cameras on the rig
        for i, ext in enumerate(extrinsics):
            t = ext.translation()
            q = ext.rotation().toQuaternion()
            f.write("e "+str(i)+" "+str(t[0])+" "+str(t[1])+" "+str(t[2])+" "+str(q.x())+" "+str(q.y())+" "+str(q.z())+" "+str(q.w())+"\n")
        #write all the poses
        for i, pose in enumerate(poses):
            t = pose.translation()
            q = pose.rotation().toQuaternion()
            f.write("x "+str(i)+" "+str(t[0])+" "+str(t[1])+" "+str(t[2])+" "+str(q.x())+" "+str(q.y())+" "+str(q.z())+" "+str(q.w())+"\n")
        for i, point in enumerate(landmarks):
            f.write("l " + str(i)+" "+str(point[0]) + " " + str(point[1]) + " " + str(point[2])+ "\n")
        #write all the landmarks
        inds = range(len(extrinsics))
        for i, pose in enumerate(poses):
            for j, point in enumerate(landmarks):
                for k in inds:
                    measurement = measurements[i, k, j]
                    if measurement[0] == 0 and measurement[1] == 0:
                        continue
                    #write the measurement to the file
                    line = "m "+str(i)+" "+str(k)+" "+str(j)+" "+str(measurement[0])+" "+str(measurement[1])+"\n"
                    f.write(line)

        for i, pose in enumerate(poses_init):
            t = pose.translation()
            q = pose.rotation().toQuaternion()
            f.write("x_i " + str(i) + " " + str(t[0]) + " " + str(t[1]) + " " + str(t[2]) +
                    " " + str(q.x())+" "+str(q.y())+" "+str(q.z())+" "+str(q.w()) + "\n")

        for i, point in enumerate(landmarks_init):
            f.write("l_i " + str(i) + " " + str(point[0]) + " " + str(point[1]) + " " + str(point[2]) + "\n")

    f.close()
