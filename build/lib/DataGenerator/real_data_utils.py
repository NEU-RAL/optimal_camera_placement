import numpy as np
import gtsam
from gtsam import  Cal3_S2
import os
import yaml


def read_april_tag_data(opti_poses_file, tag_data_file, extr_calib_dir):

    undist_intrinsics = np.genfromtxt(os.path.join(extr_calib_dir, "undistorted_intrinsics.csv"), delimiter=',', dtype=float)
    intrinsics = [Cal3_S2(undist_intrinsics[i, 0],undist_intrinsics[i, 1],0, undist_intrinsics[i, 2],undist_intrinsics[i, 3]) for i in range(undist_intrinsics.shape[0])]


    T_c_i_all = []
    ref_cam_indx = 4
    with open(os.path.join(extr_calib_dir, "camchain.yaml"), 'r') as f:
        dict_calib = yaml.safe_load(f)
        for cam_key in dict_calib.keys():
            camid = cam_key.split("_")[1][-1]
            print("camid:", camid)
            T_c_i = np.array(dict_calib[cam_key])
            T_c_i_all.append(T_c_i)
    #convert all the transforms into cam4
    T_c4_i = T_c_i_all[ref_cam_indx]
    T_c4_c = []
    for T_c_i in T_c_i_all:
        T_c4_c_np = T_c4_i @ np.linalg.inv(T_c_i)
        T_c4_c.append(gtsam.Pose3(T_c4_c_np) )

    #dirs = os.listdir(extr_calib_dir)
    # for dir in dirs:
    #     pathe = os.path.join(extr_calib_dir, dir)
    #     if os.path.isdir(pathe) and "cam" in dir:
    #         calib_file = [f for f in os.listdir(pathe) if "camchain-imucam" in f][0]
    #         with open(os.path.join(pathe, calib_file), 'r') as f:
    #             dict_calib = yaml.load(f)
    #             for cam_key in dict_calib.keys():
    #                 camid = int(dict_calib[cam_key]['rostopic'].split("/")[2][-1])
    #                 T_c_i = np.array(dict_calib[cam_key]['T_cam_imu'])
    #                 extrinsics.append(T_c_i)

    #rough estimate
    r = np.array([[-1, 0, 0],
                  [0, -1, 0],
                  [0, 0, 1]])
    t = [0, 0, 0]
    T_c4_rig = gtsam.Pose3(gtsam.Rot3(r), gtsam.Point3(t[0], t[1], t[2]))

    cols = ['time', 'rx', 'ry', 'rz', 'rw', 'tx', 'ty', 'tz']
    tagcols = ['time', 'camid', 'tagid', 'rx', 'ry', 'rz', 'rw', 'tx', 'ty', 'tz','u1', 'v1', 'u2', 'v2', 'u3', 'v3', 'u4', 'v4',]
    opti_poses_data = pd.read_csv(opti_poses_file,names=cols)
    tag_data =  pd.read_csv(tag_data_file, names=tagcols)

    #go through the rows of opti
    num_poses = opti_poses_data["time"].nunique()
    num_points = tag_data['tagid'].nunique()
    measurements = np.zeros((num_poses, len(T_c4_c), num_points, 2))

    start_index = 0
    poses = []
    points_dict = {}
    measurement_dict = {}
    tag_id_to_pointid = {}
    point_counter=0
    while start_index < len(opti_poses_data):
        start_time = opti_poses_data['time'][start_index]
        indices = opti_poses_data.index[opti_poses_data['time'] == start_time]
        opti_pose_row = opti_poses_data.iloc[start_index]
        # get the optitrack pose data
        T_rig_o = gtsam.Pose3(gtsam.Rot3(opti_pose_row['rw'], opti_pose_row['rx'], opti_pose_row['ry'], opti_pose_row['rz']), gtsam.Point3(opti_pose_row['tx'], opti_pose_row['ty'], opti_pose_row['tz']))
        T_c4_o = T_c4_rig.compose(T_rig_o)
        T_o_c4 = T_c4_o.inverse()
        poses.append(T_o_c4)

        df_tmp = tag_data.iloc[indices]
        for index, row in df_tmp.iterrows():
            #print(row)
            cam = int(row['camid'])
            tagid = int(row['tagid'])
            if tag_id_to_pointid.get(tagid)  is None:
                tag_id_to_pointid[tagid] = [point_counter] # remove list type if only one point
                point_counter = point_counter + 1
                # adding corners also as points
                tag_id_to_pointid[tagid].append(point_counter + 1)
                tag_id_to_pointid[tagid].append(point_counter + 2)
                tag_id_to_pointid[tagid].append(point_counter + 3)
                tag_id_to_pointid[tagid].append(point_counter + 4)
                point_counter = point_counter + 4
                ########################


            # sanity check : average corners = center image coordinates, project the tag center into camera to see if the coordinates matches
            c1 = np.array([row['u1'], row['v1']])
            c2 = np.array([row['u2'], row['v2']])
            c3 = np.array([row['u3'], row['v3']])
            c4 = np.array([row['u4'], row['v4']])
            corner_center = np.mean(np.vstack([c1, c2, c3, c4]), axis =0)
            # add measurements for the rest of the cotrners as well
            measurements[len(poses) - 1, cam, tag_id_to_pointid[tagid]] = c1
            measurements[len(poses) - 1, cam, tag_id_to_pointid[tagid]] = c2
            measurements[len(poses) - 1, cam, tag_id_to_pointid[tagid]] = c3
            measurements[len(poses) - 1, cam, tag_id_to_pointid[tagid]] = c4

            measurements[len(poses)-1, cam ,tag_id_to_pointid[tagid]] = corner_center
            # 3D center of the apriltag w.r.t cam it is seen in
            t_c_tag = gtsam.Point3(row['tx'], row['ty'], row['tz'])
            R_c_tag = gtsam.Rot3(row['rw'], row['rx'], row['ry'], row['rz'])
            T_c_tag = gtsam.Pose3(R_c_tag,t_c_tag)


            #sanity check
            camera = PinholeCameraCal3_S2(gtsam.Pose3(gtsam.Rot3(np.eye(3)), gtsam.Point3(0.0, 0.0, 0.0)), intrinsics[cam])
            measurement_center = camera.project(t_c_tag)
            if np.linalg.norm(measurement_center - corner_center) > 2.0 :
                print("measurement_center: ")
                print(measurement_center)
                print("computed center from corners: ")
                print(corner_center)

            # convert this into world/opti fram
            t_o_tag = T_o_c4.compose(T_c4_c[cam]).transformFrom(t_c_tag)

            #####$ corners
            P1_c = T_c_tag.transformFrom(gtsam.Point3(-5, -5, 0))
            P1_o = T_o_c4.compose(T_c4_c[cam]).transformFrom(P1_c)
            P2_c = T_c_tag.transformFrom(gtsam.Point3(-5, 5, 0))
            P2_o = T_o_c4.compose(T_c4_c[cam]).transformFrom(P2_c)
            P3_c = T_c_tag.transformFrom(gtsam.Point3(5, 5, 0))
            P3_o = T_o_c4.compose(T_c4_c[cam]).transformFrom(P3_c)
            P4_c = T_c_tag.transformFrom(gtsam.Point3(5, -5, 0))
            P4_o = T_o_c4.compose(T_c4_c[cam]).transformFrom(P4_c)

            #uncomment this section and comment the other section if not using all corners as landmarks
            # if points_dict.get(tagid) is not None:
            #     points_dict[tagid].append(t_o_tag)
            # else:
            #     points_dict[tagid] = [t_o_tag]

            if points_dict.get(tagid) is not None:
                points_dict[tagid][0].append(t_o_tag)
                points_dict[tagid][1].append(P1_o)
                points_dict[tagid][2].append(P2_o)
                points_dict[tagid][3].append(P3_o)
                points_dict[tagid][4].append(P4_o)
            else:
                points_dict[tagid] = {0:[t_o_tag]}
                points_dict[tagid]={1: [P1_o]}
                points_dict[tagid]={2: [P2_o]}
                points_dict[tagid]={3: [P3_o]}
                points_dict[tagid]={4: [P4_o]}

        #update start index
        start_index = indices.tolist()[-1] + 1

    print(poses)
    print("---------")
    points = [None] * num_points
    #if corners also there
    points = [None] * num_points
    for key, vals in points_dict.items():
        print("tag id: ", key)
        all_p = np.array(vals[1])
        mean_tag = all_p.mean(axis=0)
        points[tag_id_to_pointid[key]] = mean_tag
        print("mean tag center : ")
        print(mean_tag)
        print("variance: ")
        print(all_p.var(axis=0))

    poses_with_noise = []
    points_with_noise = []
    for p in poses:
        transformed_pose = p.retract(0.1 * np.random.randn(6, 1))
        poses_with_noise.append(transformed_pose)
    for l in points:
        transformed_point = l + 0.1 * np.random.randn(3)
        points_with_noise.append(transformed_point)

    return intrinsics, T_c4_c, poses, points, measurements, poses_with_noise, points_with_noise