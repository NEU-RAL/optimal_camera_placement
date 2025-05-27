import numpy as np
import gtsam
import matplotlib.pyplot as plt
import math

from gtsam.utils import plot

'''
Utilities
'''

def check_symmetric(a, rtol=1e-03, atol=1e-03):
    res = np.allclose(a, a.T, rtol=rtol, atol=atol)
    if not res:
        print("Norm of distace between symmetry", np.linalg.norm(a - a.T))
    return res

'''
Method that takes number of cameras, min baseline between two closest cameras and the angular separation between them
to return the camera matrices
'''
def build_cam_config(num_cams, theta, min_baseline):
    #positions
    tot_bl = (num_cams -1) * min_baseline
    bl_range = tot_bl/2
    x_pos = np.linspace(-1*bl_range, bl_range, num_cams)
    extr=[]
    fig = plt.figure(1)
    axes = fig.add_subplot(111, projection="3d")
    if num_cams == 1:
        rotmat = np.eye(3)
        R_y = np.array([[math.cos(theta), 0, math.sin(theta)],
                        [0, 1, 0],
                        [-1 * math.sin(theta), 0, math.cos(theta)]])
        rotmat_1 = rotmat @ R_y
        c1 = gtsam.Pose3(gtsam.Rot3(rotmat_1), gtsam.Point3(x_pos[0], 0, 0))
        extr.append(c1)
        plot.plot_pose3_on_axes(axes, c1, 0.1)
        #plt.show()
        return extr

    if num_cams % 2 != 0:
        centr_cam = gtsam.Pose3(gtsam.Rot3(np.eye(3)), gtsam.Point3(x_pos[ (num_cams // 2)], 0, 0))
        extr.append(centr_cam)
        plot.plot_pose3_on_axes(axes, centr_cam, 0.1)
    #go from center outwards

    for nc in range(0, num_cams//2):
        rotmat = np.eye(3)
        rot_nag=0.0
        if num_cams % 2 == 0 :
            rot_nag = (nc + 1) * theta - (theta/2)
        else:
            rot_nag = (nc+1) * theta
        R_y = np.array([[math.cos(rot_nag), 0, math.sin(rot_nag)],
                        [0, 1, 0],
                        [-1 * math.sin(rot_nag), 0, math.cos(rot_nag)]])
        R_y_neg = np.array([[math.cos(rot_nag), 0, -1.0 * math.sin(rot_nag)],
                            [0, 1, 0],
                            [math.sin(rot_nag), 0, math.cos(rot_nag)]])

        rotmat_1 = rotmat @ R_y_neg
        print(rotmat_1[:,0].dot(rotmat_1[:,2]))
        c1_ind = (num_cams//2) - (nc +1)
        c1 = gtsam.Pose3(gtsam.Rot3(rotmat_1), gtsam.Point3(x_pos[c1_ind], 0, 0))
        rotmat_2 = rotmat @ R_y
        c2_ind = (num_cams // 2) + (nc + 1)
        if num_cams % 2 == 0:
             c2_ind = c2_ind - 1

        c2 = gtsam.Pose3(gtsam.Rot3(rotmat_2), gtsam.Point3(x_pos[c2_ind], 0, 0))

        extr.append(c1)
        extr.append(c2)
        plot.plot_pose3_on_axes(axes, c1, 0.1)
        plot.plot_pose3_on_axes(axes, c2, 0.1)
    #plt.show()
    return extr

