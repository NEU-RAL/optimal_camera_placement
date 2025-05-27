import matplotlib.pyplot as plt
import sys
import numpy as np
import math
import gtsam
from gtsam.utils import plot
import time
from matplotlib import cm
figno = 2

def reset():
    global figno
    figno = 1
    plt.close("all")
def initialize_3d_plot(number=None, title='Plot', axis_labels=['x', 'y', 'z'],view=[None,None],limits=None):
    fig = plt.figure(number)
    fig.clf()
    ax = fig.add_subplot(111, projection='3d')
    fig.subplots_adjust(0,0,1,1) # Make the plot tight
    fig.suptitle(title)
    ax.set_xlabel(axis_labels[0])
    ax.set_ylabel(axis_labels[1])
    ax.set_zlabel(axis_labels[2])
    if not (limits is None):
        ax.set_xlim(*limits[0,:])
        ax.set_ylim(*limits[1,:])
        ax.set_zlim(*limits[2,:])

    ax.view_init(*view)
    if sys.version_info[:2] == (3, 5):
        ax.set_aspect('equal')
    return fig,ax

def plot_3d_points(axes, vals, line_obj=None, *args, **kwargs):
    if line_obj is None:
        graph, = axes.plot(vals[:,0], vals[:,1], vals[:,2], *args, **kwargs)
        return graph

    else:
        line_obj.set_data(vals[:,0], vals[:,1])
        line_obj.set_3d_properties(vals[:,2])
        return line_obj
def initialize_2d_plot(number=None, title='Plot', axis_labels=['x', 'y'],limits=None):
    global figno
    fig = plt.figure(figno)
    figno = figno+1
    fig.clf()
    ax = fig.add_subplot(111)
    fig.subplots_adjust(0,0,1,1) # Make the plot tight
    fig.suptitle(title)
    ax.set_xlabel(axis_labels[0])
    ax.set_ylabel(axis_labels[1])
    if not (limits is None):
        ax.set_xlim(*limits[0,:])
        ax.set_ylim(*limits[1,:])
    return fig,ax

def plot_2d_points( vals, *args, **kwargs):
    # plot
    global figno
    fig = plt.figure(figno)
    figno = figno + 1
    ax = fig.add_subplot(111)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.scatter(vals[:,0],vals[:,1], *args, **kwargs )
    plt.show()

def rotate_coord_frame():
    plt.ion()
    fig = plt.figure(8)
    axes = fig.add_subplot(111, projection='3d')

    for theta in np.linspace(0, math.pi, 5):
        # just create a stereo system for now
        rotmat = np.eye(3)
        R_y = np.array([[math.cos(theta / 2), 0, math.sin(theta / 2)],
                        [0, 1, 0],
                        [-1 * math.sin(theta / 2), 0, math.cos(theta / 2)]])
        R_y_neg = np.array([[math.cos(theta / 2), 0, -1.0 * math.sin(theta / 2)],
                            [0, 1, 0],
                            [math.sin(theta / 2), 0, math.cos(theta / 2)]])

        rotmat_1 = rotmat @ R_y_neg
        c1 = gtsam.Pose3(gtsam.Rot3(rotmat_1), gtsam.Point3(0, 0, 0))
        rotmat_2 = rotmat @ R_y
        c2 = gtsam.Pose3(gtsam.Rot3(rotmat_2), gtsam.Point3(0.1, 0, 0))
        plot.plot_pose3_on_axes(axes, c1, 0.1)
        plot.plot_pose3_on_axes(axes, c2, 0.1)
        fig.canvas.draw()
        fig.canvas.flush_events()
        axes.cla()
        time.sleep(1)
    plt.show()

def show_camconfigs(best_extr):
    global figno
    plt.ion()
    fig = plt.figure(figno)
    figno = figno + 1
    axes = fig.add_subplot(111, projection='3d')
    while (True):
        for c in best_extr:
            plot.plot_pose3_on_axes(axes, c, 0.3)
        axes.set_xlim3d([-0.5, 0.5])
        axes.set_ylim3d([-0.5, 0.5])
        axes.set_zlim3d([-0.5, 0.5])
        axes.set_xlabel("X_Axis")
        axes.set_ylabel("Y-Axis")
        axes.set_zlabel("Z-Axis")
        fig.canvas.draw()
        fig.canvas.flush_events()
        axes.cla()
        time.sleep(0.1)

def show_trajectories(poses, points, K, fignum=1, title="plot"):
    plt.ion()
    fig1, ax1 = initialize_3d_plot(number=fignum,title=title, limits=np.array([[-30, 30], [-30, 30], [-30, 30]]),
                                   view=[-90, -90])
    for i, pose in enumerate(poses):
        plot.plot_pose3_on_axes(ax1, pose, axis_length=1, P=None, scale=1)
        camera = gtsam.PinholeCameraCal3_S2(pose, K)
        num_projected = 0
        for j, point in enumerate(points):
            try:
                measurement = camera.project(point) + 1.0 * np.random.randn(2)
                if (measurement[0] > 1 and measurement[0] < 2 * K.px() and measurement[1] > 1 and measurement[
                    1] < 2 * K.py()):
                    num_projected = num_projected + 1
                    plot.plot_point3_on_axes(ax1,point,'b*')
                # else:
                # print("Measurement is out of bounds: ")
                # print(measurement)
            except Exception:
                pass
                # print("Exception at Point")
                # print(point)
        ax1.set_xlim3d([-10, 10])
        ax1.set_ylim3d([-10, 10])
        ax1.set_zlim3d([-10, 10])
        ax1.set_xlabel("X_Axis")
        ax1.set_ylabel("Y-Axis")
        ax1.set_zlabel("Z-Axis")
        fig1.canvas.draw()
        fig1.canvas.flush_events()
        time.sleep(0.1)
        ax1.cla()

def show_camconfig_with_world(best_extrs, fignum,titles, K, poses, points):
    plt.ion()
    figs=[]
    axs=[]
    #fig2, ax2 = initialize_3d_plot(number=fignum+1, limits=np.array([[-30, 30], [-30, 30], [-30, 30]]),
    #                               view=[-90, -90])
    # sanity check
    for c, best_extr in enumerate(best_extrs):
        fig1, ax1 = initialize_3d_plot(number=fignum,title=titles[c], limits=np.array([[-30, 30], [-30, 30], [-30, 30]]),
                                       view=[-90, -90])
        fignum=fignum+1
        figs.append(fig1)
        axs.append(ax1)
    while (True):
        for c, best_extr in enumerate(best_extrs):
            for i, pose in enumerate([poses[0]]):
                for comp_pose in best_extr:
                    pose_wc = pose.compose(comp_pose)
                    plot.plot_pose3_on_axes(axs[c], pose_wc, axis_length=1, P=None, scale=1)
                    camera = gtsam.PinholeCameraCal3_S2(pose_wc, K)
                    num_projected = 0
                    for j, point in enumerate(points):
                        #plot.plot_point3_on_axes(ax1, point, 'r*')
                        try:
                            measurement = camera.project(point)
                            if (measurement[0] > 1 and measurement[0] < 2 * K.px() and measurement[1] > 1 and measurement[
                                1] < 2 * K.py()):
                                num_projected = num_projected + 1
                                plot.plot_point3_on_axes(axs[c], point, 'b*')
                            #else:
                           #     plot.plot_point3_on_axes(ax1, point, 'r*')
                            # print("Measurement is out of bounds: ")
                            # print(measurement)
                        except Exception:
                            pass
        for c, fig in enumerate(figs):
            axs[c].set_xlim3d([-10, 10])
            axs[c].set_ylim3d([-10, 10])
            axs[c].set_zlim3d([-10, 10])
            axs[c].set_xlabel("X_Axis")
            axs[c].set_ylabel("Y-Axis")
            axs[c].set_zlabel("Z-Axis")
            fig.canvas.draw()
            fig.canvas.flush_events()

        time.sleep(0.1)
        axs[0].cla()
        axs[1].cla()

        # for i, pose in enumerate([poses[0]]):
        #     for comp_pose in best_extr2:
        #         pose_wc = pose.compose(comp_pose)
        #         plot.plot_pose3_on_axes(ax2, pose_wc, axis_length=1, P=None, scale=1)
        #         camera = gtsam.PinholeCameraCal3_S2(pose_wc, K)
        #         num_projected = 0
        #         for j, point in enumerate(points):
        #             #plot.plot_point3_on_axes(ax2, point, 'r*')
        #             try:
        #                 measurement = camera.project(point)
        #                 if (measurement[0] > 1 and measurement[0] < 2 * K.px() and measurement[1] > 1 and measurement[
        #                     1] < 2 * K.py()):
        #                     num_projected = num_projected + 1
        #                     plot.plot_point3_on_axes(ax2, point, 'b*')
        #                 #else:
        #                #     plot.plot_point3_on_axes(ax1, point, 'r*')
        #                 # print("Measurement is out of bounds: ")
        #                 # print(measurement)
        #             except Exception:
        #                 pass
        # fig1.canvas.draw()
        # fig1.canvas.flush_events()
        # fig2.canvas.draw()
        # fig2.canvas.flush_events()
        # ax1.cla()
        # ax2.cla()
        # ax1.set_xlabel("X_Axis")
        # ax1.set_ylabel("Y-Axis")
        # ax1.set_zlabel("Z-Axis")
        # ax2.set_xlabel("X_Axis")
        # ax2.set_ylabel("Y-Axis")
        # ax2.set_zlabel("Z-Axis")

    plt.ioff()


def draw_trajectories_configs(trajs, best_configs):

    #show the trajectories and the configs
    for traj in trajs:
        fig = plt.figure(figno)
        figno = figno + 1
        axes = fig.add_subplot(111, projection='3d')
        for p in traj:
            plot.plot_pose3_on_axes(axes, p, axis_length=1, P=None, scale=1)
        axes.set_xlim3d([-10, 10])
        axes.set_ylim3d([-10, 10])
        axes.set_zlim3d([-10, 10])
        axes.set_xlabel("X_Axis")
        axes.set_ylabel("Y-Axis")
        axes.set_zlabel("Z-Axis")
        fig.canvas.draw()
        fig.canvas.flush_events()
        time.sleep(0.1)

    #show the cam configs
    axes=[]
    axes_brute=[]
    for i, best_extr in enumerate(best_configs):
        fig = plt.figure(figno)
        figno = figno + 1
        ax = fig.add_subplot(111, projection='3d')
        axes.append(ax)

    # for i, best_extr in enumerate(best_configs_brute):
    #     fig = plt.figure(figno)
    #     figno = figno + 1
    #     ax = fig.add_subplot(111, projection='3d')
    #     axes_brute.append(ax)

    while (True):
        for i, best_extr in enumerate(best_configs):
            axes[i].cla()
            for c in best_extr:
                plot.plot_pose3_on_axes(axes[i], c, 0.3)
            axes[i].set_xlim3d([-0.5, 0.5])
            axes[i].set_ylim3d([-0.5, 0.5])
            axes[i].set_zlim3d([-0.5, 0.5])
            axes[i].set_xlabel("X_Axis")
            axes[i].set_ylabel("Y-Axis")
            axes[i].set_zlabel("Z-Axis")
        # for i, best_extr in enumerate(best_configs_brute):
        #     axes_brute[i].cla()
        #     for c in best_extr:
        #         plot.plot_pose3_on_axes(axes_brute[i], c, 0.3)
        #     axes_brute[i].set_xlim3d([-0.5, 0.5])
        #     axes_brute[i].set_ylim3d([-0.5, 0.5])
        #     axes_brute[i].set_zlim3d([-0.5, 0.5])
        #     axes_brute[i].set_xlabel("X_Axis")
        #     axes_brute[i].set_ylabel("Y-Axis")
        #     axes_brute[i].set_zlabel("Z-Axis")
        fig.canvas.draw()
        fig.canvas.flush_events()

        time.sleep(0.1)

    plt.show(block=True)

def plot_grid_data(fim, xlabel, ylabel, title, labelsX=None, labelsY=None):
    global figno
    X = np.arange(0, fim.shape[1])
    Y = np.arange(0, fim.shape[0])
    X, Y = np.meshgrid(X, Y)
    fig = plt.figure(figno)
    figno = figno + 1
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(X, Y, fim, cmap=cm.coolwarm,
                           linewidth=0, antialiased=False)
    ax.set_xlabel(xlabel, labelpad=7.0,fontsize=10, color='r')
    ax.set_ylabel(ylabel,  labelpad=10.0, fontsize=10, color='r' )
    if labelsX:
        ax.set_xticks(np.arange(fim.shape[1]), labels=labelsX)
    if labelsY:
        ax.set_yticks(np.arange(fim.shape[0]), labels=labelsY)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")
    plt.setp(ax.get_yticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")
    ax.set_title(title)
    ax.set_box_aspect((6,1,1))
    #fig.colorbar(surf, shrink=0.5, aspect=5)