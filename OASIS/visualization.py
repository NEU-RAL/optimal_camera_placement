import matplotlib.pyplot as plt
import gtsam
from gtsam.utils import plot

def visualize_3d(poses, points):
    """
    Simple 3D visualization of poses and points.
    
    Args:
        poses (list): A list of gtsam.Pose3 objects representing the poses.
        points (list or np.ndarray): A list of gtsam.Point3 objects or a NumPy array of 3D points.
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot poses
    for p in poses:
        plot.plot_pose3_on_axes(ax, p, axis_length=1)

    # Check if points are GTSAM Point3 objects or numpy arrays
    if len(points) == 0:
        print("No points to visualize.")
        return
    
    if isinstance(points[0], gtsam.Point3):  # GTSAM Point3 case
        pts = [(pt.x(), pt.y(), pt.z()) for pt in points]
    elif isinstance(points, np.ndarray):  # NumPy array case
        pts = points.tolist()  # Convert to list of tuples
    else:  # General list of tuples
        pts = [(pt[0], pt[1], pt[2]) for pt in points]

    # Unpack points for plotting
    pts_arr = list(zip(*pts))
    ax.scatter(pts_arr[0], pts_arr[1], pts_arr[2], c='r', marker='o')

    # Set axes labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('3D Pose and Point Visualization')

    # Show plot
    plt.show()
