from prettytable import PrettyTable
from camera_visualizer import CameraVisualizer
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial.transform import Rotation as R
import numpy as np

def sba_print(sba, nCams, title, xlim=[-1500, 1500], ylim=[-1500, 1500], zlim=[-100, 1500], color_palette=None):
    if not color_palette:
        color_pallette = sns.color_palette("rocket_r", nCams)
    
    x = PrettyTable()
    for row in sba.cameraArray:
        x.add_row(row)
    print(x)

    r = sba.project(sba.points3D[sba.point2DIndices], sba.cameraArray[sba.cameraIndices]) - sba.points2D
    print(r.shape)
    r = np.sqrt(np.sum(r**2, axis=1))
    plt.hist(r[r<np.percentile(r, 99)])
    plt.xlabel('Reprojection Error')
    plt.title(title)
    plt.show()

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(sba.points3D[:,0], sba.points3D[:,1], sba.points3D[:,2])
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    plt.title(title)
    for i in range(nCams):
        r_f = R.from_rotvec(-sba.cameraArray[i, 0:3]).as_matrix().copy()
        t_f = sba.cameraArray[i, 3:6].copy()
        # get inverse transformation
        r_inv = r_f.T    
        t_inv = -np.matmul(r_f, t_f)    
        ex = np.eye(4)
        ex[:3,:3] = r_inv.T
        ex[:3,3] = t_inv
        visualizer = CameraVisualizer(fig, ax)
        visualizer.extrinsic2pyramid(ex, color_palette[i], 200)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_zlim(zlim)
    plt.show()