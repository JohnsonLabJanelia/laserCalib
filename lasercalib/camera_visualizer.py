import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

class CameraVisualizer:
    def __init__(self, fig, ax):
        self.fig = fig
        self.ax = ax
        self.ax.set_aspect("auto")
        self.ax.set_xlabel('x')
        self.ax.set_ylabel('y')
        self.ax.set_zlabel('z')
        self.ax.set_zlim([-10, 2500])

    def extrinsic2pyramid(self, extrinsic, color='r', focal_len_scaled=5, aspect_ratio=0.3):
        vertex_std = np.array([[0, 0, 0, 1],
                               [focal_len_scaled * aspect_ratio, -focal_len_scaled * aspect_ratio, focal_len_scaled, 1],
                               [focal_len_scaled * aspect_ratio, focal_len_scaled * aspect_ratio, focal_len_scaled, 1],
                               [-focal_len_scaled * aspect_ratio, focal_len_scaled * aspect_ratio, focal_len_scaled, 1],
                               [-focal_len_scaled * aspect_ratio, -focal_len_scaled * aspect_ratio, focal_len_scaled, 1]])
        vertex_transformed = vertex_std @ extrinsic.T
        meshes = [[vertex_transformed[0, :-1], vertex_transformed[1][:-1], vertex_transformed[2, :-1]],
                            [vertex_transformed[0, :-1], vertex_transformed[2, :-1], vertex_transformed[3, :-1]],
                            [vertex_transformed[0, :-1], vertex_transformed[3, :-1], vertex_transformed[4, :-1]],
                            [vertex_transformed[0, :-1], vertex_transformed[4, :-1], vertex_transformed[1, :-1]],
                            [vertex_transformed[1, :-1], vertex_transformed[2, :-1], vertex_transformed[3, :-1], vertex_transformed[4, :-1]]]
        self.ax.add_collection3d(
            Poly3DCollection(meshes, facecolors=color, linewidths=0.3, edgecolors=color, alpha=0.35))


def sba_print():
    r = sba.project(sba.points3D[sba.point2DIndices], sba.cameraArray[sba.cameraIndices]) - sba.points2D
    r = np.sqrt(np.sum(r**2, axis=1))
    plt.hist(r)

    x = PrettyTable()
    for row in sba.cameraArray:
        x.add_row(row)
    print(x)

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(sba.points3D[:,0], sba.points3D[:,1], sba.points3D[:,2])
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    plt.title('initial cam params and 3D points')
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
        visualizer.extrinsic2pyramid(ex, my_palette[i], 200)
    ax.set_xlim((-1500, 1500))
    ax.set_ylim((-1500, 1500))
    ax.set_zlim((-100, 1500))
    plt.show()
