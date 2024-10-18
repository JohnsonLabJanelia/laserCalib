import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3D, Line3DCollection


class CameraVisualizer:
    def __init__(self, fig, ax):
        self.fig = fig
        self.ax = ax
        self.ax.set_aspect("auto")
        self.ax.set_xlabel('x')
        self.ax.set_ylabel('y')
        self.ax.set_zlabel('z')
        self.ax.set_zlim([-10, 2500])

    def extrinsic2pyramid(self, extrinsic, color='r', focal_len_scaled=5, aspect_ratio=0.3, axis_len=50):
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

        
        x_axis = (extrinsic @ (np.array([[0, 0, 0, 1],[axis_len,0,0,1]]).T)).T
        y_axis = (extrinsic @ (np.array([[0, 0, 0, 1],[0,axis_len,0,1]]).T)).T        
        z_axis = (extrinsic @ (np.array([[0, 0, 0, 1],[0,0,axis_len,1]]).T)).T
        coord_axes = [x_axis[:,0:3], y_axis[:,0:3], z_axis[:,0:3]]
        self.ax.add_collection3d(Line3DCollection(coord_axes, colors=['red','green','blue'], linewidth=1))
        

