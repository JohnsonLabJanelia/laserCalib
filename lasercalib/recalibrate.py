from tkinter.messagebox import NO
import numpy as np
import matplotlib.pyplot as plt
import pickle as pkl
from scipy.spatial.transform import Rotation as R
from prettytable import PrettyTable
import pySBA
from my_cam_pose_visualizer import MyCamPoseVisualizer
import seaborn as sns
import pprint
from scipy.sparse import lil_matrix
from scipy.optimize import least_squares

my_palette = sns.color_palette("rocket_r", 7)

picklefile = open('../calibres/sba_guess', 'rb')
sba = pkl.load(picklefile)
picklefile.close()

x = PrettyTable()
for row in sba.cameraArray:
    x.add_row(row)
print(x)

my_palette = sns.color_palette("rocket_r", 7)

nCams = sba.cameraArray.shape[0]

points3Dfixed_labeled = np.array([  [-1191.51,555.801,320.959],
                                [-1131.14,660.228,330.711],
                                [1132.73,680.699,284.655],
                                [1194.26,574.739,282.936],
                                [71.2041,-1381.07,200.358],
                                [-52.5972,-1381.27,205.91]]).transpose()

padding = np.ones((1,6), dtype="float")
points3Dfixed_labeled = np.vstack((points3Dfixed_labeled, padding))
sba.points3Dfixed_labeled = points3Dfixed_labeled.copy()

sba.points3Dfixed = np.vstack((sba.points3Dfixed, padding))

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(sba.points3D[:,0], sba.points3D[:,1], sba.points3D[:,2])

ax.scatter(sba.points3Dfixed[0,:], sba.points3Dfixed[1,:], sba.points3Dfixed[2,:],
    c=np.linspace(0.5, 1, sba.points3Dfixed.shape[1]), cmap='Greens', vmin=0, vmax=1, label='points3Dfixed')

ax.scatter(sba.points3Dfixed_labeled[0,:], sba.points3Dfixed_labeled[1,:], sba.points3Dfixed_labeled[2,:],
    c=np.linspace(0.5, 1, sba.points3Dfixed_labeled.shape[1]), cmap='Oranges', vmin=0, vmax=1, label='points3Dfixed_labeled')
plt.legend()
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')
plt.title('shared Intrinsics')

for i in range(nCams):
    r_f = R.from_rotvec(-sba.cameraArray[i, 0:3]).as_matrix().copy()
    t_f = sba.cameraArray[i, 3:6].copy()
    # get inverse transformation
    r_inv = r_f.T    
    t_inv = -np.matmul(r_f, t_f)    
    ex = np.eye(4)
    ex[:3,:3] = r_inv.T
    ex[:3,3] = t_inv
    visualizer = MyCamPoseVisualizer(fig, ax)
    visualizer.extrinsic2pyramid(ex, my_palette[i], 200)
plt.show()
    

def fun(params):
    r = params.reshape(3,4)
    r = np.vstack((r, [0, 0, 0, 1]))
    out_pts = np.dot(r, sba.points3Dfixed_labeled)
    res = np.sum(np.absolute((sba.points3Dfixed.ravel() - out_pts.ravel()) ** 2))
    return res
    

r1 = np.hstack((np.eye(3), np.zeros((3,1))))
params = r1.ravel()
results = least_squares(fun, params)
print("results: ", results.x)
print("cost: ", results.cost)
    
    
ax = []
fig = plt.figure()

ax.append(fig.add_subplot(1, 2, 1, projection='3d'))
ax[0].scatter(sba.points3Dfixed_labeled[0,:], sba.points3Dfixed_labeled[1,:], sba.points3Dfixed_labeled[2,:], c='b', label='original_points')
ax[0].scatter(sba.points3Dfixed[0,:], sba.points3Dfixed[1,:], sba.points3Dfixed[2,:], c='r', label='target_points')
ax[0].legend()

nRigPoints = 6

for i in range(nRigPoints):
    x = [sba.points3Dfixed_labeled[0,i], sba.points3Dfixed[0,i]]
    y = [sba.points3Dfixed_labeled[1,i], sba.points3Dfixed[1,i]]
    z = [sba.points3Dfixed_labeled[2,i], sba.points3Dfixed[2,i]]
    ax[0].plot3D(x, y, z, 'gray')
ax[0].set_xlabel('X Label')
ax[0].set_ylabel('Y Label')
ax[0].set_zlabel('Z Label')
ax[0].set_title("initial points")


transformation_matrix = np.vstack((results.x.reshape(3,4), [0, 0, 0, 1]))

transformed_pts = np.dot(transformation_matrix, sba.points3Dfixed_labeled)
ax.append(fig.add_subplot(1, 2, 2, projection='3d'))
ax[1].scatter(transformed_pts[0,:], transformed_pts[1,:], transformed_pts[2,:], c='b', label='transformed_points')
ax[1].scatter(sba.points3Dfixed[0,:], sba.points3Dfixed[1,:], sba.points3Dfixed[2,:], c='r', label='target_points')
ax[1].legend()

for i in range(nRigPoints):
    x = [transformed_pts[0,i], sba.points3Dfixed[0,i]]
    y = [transformed_pts[1,i], sba.points3Dfixed[1,i]]
    z = [transformed_pts[2,i], sba.points3Dfixed[2,i]]
    ax[1].plot3D(x, y, z, 'gray')
ax[1].set_xlabel('X Label')
ax[1].set_ylabel('Y Label')
ax[1].set_zlabel('Z Label')
ax[1].set_title("transformed points after fitting")
plt.show()
    

## test moving the cameras and 

laser_pts = sba.points3D.copy().transpose()
laser_padding = np.ones((1, laser_pts.shape[1]))
laser_pts = np.vstack((laser_pts, laser_padding))
laser_pts_transformed = np.dot(transformation_matrix, laser_pts)

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
# ax.scatter(transformed_pts[0,:], transformed_pts[1,:], transformed_pts[2,:], c='b', label='transformed_rig_points')
# ax.scatter(target_pts[0,:], target_pts[1,:], target_pts[2,:], c='r', label='target_rig_points')
ax.scatter(laser_pts_transformed[0,:], laser_pts_transformed[1,:], laser_pts_transformed[2,:],
    c=np.linspace(0.5, 1, laser_pts_transformed.shape[1]), cmap='Oranges', vmin=0, vmax=1, label='points3Dfixed_labeled')
ax.legend()
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')
ax.set_title("transformed laser points")
plt.show()
    
    
    # # added by RJ
    # def fun_camonly(self, params, n_cameras, n_points, camera_indices, point_indices, points_2d, pointWeights, points_3d, intrinsics):
    #     """Compute residuals.
    #     'params' contains camera extrinsic only"""
    #     nCamParams = 8
    #     extrinsics = params.reshape(n_cameras, nCamParams)
    #     camera_params = np.hstack((extrinsics[:,:6], intrinsics, extrinsics[:,6:]))
    #     points_proj = self.project(points_3d[point_indices], camera_params[camera_indices])
    #     # weighted_residual = pointWeights*(points_proj-points_2d) ** 2
    #     weighted_residual = pointWeights*(points_proj-points_2d) ** 2
    #     return weighted_residual.ravel()


    # # added by RJ
    # def bundle_adjustment_camonly(self):
    #     """ Returns the bundle adjusted parameters, in this case the optimized rotation and translation vectors"""
    #     numCameras = self.cameraArray.shape[0]
    #     # numCamParams = 11
    #     numCamParams = 8
    #     numPoints = self.points3D.shape[0]

    #     # x0 = self.cameraArray.ravel()
    #     x0 = np.hstack((self.cameraArray[:,:6], self.cameraArray[:,9:])).ravel()
    #     # A = self.bundle_adjustment_sparsity(numCameras, numPoints, self.cameraIndices, self.point2DIndices)

    #     # res = least_squares(self.fun_camonly, x0, jac_sparsity=A, verbose=2, x_scale='jac', ftol=1e-6, method='trf', jac='3-point',
    #     #                     args=(numCameras, numPoints, self.cameraIndices, self.point2DIndices, self.points2D, self.pointWeights, self.points3D))

    #     intrinsics = self.cameraArray[:,6:9]
    #     res = least_squares(self.fun_camonly, x0, verbose=2, ftol=1e-5, method='trf',
    #                         args=(numCameras, numPoints, self.cameraIndices, self.point2DIndices, self.points2D, self.pointWeights, self.points3D, intrinsics))

    #     extrinsics = res.x.reshape(numCameras, numCamParams)
    #     self.cameraArray = np.hstack((extrinsics[:,:6], intrinsics, extrinsics[:,6:]))
    #     # self.cameraArray = res.x.reshape(numCameras, numCamParams)
    #     return res