from tkinter.messagebox import NO
import numpy as np
import matplotlib.pyplot as plt
import pickle as pkl

from scipy.spatial.transform import Rotation as R
from prettytable import PrettyTable
import pySBA
from my_cam_pose_visualizer import MyCamPoseVisualizer
import seaborn as sns

my_palette = sns.color_palette("rocket_r", 7)


filename = 'centroids_20220701.pkl'
fileObject = open(filename, 'rb')
pts = pkl.load(fileObject)
fileObject.close()

# flip xy (regionprops orders)
pts = np.flip(pts, axis=1)
nPts = pts.shape[0]
nCams = pts.shape[2]

# initialize cameraArray for pySBA
cameraArray = np.zeros(shape=(nCams, 11))

#initial guesses
# cam0
cameraArray[0, 0:3] = [-3.14, 0, 0]
cameraArray[0, 3:6] = [0, -3.8224, 2400]
cameraArray[0, 6:9] = [1777.777, -0.015, -0.015]
cameraArray[0, 9:] = [1604, 1100]

# cam1
cameraArray[1, 0:3] = [-0.00014554, -3.0793, 0.62272]
cameraArray[1, 3:6] = [-0.074297, -375.19, 2198.6]
cameraArray[1, 6:9] = [1777.777, -0.015, -0.015]
cameraArray[1, 9:] = [1604, 1100]

#cam2
cameraArray[2, 0:3] = [0.65181, -2.4327, 1.4019]
cameraArray[2, 3:6] = [-0.030555, -341.72, 2204]
cameraArray[2, 6:9] = [1777.777, -0.015, -0.015]
cameraArray[2, 9:] = [1604, 1100]

#cam3
cameraArray[3, 0:3] = [2.409, 1.3909, -0.28182]
cameraArray[3, 3:6] = [0.018617, -373.52, 2198.8]
cameraArray[3, 6:9] = [1777.777, -0.015, -0.015]
cameraArray[3, 9:] = [1604, 1100]

#cam4
cameraArray[4, 0:3] = [1.5846, 1.5846, -0.9132]
cameraArray[4, 3:6] = [0.031478, -341.7, 2204]
cameraArray[4, 6:9] = [1777.777, -0.015, -0.015]
cameraArray[4, 9:] = [1604, 1100]

#cam5
cameraArray[5, 0:3] = [2.0397, -0.54654, 0.31497]
cameraArray[5, 3:6] = [0.0031259, -341.73, 2204]
cameraArray[5, 6:9] = [1777.777, -0.015, -0.015]
cameraArray[5, 9:] = [1604, 1100]

#cam6
cameraArray[6, 0:3] = [2.409, -1.3909, 0.28182]
cameraArray[6, 3:6] = [-0.018617, -373.52, 2198.8]
cameraArray[6, 6:9] = [1777.777, -0.015, -0.015]
cameraArray[6, 9:] = [1604, 1100]


keep = np.zeros(shape=(nPts,), dtype=bool)
for i in range(nPts):
    v = pts[i, 0, :]
    if ((np.sum(~np.isnan(v)) >=3) and (~np.isnan(v[0]))):
        keep[i] = True

inPts = pts[keep, :, :]
nPts = inPts.shape[0]
print("nPts: ", nPts)

nObs = np.sum(~np.isnan(inPts[:,0,:].ravel()))
print("nObs: ", nObs)

fig, axs = plt.subplots(1, nCams, sharey=True)
plt.title('2D points found on all cameras')
for i in range(nCams):
    colors = np.linspace(0, 1, nPts)
    axs[i].scatter(inPts[:,0,i], inPts[:,1,i], s=10, c=colors, alpha=0.5)
    axs[i].plot(inPts[:,0,i], inPts[:,1,i])
    axs[i].title.set_text('cam' + str(i))
    axs[i].invert_yaxis()
plt.show()

# create camera_ind variable
camera_ind = np.zeros(shape=(nObs,), dtype=int)
point_ind = np.zeros(shape=(nObs,), dtype=int)
points_2d = np.zeros(shape=(nObs, 2), dtype=float)

ind = 0
for i in range(nPts):
    for j in range(nCams):
        if (np.isnan(inPts[i, 0, j])):
            continue
        camera_ind[ind] = j
        point_ind[ind] = i
        points_2d[ind, :] = inPts[i, :, j].copy()
        ind += 1


# prepare points_3d variable (initializing with 2d laser points in image space on cam0)
points_3d = np.zeros(shape=(nPts, 3))
for i in range(nPts):
    if (np.isnan(inPts[i, 0, 0])):
        points_3d[i, 0:2] = [1604, 1100]
        continue
    else:
        points_3d[i, 0:2] = inPts[i, :, 0]

# # # center the world points
points_3d[:,0] = points_3d[:,0] - 1604
points_3d[:,1] = points_3d[:,1] - 1100


"""
initialize the SBA object with points and calibration (using an old calibration or just general ballpark calculated manually). 
Then optimize for the 3d positions holding all camera parameters fixed
"""
sba = pySBA.PySBA(cameraArray, points_3d, points_2d, camera_ind, point_ind)

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
    visualizer = MyCamPoseVisualizer(fig, ax)
    visualizer.extrinsic2pyramid(ex, my_palette[i], 200)
plt.show()


"""
added by RJ
first move the camera extrinsics and hold the 3D points constant
"""
# sba.bundleAdjust_transform_points_3d()

# fig = plt.figure()
# ax = fig.add_subplot(projection='3d')
# ax.scatter(sba.points3D[:,0], sba.points3D[:,1], sba.points3D[:,2])
# ax.set_xlabel('X Label')
# ax.set_ylabel('Y Label')
# ax.set_zlabel('Z Label')
# plt.title('initial cam params and 3D points')
# for i in range(nCams):
#     r_f = R.from_rotvec(-sba.cameraArray[i, 0:3]).as_matrix().copy()
#     t_f = sba.cameraArray[i, 3:6].copy()
#     # get inverse transformation
#     r_inv = r_f.T    
#     t_inv = -np.matmul(r_f, t_f)    
#     ex = np.eye(4)
#     ex[:3,:3] = r_inv.T
#     ex[:3,3] = t_inv
#     visualizer = MyCamPoseVisualizer(fig, ax)
#     visualizer.extrinsic2pyramid(ex, my_palette[i], 200)
# plt.show()


sba.bundleAdjust_nocam()

x = PrettyTable()
for row in sba.cameraArray:
    x.add_row(row)
print(x)

r = sba.project(sba.points3D[sba.point2DIndices], sba.cameraArray[sba.cameraIndices]) - sba.points2D
print(r.shape)
r = np.sqrt(np.sum(r**2, axis=1))
plt.hist(r[r<np.percentile(r, 99)])
plt.xlabel('Reprojection Error')
plt.title('no adjustment')
plt.show()

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(sba.points3D[:,0], sba.points3D[:,1], sba.points3D[:,2])
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')
plt.title('cam params held; fit 3D points')
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


# """
# Given the updated 3d positions jointly optimize the camera parameters and 3d positions to minimize reconstruction errors.  
# Use sba.bundleAdjust() if you want each camera to have separate intrinsics.
# sba.bundleAdjust_sharedcam() uses shared intrinsics but with different image centroids used for radial distortion.
# """

sba.bundleAdjust_sharedcam()

x = PrettyTable()
for row in sba.cameraArray:
    x.add_row(row)
print(x)

r = sba.project(sba.points3D[sba.point2DIndices], sba.cameraArray[sba.cameraIndices]) - sba.points2D
print(r)
r = np.sqrt(np.sum(r**2, axis=1))
plt.hist(r[r<np.percentile(r, 99)])
plt.xlabel('Reprojection Error')
plt.title('shared Intrinsics')
plt.show()


fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(sba.points3D[:,0], sba.points3D[:,1], sba.points3D[:,2])
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

sba.saveCamVecs()
picklefile = open('../calibres/old/sba_blender', 'wb')
pkl.dump(sba, picklefile)
picklefile.close()