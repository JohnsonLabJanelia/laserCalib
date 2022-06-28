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


filename = 'centroids_timeit.pkl'
# filename = 'centroids_initial_guess.pkl'

# filename = 'centroids.pkl'
fileObject = open(filename, 'rb')
pts = pkl.load(fileObject)
fileObject.close()

# flip xy (regionprops orders)
pts = np.flip(pts, axis=1)
nPts = pts.shape[0]
nCams = pts.shape[2]

# load camera data for pySBA
cameraArray = np.zeros(shape=(nCams, 11))


#initial guesses
# cam0
cameraArray[0, 0:3] = [-3.14, 0, 0]
cameraArray[0, 3:6] = [0, 0, 2600]
cameraArray[0, 6:9] = [1777.777, -0.015, -0.015]
cameraArray[0, 9:] = [1604, 1100]

# cam1
cameraArray[1, 0:3] = [0, -3.14, .63]
cameraArray[1, 3:6] = [-150, -200, 2300]
cameraArray[1, 6:9] = [1777.777, -0.015, -0.015]
cameraArray[1, 9:] = [1604, 1100]

#cam2
cameraArray[2, 0:3] = [0, -3.14, .63]
cameraArray[2, 3:6] = [150, -200, 2300]
cameraArray[2, 6:9] = [1777.777, -0.015, -0.015]
cameraArray[2, 9:] = [1604, 1100]

#cam3
cameraArray[3, 0:3] = [2.31, 1.27, -0.34]
cameraArray[3, 3:6] = [-150, -200, 2300]
cameraArray[3, 6:9] = [1777.777, -0.015, -0.015]
cameraArray[3, 9:] = [1604, 1100]

#cam4
cameraArray[4, 0:3] = [2.31, 1.27, -0.34]
cameraArray[4, 3:6] = [150, -200, 2300]
cameraArray[4, 6:9] = [1777.777, -0.015, -0.015]
cameraArray[4, 9:] = [1604, 1100]

#cam5
cameraArray[5, 0:3] = [2.31, -1.27, 0.34]
cameraArray[5, 3:6] = [-150, -200, 2300]
cameraArray[5, 6:9] = [1777.777, -0.015, -0.015]
cameraArray[5, 9:] = [1604, 1100]

#cam6
cameraArray[6, 0:3] = [2.31, -1.27, 0.34]
cameraArray[6, 3:6] = [150, -200, 2300]
cameraArray[6, 6:9] = [1777.777, -0.015, -0.015]
cameraArray[6, 9:] = [1604, 1100]


# true world points from CAD model
# these are the xyz points in rig world space (rough estimate from the blender model -- in millimeters)
rig_pts = np.array([[-1.3302, 0.64879, 0.0],
[-1.227, 0.82761, 0.0],
[1.227, 0.82761, 0.0],
[1.3302, 0.64879, 0.0],
[0.10324, -1.4764, 0.0],
[-0.10324, -1.4764, 0.0]]).transpose() * 1000.0


keep = np.zeros(shape=(nPts,), dtype=bool)
for i in range(nPts):
    v = pts[i, 0, :]
    if ((np.sum(~np.isnan(v)) >=3) and (~np.isnan(v[0]))):
        keep[i] = True

inPts = pts[keep, :, :]
nPts = inPts.shape[0]
print("nPts: ", nPts)

fig, axs = plt.subplots(1, nCams, sharey=True)
plt.title('2D points found on all cameras')
for i in range(nCams):
    colors = np.linspace(0, 1, nPts)
    axs[i].scatter(inPts[:,0,i], inPts[:,1,i], s=10, c=colors, alpha=0.5)
    axs[i].title.set_text('cam' + str(i))
    axs[i].invert_yaxis()
plt.show()


nObs = np.sum(~np.isnan(inPts[:,0,:].ravel()))
print("nObs: ", nObs)


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
sba = pySBA.PySBA(cameraArray, points_3d, points_2d, camera_ind, point_ind, rig_pts)

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
ax.scatter(sba.points3Dfixed[0,:], sba.points3Dfixed[1,:], sba.points3Dfixed[2,:],
    c=np.linspace(0.5, 1, sba.points3Dfixed.shape[1]), cmap='Oranges', vmin=0, vmax=1, label='fixed rig points')
plt.title('initial cam params and 3D points')
plt.legend()
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
ax.scatter(sba.points3Dfixed[0,:], sba.points3Dfixed[1,:], sba.points3Dfixed[2,:],
    c=np.linspace(0.5, 1, sba.points3Dfixed.shape[1]), cmap='Oranges', vmin=0, vmax=1, label='fixed rig points')
plt.legend()
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
ax.scatter(sba.points3Dfixed[0,:], sba.points3Dfixed[1,:], sba.points3Dfixed[2,:],
    c=np.linspace(0.5, 1, sba.points3Dfixed.shape[1]), cmap='Oranges', vmin=0, vmax=1, label='fixed rig points')
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

# sba.saveCamVecs()
# picklefile = open('../calibres/sba_guess', 'wb')
# pkl.dump(sba, picklefile)
# picklefile.close()


