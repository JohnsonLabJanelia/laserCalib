import numpy as np
from scipy.spatial.transform import Rotation as R
from prettytable import PrettyTable
import pySBA
import matplotlib.pyplot as plt
from my_cam_pose_visualizer import MyCamPoseVisualizer
import seaborn as sns
import pickle


my_palette = sns.color_palette()
# load laser pointer point centroids found on all cameras
with np.load('../results/all_centroids.npz') as data:
    pts = data['arr_0']

# flip xy (regionprops orders)
pts = np.flip(pts, axis=1)

nPts = pts.shape[0]
nCams = pts.shape[2]

# load camera data for pySBA
cameraArray = np.zeros(shape=(nCams, 11))

# cam0
cameraArray[0, 0:3] = [-2.2151928, -2.2044225, 0.0019530592]
cameraArray[0, 3:6] = [-15.611495, -4.2011781, 2401.2117]
cameraArray[0, 6:9] = [1777.777, -0.015, -0.015]
cameraArray[0, 9:] = [1604, 1100]

# cam1
cameraArray[1, 0:3] = [1.9783967, -2.0124056, 0.44338688]
cameraArray[1, 3:6] = [-154.71857, -351.46274, 2299.1753]
cameraArray[1, 6:9] = [1777.777, -0.015, -0.015]
cameraArray[1, 9:] = [1604, 1100]

#cam2
cameraArray[2, 0:3] = [1.9780892, -2.0034461, 0.4536269]
cameraArray[2, 3:6] = [107.51159, -330.8779, 2300.0032]
cameraArray[2, 6:9] = [1777.777, -0.015, -0.015]
cameraArray[2, 9:] = [1604, 1100]

#cam3
cameraArray[3, 0:3] = [0.81482941, 2.8825006, -0.59704882]
cameraArray[3, 3:6] = [-113.24658, -372.63889, 2271.7502]
cameraArray[3, 6:9] = [1777.777, -0.015, -0.015]
cameraArray[3, 9:] = [1604, 1100]

#cam4
cameraArray[4, 0:3] = [0.82509673, 2.8785553, -0.59267741]
cameraArray[4, 3:6] = [134.61037, -379.2637, 2273.8696]
cameraArray[4, 6:9] = [1777.777, -0.015, -0.015]
cameraArray[4, 9:] = [1604, 1100]

#cam5
cameraArray[5, 0:3] = [2.612438, 0.79784292, -0.17937534]
cameraArray[5, 3:6] = [-74.572533, -305.59396, 2283.8472]
cameraArray[5, 6:9] = [1777.777, -0.015, -0.015]
cameraArray[5, 9:] = [1604, 1100]

#cam6
cameraArray[6, 0:3] = [2.6102991, 0.79794997, -0.17989315]
cameraArray[6, 3:6] = [190.82062, -295.33585, 2283.1584]
cameraArray[6, 6:9] = [1777.777, -0.015, -0.015]
cameraArray[6, 9:] = [1604, 1100]




# fig, axs = plt.subplots(1, nCams, sharey=True)
# plt.title('2D points found on all cameras')
# for i in range(nCams):
#     colors = np.linspace(0, 1, nPts)
#     axs[i].scatter(pts[:,0,i], pts[:,1,i], s=10, c=colors, alpha=0.5)
#     axs[i].title.set_text('cam' + str(i))
#     axs[i].invert_yaxis()
# plt.show()

# # create camera_ind variable
# camera_ind = np.zeros(shape=(nPts*nCams,), dtype=int)
# point_ind = np.zeros(shape=(nPts*nCams,), dtype=int)
# points_2d = np.zeros(shape=(nPts*nCams, 2), dtype=float)
# for i in range(nCams):
#     for j in range(nPts):
#         ind = (i*nPts) + j
#         camera_ind[ind] = i
#         point_ind[ind] = j
#         points_2d[ind, :] = pts[j, :, i].copy()

# create camera_ind variable

# make sure at least 3 cams observe each observation
keep = []
for i in range(pts.shape[0]):
    r = pts[i,0,:].ravel()
    s = np.sum(~np.isnan(r))
    if (s < 3):
        pts[i,:,:] = np.nan
    else:
        keep.append(i)

keep = np.array(keep)


# prepare points_3d variable (initializing with 2d laser points in image space on cam0)
points_3d = np.zeros(shape=(nPts, 3))
# points_3d[:, :2] = pts[:,:,0].copy()

# # # center the world points
# points_3d[:,0] = points_3d[:,0] - 1604
# points_3d[:,1] = points_3d[:,1] - 1100

# for i in range(points_3d.shape[0]):
#     if np.isnan(points_3d[i, 0]):
#         points_3d[i, :] = [0, 0, 0]

camera_ind = []
point_ind = []
points_2d = []
for i in range(nCams):
    for j in keep:
        if (not np.isnan(pts[j,0,i])):
            camera_ind.append(i)
            point_ind.append(j)
            points_2d.append(pts[j,:,i])

camera_ind = np.array(camera_ind)
point_ind = np.array(point_ind)
points_2d = np.array(points_2d)

print("camera_ind shape: ", camera_ind.shape)
print("point_ind shape: ", point_ind.shape)
print("points_2d shape: ", points_2d.shape)


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


sba.bundleAdjust_nocam()

x = PrettyTable()
for row in sba.cameraArray:
    x.add_row(row)
print(x)

r = sba.project(sba.points3D[sba.point2DIndices], sba.cameraArray[sba.cameraIndices]) - sba.points2D
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


"""
Given the updated 3d positions jointly optimize the camera parameters and 3d positions to minimize reconstruction errors.  
Use sba.bundleAdjust() if you want each camera to have separate intrinsics.
sba.bundleAdjust_sharedcam() uses shared intrinsics but with different image centroids used for radial distortion.
"""

sba.bundleAdjust_sharedcam()
# sba.bundleAdjust()
x = PrettyTable()
for row in sba.cameraArray:
    x.add_row(row)
print(x)

r = sba.project(sba.points3D[sba.point2DIndices], sba.cameraArray[sba.cameraIndices]) - sba.points2D
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
picklefile = open('../calibres/sba_data_new', 'wb')
pickle.dump(sba, picklefile)
picklefile.close()





