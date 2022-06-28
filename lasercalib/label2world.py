
import numpy as np
import pySBA
import pickle
import pprint
from prettytable import PrettyTable
import seaborn as sns
import matplotlib.pyplot as plt
from my_cam_pose_visualizer import MyCamPoseVisualizer
from scipy.spatial.transform import Rotation as R

from scipy.sparse import lil_matrix
from scipy.optimize import least_squares

my_palette = sns.color_palette("rocket_r", 7)

picklefile = open('../calibres/sba_data', 'rb')
sba = pickle.load(picklefile)
picklefile.close()

x = PrettyTable()
for row in sba.cameraArray:
    x.add_row(row)
print(x)

# number of labeled points
nPts = 6
padding = np.ones((1,nPts), dtype="float")

# these are the xyz points from labeling 6 points in our labeling app (2 points in front of each robot)
label_pts = np.array([[-634.922,-1302.9,18.4702],
[-749.424,-1241.71,19.607],
[-782.37,1220.21,27.686],
[-665.717,1284.12,37.2595],
[1473.83,75.5635,41.0119],
[1473.11,-60.0179,44.0339]]).transpose()
input_pts = np.vstack((label_pts, padding))

# these are the xyz points in rig world space (rough estimate from the blender model -- in millimeters)
rig_pts = np.array([[-1.3302, 0.64879, 0.0],
[-1.227, 0.82761, 0.0],
[1.227, 0.82761, 0.0],
[1.3302, 0.64879, 0.0],
[0.10324, -1.4764, 0.0],
[-0.10324, -1.4764, 0.0]]).transpose() * 1000.0
target_pts = np.vstack((rig_pts, padding))

def fun(params):
    r = params.reshape(3,4)
    r = np.vstack((r, [0, 0, 0, 1]))
    out_pts = np.dot(r, input_pts)
    res = np.sum(np.absolute((target_pts.ravel() - out_pts.ravel()) ** 2))
    return res

r1 = np.hstack((np.eye(3), np.zeros((3,1))))
params = r1.ravel()
results = least_squares(fun, params)
print("results: ", results.x)
print("cost: ", results.cost)

ax = []
fig = plt.figure()

ax.append(fig.add_subplot(1, 2, 1, projection='3d'))
ax[0].scatter(input_pts[0,:], input_pts[1,:], input_pts[2,:], c='b', label='original_points')
ax[0].scatter(target_pts[0,:], target_pts[1,:], target_pts[2,:], c='r', label='target_points')
ax[0].legend()

for i in range(nPts):
    x = [input_pts[0,i], target_pts[0,i]]
    y = [input_pts[1,i], target_pts[1,i]]
    z = [input_pts[2,i], target_pts[2,i]]
    ax[0].plot3D(x, y, z, 'gray')
ax[0].set_xlabel('X Label')
ax[0].set_ylabel('Y Label')
ax[0].set_zlabel('Z Label')
ax[0].set_title("initial points")


transformation_matrix = np.vstack((results.x.reshape(3,4), [0, 0, 0, 1]))

transformed_pts = np.dot(transformation_matrix, input_pts)
ax.append(fig.add_subplot(1, 2, 2, projection='3d'))
ax[1].scatter(transformed_pts[0,:], transformed_pts[1,:], transformed_pts[2,:], c='b', label='transformed_points')
ax[1].scatter(target_pts[0,:], target_pts[1,:], target_pts[2,:], c='r', label='target_points')
ax[1].legend()

for i in range(nPts):
    x = [transformed_pts[0,i], target_pts[0,i]]
    y = [transformed_pts[1,i], target_pts[1,i]]
    z = [transformed_pts[2,i], target_pts[2,i]]
    ax[1].plot3D(x, y, z, 'gray')
ax[1].set_xlabel('X Label')
ax[1].set_ylabel('Y Label')
ax[1].set_zlabel('Z Label')
ax[1].set_title("transformed points after fitting")
plt.show()

np.set_printoptions(precision=5, suppress=True)
for i in range(nPts):
    print ("x, y, z, w: ", transformed_pts[:, i])

fig = plt.figure()
ax = fig.add_subplot(projection='3d')

sba_pts = sba.points3D.copy().transpose()
laser_padding = np.ones((1, sba_pts.shape[1]))
laser_pts = np.vstack((sba_pts, laser_padding))
laser_pts_transformed = np.dot(transformation_matrix, laser_pts)

print("sba.points3D shape before over-writing: ", sba.points3D.shape)

# update sba.points3D with transformed laser point positions
sba.points3D = laser_pts_transformed[:3, :].copy().transpose()

print("laser_pts_transformed shape: ", laser_pts_transformed.shape)
print("sba.points3D shape: ", sba.points3D.shape)
print("sba.cameraArray shape: ", sba.cameraArray.shape)

# update extrinsic camera parameters while keeping 3D points fixed
sba.bundle_adjustment_camonly()

# print camera params
x = PrettyTable()
for row in sba.cameraArray:
    x.add_row(row)
print(x)

ax.scatter(laser_pts_transformed[0,:], laser_pts_transformed[1,:], laser_pts_transformed[2,:], c='b', label='transformed laser points')
ax.scatter(transformed_pts[0,:], transformed_pts[1,:], transformed_pts[2,:], c=np.linspace(0.5, 1, nPts), cmap='Oranges', vmin=0, vmax=1, label='transformed rig points')
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')
ax.legend()

nCams = sba.cameraArray.shape[0]
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


# camList = []
# for i in range(nCams):
#     camList.append(pySBA.unconvertParams(sba.cameraArray[i,:]))

# pp = pprint.PrettyPrinter(indent=0)
# np.set_printoptions(precision=5)
# np.set_printoptions(suppress=True)

# for i, cam in enumerate(camList):
#     print('cam' + str(i))
#     pp.pprint(cam)
#     pp.format
#     print('\n')

# outParams = np.full((len(camList), 25), np.NaN)
# for nCam in range(len(camList)):
#     p = camList[nCam]
#     k = np.transpose(p['K']).ravel()
#     r_m = np.transpose(p['R']).ravel()
#     t = p['t']
#     d = np.hstack((p['d'], np.array([0.0, 0.0])))
#     outParams[nCam,:] = np.hstack((k, r_m, t, d))


# for i in range(nCams):
#     r = sba.cameraArray[i, 0:3].copy()
#     r_degree = np.degrees(r)
#     print("r_degree: ", r_degree)


