
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

my_palette = sns.color_palette("pastel", 7)



def rigid_transform_3D(A, B):
    # Input: expects 3xN matrix of points
    # Returns R,t
    # R = 3x3 rotation matrix
    # t = 3x1 column vector

    assert A.shape == B.shape

    num_rows, num_cols = A.shape
    if num_rows != 3:
        raise Exception(f"matrix A is not 3xN, it is {num_rows}x{num_cols}")

    num_rows, num_cols = B.shape
    if num_rows != 3:
        raise Exception(f"matrix B is not 3xN, it is {num_rows}x{num_cols}")

    # find mean column wise
    centroid_A = np.mean(A, axis=1)
    centroid_B = np.mean(B, axis=1)

    # ensure centroids are 3x1
    centroid_A = centroid_A.reshape(-1, 1)
    centroid_B = centroid_B.reshape(-1, 1)

    # subtract mean
    Am = A - centroid_A
    Bm = B - centroid_B

    H = Am @ np.transpose(Bm)

    # sanity check
    #if linalg.matrix_rank(H) < 3:
    #    raise ValueError("rank of H = {}, expecting 3".format(linalg.matrix_rank(H)))

    # find rotation
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    # special reflection case
    if np.linalg.det(R) < 0:
        print("det(R) < R, reflection detected!, correcting for it ...")
        Vt[2,:] *= -1
        R = Vt.T @ U.T

    t = -R @ centroid_A + centroid_B

    return R, t


picklefile = open('../calibres/sba_blender_2022-09-30.pkl', 'rb')
sba = pickle.load(picklefile)
picklefile.close()

x = PrettyTable()
for row in sba.cameraArray:
    x.add_row(row)
print(x)

# number of labeled points
nPts = 4 
padding = np.ones((1,nPts), dtype="float")

# these are the xyz points from labeling 4 points in our labeling app (2 points in front of each robot)
label_pts = np.array([[-0.486981,-53.6274,-59.1112],
[180.239,-52.6075,-59.8065],
[180.156,-315.251,-65.3472],
[0.543318,-315.454,-64.8314]]).transpose()
input_pts = np.vstack((label_pts, padding))

# these are the xyz points in rig world space (rough estimate from the blender model -- in millimeters)
# rig_pts = np.array([[0.0, 0.0, 0.0],
# [0.0, -182.0, 0.0],
# [-266.0, -182.0, 0.0],
# [-266.0, 0.0, 0.0]]).transpose()

rig_pts = np.array([[0.0, 0.0, 0.0],
[182.0, .0, 0.0],
[182.0, -266.0, 0.0],
[.0, -266.0, 0.0]]).transpose()
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


A = label_pts.copy()
B = rig_pts.copy()

(r_label2world, t_label2world) = rigid_transform_3D(A,B)


transform_label2world = np.hstack((r_label2world, t_label2world))
transform_label2world = np.vstack((transform_label2world, [0, 0, 0, 1]))
print(transform_label2world)


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


# transformation_matrix = np.vstack((results.x.reshape(3,4), [0, 0, 0, 1]))
transformation_matrix = transform_label2world



transformed_pts = np.dot(transformation_matrix, input_pts)

transformation_matrix[:3,3] -= transformed_pts[:3, 0]

transformed_pts = np.dot(transformation_matrix, input_pts)

print("transformation_matrix")
print(transformation_matrix)

print("transformed points")
print(transformed_pts)

chess_board_edge = transformed_pts[:3, 1].copy()
print(chess_board_edge)
mag = np.sqrt(chess_board_edge.dot(chess_board_edge))

scale_mag = np.abs(182.0 / mag)
scale_eye = np.eye(4) * scale_mag
scale_eye[3,3] = 1
print(scale_eye)

transformation_matrix = np.dot(scale_eye, transformation_matrix)

print(transformation_matrix)
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



sba_pts = sba.points3D.copy().transpose()
laser_padding = np.ones((1, sba_pts.shape[1]))
laser_pts = np.vstack((sba_pts, laser_padding))
laser_pts_transformed = np.dot(transformation_matrix, laser_pts)


# fig = plt.figure()
# ax = fig.add_subplot(projection='3d')

# ax.scatter(laser_pts_transformed[0,:], laser_pts_transformed[1,:], laser_pts_transformed[2,:], c='b', label='transformed laser points')
# ax.scatter(transformed_pts[0,:], transformed_pts[1,:], transformed_pts[2,:], c=np.linspace(0.5, 1, nPts), cmap='Oranges', vmin=0, vmax=1, label='transformed rig points')
# ax.set_xlabel('X Label')
# ax.set_ylabel('Y Label')
# ax.set_zlabel('Z Label')
# ax.set_xlim([-2400, 2400])
# ax.set_ylim([-2400, 2400])
# ax.set_zlim([-2400, 2400])
# ax.legend()


# pp = pprint.PrettyPrinter(indent=0)
# np.set_printoptions(precision=5)
# np.set_printoptions(suppress=True)
# plt.show()



nCams = 4
camList = []
for i in range(nCams):
    camList.append(pySBA.unconvertParams(sba.cameraArray[i,:]))


x, y, z = np.array([[-500,0,0],[0,-500,0],[0,0,-500]])
u, v, w = np.array([[1000,0,0],[0,1000,0],[0,0,1000]])

ax = []
fig = plt.figure()
ax.append(fig.add_subplot(1, 2, 1, projection='3d'))
cam_pts_label_space = np.zeros(shape=(3, nCams))

cam_ex_rig_space = np.zeros(shape=(4,4,nCams))

for i, cam in enumerate(camList):
    ex = np.eye(4)
    
    r_f = cam["R"]
    r_inv = r_f.T

    t_f = cam["t"]
    t_inv = -np.dot(r_f, t_f)

    ex[:3,:3] = cam["R"]
    ex[:3, 3] = t_inv

    cam_pts_label_space[:,i] = t_inv
    cam_ex_rig_space[:,:,i] = np.dot(transformation_matrix, ex)

    visualizer = MyCamPoseVisualizer(fig, ax[0])
    visualizer.extrinsic2pyramid(ex, my_palette[i], 200)

ax[0].scatter(laser_pts[0,:], laser_pts[1,:], laser_pts[2,:], color='m', alpha=0.1)
ax[0].scatter(cam_pts_label_space[0,:], cam_pts_label_space[1,:], cam_pts_label_space[2,:], color="r")


print("input_pts shape: ", input_pts.shape)
print(input_pts)
print(input_pts[:,0])

ref_pts_label_space = np.hstack((input_pts, input_pts[:, 0].reshape(-1,1)))
ax[0].plot(ref_pts_label_space[0,:], ref_pts_label_space[1,:], ref_pts_label_space[2,:], color="orange", linewidth=5, marker="o", markerfacecolor="k")
ax[0].quiver(x,y,z,u,v,w,arrow_length_ratio=0.1, color="black")
ax[0].set_xlim([-2400, 2400])
ax[0].set_ylim([-2400, 2400])
ax[0].set_zlim([-2400, 2400])
ax[0].set_title("label space")

cam_padding = np.ones((1, cam_pts_label_space.shape[1]))
cam_pts_input = np.vstack((cam_pts_label_space, cam_padding))
cam_pts_rig_space = np.dot(transformation_matrix, cam_pts_input)

ref_pts_rig_space = np.hstack((transformed_pts, transformed_pts[:, 0].reshape(-1,1)))

ax.append(fig.add_subplot(1, 2, 2, projection='3d'))
ax[1].scatter(laser_pts_transformed[0,:], laser_pts_transformed[1,:], laser_pts_transformed[2,:], color="m", alpha=0.1)
ax[1].scatter(cam_pts_rig_space[0,:], cam_pts_rig_space[1,:], cam_pts_rig_space[2,:], color="r")
ax[1].plot(ref_pts_rig_space[0,:], ref_pts_rig_space[1,:], ref_pts_rig_space[2,:], color="orange", linewidth=5, marker="o", markerfacecolor="k")


ax[1].quiver(x,y,z,u,v,w,arrow_length_ratio=0.1, color="black")

for i, cam in enumerate(camList):
    ex = np.squeeze(cam_ex_rig_space[:,:,i])
    visualizer = MyCamPoseVisualizer(fig, ax[1])
    visualizer.extrinsic2pyramid(ex, my_palette[i], 200)

    # print("cam" + str(i))
    # print(ex)
    # this_r = ex[:3,:3].copy()
    # r_vec = R.from_matrix(this_r.T).as_rotvec()
    # print("r_vec: ", r_vec)
    # t_vec = np.dot(-this_r.T, ex[:3, 3])
    # print("t_vec: ", t_vec)

    print("cam" + str(i))
    print(ex)
    this_r = ex[:3,:3].copy()
    r_vec = R.from_matrix(this_r).as_rotvec()
    print("r_vec: ", r_vec)
    t_vec = np.dot(this_r, ex[:3, 3])
    print("t_vec: ", t_vec)


ax[1].set_xlim([-2400, 2400])
ax[1].set_ylim([-2400, 2400])
ax[1].set_zlim([-2400, 2400])
ax[1].set_title("rig space")
plt.show()

############################## Loading current best cam params

picklefile = open('../calibres/sba_blender_rigspace', 'rb')
sba_rigspace = pickle.load(picklefile)
picklefile.close()

x = PrettyTable()
for row in sba_rigspace.cameraArray:
    x.add_row(row)
print(x)




#############################
"""
This code block works to put the camera parameters in rig space - but is messy (does bundle adjustment again)
"""

sba.points3D = laser_pts_transformed.copy()[:3,:].transpose()

# sba.bundle_adjustment_camonly()
sba.bundle_adjustment_camonly_shared()

new_camList = []
for i in range(nCams):
    new_camList.append(pySBA.unconvertParams(sba.cameraArray[i,:]))

fig = plt.figure()
ax = fig.add_subplot(projection='3d')

for i, cam in enumerate(new_camList):
    ex = np.eye(4)
    
    r_f = cam["R"]
    r_inv = r_f.T

    t_f = cam["t"]
    t_inv = -np.dot(r_f, t_f)

    ex[:3,:3] = cam["R"]
    ex[:3, 3] = t_inv

    visualizer = MyCamPoseVisualizer(fig, ax)
    visualizer.extrinsic2pyramid(ex, my_palette[i], 200)

ax.scatter(sba.points3D[:,0], sba.points3D[:,1], sba.points3D[:,2], color='m', alpha=0.1)
plt.show()

x = PrettyTable()
for row in sba.cameraArray:
    x.add_row(row)
print(x)
    
for i in range(nCams):
    sba.cameraArray[i, 6:9] = [1777.777, -0.015, -0.015]

sba.bundleAdjust_sharedcam()

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


r = sba.project(sba.points3D[sba.point2DIndices], sba.cameraArray[sba.cameraIndices]) - sba.points2D
print(r)
r = np.sqrt(np.sum(r**2, axis=1))
plt.hist(r[r<np.percentile(r, 99)])
plt.xlabel('Reprojection Error')
plt.title('shared Intrinsics')
plt.show()

# sba.saveCamVecs()
# picklefile = open('../calibres/sba_blender_rigspace', 'wb')
# pickle.dump(sba, picklefile)
# picklefile.close()


camList = []
for i in range(nCams):
    camList.append(pySBA.unconvertParams(sba.cameraArray[i,:]))

outParams = np.full((len(camList), 25), np.NaN)
for nCam in range(len(camList)):
    p = camList[nCam]
    k = np.transpose(p['K']).ravel()
    r_m = np.transpose(p['R']).ravel()
    t = p['t']
    d = np.hstack((p['d'], np.array([0.0, 0.0])))
    outParams[nCam,:] = np.hstack((k, r_m, t, d))


np.savetxt('../calibres/calibration_20220930_rigspace.csv', outParams, delimiter=',', newline=',\n', fmt='%f')


################################



