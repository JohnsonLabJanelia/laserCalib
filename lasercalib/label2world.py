
import numpy as np
import pySBA
import pickle
import pprint
from prettytable import PrettyTable
import seaborn as sns
import matplotlib.pyplot as plt
from camera_visualizer import CameraVisualizer
from scipy.spatial.transform import Rotation as R
from convert_params import *
from datetime import date
from rigid_body import rigid_transform_3D
import argparse
from sba_print import sba_print
import os

parser = argparse.ArgumentParser()
parser.add_argument('--root_dir', type=str, required=True)
parser.add_argument('--n_cams', type=int, required=True)

args = parser.parse_args()
root_dir = args.root_dir
nCams = args.n_cams

my_palette = sns.color_palette("pastel", nCams)
# 4 feature points on a square
rig_pts = np.array([[-692.0, 692.0, 0.0], [-692.0, -692.0, 0.0], [692, -692, 0.0], [692, 692, 0.0]]).transpose()
#label_pts = np.array([[-780.618, 741.511, 586.724], [-792.757, -748.489, 599.694], [706.665, -773.379, 607.877], [721.711, 725.702, 601.463]]).transpose()

with open(root_dir + "/results/aruco_center_3d.pkl", "rb") as f:
    label_pts = pkl.load(f)   
label_pts = label_pts.transpose()
##

with open(root_dir + '/results/sba_blender.pkl', 'rb') as f:
    sba = pickle.load(f)

# number of labeled points
nPts = 4 
padding = np.ones((1,nPts), dtype="float")
input_pts = np.vstack((label_pts, padding))
target_pts = np.vstack((rig_pts, padding))
edge_len = rig_pts[1, 0] - rig_pts[0, 0]

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


transformation_matrix = transform_label2world
transformed_pts = np.dot(transformation_matrix, input_pts)

print("transformation_matrix")
print(transformation_matrix)

print("transformed points")
print(transformed_pts)

## TODO: fit scaling matrix together
mag = np.abs(transformed_pts[1, 0] - transformed_pts[0, 0])
scale_mag = np.abs(edge_len / mag)
print("scale_mag: ", scale_mag)
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

x, y, z = np.array([[-500,0,0],[0,-500,0],[0,0,-500]])
u, v, w = np.array([[1000,0,0],[0,1000,0],[0,0,1000]])

ax = []
fig = plt.figure()
ax.append(fig.add_subplot(1, 2, 1, projection='3d'))
cam_pts_label_space = np.zeros(shape=(3, nCams))
cam_ex_rig_space = np.zeros(shape=(4,4,nCams))

camList = []
for i in range(nCams):
    camList.append(sba_to_readable_format(sba.cameraArray[i,:]))

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

    visualizer = CameraVisualizer(fig, ax[0])
    visualizer.extrinsic2pyramid(ex, my_palette[i], 200)

ax[0].scatter(laser_pts[0,:], laser_pts[1,:], laser_pts[2,:], color='m', alpha=0.1)
ax[0].scatter(cam_pts_label_space[0,:], cam_pts_label_space[1,:], cam_pts_label_space[2,:], color="r")


print("input_pts shape: ", input_pts.shape)
print(input_pts)
print(input_pts[:,0])

ref_pts_label_space = np.hstack((input_pts, input_pts[:, 0].reshape(-1,1)))
ax[0].plot(ref_pts_label_space[0,:], ref_pts_label_space[1,:], ref_pts_label_space[2,:], color="orange", linewidth=5, marker="o", markerfacecolor="k")
ax[0].quiver(x,y,z,u,v,w,arrow_length_ratio=0.1, color="black")
ax[0].set_xlim([-1500, 1500])
ax[0].set_ylim([-1500, 1500])
ax[0].set_zlim([-100, 1800])
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


new_camList = []
for i in range(nCams):
    ex = np.squeeze(cam_ex_rig_space[:,:,i])
    visualizer = CameraVisualizer(fig, ax[1])
    visualizer.extrinsic2pyramid(ex, my_palette[i], 200)
    this_r = ex[:3,:3].copy()
    r_vec = R.from_matrix(this_r).as_rotvec()
    normalized_R = R.from_rotvec(r_vec).as_matrix()
    t_vec = -np.dot(normalized_R.T, ex[:3, 3])
    print("Cam {}, R {}, t_vec {}".format(i, R.from_rotvec(r_vec).as_matrix(), t_vec))
    
    new_cam_params = {}
    new_cam_params['K'] = camList[i]['K']
    new_cam_params['d'] = camList[i]['d']
    new_cam_params['R'] = R.from_rotvec(r_vec).as_matrix()
    new_cam_params['t'] = t_vec
    new_camList.append(new_cam_params)

ax[1].set_xlim([-1500, 1500])
ax[1].set_ylim([-1500, 1500])
ax[1].set_zlim([-100, 1800])
ax[1].set_title("rig space")
plt.show()


fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(laser_pts_transformed[0,:], laser_pts_transformed[1,:], laser_pts_transformed[2,:])
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')
for i, cam in enumerate(new_camList):
    ex = np.eye(4)
    
    r_f = cam["R"]
    r_inv = r_f.T

    t_f = cam["t"]
    t_inv = -np.dot(r_f, t_f)
    print("Cam {}, t_inv {}".format(i, t_inv))

    ex[:3,:3] = cam["R"]  
    ex[:3, 3] = t_inv

    visualizer = CameraVisualizer(fig, ax)
    visualizer.extrinsic2pyramid(ex, my_palette[i], 200)
ax.set_xlim([-1500, 1500])
ax.set_ylim([-1500, 1500])
ax.set_zlim([-100, 1800])
ax.scatter(cam_pts_rig_space[0,:], cam_pts_rig_space[1,:], cam_pts_rig_space[2,:], color="r")
sns.palplot(my_palette)
plt.show()


# """
# This code block works to put the camera parameters in rig space
# """
# # Loading current best cam params
# with open(root_dir + "/results/sba_blender.pkl", 'rb') as f:
#     sba = pickle.load(f)

# sba.points3D = laser_pts_transformed.copy()[:3,:].transpose()

# sba.bundle_adjustment_camonly(1e-4)
# sba_print(sba, nCams, "Refit", zlim=[-100, 1800], color_palette=my_palette)

# new_camList = []
# for i in range(nCams):
#     new_camList.append(sba_to_readable_format(sba.cameraArray[i,:]))
#     print("Cam {}, R {}, t_vec {}".format(i, new_camList[i]['R'], new_camList[i]['t']))


save_root = root_dir + "/rigspace/results/"
with open(save_root + "calibration_rigspace.pkl", 'wb') as f:
    pkl.dump(new_camList, f)

outParams = readable_to_red_format(new_camList)
np.savetxt(save_root + "calibration_rigspace.csv", outParams, delimiter=',', newline=',\n', fmt='%f')

# save for aruco detection
aruco_folder = save_root + "/calibration_aruco/"
if not os.path.exists(aruco_folder):
   os.makedirs(save_root)
red_to_aruco(aruco_folder, nCams, outParams)
