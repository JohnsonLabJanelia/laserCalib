
import numpy as np
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
from lasercalib.camera_visualizer import CameraVisualizer
from scipy.spatial.transform import Rotation as R
from lasercalib.convert_params import *
from lasercalib.rigid_body import point_set_registration, apply_rigid_transform, invert_Rt
import argparse
from lasercalib.sba_print import sba_print
import os
import json

parser = argparse.ArgumentParser()
parser.add_argument('-c', '--config', type=str, required=True)

args = parser.parse_args()
config_dir = args.config
with open(config_dir + '/config.json', 'r') as f:
    calib_config = json.load(f)

cam_serials = calib_config['cam_serials']
cam_names = []
for cam_serial in cam_serials:
    cam_names.append("Cam" + cam_serial)
n_cams = len(cam_names)

rig_pts = np.asarray(calib_config['aruco_corners_gt'])

with open(config_dir + "/results/aruco_center_3d.pkl", "rb") as f:
    label_dict = pkl.load(f)

maker_ids = [0, 1, 2, 3]
label_pts = []
for mk_idx in maker_ids:
    label_pts.append(label_dict[mk_idx])
label_pts = np.asarray(label_pts)


scale_reg, R_reg, t_reg, mean_dist_reg = point_set_registration(label_pts, rig_pts, verbose=True)
transformed_pts = apply_rigid_transform(label_pts, R_reg, t_reg, scale_reg)

ax = []
fig = plt.figure()

ax.append(fig.add_subplot(1, 2, 1, projection='3d'))
ax[0].scatter(label_pts[:, 0], label_pts[:, 1], label_pts[:, 2], c='b', label='original_points')
ax[0].scatter(rig_pts[:, 0], rig_pts[:, 1], rig_pts[:, 2], c='r', label='target_points')
ax[0].legend()

for i in range(label_pts.shape[0]):
    x = [label_pts[i, 0], rig_pts[i, 0]]
    y = [label_pts[i, 1], rig_pts[i, 1]]
    z = [label_pts[i, 2], rig_pts[i, 2]]
    ax[0].plot3D(x, y, z, 'gray')
ax[0].set_xlabel('X Label')
ax[0].set_ylabel('Y Label')
ax[0].set_zlabel('Z Label')
ax[0].set_title("initial points")

ax.append(fig.add_subplot(1, 2, 2, projection='3d'))
ax[1].scatter(transformed_pts[:, 0], transformed_pts[:, 1], transformed_pts[:, 2], c='b', label='transformed_points')
ax[1].scatter(rig_pts[:, 0], rig_pts[:, 1], rig_pts[:, 2], c='r', label='target_points')
ax[1].legend()

for i in range(label_pts.shape[0]):
    x = [transformed_pts[i, 0], rig_pts[i, 0]]
    y = [transformed_pts[i, 1], rig_pts[i, 1]]
    z = [transformed_pts[i, 2], rig_pts[i, 2]]
    ax[1].plot3D(x, y, z, 'gray')
ax[1].set_xlabel('X Label')
ax[1].set_ylabel('Y Label')
ax[1].set_zlabel('Z Label')
ax[1].set_title("transformed points after fitting")
plt.show()

camList = []
for i in range(n_cams):
    cam_params = {}
    filename = config_dir + "/results/calibration_aruco/{}.yaml".format(cam_names[i])
    print(filename)
    fs = cv2.FileStorage(filename, cv2.FILE_STORAGE_READ)
    cam_params['camera_matrix'] = fs.getNode("camera_matrix").mat()
    cam_params['distortion_coefficients'] = fs.getNode("distortion_coefficients").mat()
    cam_params['tc_ext'] = fs.getNode("tc_ext").mat()
    cam_params['rc_ext'] = fs.getNode("rc_ext").mat()
    camList.append(cam_params)

R_inv, t_inv = invert_Rt(R_reg, t_reg)
new_camList = []
for i, cam in enumerate(camList):
    R_old = np.array(cam['rc_ext'])
    t_old = np.reshape(cam['tc_ext'], (3,1))

    R2 = np.dot(R_old, R_inv)
    t2 = np.dot(R_old, t_inv.reshape(3,-1)) + t_old.reshape(3,-1)*scale_reg   

    global_pose = {'camera_matrix':cam['camera_matrix'], 
                   'distortion_coefficients':cam['distortion_coefficients'],
                   'rc_ext':R2, 
                   'tc_ext':t2.ravel()}
    new_camList.append(global_pose)


my_palette = sns.color_palette("rocket_r", n_cams)
xlim=[-1500, 1500]
ylim=[-1500, 1500]
zlim=[-100, 2000]
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')
for i in range(n_cams):
    r_c = new_camList[i]['rc_ext']
    t_c = new_camList[i]['tc_ext']
    # get inverse transformation
    ex = np.eye(4)
    ex[:3,:3] = r_c.T
    ex[:3,3] = -np.matmul(r_c.T, t_c)    
    visualizer = CameraVisualizer(fig, ax)
    visualizer.extrinsic2pyramid(ex, my_palette[i], 200)
ax.set_xlim(xlim)
ax.set_ylim(ylim)
ax.set_zlim(zlim)
plt.show()


aruco_folder = config_dir + "/results/calibration_rig/"
if not os.path.exists(aruco_folder):
   os.makedirs(aruco_folder)
save_aruco_format(aruco_folder, n_cams, new_camList, cam_names)
