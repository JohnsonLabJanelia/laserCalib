from tkinter.messagebox import NO
import numpy as np
import matplotlib.pyplot as plt
import pickle as pkl
from datetime import date
import seaborn as sns
import pySBA
from convert_params import load_from_blender
from convert_params import *
import argparse
from sba_print import sba_print
import os
from camera_visualizer import CameraVisualizer
import cv2 as cv

parser = argparse.ArgumentParser()
parser.add_argument('--root_dir', type=str, required=True)
parser.add_argument('--cam_name_for_3d_init', type=str, required=True)
parser.add_argument('--min_num_cam_per_point', type=int, default=4)
parser.add_argument('--shift_3d', type=float, default=1.0)

args = parser.parse_args()

## inputs to the file
root_dir = args.root_dir
cam_name_for_3d_init = args.cam_name_for_3d_init
shift_3d = args.shift_3d
min_num_cam_per_point = args.min_num_cam_per_point
 
with open(root_dir + "/results/centroids.pkl", 'rb') as file:
    centroids_dict = pkl.load(file)

pts = centroids_dict['centroids']
cam_names = centroids_dict['cam_names']

# flip xy (regionprops orders)
pts = np.flip(pts, axis=1)
nPts = pts.shape[0]
nCams = pts.shape[2]
my_palette = sns.color_palette("rocket_r", nCams)

## blender initialization
# cameraArray = load_from_blender(root_dir + "/results/camera_dicts.pkl", nCams)
cameraArray = initialize_from_checkerboard(root_dir + "/calib_init/", nCams, cam_names)
camList = []
for i in range(nCams):
    camList.append(sba_to_readable_format(cameraArray[i,:]))

fig =  plt.figure()
ax = fig.add_subplot(1, 1, 1, projection='3d')

for i, cam in enumerate(camList):
    ex = np.eye(4)
    
    r_f = cam["R"]
    r_inv = r_f.T

    t_f = cam["t"]
    t_inv = -np.dot(r_f, t_f)

    ex[:3,:3] = cam["R"]  
    ex[:3, 3] = t_inv

    visualizer = CameraVisualizer(fig, ax)
    visualizer.extrinsic2pyramid(ex, my_palette[i], 200)

ax.set_xlim([-1500, 1500])
ax.set_ylim([-1500, 1500])
ax.set_zlim([-100, 1800])
ax.set_title("rig space")


cam_idx_3dpts = cam_names.index(cam_name_for_3d_init)
## use 3d triangulated points to initialize
keep = np.zeros(shape=(nPts,), dtype=bool)
for i in range(nPts):
    v = pts[i, 0, :]
    if ((np.sum(~np.isnan(v)) >= min_num_cam_per_point) and (~np.isnan(v[cam_idx_3dpts]))):
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

# # better initialization 
# points_3d = np.zeros(shape=(nPts, 3))
# # prepare points_3d variable (initializing with 2d laser points in image space on cam4)
# points_3d = np.zeros(shape=(nPts, 3))
# for i in range(nPts):
#     if (np.isnan(inPts[i, 0, cam_idx_3dpts])):
#         points_3d[i, 0:2] = [1604, 1100]
#         continue
#     else:
#         points_3d[i, 0:2] = inPts[i, :, cam_idx_3dpts]

# # # # center the world points
# points_3d[:,0] = shift_3d * (points_3d[:,0] - 1604)
# points_3d[:,1] = shift_3d * (points_3d[:,1] - 1100)


def load_camera_parameters_from_yaml(file_name):
    fs = cv2.FileStorage(file_name, cv2.FILE_STORAGE_READ)
    one_calib = {
        "camera_matrix": fs.getNode("camera_matrix").mat(),
        "distortion_coefficients": fs.getNode("distortion_coefficients").mat(),
        "rc_ext": fs.getNode("rc_ext").mat(),
        "tc_ext": fs.getNode("tc_ext").mat()
    }
    return one_calib

# use a camera in the ceiling (doesn't move much to initialize)
def Project(points, intrinsic, distortion, rotation_matrix, tvec):
    result = []
    if len(points) > 0:
        result, _ = cv.projectPoints(points, rotation_matrix, tvec,
                                  intrinsic, distortion)
    return np.squeeze(result, axis=1)

def Unproject(points, Z, intrinsic, distortion, rotation_matrix, tvec):
    """
    args: 
        points, [num_pts, coordinates (2d)] 
        Z, one number or [num_pts], select a plane or planes
        intrinsic: camera intrinsics
        distortion: camera distortion
        rotation_matrix: camera rotation matrix 
        tvec: camera translation vector
    return:
        world points, [num_pts, coordinates (3d)]
    """
    f_x = intrinsic[0, 0]
    f_y = intrinsic[1, 1]
    c_x = intrinsic[0, 2]
    c_y = intrinsic[1, 2]
   
    points_undistorted = np.array([])
    if len(points) > 0:
        points_undistorted = cv.undistortPoints(np.expand_dims(points, axis=1), intrinsic, distortion, P=intrinsic)
    points_undistorted = np.squeeze(points_undistorted, axis=1)
    
    temp_pts = []
    for idx in range(points_undistorted.shape[0]):
        u = (points_undistorted[idx, 0] - c_x) / f_x
        v = (points_undistorted[idx, 1] - c_y) / f_y
        temp_pts.append([u, v , 1])
    temp_pts = np.asarray(temp_pts)
    temp_pts = temp_pts.T
    left_eqn = np.matmul(rotation_matrix.T, temp_pts)
    right_eqn0 = np.matmul(rotation_matrix.T, tvec)

    z_camera = []
    for idx in range(points_undistorted.shape[0]):
        z_world = Z[0] if len(Z) == 1 else Z[idx]
        z_camera.append((z_world + right_eqn0[2, 0]) / left_eqn[2, idx])
    z_camera = np.asarray([z_camera])
    pts_world = np.matmul(rotation_matrix.T, temp_pts * z_camera - tvec)
    return pts_world.T


init_camera_filename = root_dir + "/calib_init/{}.yaml".format(cam_name_for_3d_init)
init_camera_parameter = load_camera_parameters_from_yaml(init_camera_filename)

projected_points = inPts[:, :, cam_idx_3dpts]
projected_points_z = np.zeros(projected_points.shape[0])
unprojected_points = Unproject(projected_points, projected_points_z, 
                            init_camera_parameter['camera_matrix'], 
                            init_camera_parameter['distortion_coefficients'], 
                            init_camera_parameter['rc_ext'], 
                            init_camera_parameter['tc_ext'])

points_3d = unprojected_points
np.set_printoptions(suppress=True)
print("Unprojected points: ", points_3d)


sba = pySBA.PySBA(cameraArray, points_3d, points_2d, camera_ind, point_ind)
sba_print(sba, nCams, "Initialization", color_palette=my_palette)

# sba.bundleAdjust_nocam(1e-6)
# sba_print(sba, nCams, "Fit 3d Points only", color_palette=my_palette)


# sba.bundle_adjustment_camonly(1e-4)
# sba_print(sba, nCams, "Fit camera parameters only", color_palette=my_palette)

sba.bundleAdjust(1e-4)
sba_print(sba, nCams, "Fit 3d Points and Camera Paramters", color_palette=my_palette)

## saving 
camList = []
for i in range(nCams):
    camList.append(sba_to_readable_format(sba.cameraArray[i,:]))
with open(root_dir + "/results/calibration.pkl", 'wb') as f:
    pkl.dump(camList, f)

# save for red
outParams = readable_to_red_format(camList)
np.savetxt(root_dir + '/results/calibration_red.csv', outParams, delimiter=',', newline=',\n', fmt='%f')


output_file = root_dir + "/results/sba.pkl"
with open(output_file, 'wb') as f:
    pkl.dump(sba, f)

print("Done fitting, saved to: {}".format(root_dir + "/results"))

# saving 
camList = []
for i in range(nCams):
    camList.append(sba_to_readable_format(sba.cameraArray[i,:]))

# save for red
outParams = readable_to_red_format(camList)
np.savetxt(args.root_dir + '/results/calibration_red.csv', outParams, delimiter=',', newline=',\n', fmt='%f')

# save for aruco detection
save_root = args.root_dir + "/results/calibration_aruco/"
if not os.path.exists(save_root):
   os.makedirs(save_root)

readable_format_to_aruco_format(save_root, nCams, camList, cam_names)