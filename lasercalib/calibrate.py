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

parser = argparse.ArgumentParser()
parser.add_argument('--root_dir', type=str, required=True)

args = parser.parse_args()
root_dir = args.root_dir

## load all the pts, and get nCams, points_3d, points_2d
with open(root_dir + "/results/points_dataset.pkl", 'rb') as file:
    points_dataset = pkl.load(file)

with open(root_dir + "/results/centroids.pkl", 'rb') as file:
    centroids_dict = pkl.load(file)
cam_names = centroids_dict['cam_names']


nCams = points_dataset[0]['nCams']

points_3d = np.vstack([points_dataset[i]['points_3d'] for i in range(len(points_dataset))])
points_2d = np.vstack([points_dataset[i]['points_2d'] for i in range(len(points_dataset))])
camera_ind = np.hstack([points_dataset[i]['camera_ind'] for i in range(len(points_dataset))])

points_ind_offset = [0]
for i in range(len(points_dataset)-1):
    points_ind_offset.append(points_dataset[i]['nPts'])
point_ind = np.hstack([points_dataset[i]['point_ind'] + points_ind_offset[i] for i in range(len(points_dataset))])

# plot 3d points
total_num_pts = points_3d.shape[0]
print("Total num of pts: ", total_num_pts)

ax = []
fig = plt.figure()
ax.append(fig.add_subplot(1, 1, 1, projection='3d'))
ax[0].scatter(points_3d[:, 0], points_3d[:, 1], points_3d[:, 2], color='m', alpha=0.1)
ax[0].set_title("laser points")
plt.show()

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