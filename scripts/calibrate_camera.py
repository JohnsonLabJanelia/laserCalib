from tkinter.messagebox import NO
import numpy as np
import matplotlib.pyplot as plt
import pickle as pkl
from datetime import date
import seaborn as sns
from lasercalib.pySBA import PySBA
from lasercalib.convert_params import load_from_blender
from lasercalib.convert_params import *
import argparse
from lasercalib.sba_print import sba_print
import os
from lasercalib.camera_visualizer import CameraVisualizer
import json
import glob

parser = argparse.ArgumentParser()
parser.add_argument('-c', '--config', type=str, required=True)

args = parser.parse_args()
config_dir = args.config
with open(config_dir + '/config.json', 'r') as f:
    calib_config = json.load(f)
calib_init = calib_config['calib_init']
laser_datasets = calib_config['lasers']
cam_serials = calib_config['cam_serials']
cam_names = []
for cam_serial in cam_serials:
    cam_names.append("Cam" + cam_serial)

## load all the pts, and get nCams, points_3d, points_2d
with open(config_dir + "/results/points_dataset.pkl", 'rb') as file:
    points_dataset = pkl.load(file)

n_cams = points_dataset[0]['n_cams']

points_3d = np.vstack([points_dataset[i]['points_3d'] for i in range(len(points_dataset))])
points_2d = np.vstack([points_dataset[i]['points_2d'] for i in range(len(points_dataset))])
camera_ind = np.hstack([points_dataset[i]['camera_ind'] for i in range(len(points_dataset))])

points_ind_offset = [0]
for i in range(len(points_dataset)-1):
    points_ind_offset.append(points_dataset[i]['n_pts'])
point_ind = np.hstack([points_dataset[i]['point_ind'] + points_ind_offset[i] for i in range(len(points_dataset))])

# plot 3d points
total_num_pts = points_3d.shape[0]
print("Total num of pts: ", total_num_pts)


my_palette = sns.color_palette("rocket_r", n_cams)

## blender initialization
# cameraArray = load_from_blender(root_dir + "/results/camera_dicts.pkl", nCams)
calibration_path = os.path.join(config_dir, calib_init)
cameraArray = initialize_from_checkerboard(calibration_path, n_cams, cam_names)
camList = []
for i in range(n_cams):
    camList.append(sba_to_readable_format(cameraArray[i,:]))


sba = PySBA(cameraArray, points_3d, points_2d, camera_ind, point_ind)
sba_print(sba, n_cams, "Initialization", color_palette=my_palette)

sba.bundleAdjust_nocam(1e-6)
sba_print(sba, n_cams, "Fit 3d Points only", color_palette=my_palette)

sba.bundle_adjustment_camonly(1e-4)
sba_print(sba, n_cams, "Fit camera parameters only", color_palette=my_palette)

sba.bundleAdjust(1e-4)
sba_print(sba, n_cams, "Fit 3d Points and Camera Paramters", color_palette=my_palette)

## saving 
camList = []
for i in range(n_cams):
    camList.append(sba_to_readable_format(sba.cameraArray[i,:]))
with open(config_dir + "/results/calibration.pkl", 'wb') as f:
    pkl.dump(camList, f)

# save for red
outParams = readable_to_red_format(camList)
np.savetxt(config_dir + '/results/calibration_red.csv', outParams, delimiter=',', newline=',\n', fmt='%f')


output_file = config_dir + "/results/sba.pkl"
with open(output_file, 'wb') as f:
    pkl.dump(sba, f)

print("Done fitting, saved to: {}".format(config_dir + "/results"))

# saving 
camList = []
for i in range(n_cams):
    camList.append(sba_to_readable_format(sba.cameraArray[i,:]))

# save for red
outParams = readable_to_red_format(camList)
np.savetxt(config_dir + '/results/calibration_red.csv', outParams, delimiter=',', newline=',\n', fmt='%f')

# save for aruco detection
save_root = config_dir + "/results/calibration_aruco/"
if not os.path.exists(save_root):
   os.makedirs(save_root)

readable_format_to_aruco_format(save_root, n_cams, camList, cam_names)