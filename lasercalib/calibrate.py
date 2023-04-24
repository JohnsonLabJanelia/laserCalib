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

parser = argparse.ArgumentParser()
parser.add_argument('--root_dir', type=str, required=True)
parser.add_argument('--cam_id_for_3d_init', type=int, required=True)
parser.add_argument('--shift_3d', type=float, default=1.0)

args = parser.parse_args()

## inputs to the file
root_dir = args.root_dir
cam_idx_3dpts = args.cam_id_for_3d_init
shift_3d = args.shift_3d
 
with open(root_dir + "/results/centroids.pkl", 'rb') as file:
    pts = pkl.load(file)

# flip xy (regionprops orders)
pts = np.flip(pts, axis=1)
nPts = pts.shape[0]
nCams = pts.shape[2]
my_palette = sns.color_palette("rocket_r", nCams)

## blender initialization
cameraArray = load_from_blender(root_dir + "/results/camera_dicts.pkl", nCams)

keep = np.zeros(shape=(nPts,), dtype=bool)
for i in range(nPts):
    v = pts[i, 0, :]
    if ((np.sum(~np.isnan(v)) >= nCams) and (~np.isnan(v[0]))):
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

# prepare points_3d variable (initializing with 2d laser points in image space on cam4)
points_3d = np.zeros(shape=(nPts, 3))
for i in range(nPts):
    if (np.isnan(inPts[i, 0, cam_idx_3dpts])):
        points_3d[i, 0:2] = [1604, 1100]
        continue
    else:
        points_3d[i, 0:2] = inPts[i, :, cam_idx_3dpts]

# # # center the world points
points_3d[:,0] = shift_3d * (points_3d[:,0] - 1604)
points_3d[:,1] = shift_3d * (points_3d[:,1] - 1100)

sba = pySBA.PySBA(cameraArray, points_3d, points_2d, camera_ind, point_ind)
sba_print(sba, nCams, "Initialization", color_palette=my_palette)

sba.bundleAdjust_nocam()
sba_print(sba, nCams, "Fit 3d Points only", color_palette=my_palette)

sba.bundleAdjust()
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
for i in range(args.n_cams):
    camList.append(sba_to_readable_format(sba.cameraArray[i,:]))

# save for red
outParams = readable_to_red_format(camList)
np.savetxt(args.root_dir + '/results/calibration_red.csv', outParams, delimiter=',', newline=',\n', fmt='%f')

# save for aruco detection
save_root = args.root_dir + "/results/calibration_aruco/"
if not os.path.exists(save_root):
   os.makedirs(save_root)

red_to_aruco(save_root, args.n_cams, outParams)