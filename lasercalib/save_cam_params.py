import numpy as np
import pickle
from convert_params import *
import cv2 
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--root_dir', type=str, required=True)
parser.add_argument('--n_cams', type=int, required=True)
args = parser.parse_args()


with open(args.root_dir + "/results/sba_blender.pkl", 'rb') as f:
    sba = pickle.load(f)

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