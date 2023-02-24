import numpy as np
import pickle
from convert_params import *
import cv2 
import os

root_dir = "/home/jinyao/Calibration/newrig8"
nCams = 8

with open(root_dir + "/results/sba_blender.pkl", 'rb') as f:
    sba = pickle.load(f)

camList = []
for i in range(nCams):
    camList.append(sba_to_readable_format(sba.cameraArray[i,:]))

# save for red
outParams = readable_to_red_format(camList)
np.savetxt(root_dir + '/results/calibration_red.csv', outParams, delimiter=',', newline=',\n', fmt='%f')

# save for aruco detection
save_root = root_dir + "/results/calibration_aruco/"
if not os.path.exists(save_root):
   os.makedirs(save_root)

save_for_aruco(save_root, 8, outParams)