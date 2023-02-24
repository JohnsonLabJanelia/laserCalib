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


aruco_cams = []
for i in range(nCams):
    with open(root_dir + "/aruco_detection/calibration_aruco/Cam{}_aruco.pkl".format(i), 'rb') as f:
        one_camera = pickle.load(f)
        aruco_cams.append(one_camera)



