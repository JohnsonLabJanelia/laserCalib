
import numpy as np
import pickle as pkl
import cv2 
import argparse
import pprint
import matplotlib.pyplot as plt
import json
from lasercalib.camera_visualizer import CameraVisualizer
import seaborn as sns
import os

parser = argparse.ArgumentParser()
parser.add_argument('-c', '--config', type=str, required=True)

args = parser.parse_args()

# load config file
config_dir = args.config
with open(config_dir + '/config.json', 'r') as f:
    calib_config = json.load(f)

# camera serial numbers
cam_serials = calib_config['cam_serials']
cam_names = []
for cam_serial in cam_serials:
    cam_names.append("Cam" + cam_serial)
n_cams = len(cam_names)

camList = []

saveCalibFolder = config_dir + "results/calibration_jarvis."
# os.makedirs( config_dir + "/results/calibration_jarvis/")
    
for i in range(n_cams):
    cam_params = {}
    filename = config_dir + "/results/calibration_rig/{}.yaml".format(cam_names[i])
    # print(filename)
    fs = cv2.FileStorage(filename, cv2.FILE_STORAGE_READ)
    
    intrinsicMatrix = fs.getNode("camera_matrix").mat()
    intrinsicMatrix = intrinsicMatrix.T
    distortionCoefficients = fs.getNode("distortion_coefficients").mat()
    distortionCoefficients = distortionCoefficients.T
    R = fs.getNode("rc_ext").mat()
    R = R.T
    T = fs.getNode("tc_ext").mat()
   
    output_filename =  config_dir + "/results/calibration_jarvis/{}.yaml".format(cam_names[i])
    s = cv2.FileStorage(output_filename, cv2.FileStorage_WRITE)
    s.write('intrinsicMatrix', intrinsicMatrix)
    s.write('distortionCoefficients', distortionCoefficients)
    s.write('R', R)
    s.write('T', T)
    s.release()
    print(output_filename)

print("converted all calibration files to jarvis format")

