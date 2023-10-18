import os 
import scipy.io
import numpy as np
import cv2
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--root_dir', type=str, required=True)
args = parser.parse_args()

cameras = os.listdir(args.root_dir + "/matlab_calibration/")
cameras.sort()

print(cameras)

for cam_name in cameras:
    mat_name = args.root_dir + "/matlab_calibration/" + cam_name + "/ext.mat"
    mat = scipy.io.loadmat(mat_name)

    camera_matrix = np.zeros((3, 3))
    camera_matrix[0, 0] = mat['fc'][0]
    camera_matrix[1, 1] = mat['fc'][1]
    camera_matrix[0, 2] = mat['cc'][0]
    camera_matrix[1, 2] = mat['cc'][1]
    camera_matrix[2, 2] = 1.0

    distortion_coefficients = np.zeros((5, 1))
    distortion_coefficients = mat['kc']

    tc_ext = np.zeros((3,1))
    tc_ext = mat['Tc_ext']

    rc_ext = np.zeros((3, 3))
    rc_ext = mat['Rc_ext']
    
    
    output_filename = args.root_dir + "/results/" + cam_name + ".yaml"
    s = cv2.FileStorage(output_filename, cv2.FileStorage_WRITE)
    s.write('image_width', 3208)
    s.write('image_height', 2200)

    s.write('camera_matrix', camera_matrix)
    s.write('distortion_coefficients', distortion_coefficients)
    s.write('tc_ext', tc_ext)
    s.write('rc_ext', rc_ext)
    s.release()