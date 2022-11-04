from __future__ import print_function
import numpy as np
import cv2 as cv
import sys
import csv

input_filename = "../calibres/calibration_20220930_rigspace.csv"
cam = []
with open(input_filename) as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0
    for row in csv_reader:
        param = [float(x) for x in row[:-1]]
        cam.append(param)
        


save_root = "../calibres/20221005/"  
for cam_idx in [0, 1, 2, 3]:
    intrinsicMatrix = np.asarray(cam[cam_idx][0:9]).reshape(3, 3)
    intrinsicMatrix = intrinsicMatrix

    distortionCoefficients = cam[cam_idx][21:25]
    # distortionCoefficients.append(0.0)
    distortionCoefficients = np.asarray(distortionCoefficients).reshape(4, 1)

    # save it using opencv 
    output_filename = save_root + 'Cam{}.yaml'.format(cam_idx)
    s = cv.FileStorage(output_filename, cv.FileStorage_WRITE)
    s.write('image_width', 3208)
    s.write('image_height', 2200)

    s.write('camera_matrix', intrinsicMatrix)
    s.write('distortion_coefficients', distortionCoefficients)
    s.release()