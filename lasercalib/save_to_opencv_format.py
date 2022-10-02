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
        


save_root = "../calibres/20220930/"  
for cam_idx in [0, 1, 2, 3]:
    intrinsicMatrix = np.asarray(cam[cam_idx][0:9]).reshape(3, 3)
    intrinsicMatrix = intrinsicMatrix.T
    RotationMatrix = np.asarray(cam[cam_idx][9:18]).reshape(3, 3)
    RotationMatrix = RotationMatrix.T # need this transpose 
    Translation = np.asarray(cam[cam_idx][18:21]).reshape(3, 1)
    distortionCoefficients = cam[cam_idx][21:25]
    distortionCoefficients.append(0.0)
    distortionCoefficients = np.asarray(distortionCoefficients).reshape(1, 5)
    
    # save it using opencv 
    output_filename = save_root + 'Cam{}.yaml'.format(cam_idx)
    s = cv.FileStorage(output_filename, cv.FileStorage_WRITE)
    s.write('intrinsicMatrix', intrinsicMatrix)
    s.write('distortionCoefficients', distortionCoefficients)
    s.write('R', RotationMatrix)
    s.write('T', Translation)
    s.release()