import numpy as np
from scipy.sparse import lil_matrix
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation as R

import pySBA
from scipy.io import loadmat
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from skimage import measure
from skimage import filters

from os import listdir, walk
from os.path import isfile, isdir, join

from scipy.spatial.transform import Rotation as R

from util.my_cam_pose_visualizer import MyCamPoseVisualizer

import pickle
import pprint

picklefile = open('sba_data', 'rb')
sba = pickle.load(picklefile)
picklefile.close()

from prettytable import PrettyTable
x = PrettyTable()
for row in sba.cameraArray:
    x.add_row(row)
print(x)

camList = []

for i in range(sba.cameraArray.shape[0]):
    camList.append(pySBA.unconvertParams(sba.cameraArray[i,:]))

pp = pprint.PrettyPrinter(indent=0)

np.set_printoptions(precision=5)
np.set_printoptions(suppress=True)

for i, cam in enumerate(camList):
    print('cam' + str(i))
    pp.pprint(cam)
    pp.format
    print('\n')