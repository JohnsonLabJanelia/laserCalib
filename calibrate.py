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

# load laser pointer point centroids found on all cameras
with np.load('good_centroids.npz') as data:
    pts = data['arr_0']

nPts = pts.shape[0]
nCams = pts.shape[2]

fig, axs = plt.subplots(1, nCams, sharey=True)

for i in range(nCams):
    colors = np.linspace(0, 1, nPts)
    axs[i].scatter(pts[:,0,i], pts[:,1,i], s=10, c=colors, alpha=0.5)
    axs[i].title.set_text('cam' + str(i))

plt.show()

# load camera data for pySBA
cameraArray = np.zeros(shape=(nCams, 11))

# cam0
cameraArray[0, 0:3] = [-2.2151928, -2.2044225, 0.0019530592]
cameraArray[0, 3:6] = [-15.611495, -4.2011781, 2401.2117]
cameraArray[0, 6:9] = [1777.777, -0.015, -0.015]
cameraArray[0, 9:] = [1608, 1100]

# cam1
cameraArray[1, 0:3] = [1.9783967, -2.0124056, 0.44338688]
cameraArray[1, 3:6] = [-154.71857, -351.46274, 2299.1753]
cameraArray[1, 6:9] = [1777.777, -0.015, -0.015]
cameraArray[1, 9:] = [1608, 1100]

#cam2
cameraArray[2, 0:3] = [1.9780892, -2.0034461, 0.4536269]
cameraArray[2, 3:6] = [107.51159, -330.8779, 2300.0032]
cameraArray[2, 6:9] = [1777.777, -0.015, -0.015]
cameraArray[2, 9:] = [1608, 1100]

#cam3
cameraArray[3, 0:3] = [0.81482941, 2.8825006, -0.59704882]
cameraArray[3, 3:6] = [-113.24658, -372.63889, 2271.7502]
cameraArray[3, 6:9] = [1777.777, -0.015, -0.015]
cameraArray[3, 9:] = [1608, 1100]

#cam4
cameraArray[4, 0:3] = [0.82509673, 2.8785553, -0.59267741]
cameraArray[4, 3:6] = [134.61037, -379.2637, 2273.8696]
cameraArray[4, 6:9] = [1777.777, -0.015, -0.015]
cameraArray[4, 9:] = [1608, 1100]

#cam5
cameraArray[5, 0:3] = [2.612438, 0.79784292, -0.17937534]
cameraArray[5, 3:6] = [-74.572533, -305.59396, 2283.8472]
cameraArray[5, 6:9] = [1777.777, -0.015, -0.015]
cameraArray[5, 9:] = [1608, 1100]

#cam6
cameraArray[6, 0:3] = [2.6102991, 0.79794997, -0.17989315]
cameraArray[6, 3:6] = [190.82062, -295.33585, 2283.1584]
cameraArray[6, 6:9] = [1777.777, -0.015, -0.015]
cameraArray[6, 9:] = [1608, 1100]

print("cameraArray: ", cameraArray)
