import numpy as np
from scipy.sparse import lil_matrix
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation as R

import pySBA
from scipy.io import loadmat
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2 as cv

from skimage import measure
from skimage import filters

from os import listdir, walk
from os.path import isfile, isdir, join

camsFolder = '/home/rob/Videos/calib_images'
folders = [d for d in listdir(camsFolder) if isdir(join(camsFolder, d))]
folders = sorted(folders)
print(folders)


for folder in folders:
    folderPath = '/home/rob/Videos/calib_images/' + folder
    onlyfiles = [f for f in listdir(folderPath) if isfile(join(folderPath, f))]
    sortedFiles = sorted(onlyfiles)
    n = len(sortedFiles)
    areas = np.empty(shape=(n,))
    centroids = np.empty(shape=(n,2))
    blobcount = np.empty(shape=(n,))

    for i in range(n):
        imgFile = folderPath + '/' + sortedFiles[i]
        img = mpimg.imread(imgFile)
        green = np.array(img[:,:,1])
        blobs = green > 0.85
        labels, num = measure.label(blobs, background=0, return_num=True)
        props = measure.regionprops(labels)
        blobcount[i] = num
        if num == 0:
            print(sortedFiles[i], "  - no region found")
            areas[i] = np.nan
            centroids[i,:] = [np.nan, np.nan]

        elif num > 1:
            print(sortedFiles[i], "  - more than one region found")
            # check distance
            ccKeep = []
            for cc in props:
                dist = (((cc.centroid[0] - 1100) ** 2) + ((cc.centroid[1] - 1608) ** 2)) ** 0.5
                if dist < 1500:
                    ccKeep.append(cc)

            imgAreas = np.empty(shape=(len(ccKeep),))
            for ccIndex in range(len(ccKeep)):
                imgAreas[ccIndex] = ccKeep[ccIndex].area

            print("imgAreas: ", imgAreas)
            if (len(imgAreas) > 0):
                index = np.argmax(imgAreas)
                areas[i] = ccKeep[index].area
                centroids[i,:] = ccKeep[index].centroid
            else:
                print(sortedFiles[i], "  - after distance control, no region found")
                areas[i] = np.nan
                centroids[i,:] = [np.nan, np.nan]

        elif num == 1:
            print(sortedFiles[i], "  - centroid: ", props[0].centroid)
            areas[i] = props[0].area
            centroids[i,:] = props[0].centroid

    outfile = 'results/' + folder + '_res'
    np.savez(outfile, areas, centroids, blobcount)

