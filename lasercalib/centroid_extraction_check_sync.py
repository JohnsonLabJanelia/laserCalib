import os
from re import L 
import numpy as np
import matplotlib.image as mpimg
from skimage import measure
import pprint
import pickle as pkl

nCams = 7
camsFolder = '/home/rob/Videos/lasercalib/calib_images'
camFolders = [d for d in os.listdir(camsFolder) if os.path.isdir(os.path.join(camsFolder, d))]
camFolders = sorted(camFolders)
print(camFolders)

csvFolder = r"/home/rob/Videos/lasercalib/lasercalib/2022-05-13_14:49:08"
csvFiles = []
for i in range(nCams):
    csvFiles.append(csvFolder + "/Cam" + str(i) + "_meta.csv")

pp = pprint.PrettyPrinter(indent=0)
np.set_printoptions(precision=5)
np.set_printoptions(suppress=True)

maxImgIdx = 0
frameIDs = []
for i in range(nCams):
    my_data = np.genfromtxt(csvFiles[i], delimiter=',').astype(int)
    frame_idx = my_data[1:,0].copy() - 1
    frameIDs.append(frame_idx)
    if frame_idx[-1] > maxImgIdx:
        maxImgIdx = frame_idx[-1]

print(maxImgIdx)

nImages = maxImgIdx + 1

centroids = np.zeros(shape=(nImages, 2, nCams))
centroids[:] = np.nan

print(centroids.shape)

for c, folder in enumerate(camFolders):
    folderPath = os.path.join(camsFolder, folder)
    onlyfiles = [f for f in os.listdir(folderPath) if os.path.isfile(os.path.join(folderPath, f))]
    sortedFiles = sorted(onlyfiles)
    
    for i, file in enumerate(sortedFiles):
        frameID = frameIDs[c][i]
        imgFile = folderPath + '/' + file
        img = mpimg.imread(imgFile)
        green = np.array(img[:,:,1])
        blobs = green > 100
        labels, num = measure.label(blobs, background=0, return_num=True)
        props = measure.regionprops(labels)

        if num == 0:
            print("image: ", i, "frameID: ", frameID, "file: ", file, "centroid: ", centroids[frameID, :, c])

        elif num > 1:
            print("image: ", i, "frameID: ", frameID, "file: ", file, "centroid: ", centroids[frameID, :, c])

        elif num == 1:
            centroids[frameID, :, c] = props[0].centroid
            print("image: ", i, "frameID: ", frameID, "file: ", file, "centroid: ", centroids[frameID, :, c])
        

outfile = 'centroids.pkl'
fileObject = open(outfile, 'wb')
pkl.dump(centroids, fileObject)
fileObject.close()
