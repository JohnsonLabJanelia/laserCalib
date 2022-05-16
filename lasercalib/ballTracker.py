import os
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from skimage import measure
import pprint
import pickle as pkl

print("hello ballTracker")

imgSize = np.array([2200, 3208, 3])

nCams = 4
background = np.empty(shape=(imgSize[0], imgSize[1], imgSize[2], nCams), dtype=int)
cams = ['cam0', 'cam1', 'cam3', 'cam5']
path = '/home/rob/Videos/tennisball/images/'

# get background images (no ball or robot visible over arena)
for i, cam in enumerate(cams):
    file = path + 'background_' + cam + '.jpg'
    background[:,:,:,i] = mpimg.imread(file)
    img = background[:,:,:,i]

# get image stack of rolling ball from 4 camera views
nImgs = 780
# imgs = np.empty(shape=(imgSize[0], imgSize[1], imgSize[2], nImgs, nCams), dtype=np.ubyte)


toshow = None

for i, cam in enumerate(cams):
    folder = path + cam
    files = [f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))]
    files = sorted(files)
    for j, file in enumerate(files):
        filepath = folder + '/' + file
        img = np.empty(shape=imgSize, dtype=int)
        # imgs[:,:,:,j,i] = mpimg.imread(filepath)
        img = mpimg.imread(filepath)
        img2 = img - background[:,:,:,i]
        img2 = np.absolute(img2)

        if (toshow is None):
            toshow = plt.imshow(img2)
        else:
            toshow = toshow.set_data(img2)

        plt.pause(0.01)
        plt.draw()
        




# nCams = 7
# camsFolder = '/home/rob/Videos/lasercalib/calib_images'
# camFolders = [d for d in os.listdir(camsFolder) if os.path.isdir(os.path.join(camsFolder, d))]
# camFolders = sorted(camFolders)
# print(camFolders)

# csvFolder = r"/home/rob/Videos/lasercalib/lasercalib/2022-05-13_14:49:08"
# csvFiles = []
# for i in range(nCams):
#     csvFiles.append(csvFolder + "/Cam" + str(i) + "_meta.csv")

# pp = pprint.PrettyPrinter(indent=0)
# np.set_printoptions(precision=5)
# np.set_printoptions(suppress=True)

# maxImgIdx = 0
# frameIDs = []
# for i in range(nCams):
#     my_data = np.genfromtxt(csvFiles[i], delimiter=',').astype(int)
#     frame_idx = my_data[1:,0].copy() - 1
#     frameIDs.append(frame_idx)
#     if frame_idx[-1] > maxImgIdx:
#         maxImgIdx = frame_idx[-1]

# print(maxImgIdx)

# nImages = maxImgIdx + 1

# centroids = np.zeros(shape=(nImages, 2, nCams))
# centroids[:] = np.nan

# print(centroids.shape)

# for c, folder in enumerate(camFolders):
#     folderPath = os.path.join(camsFolder, folder)
#     onlyfiles = [f for f in os.listdir(folderPath) if os.path.isfile(os.path.join(folderPath, f))]
#     sortedFiles = sorted(onlyfiles)
    
#     for i, file in enumerate(sortedFiles):
#         frameID = frameIDs[c][i]
#         imgFile = folderPath + '/' + file
#         img = mpimg.imread(imgFile)
#         green = np.array(img[:,:,1])
#         blobs = green > 100
#         labels, num = measure.label(blobs, background=0, return_num=True)
#         props = measure.regionprops(labels)

#         if num == 0:
#             print("image: ", i, "frameID: ", frameID, "file: ", file, "centroid: ", centroids[frameID, :, c])

#         elif num > 1:
#             print("image: ", i, "frameID: ", frameID, "file: ", file, "centroid: ", centroids[frameID, :, c])

#         elif num == 1:
#             centroids[frameID, :, c] = props[0].centroid
#             print("image: ", i, "frameID: ", frameID, "file: ", file, "centroid: ", centroids[frameID, :, c])
        

# outfile = 'centroids.pkl'
# fileObject = open(outfile, 'wb')
# pkl.dump(centroids, fileObject)
# fileObject.close()