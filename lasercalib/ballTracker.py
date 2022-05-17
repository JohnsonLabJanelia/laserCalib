import os
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from skimage import measure
from skimage.morphology import (erosion, dilation, opening, closing,  # noqa
                                white_tophat)
from skimage.morphology import black_tophat, skeletonize, convex_hull_image  # noqa
from skimage.morphology import disk  # noqa
from skimage.color import label2rgb

import pprint
import pickle as pkl

imgShape = np.array([2200, 3208, 3])
nCams = 4
nImgsPerFolder = 780
imgIdx = np.arange(400, 600, 1)
nImgs = imgIdx.size
centroids = np.empty(shape=(nImgs, 2, nCams))
centroids[:] = np.nan
background = np.empty(shape=(imgShape[0], imgShape[1], imgShape[2], nCams), dtype=np.int16)
cams = ['cam0', 'cam1', 'cam3', 'cam5']
path = '/home/rob/Videos/tennisball/images/'


# get background images (no ball or robot visible over arena)
for i, cam in enumerate(cams):
    file = path + 'background_' + cam + '.jpg'
    img = mpimg.imread(file)
    background[:,:,:,i] = img.astype(np.int16)

for c, cam in enumerate(cams):
    if c == 0:
        continue
    folder = path + cam
    files = [f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))]
    files = sorted(files)
    for j, fileIdx in enumerate(imgIdx):
        filepath = folder + '/' + files[fileIdx]
        img = mpimg.imread(filepath).astype(np.int16)

        diffA = (img - background[:,:,:,c]).sum(axis=2)
        boolA = diffA > 60
        diffB = (background[:,:,:,c] - img).sum(axis=2)
        boolB = diffB > 60

        mask0 = boolA | boolB
        mask1 = (img[:,:,0] > 100) & (img[:,:,0] < 235)
        mask2 = (img[:,:,1] > 70) & (img[:,:,1] < 235)
        mask3 = (img[:,:,2] > 40) & (img[:,:,2] < 160)
        maskColor = mask1 & mask2 & mask3

        #local search
        mask4 = np.ones(shape=(imgShape[0], imgShape[1]), dtype=bool)
        if (j > 0 & ~np.isnan(centroids[j,0,c])):
            mask4[:] = False
            row = centroids[j-1, 0, c].astype(np.int16)
            col = centroids[j-1, 1, c].astype(np.int16)
            half = 40
            mask4[row-half:row+half, col-half:col+half] = True
        
        totalMask = mask0 & mask1 & mask2 & mask3 & mask4

        footprint = disk(6)
        opened = opening(totalMask, footprint) 
        labels, num = measure.label(opened, background=0, return_num=True)

        if num == 0:
            print("no region found")
        elif num > 1:
            print("more than one region found")
            props = measure.regionprops(labels)
            areas = np.zeros(shape=(num,))
            for n in range(num):
                areas[n] = props[n].area
            idx = np.argmax(areas)
            centroids[j,:,c] = props[idx].centroid
            print("image: ", fileIdx, "cam: ", cam, "file: ", files[fileIdx], "centroid: ", centroids[j, :, c], " - largest region selected")

        elif num == 1:
            props = measure.regionprops(labels)
            centroids[j,:,c] = props[0].centroid
            print("image: ", fileIdx, "cam: ", cam, "file: ", files[fileIdx], "centroid: ", centroids[j, :, c])


        fig, axs = plt.subplots(3, 2, figsize=(16, 24))
        print("loaded " + filepath)
        axs[0,0].imshow(img.astype(np.uint8))
        axs[1,0].imshow(boolA.astype(np.uint8) * 255)
        axs[2,0].imshow(mask0.astype(np.uint8) * 255)
        axs[0,1].imshow(maskColor.astype(np.uint8) * 255)
        img2 = img.copy().astype(np.uint8)
        img2[:,:,0] = img2[:,:,0] * totalMask
        img2[:,:,1] = img2[:,:,1] * totalMask
        img2[:,:,2] = img2[:,:,2] * totalMask
        axs[1,1].imshow(img2)
        color1 = label2rgb(opened)
        axs[2,1].imshow(color1)
        plt.show()

outfile = 'ball_centroids.pkl'
fileObject = open(outfile, 'wb')
pkl.dump(centroids, fileObject)
fileObject.close()
