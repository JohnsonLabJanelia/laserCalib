import os 
import numpy as np
import matplotlib.image as mpimg
from skimage import measure


camsFolder = '/home/jinyao/Videos/calib_images'
folders = [d for d in os.listdir(camsFolder) if os.path.isdir(os.path.join(camsFolder, d))]
folders = sorted(folders)
print(folders)


for folder in folders:
    folderPath = os.path.join(camsFolder, folder)
    onlyfiles = [f for f in os.listdir(folderPath) if os.path.isfile(os.path.join(folderPath, f))]
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
                if dist < 1500:   # this was used to remove unwanted contours near the edge of the image (unsuitable images had a green power button in the shot)
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

    outfile = '../results/' + folder + '_res'
    np.savez(outfile, areas, centroids, blobcount)

