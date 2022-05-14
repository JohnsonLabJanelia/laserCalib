import os 
import numpy as np
import matplotlib.image as mpimg
from skimage import measure
import pprint

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
array = []
for i in range(nCams):
    my_data = np.genfromtxt(csvFiles[i], delimiter=',').astype(int)
    frame_idx = my_data[1:,0].copy() - 1
    array.append(frame_idx)
    if frame_idx[-1] > maxImgIdx:
        maxImgIdx = frame_idx[-1]

print(maxImgIdx)

# completeObsIdx = []
# for i in range(maxImgIdx):
#     count = 0
#     for j in range(nCams):
#         if i in array[j]:
#             count +=1
#         if count == nCams:
#             completeObsIdx.append(i)

# completeObsIdx = np.array(completeObsIdx)

# print(array)

# for c, folder in enumerate(camFolders):
#     folderPath = os.path.join(camsFolder, folder)
#     onlyfiles = [f for f in os.listdir(folderPath) if os.path.isfile(os.path.join(folderPath, f))]
#     sortedFiles = sorted(onlyfiles)
#     n = len(sortedFiles)
#     areas = np.empty(shape=(maxImgIdx + 1,))
#     centroids = np.empty(shape=(maxImgIdx + 1,2))
#     blobcount = np.empty(shape=(maxImgIdx + 1,))
#     areas[:] = np.nan
#     centroids[:] = np.nan
#     blobcount[:] = np.nan

#     for i in range(n):
#         # if array[c][i] not in completeObsIdx:
#         #     continue

#         print("i: ", i, "c: ", c)
        
#         idx = array[c][i]
#         imgFile = folderPath + '/' + sortedFiles[i]
#         img = mpimg.imread(imgFile)
#         green = np.array(img[:,:,1])
#         blobs = green > 120
#         labels, num = measure.label(blobs, background=0, return_num=True)
#         props = measure.regionprops(labels)
#         blobcount[i] = num
#         if num == 0:
#             print(sortedFiles[i], "  - no region found")

#         elif num > 1:
#             print(sortedFiles[i], "  - more than one region found")
#             # check distance
#             ccKeep = []
#             for cc in props:
#                 dist = (((cc.centroid[0] - 1100) ** 2) + ((cc.centroid[1] - 1608) ** 2)) ** 0.5
#                 if dist < 2000:   # this was used to remove unwanted contours near the edge of the image
#                     ccKeep.append(cc)

#             imgAreas = np.empty(shape=(len(ccKeep),))
#             for ccIndex in range(len(ccKeep)):
#                 imgAreas[ccIndex] = ccKeep[ccIndex].area

#             print("imgAreas: ", imgAreas)
#             if (len(imgAreas) > 0):
#                 index = np.argmax(imgAreas)
#                 areas[idx] = ccKeep[index].area
#                 centroids[idx,:] = ccKeep[index].centroid
#                 print("idx: ", idx, "  centroid: ", centroids[idx, :])
#             else:
#                 print(sortedFiles[i], "  - after distance control, no region found")

#         elif num == 1:
            
#             areas[idx] = props[0].area
#             centroids[idx,:] = props[0].centroid
#             print("idx: ", idx, "  centroid: ", centroids[idx, :])

#     outfile = '../results/' + folder + '_res'
#     np.savez(outfile, areas, centroids, blobcount, np.array(array[c][:]))
