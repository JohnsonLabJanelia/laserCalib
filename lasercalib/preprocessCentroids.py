import os
import numpy as np
import matplotlib.pyplot as plt

resFolder = '../results'
resFiles = [d for d in os.listdir(resFolder) if os.path.isfile(os.path.join(resFolder, d))]
resFiles = sorted(resFiles)
print(resFiles)

n = 1300  # number images per camera

pts = np.empty(shape=(n, 2, 7))
nanPts = np.zeros(shape=(n, 7), dtype=bool)

fig, axs = plt.subplots(1, 7, sharey=True)
for i in range(len(resFiles)):
    filename = resFolder + '/' + resFiles[i]
    with np.load(filename) as data:
        areas = data['arr_0']
        centroids = data['arr_1']
        blobcounts = data['arr_2']

    pts[:,:,i] = centroids.copy()
    nanPts[:, i] = np.isnan(areas)
    colors = np.linspace(0, 1, len(areas))
    axs[i].scatter(centroids[:,0], centroids[:,1], s=10, c=colors, alpha=0.5)
    axs[i].title.set_text(resFiles[i])

inPts = np.zeros(shape=(n,), dtype=bool)

for i in range(n):
    inPts[i] = (not any(nanPts[i, :]))

goodPts = pts[inPts, :, :]

outfile = '../data/good_centroids'
np.savez(outfile, goodPts)
plt.show()