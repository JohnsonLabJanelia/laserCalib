import os
import numpy as np
import matplotlib.pyplot as plt

resFolder = '../results'
resFiles = [d for d in os.listdir(resFolder) if os.path.isfile(os.path.join(resFolder, d))]
resFiles = sorted(resFiles)
print(resFiles)

nCams = 7
nPts = 6547  # number images per camera

pts = np.empty(shape=(nPts, 2, nCams))
pts[:] = np.nan



fig, axs = plt.subplots(1, nCams, sharey=True)
for i in range(nCams):
    filename = resFolder + '/' + resFiles[i]
    with np.load(filename) as data:
        centroids = data['arr_1']
        idx = data['arr_3']

    for j in range(len(idx)):
        pts[idx[j], :, i] = centroids[j, :].copy()
        print(centroids[idx[j], :])

    # nanPts[idx, :] = False
    colors = np.linspace(0, 1, centroids.shape[0])
    axs[i].scatter(centroids[:,0], centroids[:,1], s=10, c=colors, alpha=0.5)
    axs[i].title.set_text(resFiles[i])

outfile = '../results/all_centroids'
np.savez(outfile, pts)
plt.show()