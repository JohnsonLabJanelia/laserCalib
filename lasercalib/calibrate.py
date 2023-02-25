from tkinter.messagebox import NO
import numpy as np
import matplotlib.pyplot as plt
import pickle as pkl
from datetime import date
import seaborn as sns
from scipy.spatial.transform import Rotation as R
from prettytable import PrettyTable
import pySBA
from camera_visualizer import CameraVisualizer
from convert_params import load_from_blender

## inputs to the file
root_dir = "/home/jinyao/Calibration/newrig8"
cam_idx_3dpts = 4
a = 1.0
## 

with open(root_dir + "/results/centroids.pkl", 'rb') as file:
    pts = pkl.load(file)

# flip xy (regionprops orders)
pts = np.flip(pts, axis=1)
nPts = pts.shape[0]
nCams = pts.shape[2]
my_palette = sns.color_palette("rocket_r", nCams)
cameraArray = load_from_blender(root_dir + "/results/camera_dicts.pkl", nCams)

def sba_print(sba, nCams):
    x = PrettyTable()
    for row in sba.cameraArray:
        x.add_row(row)
    print(x)

    r = sba.project(sba.points3D[sba.point2DIndices], sba.cameraArray[sba.cameraIndices]) - sba.points2D
    print(r.shape)
    r = np.sqrt(np.sum(r**2, axis=1))
    plt.hist(r[r<np.percentile(r, 99)])
    plt.xlabel('Reprojection Error')
    plt.title('adjust points only')
    plt.show()

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(sba.points3D[:,0], sba.points3D[:,1], sba.points3D[:,2])
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    plt.title('adjust points only')
    for i in range(nCams):
        r_f = R.from_rotvec(-sba.cameraArray[i, 0:3]).as_matrix().copy()
        t_f = sba.cameraArray[i, 3:6].copy()
        # get inverse transformation
        r_inv = r_f.T    
        t_inv = -np.matmul(r_f, t_f)    
        ex = np.eye(4)
        ex[:3,:3] = r_inv.T
        ex[:3,3] = t_inv
        visualizer = CameraVisualizer(fig, ax)
        visualizer.extrinsic2pyramid(ex, my_palette[i], 200)
    ax.set_xlim((-1500, 1500))
    ax.set_ylim((-1500, 1500))
    ax.set_zlim((-100, 1500))
    plt.show()


keep = np.zeros(shape=(nPts,), dtype=bool)
for i in range(nPts):
    v = pts[i, 0, :]
    if ((np.sum(~np.isnan(v)) >= nCams) and (~np.isnan(v[0]))):
        keep[i] = True

inPts = pts[keep, :, :]
nPts = inPts.shape[0]
print("nPts: ", nPts)

nObs = np.sum(~np.isnan(inPts[:,0,:].ravel()))
print("nObs: ", nObs)

fig, axs = plt.subplots(1, nCams, sharey=True)
plt.title('2D points found on all cameras')
for i in range(nCams):
    colors = np.linspace(0, 1, nPts)
    axs[i].scatter(inPts[:,0,i], inPts[:,1,i], s=10, c=colors, alpha=0.5)
    axs[i].plot(inPts[:,0,i], inPts[:,1,i])
    axs[i].title.set_text('cam' + str(i))
    axs[i].invert_yaxis()
plt.show()

# create camera_ind variable
camera_ind = np.zeros(shape=(nObs,), dtype=int)
point_ind = np.zeros(shape=(nObs,), dtype=int)
points_2d = np.zeros(shape=(nObs, 2), dtype=float)

ind = 0
for i in range(nPts):
    for j in range(nCams):
        if (np.isnan(inPts[i, 0, j])):
            continue
        camera_ind[ind] = j
        point_ind[ind] = i
        points_2d[ind, :] = inPts[i, :, j].copy()
        ind += 1


# prepare points_3d variable (initializing with 2d laser points in image space on cam4)
points_3d = np.zeros(shape=(nPts, 3))
for i in range(nPts):
    if (np.isnan(inPts[i, 0, cam_idx_3dpts])):
        points_3d[i, 0:2] = [1604, 1100]
        continue
    else:
        points_3d[i, 0:2] = inPts[i, :, cam_idx_3dpts]

# # # center the world points
points_3d[:,0] = a * (points_3d[:,0] - 1604)
points_3d[:,1] = a * (points_3d[:,1] - 1100)

sba = pySBA.PySBA(cameraArray, points_3d, points_2d, camera_ind, point_ind)
sba_print(sba, nCams)

sba.bundleAdjust_nocam()
sba_print(sba, nCams)

sba.bundleAdjust()
sba_print(sba, nCams)

output_file = root_dir + "/results/sba_blender.pkl"
with open(output_file, 'wb') as f:
    pkl.dump(sba, f)

print("Done fitting, saved to: {}".format(output_file))