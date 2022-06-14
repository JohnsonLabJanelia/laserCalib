from tkinter.messagebox import NO
import numpy as np
import matplotlib.pyplot as plt
import pickle as pkl
import os

from scipy.spatial.transform import Rotation as R
from prettytable import PrettyTable
import pySBA
from my_cam_pose_visualizer import MyCamPoseVisualizer
import seaborn as sns

my_palette = sns.color_palette()


root_dir = '/home/rob/laser_calibration/2022-05-02_16_40_25'
results_dir = root_dir + "/results"

res_files = [os.path.join(results_dir, f) for f in os.listdir(results_dir) if os.path.isfile(os.path.join(results_dir, f))]
res_files = sorted(res_files)

fileObject = open(res_files[0], 'rb')
pts = pkl.load(fileObject)
fileObject.close()

n_pts_per_cam = pts.shape[0]
n_cams = len(res_files)
centroids = np.zeros((n_pts_per_cam, 2, n_cams))
centroids[:] = np.nan

for i, file in enumerate(res_files):
    print(file)
    fileObject = open(file, 'rb')
    centroids[:,:,i] = pkl.load(fileObject)
    fileObject.close()
    
print(centroids)
print(centroids.shape)

outfile = 'centroids_multithreaded.pkl'
fileObject = open(outfile, 'wb')
pkl.dump(centroids, fileObject)
fileObject.close()

