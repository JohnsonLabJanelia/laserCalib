from tkinter.messagebox import NO
import numpy as np
import matplotlib.pyplot as plt
import pickle as pkl

import matplotlib.image as mpimg
from skimage import measure
from skimage import morphology
from skimage import io
import pprint

from scipy.spatial.transform import Rotation as R
from prettytable import PrettyTable
import pySBA
from my_cam_pose_visualizer import MyCamPoseVisualizer
import seaborn as sns
import time


class PointBuilder:
    def __init__(self, fig, ax, collection, line, maxPts):
        self.fig = fig
        self.ax = ax
        self.collection = collection
        self.line = line
        self.maxPts = maxPts
        self.points = np.zeros(shape=(maxPts,2))
        self.point_idx = 0
        self.cid = []
        self.cid.append(collection.figure.canvas.mpl_connect('key_press_event', self))

    def __call__(self, event):
        if not (event.key == "w"):
            return

        print("point:", self.point_idx, "   xdata:", event.xdata, "   ydata:", event.ydata)
        point = np.array([event.xdata, event.ydata]).reshape(1,2)
        self.points[self.point_idx, :] = point
        xs = self.points[:self.point_idx + 1, 0]
        ys = self.points[:self.point_idx + 1, 1]

        self.point_idx += 1
        if self.point_idx == self.maxPts:
            xs = np.hstack((xs, xs[0]))
            ys = np.hstack((ys, ys[0]))

        self.line.set_data(xs, ys)
        self.line.set_color("m")
        self.line.figure.canvas.draw()
        self.collection.set_offsets(self.points)
        self.collection.set_facecolor('r')
        self.collection.figure.canvas.draw_idle()

        if (self.point_idx == self.maxPts):
            pass


my_palette = sns.color_palette("rocket_r", 7)

img_dir = "/home/rob/Videos/LaserCalibration/Calibration20220701/rig_points"
img_paths = []
nCams = 7
nRefPoints = 4

for i in range(nCams):
    img_path = img_dir + "/Cam" + str(i) + ".png"
    img_paths.append(img_path)

for i in range(nCams):
    img = io.imread(img_paths[i])
    fig, ax = plt.subplots()
    ax.set_title('click to build line segments')
    ax.imshow(img)
    collection = ax.scatter(None, None)
    line, = ax.plot([0], [0])
    pointBuilder = PointBuilder(fig, ax, collection, line, maxPts=nRefPoints)
    plt.show()
    print(collection.get_offsets())
