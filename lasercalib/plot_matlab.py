import numpy as np
import pickle as pkl
import cv2 
import argparse
import matplotlib.pyplot as plt
from camera_visualizer import CameraVisualizer
import seaborn as sns

parser = argparse.ArgumentParser()
parser.add_argument('--root_dir', type=str, required=True)
parser.add_argument('--n_cams', type=int, required=True)

args = parser.parse_args()

nCams = args.n_cams

camList = []
for i in range(nCams):
    cam_params = {}
    filename = args.root_dir + "/Cam{}.yaml".format(i)
    print(filename)
    fs = cv2.FileStorage(filename, cv2.FILE_STORAGE_READ)
    cam_params['camera_matrix'] = fs.getNode("camera_matrix").mat()
    cam_params['distortion_coefficients'] = fs.getNode("distortion_coefficients").mat()
    cam_params['tc_ext'] = fs.getNode("tc_ext").mat()
    cam_params['rc_ext'] = fs.getNode("rc_ext").mat()
    camList.append(cam_params)


color_palette = sns.color_palette("rocket_r", nCams)
xlim=[-1500, 1500]
ylim=[-1500, 1500]
zlim=[-100, 2000]
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')
for i in range(nCams):
    r_c = camList[i]['rc_ext']
    t_c = camList[i]['tc_ext'][:, 0]
    # get inverse transformation
    ex = np.eye(4)
    ex[:3,:3] = r_c.T
    ex[:3,3] = -np.matmul(r_c.T, t_c)    
    visualizer = CameraVisualizer(fig, ax)
    visualizer.extrinsic2pyramid(ex, color_palette[i], 200)
ax.set_xlim(xlim)
ax.set_ylim(ylim)
ax.set_zlim(zlim)
plt.show()