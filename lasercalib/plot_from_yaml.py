import numpy as np
import pickle as pkl
import cv2 
import argparse
import matplotlib.pyplot as plt
from camera_visualizer import CameraVisualizer
import seaborn as sns
import glob

parser = argparse.ArgumentParser()
parser.add_argument('--yaml_dir', type=str, required=True)
parser.add_argument('--ordered', type=int, default=0)

args = parser.parse_args()

cam_names = []
for file in glob.glob(args.yaml_dir + "/*.yaml"):
    file_name = file.split("/")
    cam_names.append(file_name[-1][:-5])
cam_names.sort()

nCams = len(cam_names)

if args.ordered == 1:
    cam_names = []
    for i in range(nCams):
        cam_names.append("Cam{}".format(i))


serial_to_order = {
    "2005325":0,
    "2006054":1,
    "2002486":2,
    "2006055":3,
    "2008666":4,
    "2008670":5,
    "2006052":6,
    "2008667":7,
    "2008669":8,
    "2008668":9,
    "2006050":10,
    "2006051":11,
    "2006516":12,
    "2008665":13,
    "2002487":14,
    "2006515":15
}

camList = []
for i in range(nCams):
    cam_params = {}
    filename = args.yaml_dir + "/{}.yaml".format(cam_names[i])
    print(filename)
    fs = cv2.FileStorage(filename, cv2.FILE_STORAGE_READ)
    cam_params['camera_matrix'] = fs.getNode("camera_matrix").mat()
    cam_params['distortion_coefficients'] = fs.getNode("distortion_coefficients").mat()
    cam_params['tc_ext'] = fs.getNode("tc_ext").mat()
    cam_params['rc_ext'] = fs.getNode("rc_ext").mat()
    camList.append(cam_params)


unordered_color_palette = sns.color_palette("rocket_r", nCams)

if args.ordered:
    my_palette = unordered_color_palette

else:
    my_palette = []
    for one_name in cam_names:
        cam_serial = one_name[3:]
        cam_order = serial_to_order[cam_serial]
        color_of_cam = unordered_color_palette[cam_order]
        my_palette.append(color_of_cam)


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
    visualizer.extrinsic2pyramid(ex, my_palette[i], 200)
ax.set_xlim(xlim)
ax.set_ylim(ylim)
ax.set_zlim(zlim)
plt.show()