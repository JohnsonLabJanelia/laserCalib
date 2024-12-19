import os
import cv2 as cv
import json
import numpy as np

root_folder = "/Users/yanj11/data/2024_12_13/three_cams"
all_cam_folders = os.listdir(root_folder)
all_cam_folders = [item for item in os.listdir(root_folder) if not item.startswith(".")]
print(all_cam_folders)


## load all yaml file
def read_camera_parameters(filename):
    fs = cv.FileStorage(cv.samples.findFile(filename, False), cv.FileStorage_READ)
    if fs.isOpened():
        img_width = int(fs.getNode("image_width").real())
        img_height = int(fs.getNode("image_height").real())
        img_size = [img_width, img_height]
        cam_matrix = fs.getNode("camera_matrix").mat()
        dist_coefficients = fs.getNode("distortion_coefficients").mat()
        return True, img_size, cam_matrix, dist_coefficients
    return False, [], [], []


intrinsics = {}
for cam in all_cam_folders:
    cam_path = os.path.join(root_folder, cam)
    cam_intrinsics_file = cam_path + "/outputs/{}.yaml".format(cam)
    ret, img_size, cam_matrix, dist_coefficients = read_camera_parameters(
        cam_intrinsics_file
    )
    dist_coefficients = np.squeeze(dist_coefficients)
    intrinsics_per_cam_dict = {
        "K": cam_matrix.tolist(),
        "dist": dist_coefficients.tolist(),
    }
    intrinsics[cam] = intrinsics_per_cam_dict


def json_write(filename, data):
    try:
        directory = os.path.dirname(os.path.abspath(filename))
        if not os.path.exists(directory):
            os.makedirs(directory)
        with open(os.path.abspath(filename), "w") as f:
            json.dump(data, f, indent=2)
    except ValueError:
        print("Unable to write JSON {}".format(filename))


json_write("/Users/yanj11/data/2024_12_13/rig_cams3/intrinsics.json", intrinsics)
