import os
import json
import cv2 as cv
import numpy as np


def json_read(filename):
    try:
        with open(os.path.abspath(filename)) as f:
            data = json.load(f)
        return data
    except ValueError:
        print("Unable to read JSON {}".format(filename))


def save_extrinsics_yaml(
    output_file, img_size, cam_matrix, dist_coefficients, rc_ext, tc_ext
):
    s = cv.FileStorage(output_file, cv.FileStorage_WRITE)
    s.write("image_width", img_size[0])
    s.write("image_height", img_size[1])

    s.write("camera_matrix", cam_matrix)
    s.write("distortion_coefficients", dist_coefficients)

    s.write("tc_ext", tc_ext)
    s.write("rc_ext", rc_ext)
    s.release()


root_dir = "/Users/yanj11/data/rig5cams"
global_poses_file = root_dir + "/output/global_registration/global_poses.json"
global_poses = json_read(global_poses_file)

output_dir = root_dir + "/output/rig_space"
os.makedirs(output_dir, exist_ok=True)

r_x_c_180 = np.asarray([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
r_z_c_90 = np.asarray([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
r_t = r_z_c_90 @ r_x_c_180

for key, value in global_poses.items():
    output_file = output_dir + "/Cam{}.yaml".format(key)

    new_r = np.asarray(value["R"]) @ (r_t.T)
    new_t = np.asarray(value["t"])

    save_extrinsics_yaml(
        output_file,
        [3208, 2200],
        np.asarray(value["K"]),
        np.asarray(value["dist"]),
        new_r,
        new_t,
    )
