#!/usr/bin/env python
import argparse
import numpy as np
import cv2 as cv
from scipy.spatial.transform import Rotation as R
import os
import json


def json_write(filename, data):
    try:
        directory = os.path.dirname(os.path.abspath(filename))
        if not os.path.exists(directory):
            os.makedirs(directory)
        with open(os.path.abspath(filename), "w") as f:
            json.dump(data, f, indent=2)
    except ValueError:
        print("Unable to write JSON {}".format(filename))


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


def json_read(filename):
    try:
        with open(os.path.abspath(filename)) as f:
            data = json.load(f)
        return data
    except ValueError:
        print("Unable to read JSON {}".format(filename))


def get_charuco_extrinsics(
    charuco_setup, img_path, cam_intrinsic_file, output_folder, cam_name
):
    """
    args:
    charuco_setup: charuco json file
    img_path: path to image
    cam_intrinsic_file: camera_intrinsic

    charuco_setup fields
    width: Number of squares in X direction
    height: Number of squares in Y direction
    square_len: Square side length
    marker_len: Marker side length
    dict: dictionary: DICT_4X4_50=0, DICT_4X4_100=1, DICT_4X4_250=2,  DICT_4X4_1000=3,"
        "DICT_5X5_50=4, DICT_5X5_100=5, DICT_5X5_250=6, DICT_5X5_1000=7, DICT_6X6_50=8,"
        "DICT_6X6_100=9, DICT_6X6_250=10, DICT_6X6_1000=11, DICT_7X7_50=12, DICT_7X7_100=13,"
        "DICT_7X7_250=14, DICT_7X7_1000=15, DICT_ARUCO_ORIGINAL = 16
    """
    cam_matrix = []
    dist_coefficients = []

    yaml_read_return, img_size, cam_matrix, dist_coefficients = read_camera_parameters(
        cam_intrinsic_file
    )
    if not yaml_read_return:
        raise ValueError("Can't read camera intrinsics.")

    image = cv.imread(cv.samples.findFile(img_path, False))
    if image is None:
        raise ValueError("Error: unable to open video/image source")

    charuco_config = json_read(charuco_setup)
    width = charuco_config["w"]
    height = charuco_config["h"]
    square_len = charuco_config["square_side_length"]
    marker_len = charuco_config["marker_side_length"]
    dict = charuco_config["dictionary"]

    aruco_dict = cv.aruco.getPredefinedDictionary(dict)
    board_size = (width, height)
    board = cv.aruco.CharucoBoard(board_size, square_len, marker_len, aruco_dict)
    charuco_detector = cv.aruco.CharucoDetector(board)

    charuco_corners, charuco_ids, marker_corners, marker_ids = (
        charuco_detector.detectBoard(image)
    )
    if (marker_ids is not None) and len(marker_ids) > 0:
        cv.aruco.drawDetectedMarkers(image, marker_corners)
    if (charuco_ids is not None) and len(charuco_ids) > 0:
        cv.aruco.drawDetectedCornersCharuco(image, charuco_corners, charuco_ids)
        if len(cam_matrix) > 0 and len(charuco_ids) >= 4:
            try:
                obj_points, img_points = board.matchImagePoints(
                    charuco_corners, charuco_ids
                )

                temp = np.zeros_like(obj_points)
                temp[:, 0, 0] = obj_points[:, 0, 1]
                temp[:, 0, 1] = obj_points[:, 0, 0]
                obj_points = temp

                flag, rvec, tvec = cv.solvePnP(
                    obj_points, img_points, cam_matrix, dist_coefficients
                )
                r = R.from_rotvec(rvec[:, 0])
                rotation_matrix = r.as_matrix()

                cam_extrinsics_file = output_folder + "/{}.yaml".format(cam_name)
                save_extrinsics_yaml(
                    cam_extrinsics_file,
                    img_size,
                    cam_matrix,
                    dist_coefficients,
                    rotation_matrix,
                    tvec,
                )
                if flag:
                    for pts_idx in range(img_points.shape[0]):
                        cv.circle(
                            image,
                            (
                                int(img_points[pts_idx, 0, 0]),
                                int(img_points[pts_idx, 0, 1]),
                            ),
                            10,
                            (255, 0, 255),
                            -1,
                        )
                        cv.putText(
                            image,
                            str(pts_idx),
                            (
                                int(img_points[pts_idx, 0, 0]),
                                int(img_points[pts_idx, 0, 1]),
                            ),
                            cv.FONT_HERSHEY_SIMPLEX,
                            1,
                            (255, 255, 0),
                            3,
                            cv.LINE_AA,
                        )
                    cv.drawFrameAxes(
                        image,
                        cam_matrix,
                        dist_coefficients,
                        rotation_matrix,
                        tvec,
                        length=100,
                        thickness=15,
                    )
            except cv.error as error_inst:
                print(
                    "SolvePnP recognize calibration pattern as non-planar pattern. To process this need to use minimum 6 points. The planar pattern may be mistaken for non-planar if the pattern is deformed or incorrect camera parameters are used."
                )
                print(error_inst.err)

        image_resize = cv.resize(image, (1604, 1100))
        cv.imshow("World Coordinates", image_resize)
        cv.waitKey(0)
        cv.destroyAllWindows()


root_folder = "/Users/yanj11/data/rig5cams"
cam_name = "710038"
camera_intrinsic_file = root_folder + "/output/intrinsics/710038.yaml"
charuco_setup_file = os.path.join(root_folder, "charuco_setup.json")
img_path = "/Users/yanj11/data/2024_12_18_2/710038_16_35_55_349.tiff"
output_folder = root_folder + "/output/extrinsics"
os.makedirs(output_folder, exist_ok=True)
get_charuco_extrinsics(
    charuco_setup_file, img_path, camera_intrinsic_file, output_folder, cam_name
)
