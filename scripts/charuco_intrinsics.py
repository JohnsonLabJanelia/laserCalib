import cv2 as cv
import numpy as np
import os
import argparse
from lasercalib.utils import probe_monotonicity
import matplotlib.pyplot as plt
import json
import pickle


def read_chessboards(images, board, aruco_dict, verbose):
    """
    Charuco base pose estimation.
    """
    all_corners = []
    all_ids = []
    # SUB PIXEL CORNER DETECTION CRITERION
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 200, 0.00001)
    wait_time = 200

    charuco_detector = cv.aruco.CharucoDetector(board)
    objpoints = []
    imgpoints = []

    frame_0 = cv.imread(images[0])
    imsize = frame_0.shape[:2]
    all_im_ids = []
    for im in images:
        print("=> Processing image {0}".format(im))
        frame = cv.imread(im)
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        corners, ids, rejectedImgPoints = cv.aruco.detectMarkers(gray, aruco_dict)

        if len(corners) > 0:
            charuco_corners, charuco_ids, marker_corners, marker_ids = (
                charuco_detector.detectBoard(frame)
            )

            if charuco_corners is not None and charuco_ids is not None:
                obj_points, img_points = board.matchImagePoints(
                    charuco_corners, charuco_ids
                )

                # SUB PIXEL DETECTION
                for corner in corners:
                    cv.cornerSubPix(
                        gray,
                        corner,
                        winSize=(3, 3),
                        zeroZone=(-1, -1),
                        criteria=criteria,
                    )

                res2 = cv.aruco.interpolateCornersCharuco(corners, ids, gray, board)
                if res2[1] is not None and res2[2] is not None and len(res2[1]) > 3:
                    im_name = im.split("/")[-1]
                    all_im_ids.append("_".join(im_name.split("_")[1:]))
                    all_corners.append(res2[1])
                    all_ids.append(res2[2])

                    objpoints.append(obj_points)
                    imgpoints.append(img_points)

                    if verbose:
                        image_copy = np.copy(frame)

                        for pts_idx in range(res2[1].shape[0]):
                            cv.circle(
                                image_copy,
                                (
                                    int(res2[1][pts_idx, 0, 0]),
                                    int(res2[1][pts_idx, 0, 1]),
                                ),
                                25,
                                (255, 0, 255),
                                -1,
                            )
                        image_resize = cv.resize(image_copy, (1604, 1100))
                        cv.imshow("{}".format(im), image_resize)
                        key = cv.waitKey(wait_time)
                        if key == ord("q"):
                            break

    print("=> Number of valid image: {}.".format(len(all_im_ids)))
    return all_corners, all_ids, imsize, objpoints, imgpoints, all_im_ids


def calibrate_camera(board, all_corners, all_ids, imsize, cam_name):
    """
    Calibrates the camera using the dected corners.
    """
    flags = 0
    flags += cv.CALIB_USE_INTRINSIC_GUESS + cv.CALIB_FIX_ASPECT_RATIO
    # flags += (
    #     cv.CALIB_USE_INTRINSIC_GUESS
    #     + cv.CALIB_FIX_ASPECT_RATIO
    #     + cv.CALIB_RATIONAL_MODEL
    # )

    if cam_name == "710038":
        focal_length_init = 1780
    else:
        focal_length_init = 2300

    cameraMatrixInit = np.array(
        [
            [focal_length_init, 0.0, imsize[1] / 2.0],
            [0.0, focal_length_init, imsize[0] / 2.0],
            [0.0, 0.0, 1.0],
        ]
    )
    distCoeffsInit = np.zeros((5, 1))

    (
        ret,
        camera_matrix,
        distortion_coefficients0,
        rotation_vectors,
        translation_vectors,
        stdDeviationsIntrinsics,
        stdDeviationsExtrinsics,
        perViewErrors,
    ) = cv.aruco.calibrateCameraCharucoExtended(
        charucoCorners=all_corners,
        charucoIds=all_ids,
        board=board,
        imageSize=imsize,
        cameraMatrix=cameraMatrixInit,
        distCoeffs=distCoeffsInit,
        flags=flags,
        criteria=(cv.TERM_CRITERIA_EPS & cv.TERM_CRITERIA_COUNT, 10000, 1e-9),
    )

    return (
        ret,
        camera_matrix,
        distortion_coefficients0,
        rotation_vectors,
        translation_vectors,
        stdDeviationsIntrinsics,
        stdDeviationsExtrinsics,
        perViewErrors,
    )


def save_instrinsics_yaml(
    output_file, img_width, img_height, intrinsic_matrix, distortion_coefficients
):
    # distortion_coefficients = np.asarray(distortion_coefficients).reshape(5, 1)

    # save it using opencv
    s = cv.FileStorage(output_file, cv.FileStorage_WRITE)
    s.write("image_width", img_width)
    s.write("image_height", img_height)

    s.write("camera_matrix", intrinsic_matrix)
    s.write("distortion_coefficients", distortion_coefficients)
    s.release()


def json_read(filename):
    try:
        with open(os.path.abspath(filename)) as f:
            data = json.load(f)
        return data
    except ValueError:
        print("Unable to read JSON {}".format(filename))


def get_charuco_intrinsics(cam_name, charuco_setup, images, output_folder):
    """
    args:
    charuco_setup: json file
    images: list of path to images
    output_folder: path of folder to save results

    charuco_setup fileds:
    "w", int: Number of squares in X direction
    "h", int:Number of squares in Y direction
    "square_side_length", float
    "marker_side_length", float
    "dictionary",int:
        dictionary: DICT_4X4_50=0, DICT_4X4_100=1, DICT_4X4_250=2,  DICT_4X4_1000=3,
        DICT_5X5_50=4, DICT_5X5_100=5, DICT_5X5_250=6, DICT_5X5_1000=7, DICT_6X6_50=8,
        DICT_6X6_100=9, DICT_6X6_250=10, DICT_6X6_1000=11, DICT_7X7_50=12, DICT_7X7_100=13,
        DICT_7X7_250=14, DICT_7X7_1000=15, DICT_ARUCO_ORIGINAL = 16
    """
    charuco_config = json_read(charuco_setup)
    width = charuco_config["w"]
    height = charuco_config["h"]
    square_len = charuco_config["square_side_length"]
    marker_len = charuco_config["marker_side_length"]
    dict = charuco_config["dictionary"]

    aruco_dict = cv.aruco.getPredefinedDictionary(dict)
    board_size = (width, height)
    board = cv.aruco.CharucoBoard(board_size, square_len, marker_len, aruco_dict)

    all_corners, all_ids, imsize, objpoints, imgpoints, all_im_ids = read_chessboards(
        images, board, aruco_dict, False
    )

    landmark = {}
    for i, im_id in enumerate(all_im_ids):
        landmark[im_id] = {
            "corners": all_corners[i],
            "ids": all_ids[i],
            "objpoints": objpoints[i],
        }
    with open(output_folder + "/landmarks_{}.pkl".format(cam_name), "wb") as f:
        pickle.dump(landmark, f)

    (
        ret,
        mtx,
        dist,
        rvecs,
        tvecs,
        std_dev_intrisics,
        std_dev_extrinsics,
        per_view_errors,
    ) = calibrate_camera(board, all_corners, all_ids, imsize, cam_name)

    # add metrics
    def reprojection_error(mtx, distCoeffs, rvecs, tvecs):
        # print reprojection error
        reproj_error = 0
        for i in range(len(objpoints)):
            imgpoints2, _ = cv.projectPoints(
                objpoints[i], rvecs[i], tvecs[i], mtx, distCoeffs
            )
            reproj_error += cv.norm(imgpoints[i], imgpoints2, cv.NORM_L2) / len(
                imgpoints2
            )
        reproj_error /= len(objpoints)
        return reproj_error

    reproj_error = reprojection_error(mtx, dist, rvecs, tvecs)
    print(
        "RMS Reprojection Error: {}, Total Reprojection Error: {}".format(
            ret, reproj_error
        )
    )

    alpha = 0.95
    newcameramtx, roi = cv.getOptimalNewCameraMatrix(
        mtx, dist, imsize, alpha, imsize, centerPrincipalPoint=False
    )

    grid_norm, is_monotonic = probe_monotonicity(
        mtx, dist, newcameramtx, imsize, N=100, M=100
    )
    if not np.all(is_monotonic):
        print("-" * 50)
        print(
            " The distortion function is not monotonous for alpha={:0.2f}!".format(
                alpha
            )
        )
        print(
            " To fix this we suggest sampling more precise points on the corner of the image first."
        )
        print(
            " If this is not enough, use the option Rational Camera Model which more adpated to wider lenses."
        )
        print("-" * 50)

    frame = cv.imread(images[0])
    plt.figure()
    plt.imshow(cv.undistort(frame, mtx, dist, None, newcameramtx))
    grid = (
        grid_norm * newcameramtx[[0, 1], [0, 1]][None]
        + newcameramtx[[0, 1], [2, 2]][None]
    )
    plt.plot(
        grid[is_monotonic, 0],
        grid[is_monotonic, 1],
        ".g",
        label="monotonic",
        markersize=1.5,
    )
    plt.plot(
        grid[~is_monotonic, 0],
        grid[~is_monotonic, 1],
        ".r",
        label="not monotonic",
        markersize=1.5,
    )
    plt.legend()
    plt.grid()
    plt.savefig(
        os.path.join(output_folder, "monotonicity_{}.jpg".format(cam_name)),
        bbox_inches="tight",
    )

    output_file = os.path.join(output_folder, "{}.yaml".format(cam_name))
    save_instrinsics_yaml(output_file, imsize[1], imsize[0], mtx, dist)


root_folder = "/Users/yanj11/data/rig5cams"
img_path = "/Users/yanj11/data/2024_12_18_2"
output_folder = os.path.join(root_folder, "output/intrinsics")

os.makedirs(output_folder, exist_ok=True)

images = []
for f in os.listdir(img_path):
    if f.endswith(".tiff"):
        images.append(f)

cam_names = []
for image in images:
    cam_names.append(image.split("_")[0])
cam_names = sorted(np.unique(cam_names).tolist())

charuco_setup_file = os.path.join(root_folder, "charuco_setup.json")
for cam in cam_names:
    images_per_cam = []
    for image in images:
        this_image_cam_name = image.split("_")[0]
        if this_image_cam_name == cam:
            image_name = "_".join([cam, image])
            images_per_cam.append(os.path.join(img_path, image))
    print("=====> Camera: {}, Number of images: {}".format(cam, len(images_per_cam)))
    get_charuco_intrinsics(cam, charuco_setup_file, images_per_cam, output_folder)
