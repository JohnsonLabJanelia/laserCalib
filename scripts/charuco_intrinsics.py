import cv2 as cv
import numpy as np
import os
import argparse
import sys
from lasercalib.utils import probe_monotonicity
import matplotlib.pyplot as plt

def read_chessboards(images, board, aruco_dict, verbose):
    """
    Charuco base pose estimation.
    """
    print("POSE ESTIMATION STARTS:")
    all_corners = []
    all_Ids = []
    decimator = 0
    # SUB PIXEL CORNER DETECTION CRITERION
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 200, 0.00001)
    wait_time = 100


    charuco_detector = cv.aruco.CharucoDetector(board)
    objpoints = []
    imgpoints = []
    
    for im in images:
        print("=> Processing image {0}".format(im))
        frame = cv.imread(im)
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        corners, ids, rejectedImgPoints = cv.aruco.detectMarkers(gray, aruco_dict)

        if len(corners)>0:
            charuco_corners, charuco_ids, marker_corners, marker_ids = charuco_detector.detectBoard(frame)
            obj_points, img_points = board.matchImagePoints(charuco_corners, charuco_ids)

            objpoints.append(obj_points)
            imgpoints.append(img_points)

            # SUB PIXEL DETECTION
            for corner in corners:
                cv.cornerSubPix(gray, corner,
                                 winSize = (3,3),
                                 zeroZone = (-1,-1),
                                 criteria = criteria)
            res2 = cv.aruco.interpolateCornersCharuco(corners,ids,gray,board)
            if res2[1] is not None and res2[2] is not None and len(res2[1])>3 and decimator%1==0:
                all_corners.append(res2[1])
                all_Ids.append(res2[2])
                
                if verbose:
                    image_copy = np.copy(frame)

                    for pts_idx in range(res2[1].shape[0]):
                        cv.circle(image_copy, (int(res2[1][pts_idx, 0, 0]), int(res2[1][pts_idx, 0, 1])), 25, (255, 0, 255), -1)
                    image_resize = cv.resize(image_copy, (1604, 1100))
                    cv.imshow("image", image_resize)
                    key = cv.waitKey(wait_time)
                    if key == 27:
                        break
        decimator+=1

    imsize = gray.shape
    return all_corners, all_Ids, imsize, objpoints, imgpoints

def calibrate_camera(board, all_corners, all_Ids, imsize, focal_length_init):
    """
    Calibrates the camera using the dected corners.
    """
    cameraMatrixInit = np.array([[ focal_length_init,    0., imsize[1]/2.],
                                 [    0., focal_length_init, imsize[0]/2.],
                                 [    0.,    0.,           1.]])

    distCoeffsInit = np.zeros((5,1))
    # flags = (cv.CALIB_USE_INTRINSIC_GUESS + cv.CALIB_RATIONAL_MODEL + cv.CALIB_FIX_ASPECT_RATIO)
    # flags = (cv.CALIB_USE_INTRINSIC_GUESS  + cv.CALIB_FIX_ASPECT_RATIO + cv.CALIB_FIX_K3 )
    flags = 0
    # flags += cv.CALIB_RATIONAL_MODEL
    (ret, camera_matrix, distortion_coefficients0,
     rotation_vectors, translation_vectors,
     stdDeviationsIntrinsics, stdDeviationsExtrinsics,
     perViewErrors) = cv.aruco.calibrateCameraCharucoExtended(
                      charucoCorners=all_corners,
                      charucoIds=all_Ids,
                      board=board,
                      imageSize=imsize,
                      cameraMatrix=cameraMatrixInit,
                      distCoeffs=distCoeffsInit,
                      flags=flags,
                      criteria=(cv.TERM_CRITERIA_EPS & cv.TERM_CRITERIA_COUNT, 10000, 1e-9))
        
    return ret, camera_matrix, distortion_coefficients0, rotation_vectors, translation_vectors, stdDeviationsIntrinsics, stdDeviationsExtrinsics, perViewErrors


def save_instrinsics_yaml(output_file, img_width, img_height, intrinsic_matrix, distortion_coefficients):
    
    # distortion_coefficients = np.asarray(distortion_coefficients).reshape(5, 1)

    # save it using opencv 
    s = cv.FileStorage(output_file, cv.FileStorage_WRITE)
    s.write('image_width', img_width)
    s.write('image_height', img_height)

    s.write('camera_matrix', intrinsic_matrix)
    s.write('distortion_coefficients', distortion_coefficients)
    s.release()
    

def main():

    parser = argparse.ArgumentParser(description="Get camera intrinsics from multiple images.", add_help=False)
    parser.add_argument("-H", "--help", help="show help", action="store_true", dest="show_help")
    parser.add_argument("-i", "--images", help="Folder of input images", default="", action="store", dest="img_path")
    parser.add_argument("-o", "--output", help="YAML file saves to the output folder", default="", type=str, action="store", dest="output_folder")
    parser.add_argument("-w", help="Number of squares in X direction", default="5", action="store", dest="w", type=int)
    parser.add_argument("-h", help="Number of squares in Y direction", default="7", action="store", dest="h", type=int)
    parser.add_argument("-sl", help="Square side length", default="120.0", action="store", dest="sl", type=float)
    parser.add_argument("-ml", help="Marker side length", default="60.0", action="store", dest="ml", type=float)
    parser.add_argument("-d", help="dictionary: DICT_4X4_50=0, DICT_4X4_100=1, DICT_4X4_250=2,  DICT_4X4_1000=3,"
                                   "DICT_5X5_50=4, DICT_5X5_100=5, DICT_5X5_250=6, DICT_5X5_1000=7, DICT_6X6_50=8,"
                                   "DICT_6X6_100=9, DICT_6X6_250=10, DICT_6X6_1000=11, DICT_7X7_50=12, DICT_7X7_100=13,"
                                   "DICT_7X7_250=14, DICT_7X7_1000=15, DICT_ARUCO_ORIGINAL = 16}",
                        default="5", action="store", dest="d", type=int)
    
    args = parser.parse_args()
    show_help = args.show_help
    if show_help:
        parser.print_help()
        sys.exit()
    
    width = args.w
    height = args.h
    square_len = args.sl
    marker_len = args.ml
    dict = args.d
    img_path = args.img_path
    
    # parse camera name
    cam_name = img_path.split("/")[-2]

    output_folder = args.output_folder

    aruco_dict = cv.aruco.getPredefinedDictionary(dict)
    board_size = (width, height)
    board = cv.aruco.CharucoBoard(board_size, square_len, marker_len, aruco_dict)

    images = np.array([img_path + f for f in os.listdir(img_path) if f.endswith(".tif") ])

    all_corners, all_Ids, imsize, objpoints, imgpoints =read_chessboards(images, board, aruco_dict, False)
    ret, mtx, dist, rvecs, tvecs, std_dev_intrisics, std_dev_extrinsics, per_view_errors = calibrate_camera(board, all_corners, all_Ids, imsize, 1700)

    # add metrics 
    def reprojection_error(mtx, distCoeffs, rvecs, tvecs):
        # print reprojection error
        reproj_error = 0
        for i in range(len(objpoints)):
            imgpoints2, _ = cv.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, distCoeffs)
            reproj_error += cv.norm(imgpoints[i],imgpoints2,cv.NORM_L2)/len(imgpoints2)
        reproj_error /= len(objpoints) 
        return reproj_error
    reproj_error = reprojection_error(mtx, dist, rvecs, tvecs)
    print("RMS Reprojection Error: {}, Total Reprojection Error: {}".format(ret, reproj_error))

    alpha = 0.95
    newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, imsize, alpha, 
                                                      imsize, centerPrincipalPoint=False)
    

    grid_norm, is_monotonic = probe_monotonicity(mtx, dist, newcameramtx, imsize, N=100, M=100)
    if not np.all(is_monotonic):
        print("-"*50)
        print(" The distortion function is not monotonous for alpha={:0.2f}!".format(alpha))
        print(" To fix this we suggest sampling more precise points on the corner of the image first.")
        print(" If this is not enough, use the option Rational Camera Model which more adpated to wider lenses.")
        print("-"*50)

    frame = cv.imread(images[0])
    plt.figure()
    plt.imshow(cv.undistort(frame, mtx, dist, None, newcameramtx))
    grid = grid_norm*newcameramtx[[0,1],[0,1]][None]+newcameramtx[[0,1],[2,2]][None]
    plt.plot(grid[is_monotonic, 0], grid[is_monotonic, 1], '.g', label='monotonic', markersize=1.5)
    plt.plot(grid[~is_monotonic, 0], grid[~is_monotonic, 1], '.r', label='not monotonic', markersize=1.5)
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(output_folder, "monotonicity.jpg"), bbox_inches='tight')


    output_file = output_folder + cam_name + '.yaml'
    print(output_file)
    save_instrinsics_yaml(output_file, imsize[1], imsize[0], mtx, dist)



if __name__ == "__main__":
    main()
