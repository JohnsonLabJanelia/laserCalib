import numpy as np
import pickle as pkl
import cv2 
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--root_dir', type=str, required=True)
parser.add_argument('--n_cams', type=int, required=True)
args = parser.parse_args()

nCams = args.n_cams

with open(args.root_dir + "/results/calibration_blender.pkl", "rb") as f:
    camList = pkl.load(f)

aruco_loc = []
for i in range(nCams):
    with open(args.root_dir + "/results/Cam{}_aruco.pkl".format(i), 'rb') as f:
        one_camera = pkl.load(f)
        aruco_loc.append(one_camera)

centers = []
for cam in aruco_loc:
    points = []
    for marker_id, marker_coord in cam.items():
        points.append(marker_coord.mean(axis=0))
    points = np.asarray(points)
    centers.append(points)
centers = np.asarray(centers)

undistorted_pts = []
for cam_idx in range(nCams):
    distortion = np.zeros((5,1))
    distortion[0:2, 0] = camList[cam_idx]['d']
    ideal_points = cv2.undistortPoints(centers[cam_idx], camList[cam_idx]['K'].T, distortion, None, camList[cam_idx]['K'].T)[:, 0, :]
    undistorted_pts.append(ideal_points)
undistorted_pts = np.asarray(undistorted_pts)

pts_3d = []
for center_idx in range(4):
    # triangulation 
    # https://filebox.ece.vt.edu/~jbhuang/teaching/ece5554-4554/fa17/lectures/Lecture_15_StructureFromMotion.pdf
    # https://www.cs.cmu.edu/~16385/s17/Slides/11.4_Triangulation.pdf
    A = np.zeros([nCams*2, 4])
    for cam_idx in range(nCams):
        proj_matrix = np.zeros((3, 4))
        proj_matrix[0:3, 0:3] = camList[cam_idx]['R'].T
        proj_matrix[:, 3] = camList[cam_idx]['t']
        proj_matrix = np.matmul(camList[cam_idx]['K'].T, proj_matrix)
        p0 = proj_matrix[0]
        p1 = proj_matrix[1]
        p2 = proj_matrix[2]    
        x, y = undistorted_pts[cam_idx, center_idx,  :]
        A[cam_idx*2] = y * p2 - p1
        A[cam_idx*2 + 1] = x * p2 - p0

    w, u, vt = cv2.SVDecomp(A)
    X =  vt.T
    X = X[:, -1]
    X = X / X[-1]
    X = X[:3]
    pts_3d.append(X)

# 4 * 3 
pts_3d = np.asarray(pts_3d)
print("3d aruco centers: \n", pts_3d)

with open(args.root_dir + "/results/aruco_center_3d.pkl", 'wb') as f:
    pkl.dump(pts_3d, f)
