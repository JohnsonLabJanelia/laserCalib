import numpy as np
import pickle as pkl
import cv2 
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--root_dir', type=str, required=True)
parser.add_argument('--n_cams', type=int, required=True)
parser.add_argument('--side_len', type=float, required=120.0)

args = parser.parse_args()

nCams = args.n_cams

camList = []
for i in range(nCams):
    cam_params = {}
    filename = args.root_dir + "/calibration_aruco/Cam{}.yaml".format(i)
    print(filename)
    fs = cv2.FileStorage(filename, cv2.FILE_STORAGE_READ)
    cam_params['camera_matrix'] = fs.getNode("camera_matrix").mat()
    cam_params['distortion_coefficients'] = fs.getNode("distortion_coefficients").mat()
    cam_params['tc_ext'] = fs.getNode("tc_ext").mat()
    cam_params['rc_ext'] = fs.getNode("rc_ext").mat()
    camList.append(cam_params)


aruco_loc = []
for i in range(nCams):
    with open(args.root_dir + "/results/aruco_corners/Cam{}_aruco.pkl".format(i), 'rb') as f:
        one_camera = pkl.load(f)
        aruco_loc.append(one_camera)

features = []
for cam in aruco_loc:
    temp_list = []
    for marker_id, marker_coord in cam.items():
        temp_list.append(marker_coord)
    features.append(np.asarray(temp_list))
features = np.asarray(features)


undistorted_pts = []
for cam_idx in range(nCams):
    temp_list = []
    for mk_idx in range(4):        
        for corner in range(4):
            ideal_points = cv2.undistortPoints(features[cam_idx][mk_idx][corner], camList[cam_idx]['camera_matrix'], camList[cam_idx]['distortion_coefficients'], None, camList[cam_idx]['camera_matrix'])[:, 0, :]
            temp_list.append(ideal_points[0])
    
    undistorted_pts.append(np.asarray(temp_list))

undistorted_pts = np.asarray(undistorted_pts)    
        
features_3d = []

# for all corners of all marker (4*num_markers)
for pts in range(undistorted_pts.shape[1]):

    # triangulation 
    # https://filebox.ece.vt.edu/~jbhuang/teaching/ece5554-4554/fa17/lectures/Lecture_15_StructureFromMotion.pdf
    # https://www.cs.cmu.edu/~16385/s17/Slides/11.4_Triangulation.pdf
    A = np.zeros([nCams*2, 4])
    for cam_idx in range(nCams):
        proj_matrix = np.zeros((3, 4))
        proj_matrix[0:3, 0:3] = camList[cam_idx]['rc_ext']
        proj_matrix[:, 3] = camList[cam_idx]['tc_ext'][:, 0]
        proj_matrix = np.matmul(camList[cam_idx]['camera_matrix'], proj_matrix)
        p0 = proj_matrix[0]
        p1 = proj_matrix[1]
        p2 = proj_matrix[2]    
        x, y = undistorted_pts[cam_idx, pts,  :]
        A[cam_idx*2] = y * p2 - p1
        A[cam_idx*2 + 1] = x * p2 - p0

    w, u, vt = cv2.SVDecomp(A)
    X =  vt.T
    X = X[:, -1]
    X = X / X[-1]
    X = X[:3]
    features_3d.append(X)

# 4 * 3 
features_3d = np.asarray(features_3d)
features_3d = np.reshape(features_3d,(4,4,3))
print(features_3d)


delta_pts = []
for mk_idx in range(4):
    delta_pts.append(features_3d[mk_idx,0,:] - features_3d[mk_idx,1,:])
    delta_pts.append(features_3d[mk_idx,1,:] - features_3d[mk_idx,2,:])
    delta_pts.append(features_3d[mk_idx,2,:] - features_3d[mk_idx,3,:])
    delta_pts.append(features_3d[mk_idx,3,:] - features_3d[mk_idx,0,:])

pts = []
for pt in delta_pts:
    pts.append(np.linalg.norm(pt))
    print("side length (mm) :", np.linalg.norm(pt))

print("Ratio of real to estimated side length of aruco marker: ", args.side_len/np.mean(pts))
with open(args.root_dir + "/results/aruco_corners_3d.pkl", 'wb') as f:
    pkl.dump(features_3d, f)