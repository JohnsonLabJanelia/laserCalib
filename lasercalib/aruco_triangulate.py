import numpy as np
import pickle as pkl
import cv2 
import argparse
import pprint

parser = argparse.ArgumentParser()
parser.add_argument('--root_dir', type=str, required=True)
parser.add_argument('--n_cams', type=int, required=True)
parser.add_argument('--side_len', type=float, required=120.0)
parser.add_argument('--load_matlab', type=bool, default=False)

args = parser.parse_args()
nCams = args.n_cams

with open(args.root_dir + "/results/centroids.pkl", 'rb') as file:
    centroids_dict = pkl.load(file)
cam_names = centroids_dict['cam_names']

if args.load_matlab:
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
else:
    with open(args.root_dir + "/results/calibration.pkl", "rb") as f:
        camList = pkl.load(f)

aruco_loc = []
for i in range(nCams):
    with open(args.root_dir + "/results/aruco_corners/{}_aruco.pkl".format(cam_names[i]), 'rb') as f:
        one_camera = pkl.load(f)
        aruco_loc.append(one_camera)

# TODO: pass marker ids 
marker_ids = [0, 1, 2, 3]

# organize data
features = {}
for mk_idx in marker_ids:
    # organize marker features
    pts_list = []
    cam_id_list = []
    for cam_idx in range(nCams):
        if mk_idx in aruco_loc[cam_idx].keys():
            pts_list.append(aruco_loc[cam_idx][mk_idx])
            cam_id_list.append(cam_idx)
    pts_list = np.asarray(pts_list)
    cam_id_list = np.asarray(cam_id_list)
    dict_per_marker = {'pts': pts_list, 'cam_ids': cam_id_list}
    features[mk_idx] = dict_per_marker


for mk_idx in features.keys():
    undistorted_pts = []
    for idx in range(len(features[mk_idx]['cam_ids'])):
        cam_idx = features[mk_idx]['cam_ids'][idx]
        distortion = np.zeros((5,1))
        distortion[0:2, 0] = camList[cam_idx]['d']
        original_points = features[mk_idx]['pts'][idx]
        ideal_points = cv2.undistortPoints(original_points, camList[cam_idx]['K'].T, distortion, None, camList[cam_idx]['K'].T)[:, 0, :]
        undistorted_pts.append(ideal_points)
    features[mk_idx]['pts_undistorted'] = np.asarray(undistorted_pts)

features_3d = {} 
for mk_idx in features.keys():
    features_3d_per_marker = []
    undistorted_pts = features[mk_idx]['pts_undistorted']
    camera_list = features[mk_idx]['cam_ids']
    num_views = undistorted_pts.shape[0]
    for pts_idx in range(4):
        # triangulation 
        # https://filebox.ece.vt.edu/~jbhuang/teaching/ece5554-4554/fa17/lectures/Lecture_15_StructureFromMotion.pdf
        # https://www.cs.cmu.edu/~16385/s17/Slides/11.4_Triangulation.pdf
        A = np.zeros([num_views*2, 4])
        for idx in range(num_views):
            cam_idx = camera_list[idx]
            proj_matrix = np.zeros((3, 4))
            proj_matrix[0:3, 0:3] = camList[cam_idx]['R'].T
            proj_matrix[:, 3] = camList[cam_idx]['t']
            proj_matrix = np.matmul(camList[cam_idx]['K'].T, proj_matrix)
            p0 = proj_matrix[0]
            p1 = proj_matrix[1]
            p2 = proj_matrix[2]    
            x, y = undistorted_pts[idx, pts_idx]
            A[idx*2] = y * p2 - p1
            A[idx*2 + 1] = x * p2 - p0
        w, u, vt = cv2.SVDecomp(A)
        X =  vt.T
        X = X[:, -1]
        X = X / X[-1]
        X = X[:3]
        features_3d_per_marker.append(X)
    features_3d[mk_idx] = np.asarray(features_3d_per_marker)


features_center_3d = {}
for mk_idx in features.keys():
    features_3d_center_per_marker = []
    undistorted_pts = features[mk_idx]['pts_undistorted']
    camera_list = features[mk_idx]['cam_ids']
    num_views = undistorted_pts.shape[0]
    pts_center = undistorted_pts.mean(axis=1)
    A = np.zeros([num_views*2, 4])
    for idx in range(num_views):
        cam_idx = camera_list[idx]
        proj_matrix = np.zeros((3, 4))
        proj_matrix[0:3, 0:3] = camList[cam_idx]['R'].T
        proj_matrix[:, 3] = camList[cam_idx]['t']
        proj_matrix = np.matmul(camList[cam_idx]['K'].T, proj_matrix)
        p0 = proj_matrix[0]
        p1 = proj_matrix[1]
        p2 = proj_matrix[2]  
        x, y = pts_center[idx]
        A[idx*2] = y * p2 - p1
        A[idx*2 + 1] = x * p2 - p0
    w, u, vt = cv2.SVDecomp(A)
    X =  vt.T
    X = X[:, -1]
    X = X / X[-1]
    X = X[:3]
    features_center_3d[mk_idx] = X

print("Marker ID: center coordinats:")
pp = pprint.PrettyPrinter(depth=4)
pp.pprint(features_center_3d)
print("Marker ID: corner coordinats:")
pp.pprint(features_3d)

delta_pts = []
for mk_idx in features_3d.keys():
    delta_pts.append(features_3d[mk_idx][0,:] - features_3d[mk_idx][1,:])
    delta_pts.append(features_3d[mk_idx][1,:] - features_3d[mk_idx][2,:])
    delta_pts.append(features_3d[mk_idx][2,:] - features_3d[mk_idx][3,:])
    delta_pts.append(features_3d[mk_idx][3,:] - features_3d[mk_idx][0,:])

pts = []
for pt in delta_pts:
    pts.append(np.linalg.norm(pt))
    print("side length (mm) :", np.linalg.norm(pt))

print("Ratio of real to estimated side length of aruco marker: ", args.side_len/np.mean(pts))
with open(args.root_dir + "/results/aruco_corners_3d.pkl", 'wb') as f:
    pkl.dump(features_3d, f)

with open(args.root_dir + "/results/aruco_center_3d.pkl", 'wb') as f:
    pkl.dump(features_center_3d, f)