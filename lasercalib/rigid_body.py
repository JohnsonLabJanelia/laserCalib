import numpy as np
import cv2 as cv

def get_inverse_transformation(rotation_matrix, translation):
    r_inv = rotation_matrix.T    
    t_inv = -np.matmul(rotation_matrix.T, translation)
    return r_inv, t_inv

def rigid_transform_3D(A, B):
    # https://nghiaho.com/?page_id=671
    # https://github.com/nghiaho12/rigid_transform_3D/blob/master/rigid_transform_3D.py
    # Input: expects 3xN matrix of points
    # Returns R,t
    # R = 3x3 rotation matrix
    # t = 3x1 column vector

    assert A.shape == B.shape

    num_rows, num_cols = A.shape
    if num_rows != 3:
        raise Exception(f"matrix A is not 3xN, it is {num_rows}x{num_cols}")

    num_rows, num_cols = B.shape
    if num_rows != 3:
        raise Exception(f"matrix B is not 3xN, it is {num_rows}x{num_cols}")

    # find mean column wise
    centroid_A = np.mean(A, axis=1)
    centroid_B = np.mean(B, axis=1)

    # ensure centroids are 3x1
    centroid_A = centroid_A.reshape(-1, 1)
    centroid_B = centroid_B.reshape(-1, 1)

    # subtract mean
    Am = A - centroid_A
    Bm = B - centroid_B

    H = Am @ np.transpose(Bm)

    # sanity check
    #if linalg.matrix_rank(H) < 3:
    #    raise ValueError("rank of H = {}, expecting 3".format(linalg.matrix_rank(H)))

    # find rotation
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    # special reflection case
    if np.linalg.det(R) < 0:
        print("det(R) < R, reflection detected!, correcting for it ...")
        Vt[2,:] *= -1
        R = Vt.T @ U.T

    t = -R @ centroid_A + centroid_B

    return R, t


def load_camera_parameters_from_yaml(file_name):
    fs = cv.FileStorage(file_name, cv.FILE_STORAGE_READ)
    one_calib = {
        "camera_matrix": fs.getNode("camera_matrix").mat(),
        "distortion_coefficients": fs.getNode("distortion_coefficients").mat(),
        "rc_ext": fs.getNode("rc_ext").mat(),
        "tc_ext": fs.getNode("tc_ext").mat()
    }
    return one_calib

def Project(points, intrinsic, distortion, rotation_matrix, tvec):
    result = []
    if len(points) > 0:
        result, _ = cv.projectPoints(points, rotation_matrix, tvec,
                                  intrinsic, distortion)
    return np.squeeze(result, axis=1)

def Unproject(points, Z, intrinsic, distortion, rotation_matrix, tvec):
    """
    args: 
        points, [num_pts, coordinates (2d)] 
        Z, one number or [num_pts], select a plane or planes
        intrinsic: camera intrinsics
        distortion: camera distortion
        rotation_matrix: camera rotation matrix 
        tvec: camera translation vector
    return:
        world points, [num_pts, coordinates (3d)]
    """
    f_x = intrinsic[0, 0]
    f_y = intrinsic[1, 1]
    c_x = intrinsic[0, 2]
    c_y = intrinsic[1, 2]
   
    points_undistorted = np.array([])
    if len(points) > 0:
        points_undistorted = cv.undistortPoints(np.expand_dims(points, axis=1), intrinsic, distortion, P=intrinsic)
    points_undistorted = np.squeeze(points_undistorted, axis=1)
    
    temp_pts = []
    for idx in range(points_undistorted.shape[0]):
        u = (points_undistorted[idx, 0] - c_x) / f_x
        v = (points_undistorted[idx, 1] - c_y) / f_y
        temp_pts.append([u, v , 1])
    temp_pts = np.asarray(temp_pts)
    temp_pts = temp_pts.T
    left_eqn = np.matmul(rotation_matrix.T, temp_pts)
    right_eqn0 = np.matmul(rotation_matrix.T, tvec)

    z_camera = []
    for idx in range(points_undistorted.shape[0]):
        z_world = Z[0] if len(Z) == 1 else Z[idx]
        z_camera.append((z_world + right_eqn0[2, 0]) / left_eqn[2, idx])
    z_camera = np.asarray([z_camera])
    pts_world = np.matmul(rotation_matrix.T, temp_pts * z_camera - tvec)
    return pts_world.T
