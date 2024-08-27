import numpy as np
import cv2 as cv

def invert_Rt(R, t):
    Ri = R.T
    ti = np.dot(-Ri, t)
    return Ri, ti

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
    if np.linalg.matrix_rank(H) < 3:
       print("rank of H = {}, expecting 3".format(np.linalg.matrix_rank(H)))

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

def average_distance(X, Y):
    return np.linalg.norm(X-Y, axis=1).mean()


def apply_rigid_transform(X, R, t, scale):
    return np.dot(X*scale, R.T) + t[None]

def procrustes_registration(src, dst):
    """
    Estimates rotation translation and scale of two point sets
    using Procrustes analysis
    
    dst = (src*scale x R.T) + t + residuals
    
    Parameters:
    ----------
    src : numpy.ndarray (N,3)
        transformed points set
    dst : numpy.ndarray (N,3)
        target points set   
        
    Return:
    -------
    scale, rotation matrix, translation and average distance
    between the alligned points sets
    """
    from scipy.linalg import orthogonal_procrustes
    
    assert src.shape[0]==dst.shape[0]
    assert src.shape[1]==dst.shape[1]
    assert src.shape[1]==3    

    P = src.copy()
    Q = dst.copy()

    m1 = np.mean(P, 0) 
    m2 = np.mean(Q, 0)

    P -= m1
    Q -= m2

    norm1 = np.linalg.norm(P)
    norm2 = np.linalg.norm(Q)

    if norm1 == 0 or norm2 == 0:
        raise ValueError("Input matrices must contain >1 unique points")

    # change scaling of data (in rows) such that trace(mtx*mtx') = 1
    P /= norm1
    Q /= norm2
    R, s = orthogonal_procrustes(Q, P)
    
    scale = s*norm2/norm1
    t = m2-np.dot(m1*scale, R.T)
    
    mean_dist = average_distance(apply_rigid_transform(src, R, t, scale), dst)
    
    return scale, R, t, mean_dist


def point_set_registration(src, dst, fixed_scale=None, verbose=True):
    """
    code adapted from
    https://github.com/cvlab-epfl/multiview_calib

    """
    from scipy.optimize import minimize
    assert src.shape[0] == dst.shape[0]
    assert src.shape[1] == dst.shape[1]
    assert src.shape[1] == 3 
    
    def pack_params(R, t, scale):
        rvec = cv.Rodrigues(R)[0]
        if fixed_scale is not None:
            return np.concatenate([rvec.ravel(), t, [fixed_scale]])
        else:
            return np.concatenate([rvec.ravel(), t, [scale]])
    
    def unpack_params(params):
        R, t, scale = cv.Rodrigues(params[:3])[0], params[3:6], params[-1]
        if fixed_scale is not None:
            return R, t, fixed_scale
        else:
            return R, t, scale    

    _src, _dst = src.copy().astype(np.float32), dst.copy().astype(np.float32)
    
    if fixed_scale is not None:
        _, R, t, _ = procrustes_registration(_src*fixed_scale, _dst)
        scale = fixed_scale
    else:
        scale, R, t, _ = procrustes_registration(_src, _dst)
        
    mean_dist = average_distance(apply_rigid_transform(_src, R, t, scale), _dst)
    
    if verbose:
        print("Initial guess using Procrustes registration:")
        print("\t Mean error distance: {:0.3f} [unit of destination (dst) point set]".format(mean_dist))
        
    if np.linalg.det(R)<0:
        print("!"*20)
        print("Procrusted produced a rotation matrix with negative determinant.")
        print("This implies that the coordinate systems of src and dst have different handedness.")
        print("To fix this you have to flip one or more of the axis of your input..for example by negating them.")
        print("!"*20)
    
    def funct(x):
        R, t, scale = unpack_params(x)
        src_transf = apply_rigid_transform(_src, R, t, scale)
        return average_distance(src_transf, _dst)  

    x0 = pack_params(R, t, scale)        
    res = minimize(funct, x0, method='Nelder-Mead', 
                   options={'maxiter':10000, 'disp':True}, 
                   tol=1e-24)
    
    R, t, scale = unpack_params(res.x)
    mean_dist = average_distance(apply_rigid_transform(_src, R, t, scale), _dst) 

    if verbose:
        print("Final registration:")
        print("\t Mean error distance: {:0.3f} [unit of destination (dst) point set]".format(mean_dist))     
    return scale, R, t, mean_dist

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
