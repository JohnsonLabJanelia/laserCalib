import numpy as np
import pickle as pkl
from lasercalib.rigid_body import get_inverse_transformation
from scipy.spatial.transform import Rotation as R
import cv2

def readable_to_red_format(camList):
    outParams = np.full((len(camList), 25), np.NaN)
    for nCam in range(len(camList)):
        p = camList[nCam]
        k = np.transpose(p['K']).ravel()
        r_m = np.transpose(p['R']).ravel()
        t = p['t']
        d = np.hstack((p['d'], np.array([0.0, 0.0])))
        outParams[nCam,:] = np.hstack((k, r_m, t, d))
    return outParams

def sba_to_readable_format(camParamVec):
    thisK = np.full((3, 3), 0.0)
    thisK[0, 0] = camParamVec[6]
    thisK[1,1] = camParamVec[6] 
    thisK[2,2] = 1
    thisK[2,:2] = camParamVec[9:]
    r = R.from_rotvec(-camParamVec[:3]).as_matrix()
    t = camParamVec[3:6]
    d = camParamVec[7:9]
    return {'K': thisK, 'R':r, 't':t, 'd':d}

def getCameraArray(allCameras = ['lBack', 'lFront', 'lTop', 'rBack', 'rFront', 'rTop']):
    # Camera parameters are 3 rotation angles, 3 translations, 1 focal distance, 2 distortion params, and x,y principal points
    # Following notes outlined in evernote, 'bundle adjustment', later updated using optimized values
    camMatDict = {
        'lBack': np.array([0.86, -1.95, 1.69, 0.012, 0.091, 1.38, 1779, -0.021, -0.026, 1408, 704]),
        'lFront': np.array([1.96, -.66, .72, -0.039, .068, 1.40, 1779, -0.021, -0.026, 1408, 704]),
        'lTop': np.array([1.92, -1.77, 0.84, -.038, 0.039, 1.69, 1779, -0.021, -0.026, 1408, 848]),
        'rBack': np.array([0.96, 2.14, -1.67, 0.035, 0.077, 1.42, 1779, -0.021, -0.026, 1408, 704]),
        'rFront': np.array([1.966, .84, -.64, 0.056, 0.1399, 1.48, 1779, -0.021, -0.026, 1408, 704]),
        'rTop': np.array([2.02, 1.95, -0.71, 0.0377, 0.0047, 1.74, 1779, -0.021, -0.026, 1408, 848]),
    }
    cameraArray = np.full((len(allCameras), 11), np.NaN)
    for i, e in enumerate(allCameras):
        cameraArray[i,:] = camMatDict[e]
    return cameraArray

def load_from_blender(filename, nCams):
    # import camera parameters from blender to cameraArray for pySBA
    with open(filename, 'rb') as file: 
        camera_params = pkl.load(file)
    cameraArray = np.zeros(shape=(nCams, 11))
    for i in range(nCams):
        rotation_matrix = np.zeros((3, 3))
        rotation_matrix[:, 0] = camera_params[i]['3x3'][:, 0] 
        rotation_matrix[:, 1] = -camera_params[i]['3x3'][:, 1] 
        rotation_matrix[:, 2] = -camera_params[i]['3x3'][:, 2] 
        calib_Rmatrix, calib_P = get_inverse_transformation(rotation_matrix, camera_params[i]['location'] * 1000)
        calib_R = R.from_matrix(calib_Rmatrix).as_rotvec()
        cameraArray[i][0:3] = calib_R
        cameraArray[i][3:6] = calib_P
        cameraArray[i][6:9] = [1500, 0, 0]
        cameraArray[i][9:11] = [1604, 1100]
    return cameraArray

def initialize_from_checkerboard(filedir, nCams, cam_names):
    # load files 
    calib_data_all = []
    for idx in range(nCams):
        fs = cv2.FileStorage(filedir + "/{}.yaml".format(cam_names[idx]), cv2.FILE_STORAGE_READ)
        one_calib = {
            "camera_matrix": fs.getNode("camera_matrix").mat(),
            "distortion_coefficients": fs.getNode("distortion_coefficients").mat(),
            "R": fs.getNode("rc_ext").mat(),
            "T": fs.getNode("tc_ext").mat()
        }
        calib_data_all.append(one_calib)

    cameraArray = np.zeros(shape=(nCams, 11))
    for i in range(nCams):
        calib_R = R.from_matrix(calib_data_all[i]['R']).as_rotvec()
        cameraArray[i][0:3] = calib_R
        cameraArray[i][3:6] = calib_data_all[i]['T'][:, 0]
        cameraArray[i][6:9] = [calib_data_all[i]["camera_matrix"][0, 0], calib_data_all[i]["distortion_coefficients"][0, 0], calib_data_all[i]["distortion_coefficients"][1, 0]]
        cameraArray[i][9:11] = [calib_data_all[i]["camera_matrix"][0, 2], calib_data_all[i]["camera_matrix"][1, 2]]
    return cameraArray

def red_to_aruco(save_root, nCams, cam, cam_names):
    ## Depreciate 
    for cam_idx in range(nCams):
        intrinsicMatrix = np.asarray(cam[cam_idx][0:9]).reshape(3, 3)
        intrinsicMatrix = intrinsicMatrix

        distortionCoefficients = cam[cam_idx][21:25]
        distortionCoefficients = np.asarray(distortionCoefficients).reshape(4, 1)

        # save it using opencv 
        output_filename = save_root + '{}.yaml'.format(cam_names[cam_idx])
        s = cv2.FileStorage(output_filename, cv2.FileStorage_WRITE)
        s.write('image_width', 3208)
        s.write('image_height', 2200)

        s.write('camera_matrix', intrinsicMatrix)
        s.write('distortion_coefficients', distortionCoefficients)
        s.release()


def readable_format_to_aruco_format(save_root, nCams, camList, cam_names):
    for i in range(nCams):
        output_filename = save_root + '{}.yaml'.format(cam_names[i])
        s = cv2.FileStorage(output_filename, cv2.FileStorage_WRITE)
        s.write('camera_matrix', camList[i]['K'].T)
        s.write('distortion_coefficients', np.asarray([camList[i]['d'][0], camList[i]['d'][1], 0, 0, 0]))
        s.write('rc_ext', camList[i]['R'].T)
        s.write('tc_ext', camList[i]['t'])
        s.release()

def save_aruco_format(save_root, nCams, aruco_cam_list, cam_names):
    for i in range(nCams):
        output_filename = save_root + '{}.yaml'.format(cam_names[i])
        s = cv2.FileStorage(output_filename, cv2.FileStorage_WRITE)
        s.write('camera_matrix', aruco_cam_list[i]['camera_matrix'])
        s.write('distortion_coefficients', aruco_cam_list[i]['distortion_coefficients'])
        s.write('rc_ext', aruco_cam_list[i]['rc_ext'])
        s.write('tc_ext', aruco_cam_list[i]['tc_ext'])
        s.release()