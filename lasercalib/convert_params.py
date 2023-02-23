import numpy as np
import pickle as pkl
from rigid_body import get_inverse_transformation
from scipy.spatial.transform import Rotation as R

def convertParams(camParams):
    allParams = np.full((len(camParams), 11), np.NaN)
    for nCam in range(len(camParams)):
        p = camParams[nCam][0]
        f = p['K'][0,0]/2 + p['K'][1,1]/2
        r = -R.from_matrix(p['r']).as_rotvec()
        t = p['t']
        c = p['K'][2,0:2]
        d = p['RDistort']
        allParams[nCam,:] = np.hstack((r,t,f,d,c))
    return allParams

def unconvertParams(camParamVec):
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
    with open(filename, 'rb') as file: 
        camera_params = pkl.load(file)

    # import camera parameters from blender to cameraArray for pySBA
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