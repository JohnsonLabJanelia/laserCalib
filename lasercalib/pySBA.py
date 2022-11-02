"""
MIT License (MIT)
Copyright (c) FALL 2016, Jahdiel Alvarez
Author: Jahdiel Alvarez
Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
documentation files (the "Software"), to deal in the Software without restriction, including without limitation
the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be
included in all copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE
FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
Based on Scipy's cookbook:
http://scipy-cookbook.readthedocs.io/items/bundle_adjustment.html
"""

import numpy as np
from scipy.sparse import lil_matrix
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation as R

#%%
class PySBA:
    """Python class for Simple Bundle Adjustment"""

    def __init__(self, cameraArray, points3D, points2D, cameraIndices, point2DIndices, points3Dfixed=None, pointWeights=None):
        """Intializes all the class attributes and instance variables.
            Write the specifications for each variable:
            cameraArray with shape (n_cameras, 11) contains initial estimates of parameters for all cameras.
                    First 3 components in each row form a rotation vector,
                    next 3 components form a translation vector,
                    then a focal distance and two distortion parameters,
                    then x,y image center coordinates
            points_3d with shape (n_points, 3)
                    contains initial estimates of point coordinates in the world frame.
            camera_ind with shape (n_observations,)
                    contains indices of cameras (from 0 to n_cameras - 1) involved in each observation.
            point_ind with shape (n_observations,)
                    contatins indices of points (from 0 to n_points - 1) involved in each observation.
            points_2d with shape (n_observations, 2)
                    contains measured 2-D coordinates of points projected on images in each observations.
            points_3d_fixed with shape (n_fixed_points, 3)
                    contains estimate for 3D points from a CAD model of the rig
            pointWeights with shape (n_observations, )
                    contains cost function weights for each observation point.

        """
        self.cameraArray = cameraArray
        self.points3D = points3D
        self.points2D = points2D
        self.cameraIndices = cameraIndices
        self.point2DIndices = point2DIndices
        self.points3Dfixed = points3Dfixed
        if pointWeights is None:
            pointWeights = np.full_like(point2DIndices, 1)
        self.pointWeights = pointWeights.reshape((-1, 1))
        self.points3Dfixed_labeled = None

    def rotate(self, points, rot_vecs):
        """Rotate points by given rotation vectors.
        Rodrigues' rotation formula is used.
        """
        theta = np.linalg.norm(rot_vecs, axis=1)[:, np.newaxis]
        with np.errstate(invalid='ignore'):
            v = rot_vecs / theta
            v = np.nan_to_num(v)
        dot = np.sum(points * v, axis=1)[:, np.newaxis]
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)

        return cos_theta * points + sin_theta * np.cross(v, points) + dot * (1 - cos_theta) * v


    def project(self, points, cameraArray):
        """Convert 3-D points to 2-D by projecting onto images."""
        points_proj = self.rotate(points, cameraArray[:, :3])
        points_proj += cameraArray[:, 3:6]
        points_proj = points_proj[:, :2] / points_proj[:, 2, np.newaxis]
        # points_proj -= cameraArray[:, 9:] / 1778
        f = cameraArray[:, 6]
        k1 = cameraArray[:, 7]
        k2 = cameraArray[:, 8]
        n = np.sum(points_proj ** 2, axis=1)
        r = 1 + k1 * n + k2 * n ** 2
        points_proj *= (r * f)[:, np.newaxis]
        points_proj += cameraArray[:, 9:]
        return points_proj


    def fun(self, params, n_cameras, n_points, camera_indices, point_indices, points_2d, pointWeights):
        """Compute residuals.
        `params` contains camera parameters and 3-D coordinates.
        """
        nCamParams = 11
        camera_params = params[:n_cameras * nCamParams].reshape((n_cameras, nCamParams))
        points_3d = params[n_cameras * nCamParams:].reshape((n_points, 3))
        points_proj = self.project(points_3d[point_indices], camera_params[camera_indices])
        weighted_residual = pointWeights*(points_proj-points_2d)
        return weighted_residual.ravel()

    def bundle_adjustment_sparsity(self, numCameras, numPoints, cameraIndices, pointIndices):
        nCamParams = 11
        m = cameraIndices.size * 2
        n = numCameras * nCamParams + numPoints * 3
        A = lil_matrix((m, n), dtype=int)

        i = np.arange(cameraIndices.size)
        for s in range(nCamParams):
            A[2 * i, cameraIndices * nCamParams + s] = 1
            A[2 * i + 1, cameraIndices * nCamParams + s] = 1

        for s in range(3):
            A[2 * i, numCameras * nCamParams + pointIndices * 3 + s] = 1
            A[2 * i + 1, numCameras * nCamParams + pointIndices * 3 + s] = 1

        return A


    def optimizedParams(self, params, n_cameras, n_points):
        """
        Retrieve camera parameters and 3-D coordinates.
        """
        nCamParams = 11
        camera_params = params[:n_cameras * nCamParams].reshape((n_cameras, nCamParams))
        points_3d = params[n_cameras * nCamParams:].reshape((n_points, 3))

        return camera_params, points_3d


    def bundleAdjust(self):
        """ Returns the bundle adjusted parameters, in this case the optimized
         rotation and translation vectors. """
        numCameras = self.cameraArray.shape[0]
        numPoints = self.points3D.shape[0]

        x0 = np.hstack((self.cameraArray.ravel(), self.points3D.ravel()))
        A = self.bundle_adjustment_sparsity(numCameras, numPoints, self.cameraIndices, self.point2DIndices)

        #res = least_squares(self.fun, x0, jac_sparsity=A, verbose=2, x_scale='jac', ftol=1e-4, method='trf',
        #                    args=(numCameras, numPoints, self.cameraIndices, self.point2DIndices, self.points2D))
        res = least_squares(self.fun, x0, jac_sparsity=A, verbose=2, x_scale='jac', ftol=1e-7, method='trf', jac='3-point',
                            args=(numCameras, numPoints, self.cameraIndices, self.point2DIndices, self.points2D, self.pointWeights))

        camera_params, points_3d = self.optimizedParams(res.x, numCameras, numPoints)
        self.cameraArray = camera_params
        self.points3D = points_3d

        return res





    # added by RJ
    def fun_camonly(self, params, n_cameras, n_points, camera_indices, point_indices, points_2d, pointWeights, points_3d):
        """Compute residuals.
        'params' contains camera extrinsic only"""
        nCamParams = 11
        camera_params = params.reshape(n_cameras, nCamParams)
        points_proj = self.project(points_3d[point_indices], camera_params[camera_indices])
        # weighted_residual = pointWeights*(points_proj-points_2d) ** 2
        weighted_residual = pointWeights*(points_proj-points_2d) ** 2
        return weighted_residual.ravel()


    # added by RJ
    def bundle_adjustment_camonly(self):
        """ Returns the bundle adjusted parameters, in this case the optimized rotation and translation vectors"""
        numCameras = self.cameraArray.shape[0]
        numCamParams = 11
        numPoints = self.points3D.shape[0]

        x0 = self.cameraArray.ravel()
        # A = self.bundle_adjustment_sparsity(numCameras, numPoints, self.cameraIndices, self.point2DIndices)

        # res = least_squares(self.fun_camonly, x0, jac_sparsity=A, verbose=2, x_scale='jac', ftol=1e-6, method='trf', jac='3-point',
        #                     args=(numCameras, numPoints, self.cameraIndices, self.point2DIndices, self.points2D, self.pointWeights, self.points3D))
        res = least_squares(self.fun_camonly, x0, verbose=2, ftol=1e-7, method='trf',
                            args=(numCameras, numPoints, self.cameraIndices, self.point2DIndices, self.points2D, self.pointWeights, self.points3D))

        self.cameraArray = res.x.reshape(numCameras, numCamParams)
        return res



    # added by RJ
    def fun_camonly_shared(self, params, n_cameras, n_points, camera_indices, point_indices, points_2d, pointWeights, points_3d, intrinsics):
        """Compute residuals.
        'params' contains camera extrinsic only"""
        nCamIntrinsic = 3
        nCamExtrinsic = 6
        nCamCentroid = 2
        nCamUnique = nCamExtrinsic + nCamCentroid
        nCamParams = n_cameras * nCamUnique + nCamIntrinsic

        cam_shared_intrinsic = params[:nCamIntrinsic]
        camera_extrinsic = params[nCamIntrinsic:nCamIntrinsic+n_cameras*nCamExtrinsic].reshape((n_cameras, nCamExtrinsic))
        camera_centroid = params[nCamIntrinsic+n_cameras*nCamExtrinsic : nCamParams].reshape((n_cameras, nCamCentroid))
        camera_params = np.concatenate((camera_extrinsic, np.tile(cam_shared_intrinsic, (n_cameras,1)), camera_centroid), axis=1)

        points_proj = self.project(points_3d[point_indices], camera_params[camera_indices])
        weighted_residual = pointWeights * (points_proj - points_2d)
        return weighted_residual.ravel()


    # added by RJ
    def bundle_adjustment_camonly_shared(self):
        """ Returns the bundle adjusted parameters, in this case the optimized rotation and translation vectors"""
        numCameras = self.cameraArray.shape[0]
        numPoints = self.points3D.shape[0]

        nCamIntrinsic = 3
        nCamExtrinsic = 6
        nCamCentroid = 2
        nCamUnique = nCamExtrinsic + nCamCentroid
        nCamParams = numCameras * nCamUnique + nCamIntrinsic

        camera_shared_intrinsic = np.mean(self.cameraArray[:, 6:9], axis=0).ravel()
        camera_extrinsic = self.cameraArray[:,:6].ravel()
        camera_centroids = self.cameraArray[:,9:].ravel()

        x0 = np.hstack((camera_shared_intrinsic, camera_extrinsic, camera_centroids))
        
        res = least_squares(self.fun_camonly_shared, x0, verbose=2, ftol=1e-6, method='trf',
                            args=(numCameras, numPoints, self.cameraIndices, self.point2DIndices, self.points2D, self.pointWeights, self.points3D, camera_shared_intrinsic))

        cam_shared_intrinsic = res.x[:nCamIntrinsic]
        camera_extrinsic = res.x[nCamIntrinsic:nCamIntrinsic + numCameras * nCamExtrinsic].reshape((numCameras, nCamExtrinsic))
        camera_centroid = res.x[nCamIntrinsic+numCameras*nCamExtrinsic : nCamParams].reshape((numCameras, nCamCentroid))
        camera_params = np.concatenate((camera_extrinsic, np.tile(cam_shared_intrinsic, (numCameras, 1)), camera_centroid), axis=1)

        self.cameraArray = camera_params
        return res

    def fun_transform_points_3d(self, params, numCameras, n_points, camera_params, camera_indices, point_indices, points_2d, pointWeights, points_3d):
        r = params.reshape(3,4)
        r = np.vstack((r, [0, 0, 0, 1]))
        input_pts = points_3d.transpose()
        padding = np.ones(shape=(1, n_points))
        input_pts = np.vstack((input_pts, padding))
        transformed_points = np.dot(r, input_pts)
        transformed_points3d = transformed_points.transpose()[:,:3]
        points_proj = self.project(transformed_points3d[point_indices], camera_params[camera_indices])
        weighted_residual = pointWeights*(points_proj-points_2d) ** 2
        # weighted_residual = pointWeights*(points_proj-points_2d)
        return weighted_residual.ravel()

    
    def bundleAdjust_transform_points_3d(self):
        numCameras = self.cameraArray.shape[0]
        numPoints = self.points3D.shape[0]
        x0 = np.hstack((np.eye(3), np.zeros((3,1)))).ravel()
        res = least_squares(self.fun_transform_points_3d, x0, verbose=2, ftol=1e-3, method='trf',
                            args=(numCameras, numPoints, self.cameraArray, self.cameraIndices, self.point2DIndices, self.points2D, self.pointWeights, self.points3D))

        transformation_matrix = np.vstack((res.x.reshape(3,4), [0, 0, 0, 1]))
        
        input_pts = self.points3D.transpose()
        padding = np.ones(shape=(1, numPoints))
        input_pts = np.vstack((input_pts, padding))
        transformed_points = np.dot(transformation_matrix, input_pts)
        transformed_points3d = transformed_points.transpose()[:,:3]
        self.points3D = transformed_points3d
        return res

    def getResiduals(self):
        """Gets residuals given current camera parameters and 3d locations"""
        numCameras = self.cameraArray.shape[0]
        numPoints = self.points3D.shape[0]
        x0 = np.hstack((self.cameraArray.ravel(), self.points3D.ravel()))
        f0 = self.fun(x0, numCameras, numPoints, self.cameraIndices, self.point2DIndices, self.points2D, np.full_like(self.point2DIndices, 1))
        return f0


    def bundle_adjustment_sparsity_nocam(self, numPoints, pointIndices):
        m = pointIndices.size * 2
        n = numPoints * 3
        A = lil_matrix((m, n), dtype=int)

        i = np.arange(pointIndices.size)
        for s in range(3):
            A[2 * i, pointIndices * 3 + s] = 1
            A[2 * i + 1, pointIndices * 3 + s] = 1

        return A

    def fun_nocam(self, params, camera_params, n_points, camera_indices, point_indices, points_2d, pointWeights):
        """Compute residuals.
        `params` contains 3-D coordinates only.
        """
        points_3d = params.reshape((n_points, 3))
        points_proj = self.project(points_3d[point_indices], camera_params[camera_indices])
        weighted_residual = pointWeights * (points_proj - points_2d)
        return weighted_residual.ravel()

    def bundleAdjust_nocam(self):
        """ Returns the optimized 3d positions given current camera parameters,
        without adjusting the camera parameters themselves. """
        numCameras = self.cameraArray.shape[0]
        numPoints = self.points3D.shape[0]
        camera_params = self.cameraArray

        x0 = self.points3D.ravel()
        A = self.bundle_adjustment_sparsity_nocam(numPoints, self.point2DIndices)
        res = least_squares(self.fun_nocam, x0, jac_sparsity=A, verbose=2, x_scale='jac', ftol=1e-7, method='trf', jac='3-point',
                            args=(camera_params, numPoints, self.cameraIndices, self.point2DIndices, self.points2D, self.pointWeights))
        self.points3D = res.x.reshape((numPoints, 3))

        return res



    def bundle_adjustment_sparsity_sharedcam(self, numCameras, numPoints, cameraIndices, pointIndices):
        m = cameraIndices.size * 2
        nCamIntrinsic = 3
        nCamExtrinsic = 6
        nCamCentroid = 2
        nCamParams = numCameras * nCamExtrinsic + numCameras * nCamCentroid + nCamIntrinsic
        n = nCamParams + numPoints * 3
        A = lil_matrix((m, n), dtype=int)

        i = np.arange(cameraIndices.size)
        A[2*i, 0:nCamIntrinsic] = 1
        A[2*i + 1, 0:nCamIntrinsic] = 1
        for s in range(nCamExtrinsic):
            A[2 * i, nCamIntrinsic + cameraIndices * nCamExtrinsic + s] = 1
            A[2 * i + 1, nCamIntrinsic + cameraIndices * nCamExtrinsic + s] = 1
        for s in range(nCamCentroid):
            A[2 * i, nCamIntrinsic + numCameras*nCamExtrinsic + cameraIndices * nCamCentroid + s] = 1
            A[2 * i + 1, nCamIntrinsic + numCameras*nCamExtrinsic + cameraIndices * nCamCentroid + s] = 1

        for s in range(3):
            A[2 * i, nCamParams + pointIndices * 3 + s] = 1
            A[2 * i + 1, nCamParams + pointIndices * 3 + s] = 1

        return A

    def fun_sharedcam(self, params, n_cameras, n_points, camera_indices, point_indices, points_2d, pointWeights):
        """Compute residuals.
        `params` contains camera parameters and 3-D coordinates.
        """
        nCamIntrinsic = 3
        nCamExtrinsic = 6
        nCamCentroid = 2
        nCamUnique = nCamExtrinsic + nCamCentroid
        nCamParams = n_cameras * nCamUnique + nCamIntrinsic

        cam_shared_intrinsic = params[:nCamIntrinsic]
        camera_extrinsic = params[nCamIntrinsic:nCamIntrinsic+n_cameras*nCamExtrinsic].reshape((n_cameras, nCamExtrinsic))
        camera_centroid = params[nCamIntrinsic+n_cameras*nCamExtrinsic : nCamParams].reshape((n_cameras, nCamCentroid))
        camera_params = np.concatenate((camera_extrinsic, np.tile(cam_shared_intrinsic, (n_cameras,1)), camera_centroid), axis=1)

        points_3d = params[nCamParams:].reshape((n_points, 3))
        points_proj = self.project(points_3d[point_indices], camera_params[camera_indices])
        weighted_residual = pointWeights * (points_proj - points_2d)
        return weighted_residual.ravel()

    def bundleAdjust_sharedcam(self):
        """ Returns the bundle adjusted parameters, in this case the optimized
         rotation and translation vectors. """
        numCameras = self.cameraArray.shape[0]
        numPoints = self.points3D.shape[0]

        nCamIntrinsic = 3
        nCamExtrinsic = 6
        nCamCentroid = 2
        nCamUnique = nCamExtrinsic + nCamCentroid
        nCamParams = numCameras * nCamUnique + nCamIntrinsic

        camera_shared_intrinsic = np.mean(self.cameraArray[:, 6:9], axis=0).ravel()
        camera_extrinsic = self.cameraArray[:,:6].ravel()
        camera_centroids = self.cameraArray[:,9:].ravel()

        x0 = np.hstack((camera_shared_intrinsic, camera_extrinsic, camera_centroids, self.points3D.ravel()))
        A = self.bundle_adjustment_sparsity_sharedcam(numCameras, numPoints, self.cameraIndices, self.point2DIndices)
        res = least_squares(self.fun_sharedcam, x0, jac_sparsity=A, verbose=2, x_scale='jac', ftol=1e-8, method='trf', jac='3-point',
                            args=(numCameras, numPoints, self.cameraIndices, self.point2DIndices, self.points2D, self.pointWeights))

        cam_shared_intrinsic = res.x[:nCamIntrinsic]
        camera_extrinsic = res.x[nCamIntrinsic:nCamIntrinsic + numCameras * nCamExtrinsic].reshape((numCameras, nCamExtrinsic))
        camera_centroid = res.x[nCamIntrinsic+numCameras*nCamExtrinsic : nCamParams].reshape((numCameras, nCamCentroid))
        camera_params = np.concatenate((camera_extrinsic, np.tile(cam_shared_intrinsic, (numCameras, 1)), camera_centroid), axis=1)
        points_3d = res.x[nCamParams:].reshape((numPoints, 3))
        self.cameraArray = camera_params
        self.points3D = points_3d
        return res

    
    def saveCamVecs(self):
        outfile = 'sba_cameraArray'
        print(outfile)
        np.savez(outfile, self.cameraArray)
        print("cameraArray saved to ", outfile)




#%%
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


