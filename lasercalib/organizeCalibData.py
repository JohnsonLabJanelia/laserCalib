import numpy as np
import pySBA
import pickle
import pprint
from prettytable import PrettyTable
import seaborn as sns
import matplotlib.pyplot as plt
from my_cam_pose_visualizer import MyCamPoseVisualizer
from scipy.spatial.transform import Rotation as R


my_palette = sns.color_palette()

picklefile = open('../calibres/sba_data_new', 'rb')
sba = pickle.load(picklefile)
picklefile.close()

x = PrettyTable()
for row in sba.cameraArray:
    x.add_row(row)
print(x)


fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(sba.points3D[:,0], sba.points3D[:,1], sba.points3D[:,2])
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

nCams = sba.cameraArray.shape[0]
for i in range(nCams):
    r_f = R.from_rotvec(-sba.cameraArray[i, 0:3]).as_matrix().copy()
    t_f = sba.cameraArray[i, 3:6].copy()
    # get inverse transformation
    r_inv = r_f.T    
    t_inv = -np.matmul(r_f, t_f)    
    ex = np.eye(4)
    ex[:3,:3] = r_inv.T
    ex[:3,3] = t_inv
    visualizer = MyCamPoseVisualizer(fig, ax)
    visualizer.extrinsic2pyramid(ex, my_palette[i], 200)
plt.show()



camList = []
for i in range(nCams):
    camList.append(pySBA.unconvertParams(sba.cameraArray[i,:]))

pp = pprint.PrettyPrinter(indent=0)

np.set_printoptions(precision=5)
np.set_printoptions(suppress=True)

for i, cam in enumerate(camList):
    print('cam' + str(i))
    pp.pprint(cam)
    pp.format
    print('\n')


# stack for saving out the camera parameters

allParams = np.full((len(camList), 17), np.NaN)
for nCam in range(len(camList)):
    p = camList[nCam]
    f = p['K'][0,0]/2 + p['K'][1,1]/2
    r_m = np.transpose(p['R']).ravel()
    t = p['t']
    c = p['K'][2,0:2]
    d = p['d']
    allParams[nCam,:] = np.hstack((r_m,t,f,d,c))

np.savetxt('../calibres/calib_new.csv', allParams, delimiter=',')