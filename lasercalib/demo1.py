# from util.my_test import *


# test_func()

from util.camera_pose_visualizer import *


# argument : the minimum/maximum value of x, y, z
visualizer = CamPoseVisualizer([-50, 50], [-50, 50], [0, 50])

# argument : extrinsic matrix, color, scaled focal length(z-axis length of frame body of camera
visualizer.extrinsic2pyramid(np.eye(4), 'c', 10)

visualizer.show()