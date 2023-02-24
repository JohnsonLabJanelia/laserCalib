import queue
from threading import Thread
import cv2
from skimage import morphology
from skimage import measure
import numpy as np
from feature_detection import green_laser_finder
import pickle as pkl

def concat_vh(list_2d):
    return cv2.vconcat([cv2.hconcat(list_h) for list_h in list_2d])

class VideoGet:
    def __init__(self, src_dir, q, cam_idx, laser=False, aruco=False):
        
        self.src_dir = src_dir
        video_source = src_dir + "/Cam{}.mp4".format(cam_idx)
        print(video_source)
        self.stream = cv2.VideoCapture(video_source)
        if (self.stream.isOpened()== False): 
            print("Error opening video: ", video_source)

        (self.grabbed, self.frame) = self.stream.read()
        self.stopped = False
        self.q = q
        self.current_frame = 0
        self.laser = laser
        self.aruco = aruco
        self.cam_idx = cam_idx
        if self.aruco:
            cam_params_file = src_dir + "/calibration_aruco/Cam{}.yaml".format(i)
            self.cam_matrix, self.distortion = self.load_camera_intrinsic(cam_params_file)
            # keep a moving average
            self.corner_average = {
                0: np.zeros((4, 2)),
                1: np.zeros((4, 2)),
                2: np.zeros((4, 2)),
                3: np.zeros((4, 2))
            }

    def load_camera_intrinsic(self, cam_params_file):
        fs = cv2.FileStorage(cam_params_file, cv2.FILE_STORAGE_READ)
        m_matrix_coefficients = fs.getNode("intrinsicMatrix")
        m_distortion_coefficients = fs.getNode("distortionCoefficients")
        cam_matrix = m_matrix_coefficients.mat().T
        distortion = m_distortion_coefficients.mat()[0]
        return cam_matrix, distortion

    def start(self):    
        Thread(target=self.get, args=()).start()
        return self

    def get(self):
        while not self.stopped:
            if not self.grabbed:
                self.q.put((0, None))
                self.stop()
            else:
                (self.grabbed, frame) = self.stream.read()
                
                if self.laser:
                    # process frames
                    laser_coord = green_laser_finder(frame, 70, 1100, morphology.disk(1), morphology.disk(4))
                    if laser_coord:
                        frame = cv2.circle(frame, (int(laser_coord[1]), int(laser_coord[0])), 50, (0, 0, 255), 5)

                if self.aruco:
                    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_100)
                    aruco_params = cv2.aruco.DetectorParameters_create() 
                    aruco_params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
                    corners, ids, rejectedImgPoints = cv2.aruco.detectMarkers(frame_gray, aruco_dict, parameters=aruco_params)
                    if len(corners) > 0:
                        # flatten the ArUco IDs list
                        ids = ids.flatten()
                        # loop over the detected ArUCo corners
                        for (markerCorner, markerID) in zip(corners, ids):
                            # rvec, tvec, markerPoints = cv2.aruco.estimatePoseSingleMarkers(markerCorner, 100, self.cam_matrix, self.distortion)
                            # cv2.drawFrameAxes(frame, self.cam_matrix, self.distortion, rot, trans, 100)  # Draw Axis
                            if np.sum(self.corner_average[markerID]) == 0:
                                self.corner_average[markerID] = markerCorner.reshape((4, 2))
                            else:
                                self.corner_average[markerID] = (markerCorner.reshape((4, 2)) + self.corner_average[markerID])/2

                    for markerID, markerCorner in self.corner_average.items():        
                        (topLeft, topRight, bottomRight, bottomLeft) = markerCorner
                        # convert each of the (x, y)-coordinate pairs to integers
                        topRight = (int(topRight[0]), int(topRight[1]))
                        bottomRight = (int(bottomRight[0]), int(bottomRight[1]))
                        bottomLeft = (int(bottomLeft[0]), int(bottomLeft[1]))
                        topLeft = (int(topLeft[0]), int(topLeft[1]))

                        # draw the bounding box of the ArUCo detection
                        cv2.line(frame, topLeft, topRight, (0, 255, 0), 5)
                        cv2.line(frame, topRight, bottomRight, (0, 0, 255), 5)
                        cv2.line(frame, bottomRight, bottomLeft, (0, 0, 255), 5)
                        cv2.line(frame, bottomLeft, topLeft, (0, 0, 255), 5)
                        # compute and draw the center (x, y)-coordinates of the ArUco marker                        
                        cX = int((topLeft[0] + bottomRight[0]) / 2.0)
                        cY = int((topLeft[1] + bottomRight[1]) / 2.0)
                        cv2.circle(frame, (cX, cY), 5, (0, 0, 255), -1)                        
                        # draw the ArUco marker ID on the frame
                        cv2.putText(frame, str(markerID),
                            (topLeft[0], topLeft[1] - 15),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            3, (0, 255, 0), 5)
                        
                        # # coplanar points from one view for pose estimation suffers from ambiguity: https://github.com/opencv/opencv/issues/8813 
                        # rvecs, tvecs, markerPoints = cv2.aruco.estimatePoseSingleMarkers(markerCorner.reshape((1, 4, 2)), 120, self.cam_matrix, self.distortion)
                        # rot, trans = rvecs[0].ravel(), tvecs[0].ravel()
                        # cv2.drawFrameAxes(frame, self.cam_matrix, self.distortion, rot, trans, 120)  # Draw Axis    
                            

                self.current_frame = self.current_frame + 1
                frame_small = cv2.resize(frame, (802, 550), interpolation = cv2.INTER_AREA)
                self.q.put((self.current_frame, frame_small))

    def stop(self):
        # save the corners 
        if self.aruco:
            with open(self.src_dir + "/calibration_aruco/Cam{}_aruco.pkl".format(self.cam_idx), 'wb') as f:
                pkl.dump(self.corner_average, f)
            print("Aruco corner detection saved.")
        self.stopped = True


view_queues = []
num_cams = 8

for i in range(num_cams):
    view_queues.append(queue.Queue())

threadpool = []
for i in range(num_cams):
    source_dir = "/home/jinyao/Calibration/newrig8/aruco_detection"
    threadpool.append(VideoGet(source_dir, view_queues[i], i, False, True))

for thread in threadpool:
    thread.start()

imgs = [None] * num_cams

while True:
    key = cv2.waitKey(1) & 0xFF
    
    if key == ord('q'):    
        break
    
    if key == ord('p'):
        cv2.waitKey(-1) #wait until any key is pressed

    for i in range(num_cams):
        frame_id, img = view_queues[i].get()
        cv2.putText(img, "Cam {}: {:.0f}".format(i, frame_id),
            (10, 450), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255))
        imgs[i] = img

    if (img is None):
            break
    
    img_tile = concat_vh([[imgs[0], imgs[1], imgs[2], imgs[3]], 
                        [imgs[4], imgs[5], imgs[6], imgs[7]]])
    cv2.imshow('Frame', img_tile)


for i, thread in enumerate(threadpool):
    thread.stop()

cv2.destroyAllWindows()