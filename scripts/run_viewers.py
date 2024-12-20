import queue
from threading import Thread
import cv2
from skimage import morphology
from skimage import measure
import numpy as np
from lasercalib.feature_detection import *
import pickle as pkl
import argparse
import os
import json

def concat_vh(list_2d):
    return cv2.vconcat([cv2.hconcat(list_h) for list_h in list_2d])

class VideoGet:
    def __init__(self, src_dir, save_dir, q, cam_name, laser=False, aruco=False):
        
        self.src_dir = src_dir
        self.save_dir = save_dir
        if aruco:
            video_source = video_dir + "/{}.mp4".format(cam_name)
            
        else:
            video_source = video_dir + "/{}.mp4".format(cam_name)
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
        self.cam_name = cam_name
        if self.aruco:
            # cam_params_file = src_dir + "/results/calibration_aruco/Cam{}.yaml".format(i)
            # print(cam_params_file)
            # self.cam_matrix, self.distortion = self.load_camera_intrinsic(cam_params_file)
            # keep a moving average
            self.corner_average = {}

    def load_camera_intrinsic(self, cam_params_file):
        fs = cv2.FileStorage(cam_params_file, cv2.FILE_STORAGE_READ)
        m_matrix_coefficients = fs.getNode("camera_matrix")
        m_distortion_coefficients = fs.getNode("distortion_coefficients")
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
                    laser_coord = green_laser_finder_faster(frame, 50)
                    if laser_coord:
                        frame = cv2.circle(frame, (int(laser_coord[1]), int(laser_coord[0])), 100, (0, 0, 255), 25)

                if self.aruco:
                    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_100)
                    aruco_params = cv2.aruco.DetectorParameters()
                    aruco_detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)
                    aruco_params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
                    corners, ids, rejectedImgPoints = aruco_detector.detectMarkers(frame_gray)

                    if len(corners) > 0:
                        # flatten the ArUco IDs list
                        ids = ids.flatten()
                        # loop over the detected ArUCo corners
                        for (markerCorner, markerID) in zip(corners, ids):
                            # rvec, tvec, markerPoints = cv2.aruco.estimatePoseSingleMarkers(markerCorner, 100, self.cam_matrix, self.distortion)
                            # cv2.drawFrameAxes(frame, self.cam_matrix, self.distortion, rot, trans, 100)  # Draw Axis
                            if markerID in self.corner_average.keys():
                                self.corner_average[markerID] = (markerCorner.reshape((4, 2)) + self.corner_average[markerID])/2
                            else:
                                self.corner_average[markerID] = markerCorner.reshape((4, 2))

                    for markerID, markerCorner in self.corner_average.items():        
                        (topLeft, topRight, bottomRight, bottomLeft) = markerCorner
                        # convert each of the (x, y)-coordinate pairs to integers
                        topRight = (int(topRight[0]), int(topRight[1]))
                        bottomRight = (int(bottomRight[0]), int(bottomRight[1]))
                        bottomLeft = (int(bottomLeft[0]), int(bottomLeft[1]))
                        topLeft = (int(topLeft[0]), int(topLeft[1]))

                        # draw the bounding box of the ArUCo detection
                        cv2.line(frame, topLeft, topRight, (0, 255, 0), 15)
                        cv2.line(frame, topRight, bottomRight, (0, 0, 255), 15)
                        cv2.line(frame, bottomRight, bottomLeft, (0, 0, 255), 15)
                        cv2.line(frame, bottomLeft, topLeft, (0, 0, 255), 15)
                        # compute and draw the center (x, y)-coordinates of the ArUco marker                        
                        cX = int((topLeft[0] + bottomRight[0]) / 2.0)
                        cY = int((topLeft[1] + bottomRight[1]) / 2.0)
                        cv2.circle(frame, (cX, cY), 15, (0, 0, 255), -1)                        
                        # draw the ArUco marker ID on the frame
                        cv2.putText(frame, str(markerID),
                            (topLeft[0], topLeft[1] - 15),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            3, (0, 255, 0), 15)
                        
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
            aruco_marker_folder = self.save_dir + "/results/aruco_corners/" 
            if not os.path.exists(aruco_marker_folder):
                os.makedirs(aruco_marker_folder)
            with open(aruco_marker_folder + "{}_aruco.pkl".format(self.cam_name), 'wb') as f:
                pkl.dump(self.corner_average, f)
            print("Aruco corner detection saved.")
        self.stopped = True


parser = argparse.ArgumentParser()
parser.add_argument('-c', '--config', type=str, required=True)
parser.add_argument('-m', '--mode', choices=['laser', 'aruco'], required=True)
parser.add_argument('-i', '--dataset_idx', type=int, required=False, default=0)
args = parser.parse_args()
dataset_idx = args.dataset_idx

config_dir = args.config
with open(config_dir + '/config.json', 'r') as f:
    calib_config = json.load(f)

root_dir = calib_config['root_dir']
cam_serials = calib_config['cam_serials']
cam_names = []
for cam_serial in cam_serials:
    cam_names.append("Cam" + cam_serial)
num_cams = len(cam_names)

view_queues = []

for i in range(num_cams):
    view_queues.append(queue.Queue())

threadpool = []
for i in range(num_cams):
    if args.mode == "aruco":
        aruco_dataset = calib_config['aruco']
        video_dir = os.path.join(root_dir + aruco_dataset)
        threadpool.append(VideoGet(video_dir, config_dir, view_queues[i], cam_names[i], laser=False, aruco=True))
    else:
        laser_datasets = calib_config['lasers']
        video_dir = os.path.join(root_dir + laser_datasets[dataset_idx])
        threadpool.append(VideoGet(video_dir, config_dir, view_queues[i], cam_names[i], laser=True, aruco=False))

for thread in threadpool:
    thread.start()


if num_cams == 1:
    imgs = [None]
else:
    imgs = [None] * int(np.ceil(num_cams / 4.0) * 4)

while True:
    key = cv2.waitKey(1) & 0xFF
    
    if key == ord('q'):    
        break
    
    if key == ord('p'):
        cv2.waitKey(-1) #wait until any key is pressed

    for i in range(len(imgs)):
        if (i < num_cams):
            frame_id, img = view_queues[i].get()
            cv2.putText(img, "{}: {:.0f}".format(cam_names[i], frame_id),
                (10, 450), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255))
            imgs[i] = img
        else:
            blank_image = np.zeros((550, 802, 3), np.uint8)
            imgs[i] = blank_image

    if (img is None):
            break
    
    if num_cams != 1:
        layout = []
        for i in range(len(imgs)):
            if i % 4 == 0:
                temp = []
            temp.append(imgs[i])
            if i % 4 == 3:
                layout.append(temp.copy())
            i = i + 1
        img_tile = concat_vh(layout)
        img_tile_resize = cv2.resize(img_tile, (2100, 1200))
        cv2.imshow('Frame', img_tile_resize)
    else:
        cv2.imshow('Frame', imgs[0])
    
for i, thread in enumerate(threadpool):
    thread.stop()

cv2.destroyAllWindows()