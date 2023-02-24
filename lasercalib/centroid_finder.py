import cv2
import threading
from skimage import morphology
import numpy as np
from feature_detection import green_laser_finder


class CentroidFinder(threading.Thread):
    def __init__(self, root_dir, cam_name, thread_idx, q, centroids):
        threading.Thread.__init__(self)
        self.thread_idx = thread_idx
        self.thread_name = cam_name + "_thread" + str(thread_idx)
        self.root_dir = root_dir
        self.cam_name = cam_name
        self.q = q
        self.centroids = centroids

    def process_queue(self):
        small_footprint = morphology.disk(1)
        big_footprint = morphology.disk(4)
        laser_intensity_thresh = 70
        centroid_dist_thresh = 1100
        
        while True:
            frame_idx, img = self.q.get()
            if (img is None):
                break
            
            laser_coord = green_laser_finder(
                                    img, 
                                    laser_intensity_thresh,  
                                    centroid_dist_thresh, 
                                    small_footprint, 
                                    big_footprint)
            if laser_coord:
                print(self.thread_name, "frame: ", frame_idx, "centroid:", laser_coord)
                self.centroids[frame_idx, :] = laser_coord

    def run(self):
        # print(self.thread_name, "started")
        self.process_queue()
        # print(self.thread_name, "finished")

