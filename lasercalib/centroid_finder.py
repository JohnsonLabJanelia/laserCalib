import cv2
import threading
from skimage import morphology
from skimage import measure
import numpy as np

def laser_finder(img, 
                laser_intensity_thresh=70, 
                centroid_dist_thresh=1100, 
                small_footprint=morphology.disk(1), 
                big_footprint=morphology.disk(4)):
    
    green = img[:,:,1]
    cc = green > laser_intensity_thresh

    cc = morphology.binary_erosion(cc, small_footprint)
    cc = morphology.binary_dilation(cc, small_footprint)
    cc = morphology.binary_closing(cc, big_footprint)

    ## roi 
    mask_roi = np.zeros_like(green)
    mask_roi = cv2.circle(mask_roi, (1604,1100), 1100, 255, -1)
    cc = mask_roi * cc

    labels = measure.label(cc, background=0, return_num=False)
    props = measure.regionprops(labels)

    # exclude connected components that are too far from image center
    n = len(props)
    idx = []
    
    for i in range(n):
        dist = ((props[i].centroid[0] - 1100)**2 + (props[i].centroid[1] - 1604)**2)**0.5
        if dist > centroid_dist_thresh:
            continue            
        idx.append(i)

    if len(idx) == 1:
        return props[idx[0]].centroid
    else:
        return None


class CentroidFinder(threading.Thread):
    def __init__(self, root_dir, cam_name, thread_idx, q, centroids):
        threading.Thread.__init__(self)
        self.thread_idx = thread_idx
        self.thread_name = cam_name + "_thread" + str(thread_idx)
        self.root_dir = root_dir
        self.cam_name = cam_name
        self.q = q
        self.small_footprint = morphology.disk(1)
        self.big_footprint = morphology.disk(4)
        self.laser_intensity_thresh = 70
        self.centroid_dist_thresh = 1100
        self.centroids = centroids

    def process_queue(self):
        while True:
            frame_idx, img = self.q.get()
            if (img is None):
                break
            
            laser_coord = laser_finder(
                                    img, 
                                    self.laser_intensity_thresh,  
                                    self.centroid_dist_thresh, 
                                    self.small_footprint, 
                                    self.big_footprint)
            if laser_coord:
                print(self.thread_name, "frame: ", frame_idx, "centroid:", laser_coord)
                self.centroids[frame_idx, :] = laser_coord

    def run(self):
        # print(self.thread_name, "started")
        self.process_queue()
        # print(self.thread_name, "finished")

