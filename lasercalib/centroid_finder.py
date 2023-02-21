import cv2
import threading
from skimage import morphology
from skimage import measure
import numpy as np

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

            green = img[:,:,1]
            cc = green > self.laser_intensity_thresh
            cc = morphology.binary_erosion(cc, self.small_footprint)
            cc = morphology.binary_dilation(cc, self.small_footprint)
            cc = morphology.binary_closing(cc, self.big_footprint)

            ## roi 
            mask_roi = np.zeros_like(green)
            mask_roi = cv2.circle(mask_roi, (1604,1100), 1100, 255, -1)
            cc = mask_roi * cc

            labels = measure.label(cc, background=0, return_num=False)
            props = measure.regionprops(labels)
            self.process_props(props, frame_idx)
            self.safe_print(self.thread_name, "frame: ", frame_idx, "centroid:", self.centroids[frame_idx, :])

    def process_props(self, props, frame_idx):
        # exclude connected components that are too far from image center
        n = len(props)
        idx = []
        
        for i in range(n):
            keep = True
            dist = ((props[i].centroid[0] - 1100)**2 + (props[i].centroid[1] - 1604)**2)**0.5
            if dist > self.centroid_dist_thresh:
                continue
            # legacy 
            # if (self.cam_name == "Cam3"):
            #     if props[i].centroid[0] < 70:
            #         continue
            
            idx.append(i)
                
        if len(idx) == 1:
            self.centroids[frame_idx, :] = props[idx[0]].centroid
            self.safe_print(self.thread_name, "frame: ", frame_idx, "centroid:", self.centroids[frame_idx, :])
        else:
            self.safe_print(self.thread_name, "frame: ", frame_idx, "objects found: ", len((idx)))

    def run(self):
        self.safe_print(self.thread_name, "started")
        self.process_queue()
        self.safe_print(self.thread_name, "finished")


    def safe_print(*args, sep=" ", end="", **kwargs):
        joined_string = sep.join([ str(arg) for arg in args ])
        print(joined_string  + "\n", sep=sep, end=end, **kwargs)