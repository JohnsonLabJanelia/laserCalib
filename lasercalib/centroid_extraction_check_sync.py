import os
from re import L 
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from skimage import measure
from skimage import morphology
import pprint
import pickle as pkl
import threading, time
import subprocess as sp
from multiprocessing import Process
import signal


class MovieCentroidProcessor(threading.Thread):
    def __init__(self, threadID, root_dir, cam_name, use_mask=False):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.thread_name = "Thread-" + str(self.threadID)
        self.root_dir = root_dir
        self.cam_name = cam_name
        self.use_mask = use_mask
        self.movie_path = self.root_dir + "/movies/" + self.cam_name + ".mp4"
        self.metadata_path = self.root_dir + "/metadata/" + self.cam_name + "_meta.csv"
        self.results_file = self.root_dir + "/results/" + self.cam_name + "_centroids.pkl"
        self.curr_img_num = 0
        self.curr_img = np.zeros((2200, 3208, 3), dtype='uint8')
        self.frameRange = [400, 500]
        self.nFramesAnalyzed = self.frameRange[1] - self.frameRange[0]
        self.centroids = np.zeros((self.nFramesAnalyzed, 2), dtype=np.float)
        self.centroids[:] = np.nan
        self.laser_intensity_threshold = 200
        self.mask = []
        self.footprint = morphology.disk(4)

    def mask_setup(self):
        print("creating mask to spatially limit search for laser pointer point to a central circle ", self.cam_name)
        self.mask = np.zeros((2200, 3208), dtype=bool)
        for i in range(self.mask.shape[0]):
            for j in range(self.mask.shape[1]):
                if (((i-1100)**2 + (j-1604)**2)**0.5 < 1400):
                    self.mask[i,j] = True

    def ffmpeg_loader(self):
        FFMPEG_BIN = "ffmpeg"
        MOVIE_PATH = self.movie_path
        command = [ FFMPEG_BIN,
            '-i', MOVIE_PATH,
            '-f', 'image2pipe',
            '-pix_fmt', 'rgb24',
            '-vcodec', 'rawvideo', '-']
        
        pipe = sp.Popen(command, stdout = sp.PIPE, bufsize=3208*2200*3)

        # skip unwanted initial frames
        trash = pipe.stdout.read(3208*2200*3*self.frameRange[0])
        pipe.stdout.flush()

        for i in np.arange(self.frameRange[0], self.frameRange[1]):
            self.curr_img_idx = i - self.frameRange[0]
            raw_image = pipe.stdout.read(3208*2200*3)
            self.curr_img = np.frombuffer(raw_image, dtype='uint8').reshape((2200, 3208,3))
            pipe.stdout.flush()
            self.process_curr_img()

    def process_curr_img(self):
        green = self.curr_img[:,:,1]
        if (self.use_mask):
            cc = green * self.mask

        cc = green > self.laser_intensity_threshold
        cc = morphology.binary_closing(cc, self.footprint)
        labels, num = measure.label(cc, background=0, return_num=True)
        props = measure.regionprops(labels)
        
        if num == 0:
            print(self.cam_name, "image: ", self.curr_img_idx, " regions found: ", num)

        elif num > 1:
            print(self.cam_name, "image: ", self.curr_img_idx, " regions found: ", num)

        elif num == 1:
            print(self.cam_name, "image: ", self.curr_img_idx, " regions found: ", num, "centroid: ", props[0].centroid, "size: ", props[0].area)
            self.centroids[self.curr_img_idx, :] = props[0].centroid

    def save_results(self):
        fileObject = open(self.results_file, 'wb')
        pkl.dump(self.centroids, fileObject)
        fileObject.close()

    def run(self):
        print("starting ", self.thread_name)
        if (self.use_mask):
            self.mask_setup()
        self.ffmpeg_loader()
        self.save_results()
        print(self.thread_name, " finished")


######## FIND CENTROIDS AND SAVE THEIR POSITIONS #########
n_cams = 7

"""
    root_dir should have 3 subdirectories:
        ---> /movies (contains movies with name: [cam_name].mp4)
        ---> /metadata (contains [cam_name].csv file with frame_number and time_stamp on each line)
        ---> /results (empty folder to save pickle files)
"""
root_dir = '/home/rob/laser_calibration/2022-05-02_16_40_25'

pp = pprint.PrettyPrinter(indent=0)
np.set_printoptions(precision=5)
np.set_printoptions(suppress=True)

threadpool = []

for i in range(n_cams):
    cam_name = "Cam" + str(i)
    threadpool.append(MovieCentroidProcessor(i, root_dir, cam_name, use_mask=True))

for thread in threadpool:
    thread.start()

