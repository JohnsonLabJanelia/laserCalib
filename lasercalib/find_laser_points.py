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
import time
import queue

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
        self.laser_intensity_thresh = 60
        self.centroid_dist_thresh = 1300
        self.centroids = centroids

    def process_queue(self):
        while True:
            frame_idx, img = self.q.get()
            if (img is None):
                break

            green = img[:,:,1]
            cc = green > self.laser_intensity_thresh
            # cc = morphology.binary_erosion(cc, self.small_footprint)
            cc = morphology.binary_closing(cc, self.big_footprint)
            labels = measure.label(cc, background=0, return_num=False)
            props = measure.regionprops(labels)
            self.process_props(props, frame_idx)
            print(self.thread_name, "frame: ", frame_idx, "centroid:", self.centroids[frame_idx, :])

    def process_props(self, props, frame_idx):
        # exclude connected components that are too far from image center
        n = len(props)
        idx = []
        
        for i in range(n):
            keep = True
            dist = ((props[i].centroid[0] - 1100)**2 + (props[i].centroid[1] - 1604)**2)**0.5
            if dist > self.centroid_dist_thresh:
                continue
            if (self.cam_name == "Cam3"):
                if props[i].centroid[0] < 70:
                    continue
            
            idx.append(i)
                
        if len(idx) == 1:
            self.centroids[frame_idx, :] = props[idx[0]].centroid
            print(self.thread_name, "frame: ", frame_idx, "centroid:", self.centroids[frame_idx, :])
        else:
            print(self.thread_name, "frame: ", frame_idx, "objects found: ", len((idx)))

    def run(self):
        print(self.thread_name, "started")
        self.process_queue()
        print(self.thread_name, "finished")


class SingleMovieManager(threading.Thread):
    def __init__(self, threadID, root_dir, cam_name):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.thread_name = "Thread-" + str(self.threadID)
        self.root_dir = root_dir
        self.cam_name = cam_name
        self.movie_path = self.root_dir + "/laser_movies/" + self.cam_name + ".mp4"
        self.metadata_path = self.root_dir + "/metadata/" + self.cam_name + "_meta.csv"
        self.results_file = self.root_dir + "/results/" + self.cam_name + "_centroids.pkl"
        self.curr_img_num = 0
        self.frameRange = [0, 2600]
        self.nFramesAnalyzed = self.frameRange[1] - self.frameRange[0]
        self.centroids = np.zeros((self.nFramesAnalyzed, 2), dtype=float)
        self.centroids[:] = np.nan
        self.mask = np.zeros((2200, 3208), dtype='uint8')
        self.small_footprint = morphology.disk(1)
        self.big_footprint = morphology.disk(4)
        self.centroid_threads = []
        self.q = queue.Queue()

    def ffmpeg_loader(self):
        FFMPEG_BIN = "ffmpeg"
        MOVIE_PATH = self.movie_path
        command = [ FFMPEG_BIN,
            '-i', MOVIE_PATH,
            '-threads', str(8),
            '-f', 'image2pipe',
            '-pix_fmt', 'rgb24',
            '-vcodec', 'rawvideo', '-']
        
        pipe = sp.Popen(command, stdout = sp.PIPE, bufsize=3208*2200*3)

        # skip unwanted initial frames
        _ = pipe.stdout.read(3208*2200*3*self.frameRange[0])
        pipe.stdout.flush()

        self.start_centroid_threads()

        for i in np.arange(self.frameRange[0], self.frameRange[1]):
            self.curr_frame_idx = i - self.frameRange[0]
            raw_image = pipe.stdout.read(3208*2200*3)

            frame_idx = self.curr_frame_idx
            self.q.put((frame_idx, np.frombuffer(raw_image, dtype='uint8').reshape((2200, 3208,3))))
            pipe.stdout.flush()
        
        # send termination signal to worker threads
        for i, thread in enumerate(self.centroid_threads):
            self.q.put((0, None))
        
        for i, thread in enumerate(self.centroid_threads):
            thread.join()
        
        pipe.stdout.flush()
        pipe.terminate()

    def save_results(self):
        fileObject = open(self.results_file, 'wb')
        pkl.dump(self.centroids, fileObject)
        fileObject.close()

    def start_centroid_threads(self):
        n_threads = 8
        for i in range(n_threads):
            self.centroid_threads.append(CentroidFinder(self.root_dir, self.cam_name, i, self.q, self.centroids))
        
        for thread in self.centroid_threads:
            thread.start()

    def run(self):
        print("starting ", self.thread_name)
        self.ffmpeg_loader()
        self.save_results()
        print(self.thread_name, " finished")


######## FIND CENTROIDS AND SAVE THEIR POSITIONS #########

start_time = time.time()

n_cams = 7

"""
    root_dir should have 3 subdirectories:
        ---> /movies (contains movies with name: [cam_name].mp4)
        ---> /metadata (contains [cam_name].csv file with frame_number and time_stamp on each line)
        ---> /results (empty folder to save pickle files)

"""

root_dir = '/home/jinyao/calibration/Calibration20220930'

pp = pprint.PrettyPrinter(indent=0)
np.set_printoptions(precision=5)
np.set_printoptions(suppress=True)

threadpool = []

for i in range(n_cams):
    cam_name = "Cam" + str(i)
    threadpool.append(SingleMovieManager(i, root_dir, cam_name))

for thread in threadpool:
    thread.start()

for thread in threadpool:
    thread.join()

results_dir = root_dir + "/results"
res_files = [os.path.join(results_dir, f) for f in os.listdir(results_dir) if os.path.isfile(os.path.join(results_dir, f))]
res_files = sorted(res_files)

fileObject = open(res_files[0], 'rb')
pts = pkl.load(fileObject)
fileObject.close()

n_pts_per_cam = pts.shape[0]
n_cams = len(res_files)
centroids = np.zeros((n_pts_per_cam, 2, n_cams))
centroids[:] = np.nan

for i, file in enumerate(res_files):
    print(file)
    fileObject = open(file, 'rb')
    centroids[:,:,i] = pkl.load(fileObject)
    fileObject.close()
    
print(centroids)
print(centroids.shape)

outfile = 'centroids_20220930.pkl'
fileObject = open(outfile, 'wb')
pkl.dump(centroids, fileObject)
fileObject.close()

end_time = time.time()
print("time elapsed: ", end_time - start_time)