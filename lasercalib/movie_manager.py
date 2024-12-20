from centroid_finder import *
import threading
import pickle as pkl
import subprocess as sp
import queue
import signal
import cv2 as cv
from tqdm import tqdm


class SingleMovieManager(threading.Thread):
    def __init__(self, threadID, root_dir, cam_name, frame_range):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.thread_name = "Thread-" + str(self.threadID)
        self.root_dir = root_dir
        self.cam_name = cam_name
        self.movie_path = self.root_dir + "/movies/" + self.cam_name + ".mp4"
        self.results_file = self.root_dir + "/results/laser_points/" + self.cam_name + "_centroids.pkl"
        self.frameRange = frame_range
        self.nFramesAnalyzed = self.frameRange[1] - self.frameRange[0]
        self.centroids = np.zeros((self.nFramesAnalyzed, 2), dtype=float)
        self.centroids[:] = np.nan
        self.centroid_threads = []
        self.q = queue.Queue()
        self.get_width_and_height()

    def get_width_and_height(self):
        video_stream = cv2.VideoCapture(self.movie_path)
        if (video_stream.isOpened()== False): 
            print("Error opening video: ", self.movie_path)
        else:
            self.width  = int(video_stream.get(cv.CAP_PROP_FRAME_WIDTH))   # float `width`
            self.height = int(video_stream.get(cv.CAP_PROP_FRAME_HEIGHT))  # float `height`
        video_stream.release()


    def ffmpeg_loader(self):
        FFMPEG_BIN = "ffmpeg"
        MOVIE_PATH = self.movie_path
        command = [ FFMPEG_BIN,
            '-nostdin',
            '-hide_banner',
            '-loglevel', 'error',
            '-i', MOVIE_PATH,
            '-vframes', str(self.frameRange[1]), 
            '-threads', str(1),
            '-f', 'image2pipe',
            '-pix_fmt', 'rgb24',
            '-vcodec', 'rawvideo', '-']
        
        pipe = sp.Popen(command, stdout=sp.PIPE, stderr=sp.PIPE, bufsize=self.width*self.height*3)

        # skip unwanted initial frames
        _ = pipe.stdout.read(self.width*self.height*3*self.frameRange[0])
        pipe.stdout.flush()

        self.start_centroid_threads()

        for i in range(self.frameRange[0], self.frameRange[1]):
            self.curr_frame_idx = i - self.frameRange[0]
            raw_image = pipe.stdout.read(self.width*self.height*3)

            frame_idx = self.curr_frame_idx
            self.q.put((frame_idx, np.frombuffer(raw_image, dtype='uint8').reshape((self.height, self.width, 3))))
            pipe.stdout.flush()
        
        # send termination signal to worker threads
        for i, thread in enumerate(self.centroid_threads):
            self.q.put((0, None))
        
        for i, thread in enumerate(self.centroid_threads):
            thread.join()
        
        pipe.stdout.flush()
        pipe.send_signal(signal.SIGINT)
        pipe.wait()

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
        # print("starting ", self.thread_name)
        self.ffmpeg_loader()
        self.save_results()
        # print(self.thread_name, " finished")
