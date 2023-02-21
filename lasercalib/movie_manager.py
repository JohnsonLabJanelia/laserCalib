from centroid_finder import *
import threading
import pickle as pkl
import subprocess as sp
import queue
import signal

class SingleMovieManager(threading.Thread):
    def __init__(self, threadID, root_dir, cam_name, frame_range):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.thread_name = "Thread-" + str(self.threadID)
        self.root_dir = root_dir
        self.cam_name = cam_name
        self.movie_path = self.root_dir + "/" + self.cam_name + ".mp4"
        self.results_file = self.root_dir + "/results/" + self.cam_name + "_centroids.pkl"
        self.curr_img_num = 0
        self.frameRange = frame_range
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
            '-nostdin',
            '-hide_banner',
            '-loglevel', 'error',
            '-i', MOVIE_PATH,
            '-vframes', str(self.frameRange[1]), 
            '-threads', str(8),
            '-f', 'image2pipe',
            '-pix_fmt', 'rgb24',
            '-vcodec', 'rawvideo', '-']
        
        pipe = sp.Popen(command, stdout=sp.PIPE, stderr=sp.PIPE, bufsize=3208*2200*3)

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
        self.safe_print("starting ", self.thread_name)
        self.ffmpeg_loader()
        self.save_results()
        self.safe_print(self.thread_name, " finished")


    def safe_print(*args, sep=" ", end="", **kwargs):
        joined_string = sep.join([ str(arg) for arg in args ])
        print(joined_string  + "\n", sep=sep, end=end, **kwargs)