import os
from re import L 
import numpy as np
import pprint
import pickle as pkl
import time
import argparse
from datetime import date
from movie_manager import SingleMovieManager
import glob

parser = argparse.ArgumentParser()
parser.add_argument('--root_dir', type=str, required=True)
parser.add_argument('--frame_range', nargs="+", type=int, required=True)
parser.add_argument('--width', type=int, required=True)
parser.add_argument('--height', type=int, required=True)

args = parser.parse_args()

start_time = time.time()

pp = pprint.PrettyPrinter(indent=0)
np.set_printoptions(precision=5)
np.set_printoptions(suppress=True)


results_dir = args.root_dir + "/results"
if not os.path.exists(results_dir):
   os.makedirs(results_dir)


mp4_files = []
for file in glob.glob(args.root_dir + "/movies/*.mp4"):
    file_name = file.split("/")
    mp4_files.append(file_name[-1][:-4])
n_cams = len(mp4_files)

print("Number of cameras: ", n_cams)

threadpool = []

for i in range(n_cams):
    cam_name = mp4_files[i]
    threadpool.append(SingleMovieManager(i, args.root_dir, cam_name, args.frame_range, args.width, args.height))

for thread in threadpool:
    thread.start()

for thread in threadpool:
    thread.join()

res_files = []
for i in range(n_cams):
    f = "{}_centroids.pkl".format(mp4_files[i])
    res_files.append(results_dir + '/laser_points/' + f)

with open(res_files[0], 'rb') as f:
    pts = pkl.load(f)

n_pts_per_cam = pts.shape[0]
centroids = np.zeros((n_pts_per_cam, 2, n_cams))
centroids[:] = np.nan

for i, file in enumerate(res_files):
    with open(file, 'rb') as f:
        centroids[:,:,i] = pkl.load(f)

with open(results_dir + '/centroids.pkl', 'wb') as f:
    pkl.dump(centroids, f)

end_time = time.time()
print("time elapsed: {:.2f} second".format(end_time - start_time))
