import os
from re import L 
import numpy as np
import pprint
import pickle as pkl
import time
import argparse
from datetime import date
from movie_manager import SingleMovieManager

# python find_laser_points.py --n_cams=8 --root_dir=/home/jinyao/Calibration/newrig8 --frame_range 0 50 --width 3208 --height 2200

parser = argparse.ArgumentParser()
parser.add_argument('--n_cams', type=int, required=True)
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

threadpool = []

for i in range(args.n_cams):
    cam_name = "Cam" + str(i)
    threadpool.append(SingleMovieManager(i, args.root_dir, cam_name, args.frame_range, args.width, args.height))

for thread in threadpool:
    thread.start()

for thread in threadpool:
    thread.join()

res_files = []
for i in range(args.n_cams):
    f = "Cam{}_centroids.pkl".format(i)
    res_files.append(os.path.join(results_dir, f))

fileObject = open(res_files[0], 'rb')
pts = pkl.load(fileObject)
fileObject.close()

n_pts_per_cam = pts.shape[0]
centroids = np.zeros((n_pts_per_cam, 2, args.n_cams))
centroids[:] = np.nan

for i, file in enumerate(res_files):
    fileObject = open(file, 'rb')
    centroids[:,:,i] = pkl.load(fileObject)
    fileObject.close()

outfile = results_dir + '/centroids_{}.pkl'.format(str(date.today()))
print(outfile)

fileObject = open(outfile, 'wb')
pkl.dump(centroids, fileObject)
fileObject.close()

end_time = time.time()
print("time elapsed: {:.2f} second".format(end_time - start_time))
