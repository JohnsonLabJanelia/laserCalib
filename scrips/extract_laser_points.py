import os
import numpy as np
import pickle as pkl
import argparse
import glob
from lasercalib.feature_detection import *
from multiprocessing import Pool
from tqdm import tqdm
import cv2 as cv
import time 
from tqdm.contrib.concurrent import process_map

parser = argparse.ArgumentParser()
parser.add_argument('--root_dir', type=str, required=True)
parser.add_argument('--frame_range', nargs="+", type=int, required=True)
parser.add_argument('--n_cams', type=int, required=False, default=0)


args = parser.parse_args()
root_dir = args.root_dir
frame_start = args.frame_range[0]
frame_end = args.frame_range[1]
n_cams = args.n_cams

cam_names = []
for file in glob.glob(args.root_dir + "/movies/*.mp4"):
    file_name = file.split("/")
    cam_names.append(file_name[-1].split('.')[0])
cam_names.sort()

if (n_cams==0):
    n_cams = len(cam_names)

print("Number of cameras: ", n_cams)

def extract_laser_points_per_camera(camera_idx):
    video_file = os.path.join(root_dir, "movies/" + cam_names[camera_idx] + ".mp4")
    results_file = os.path.join(root_dir, "results/laser_points/" + cam_names[camera_idx] + "_centroids.pkl")
    vr = cv.VideoCapture(video_file)

    centroids = np.zeros((frame_end - frame_start, 2), dtype=float)
    centroids[:] = np.nan

    count = 0

    if frame_start != 0:
        vr.set(cv.CAP_PROP_POS_FRAMES, frame_start)

    for i in range(frame_start, frame_end):
        ret, frame = vr.read()
        if not ret:
            break
        laser_coord = green_laser_finder_faster(frame, 50)
        if laser_coord:
            centroids[i] = laser_coord
            count = count + 1

    with open(results_file, 'wb') as f:
        pkl.dump(centroids, f)
    
    return camera_idx, count


if __name__ == '__main__':
    start_time = time.time()
    results = process_map(extract_laser_points_per_camera, range(n_cams), max_workers=4)
    end_time = time.time()
    for cam_idx, num_frames in results:
        print(cam_names[cam_idx] + ": {}".format(num_frames))
    print("time elapsed: {:.2f} second".format(end_time - start_time))