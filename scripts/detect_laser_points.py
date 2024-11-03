import os
import numpy as np
import pickle as pkl
import argparse
from lasercalib.feature_detection import *
from multiprocessing import Pool
from tqdm import tqdm
import cv2 as cv
import time 
from tqdm.contrib.concurrent import process_map
import json

parser = argparse.ArgumentParser()
parser.add_argument('-c', '--config', type=str, required=True)
parser.add_argument('-i', '--dataset_idx', type=int, required=False, default=-1)

args = parser.parse_args()

config_dir = args.config
with open(config_dir + '/config.json', 'r') as f:
    calib_config = json.load(f)
dataset_idx = args.dataset_idx

root_dir = calib_config['root_dir']
laser_datasets = calib_config['lasers']
cam_serials = calib_config['cam_serials']
cam_names = []
for cam_serial in cam_serials:
    cam_names.append("Cam" + cam_serial)
n_cams = len(cam_names)
print("Number of cameras: ", n_cams)

def extract_laser_points_per_camera(dataset_camera_idx):
    dataset_idx, camera_idx = dataset_camera_idx
    video_file = os.path.join(root_dir, "{}/".format(laser_datasets[dataset_idx]) + cam_names[camera_idx] + ".mp4")
    output_file = os.path.join(config_dir, "results/{}/".format(laser_datasets[dataset_idx]) + cam_names[camera_idx] + "_centroids.pkl")
    vr = cv.VideoCapture(video_file)

    centroids = np.zeros((frame_end - frame_start, 2), dtype=float)
    centroids[:] = np.nan

    count = 0

    if frame_start != 0:
        vr.set(cv.CAP_PROP_POS_FRAMES, frame_start)

    for i in range(centroids.shape[0]):
        ret, frame = vr.read()
        if not ret:
            break
        laser_coord = green_laser_finder_faster(frame, 50)
        if laser_coord:
            centroids[i] = laser_coord
            count = count + 1

    with open(output_file, 'wb') as f:
        pkl.dump(centroids, f)
    
    return camera_idx, count


if __name__ == '__main__':
    if dataset_idx != -1:
        print(laser_datasets[dataset_idx])
        frame_start = calib_config['frames'][dataset_idx][0]
        frame_end = calib_config['frames'][dataset_idx][1]

        start_time = time.time()
        # make folder 
        results_dir = os.path.join(config_dir, 'results')
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)

        laser_points_folder =  os.path.join(results_dir, laser_datasets[dataset_idx])
        if not os.path.exists(laser_points_folder):
            os.makedirs(laser_points_folder)

        dataset_camera_idx = [[dataset_idx, cam_idx] for cam_idx in range(n_cams)]
        results = process_map(extract_laser_points_per_camera, dataset_camera_idx, max_workers=4)
        end_time = time.time()
        for cam_idx, num_frames in results:
            print(cam_names[cam_idx] + ": {}".format(num_frames))
        print("time elapsed: {:.2f} second".format(end_time - start_time))
    else:
        for dataset_idx in range(len(laser_datasets)):
            print(laser_datasets[dataset_idx])
            frame_start = calib_config['frames'][dataset_idx][0]
            frame_end = calib_config['frames'][dataset_idx][1]

            start_time = time.time()
            # make folder 
            results_dir = os.path.join(config_dir, 'results')
            if not os.path.exists(results_dir):
                os.makedirs(results_dir)

            laser_points_folder =  os.path.join(results_dir, laser_datasets[dataset_idx])
            if not os.path.exists(laser_points_folder):
                os.makedirs(laser_points_folder)

            dataset_camera_idx = [[dataset_idx, cam_idx] for cam_idx in range(n_cams)]
            results = process_map(extract_laser_points_per_camera, dataset_camera_idx, max_workers=4)
            end_time = time.time()
            for cam_idx, num_frames in results:
                print(cam_names[cam_idx] + ": {}".format(num_frames))
            print("time elapsed: {:.2f} second".format(end_time - start_time))
