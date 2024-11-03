import argparse
import pickle as pkl
import numpy as np
import matplotlib.pyplot as plt
from lasercalib.rigid_body import load_camera_parameters_from_yaml, Unproject
import os
import json
import glob
import seaborn as sns

parser = argparse.ArgumentParser()
parser.add_argument('-c', '--config', type=str, required=True)

args = parser.parse_args()
config_dir = args.config

with open(config_dir + '/config.json', 'r') as f:
    calib_config = json.load(f)

root_dir = calib_config['root_dir']
laser_datasets = calib_config['lasers']
cam_name_for_3d_init = calib_config['cam_name_for_3d_init']
min_num_cam_per_point = calib_config['min_num_cam_per_point']
z_gt = calib_config['z_gt']
cam_serials = calib_config['cam_serials']
calib_init = calib_config['calib_init']
cam_names = []
for cam_serial in cam_serials:
    cam_names.append("Cam" + cam_serial)
n_cams = len(cam_names)
print("Number of cameras: ", n_cams)

dataset_all = []
for dataset_idx in range(len(laser_datasets)):
    frame_start = calib_config['frames'][dataset_idx][0]
    frame_end = calib_config['frames'][dataset_idx][1]
    n_pts = frame_end - frame_start

    centroids = np.zeros((n_pts, 2, n_cams))
    centroids[:] = np.nan

    for camera_idx, cam_name in enumerate(cam_names):
        one_centroid_file = config_dir + "/results/{}/{}_centroids.pkl".format(laser_datasets[dataset_idx], cam_names[camera_idx])
        with open(one_centroid_file, 'rb') as f:
            centroids[:,:,camera_idx] = pkl.load(f)

    # flip xy (regionprops orders)
    centroids = np.flip(centroids, axis=1)
    cam_idx_3dpts = cam_names.index(cam_name_for_3d_init)
    # filter points
    # TODO: add assert, half of the points available on each camera 
    keep = np.zeros(shape=(n_pts,), dtype=bool)
    for i in range(n_pts):
        v = centroids[i, 0, :]
        if ((np.sum(~np.isnan(v)) >= min_num_cam_per_point) and (~np.isnan(v[cam_idx_3dpts]))):
            keep[i] = True

    in_pts = centroids[keep, :, :]
    n_in_pts = in_pts.shape[0]
    print("Number of kept points: ", n_in_pts)
    n_obs = np.sum(~np.isnan(in_pts[:,0,:].ravel()))

    fig, axs = plt.subplots(1, n_cams, sharey=True)
    plt.title('2D points found on all cameras')
    for i in range(n_cams):
        colors = np.linspace(0, 1, n_in_pts)
        axs[i].scatter(in_pts[:,0,i], in_pts[:,1,i], s=10, c=colors, alpha=0.5)
        axs[i].plot(in_pts[:,0,i], in_pts[:,1,i])
        axs[i].title.set_text(cam_names[i])
        axs[i].invert_yaxis()
    plt.show()

    # create camera_ind variable
    camera_ind = np.zeros(shape=(n_obs,), dtype=int)
    point_ind = np.zeros(shape=(n_obs,), dtype=int)
    points_2d = np.zeros(shape=(n_obs, 2), dtype=float)

    ind = 0
    for i in range(in_pts.shape[0]):
        for j in range(n_cams):
            if (np.isnan(in_pts[i, 0, j])):
                continue
            camera_ind[ind] = j
            point_ind[ind] = i
            points_2d[ind, :] = in_pts[i, :, j].copy()
            ind += 1

    init_camera_filename = os.path.join(config_dir, calib_init) + "/{}.yaml".format(cam_name_for_3d_init)
    init_camera_parameter = load_camera_parameters_from_yaml(init_camera_filename)

    projected_points = in_pts[:, :, cam_idx_3dpts]
    projected_points_z = np.zeros(projected_points.shape[0]) + z_gt[dataset_idx]
    unprojected_points = Unproject(projected_points, projected_points_z, 
                                init_camera_parameter['camera_matrix'], 
                                init_camera_parameter['distortion_coefficients'], 
                                init_camera_parameter['rc_ext'], 
                                init_camera_parameter['tc_ext'])

    points_3d = unprojected_points
    np.set_printoptions(suppress=True)
    print("Unprojected points: ", points_3d)
    one_dataset = {
        "n_cams": n_cams,
        "n_pts": n_in_pts,
        "points_2d": points_2d,
        "points_3d": points_3d,
        "camera_ind": camera_ind,
        "point_ind": point_ind
    }
    dataset_all.append(one_dataset)

my_palette = sns.color_palette("rocket_r", len(dataset_all))

ax = []
fig = plt.figure()
ax.append(fig.add_subplot(1, 1, 1, projection='3d'))

for i in range(len(dataset_all)):
    points_3d = dataset_all[i]['points_3d']
    ax[0].scatter(points_3d[:, 0], points_3d[:, 1], points_3d[:, 2], color=my_palette[i], alpha=0.1)
ax[0].set_title("laser points")
plt.show()


output_dir = os.path.join(config_dir, 'results')
with open(output_dir + '/points_dataset.pkl', 'wb') as f:
    pkl.dump(dataset_all, f)