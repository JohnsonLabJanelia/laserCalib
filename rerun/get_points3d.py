import argparse
import pickle as pkl
import numpy as np
import matplotlib.pyplot as plt
from rigid_body import load_camera_parameters_from_yaml, Unproject
import os

parser = argparse.ArgumentParser()
parser.add_argument('-f', '--folders', nargs='+', default=[])
parser.add_argument('-z', '--elevate', nargs="+", type=int)
parser.add_argument('-o', '--output_dir', type=str, required=True)
parser.add_argument('-n', '--cam_name_for_3d_init', type=str, required=True)
parser.add_argument('--min_num_cam_per_point', type=int, default=4)
parser.add_argument('--shift_3d', type=float, default=1.0)

args = parser.parse_args()
input_folders = args.folders
z_per_dataset = args.elevate
cam_name_for_3d_init = args.cam_name_for_3d_init
shift_3d = args.shift_3d
min_num_cam_per_point = args.min_num_cam_per_point
output_dir = args.output_dir

dataset_all = []
for dataset_idx, root_dir in enumerate(input_folders):
    print(root_dir)
    with open(root_dir + "/results/centroids.pkl", 'rb') as file:
        centroids_dict = pkl.load(file)

    pts = centroids_dict['centroids']
    cam_names = centroids_dict['cam_names']
    
    import pdb; pdb.set_trace()
    # flip xy (regionprops orders)
    pts = np.flip(pts, axis=1)
    nPts = pts.shape[0]
    nCams = pts.shape[2]


    cam_idx_3dpts = cam_names.index(cam_name_for_3d_init)
    ## use 3d triangulated points to initialize
    keep = np.zeros(shape=(nPts,), dtype=bool)
    for i in range(nPts):
        v = pts[i, 0, :]
        if ((np.sum(~np.isnan(v)) >= min_num_cam_per_point) and (~np.isnan(v[cam_idx_3dpts]))):
            keep[i] = True

    inPts = pts[keep, :, :]
    nPts = inPts.shape[0]
    print("nPts: ", nPts)

    nObs = np.sum(~np.isnan(inPts[:,0,:].ravel()))
    print("nObs: ", nObs)

    fig, axs = plt.subplots(1, nCams, sharey=True)
    plt.title('2D points found on all cameras')
    for i in range(nCams):
        colors = np.linspace(0, 1, nPts)
        axs[i].scatter(inPts[:,0,i], inPts[:,1,i], s=10, c=colors, alpha=0.5)
        axs[i].plot(inPts[:,0,i], inPts[:,1,i])
        axs[i].title.set_text('cam' + str(i))
        axs[i].invert_yaxis()
    plt.show()

    # create camera_ind variable
    camera_ind = np.zeros(shape=(nObs,), dtype=int)
    point_ind = np.zeros(shape=(nObs,), dtype=int)
    points_2d = np.zeros(shape=(nObs, 2), dtype=float)

    ind = 0
    for i in range(nPts):
        for j in range(nCams):
            if (np.isnan(inPts[i, 0, j])):
                continue
            camera_ind[ind] = j
            point_ind[ind] = i
            points_2d[ind, :] = inPts[i, :, j].copy()
            ind += 1


    init_camera_filename = root_dir + "/calib_init/{}.yaml".format(cam_name_for_3d_init)
    init_camera_parameter = load_camera_parameters_from_yaml(init_camera_filename)

    projected_points = inPts[:, :, cam_idx_3dpts]
    projected_points_z = np.zeros(projected_points.shape[0]) + z_per_dataset[dataset_idx]
    unprojected_points = Unproject(projected_points, projected_points_z, 
                                init_camera_parameter['camera_matrix'], 
                                init_camera_parameter['distortion_coefficients'], 
                                init_camera_parameter['rc_ext'], 
                                init_camera_parameter['tc_ext'])

    points_3d = unprojected_points
    np.set_printoptions(suppress=True)
    print("Unprojected points: ", points_3d)
    one_dataset = {
        "nCams": nCams,
        "nPts": nPts,
        "points_2d": points_2d,
        "points_3d": points_3d,
        "camera_ind": camera_ind,
        "point_ind": point_ind
    }
    dataset_all.append(one_dataset)


results_dir = os.path.join(output_dir, 'results')
with open(results_dir + '/points_dataset.pkl', 'wb') as f:
    pkl.dump(dataset_all, f)
