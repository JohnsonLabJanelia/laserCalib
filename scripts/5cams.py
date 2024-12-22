import rerun as rr
import numpy as np
import argparse
import cv2


parser = argparse.ArgumentParser()
parser.add_argument("-i", "--calibration_dir", type=str, required=True)
args = parser.parse_args()


calibration_dir = args.calibration_dir

serial_to_order = {
    "2002496": 0,
    "2002488": 2,
    "2002489": 4,
    "2002490": 6,
    "710038": 16,
}


def load_yaml_file(yaml_cam_name):
    cam_params = {}
    fs = cv2.FileStorage(yaml_cam_name, cv2.FILE_STORAGE_READ)
    cam_params["camera_matrix"] = fs.getNode("camera_matrix").mat()
    cam_params["distortion_coefficients"] = fs.getNode("distortion_coefficients").mat()
    cam_params["tc_ext"] = fs.getNode("tc_ext").mat()
    cam_params["rc_ext"] = fs.getNode("rc_ext").mat()
    return cam_params


rr.init("rerun_example_dna_abacus")
rr.spawn()

rr.set_time_sequence("stable_time", 0)
rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Y_UP, static=True)
rr.log("arena", rr.Boxes3D(centers=[0, 0, -174.6], half_sizes=[762, 762, 174.6]))
rr.log("shelter", rr.Boxes3D(centers=[862, 0, -174.6], half_sizes=[100, 100, 174.6]))


for serial, order in serial_to_order.items():
    ## load yaml file
    yaml_file_name = calibration_dir + "/Cam{}.yaml".format(serial)
    cam_params = load_yaml_file(yaml_file_name)

    # compute camera pose
    rotation = cam_params["rc_ext"].T
    translation = -np.matmul(rotation, cam_params["tc_ext"][:, 0])

    resolution = [3208, 2200]

    # rr.log("world/camera/{}_{}".format(order, calib_date), rr.Transform3D(translation=translation, mat3x3=rotation))
    rr.log(
        "world/camera/{}".format(order),
        rr.Transform3D(translation=translation, mat3x3=rotation),
    )

    rr.log(
        # "world/camera/{}_{}".format(order, calib_date),
        "world/camera/{}".format(order),
        rr.Pinhole(
            resolution=resolution,
            image_from_camera=cam_params["camera_matrix"],
            camera_xyz=rr.ViewCoordinates.RDF,
        ),
    )
