import os
import json
import numpy as np
import pickle

root_folder = "/Users/yanj11/data/rig5cams"
world_coordinate_img = "16_35_55_349.tiff"
charuco_setup_file = os.path.join(root_folder, "charuco_setup.json")


def verify_landmarks(landmarks):
    for view, val in landmarks.items():
        if "ids" not in val:
            return False, "landmarks file must contain the 'ids'"
        if "landmarks" not in val:
            return False, "landmarks file must contain the 'landmarks'"
        unique = len(set(val["ids"])) == len(val["ids"])
        if not unique:
            return False, "landmarks file contains duplicate IDs!"
        number = len(val["landmarks"]) == len(val["ids"])
        if not number:
            return (
                False,
                "the number of IDs and landmarks in landmark file is not the same!",
            )
        if len(val["landmarks"][0]) != 2:
            return (
                False,
                "the landmarks must be defined in a two-dimensional space! {} wrong.".format(
                    view
                ),
            )
    return True, ""


def json_write(filename, data):
    try:
        directory = os.path.dirname(os.path.abspath(filename))
        if not os.path.exists(directory):
            os.makedirs(directory)
        with open(os.path.abspath(filename), "w") as f:
            json.dump(data, f, indent=2)
    except ValueError:
        print("Unable to write JSON {}".format(filename))


def json_read(filename):
    try:
        with open(os.path.abspath(filename)) as f:
            data = json.load(f)
        return data
    except ValueError:
        print("Unable to read JSON {}".format(filename))


charuco_config = json_read(charuco_setup_file)
width = charuco_config["w"]
height = charuco_config["h"]

all_cams = []
for f in os.listdir(root_folder + "/output/intrinsics"):
    if f.endswith(".yaml"):
        all_cams.append(f.split(".")[0])

landmarks = {}
for cam in all_cams:
    landmarks_file = root_folder + "/output/intrinsics/landmarks_{}.pkl".format(cam)
    with open(landmarks_file, "rb") as f:
        landmarks_per_cam_dict = pickle.load(f)
    landmarks[cam] = landmarks_per_cam_dict

all_img_names = []
for key, value in landmarks.items():
    all_img_names.extend(list(value.keys()))
unique_names, counts = np.unique(all_img_names, return_counts=True)

# assign a unique id for each image, and each corner
img_id = 0
landmarks_final = {}
landmarks_global = {}
for i, img_name in enumerate(unique_names):
    if counts[i] > 1:
        for cam in landmarks.keys():
            per_cam_ids = []
            per_cam_landmarks = []
            per_cam_global_ids = []

            if img_name in landmarks[cam]:
                marker_dict = landmarks[cam][img_name]

                for j in range(len(marker_dict["ids"])):
                    point_unique_id = img_id * (width - 1) * (height - 1) + int(
                        marker_dict["ids"][j][0]
                    )
                    per_cam_ids.append(point_unique_id)
                    per_cam_landmarks.append(marker_dict["corners"][j][0].tolist())

                    if img_name == world_coordinate_img:
                        per_cam_global_ids.append(point_unique_id)

                if cam in landmarks_final.keys():
                    landmarks_final[cam]["ids"].extend(per_cam_ids)
                    landmarks_final[cam]["landmarks"].extend(per_cam_landmarks)
                else:
                    landmarks_final[cam] = {
                        "ids": per_cam_ids,
                        "landmarks": per_cam_landmarks,
                    }

                if img_name == world_coordinate_img:
                    landmarks_global[cam] = {
                        "ids": per_cam_global_ids,
                        "landmarks_global": np.squeeze(
                            marker_dict["objpoints"]
                        ).tolist(),
                    }
        img_id = img_id + 1


res, msg = verify_landmarks(landmarks_final)
if not res:
    raise ValueError(msg)
json_write(root_folder + "/landmarks.json", landmarks_final)


max_number_ids = 0
cam_with_max_number_ids = next(iter(landmarks_global.keys()))
for cam, marker in landmarks_global.items():
    if len(marker["ids"]) > max_number_ids:
        cam_with_max_number_ids = cam
        max_number_ids = len(marker["ids"])

json_write(
    root_folder + "/landmarks_global.json", landmarks_global[cam_with_max_number_ids]
)
