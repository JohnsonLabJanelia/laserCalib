import os
import json
import numpy as np

root_folder = "/Users/yanj11/data/2024_12_13/three_cams"
all_cam_folders = os.listdir(root_folder)
all_cam_folders = [item for item in os.listdir(root_folder) if not item.startswith(".")]
print(all_cam_folders)

## load all landmarks and format it
landmarks = {}

for cam in all_cam_folders:
    cam_path = os.path.join(root_folder, cam)
    landmarks_file = cam_path + "/outputs/landmarks.npz"
    landmarks_per_cam = np.load(landmarks_file)
    landmarks_per_cam_dict = {
        "ids": landmarks_per_cam["ids"].tolist(),
        "landmarks": landmarks_per_cam["landmarks"].tolist(),
    }
    landmarks[cam] = landmarks_per_cam_dict


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


res, msg = verify_landmarks(landmarks)
if not res:
    raise ValueError(msg)


def json_write(filename, data):
    try:
        directory = os.path.dirname(os.path.abspath(filename))
        if not os.path.exists(directory):
            os.makedirs(directory)
        with open(os.path.abspath(filename), "w") as f:
            json.dump(data, f, indent=2)
    except ValueError:
        print("Unable to write JSON {}".format(filename))


json_write("/Users/yanj11/data/2024_12_13/rig_cams3/landmarks.json", landmarks)
