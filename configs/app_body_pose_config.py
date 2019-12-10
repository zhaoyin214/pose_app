import os

static_dir = "static"
app_config_name = "poses.hand_pose"
app_name = app_config_name.split(".")[-1]
base_dir = os.path.join("images", *app_config_name.split("."))
img_dir = os.path.join(base_dir, "img")
upload_dir = os.path.join(base_dir, "upload")
output_dir = os.path.join(base_dir, "output")

POSE_PAIRS = [
    [0, 1], [1, 2], [2, 3], [3, 4],
    [0, 5], [5, 6], [6, 7], [7, 8],
    [0, 9], [9, 10], [10, 11], [11, 12],
    [0, 13], [13, 14], [14, 15], [15, 16],
    [0, 17], [17, 18], [18, 19], [19, 20]
]

PRETRAINED_MODEL = {
    "proto": "./poses/hand_pose/hand_pose/models/pose_deploy.prototxt",
    "weights": "./poses/hand_pose/hand_pose/models/pose_iter_102000.caffemodel",
    "input_height": 368,
    "num_keypoints": 21,
    "pose_pairs": POSE_PAIRS,
}

THRESHOLD = 0.1

