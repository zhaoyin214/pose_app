import os

static_dir = "static"
app_config_name = "poses.openpose"
app_name = app_config_name.split(".")[-1]
base_dir = os.path.join("images", *app_config_name.split("."))
img_dir = os.path.join(base_dir, "img")
upload_dir = os.path.join(base_dir, "upload")
output_dir = os.path.join(base_dir, "output")

"""
COCO Keypoints challenge
Nose – 0, Neck – 1, Right Shoulder – 2, Right Elbow – 3, Right Wrist – 4,
Left Shoulder – 5, Left Elbow – 6, Left Wrist – 7, Right Hip – 8,
Right Knee – 9, Right Ankle – 10, Left Hip – 11, Left Knee – 12,
LAnkle – 13, Right Eye – 14, Left Eye – 15, Right Ear – 16,
Left Ear – 17, Background – 18
"""
POSE_PAIRS = [
    [0, 1],
    [0, 14], [14, 16],
    [0, 15], [15, 17],
    [1, 2], [2, 3], [3, 4],
    [1, 5], [5, 6], [6, 7],
    [1, 8], [8, 9], [9, 10],
    [1, 11], [11, 12], [12, 13],
    [2, 17],
    [5, 16],
]

# index of pafs correspoding to the POSE_PAIRS
# e.g for POSE_PAIR(1, 2), the PAFs are located at indices (31, 32) of output, Similarly, (1, 5) -> (39, 40) and so on.
PAFS = [
    [47, 48],
    [49, 50], [53, 54],
    [51, 52], [55, 56],
    [31, 32], [33, 34], [35, 36],
    [39, 40], [41, 42], [43, 44],
    [19, 20], [21, 22], [23, 24],
    [25, 26], [27, 28], [29, 30],
    [37, 38],
    [45, 46],
]

PRETRAINED_MODEL = {
    "proto": "./openpose/models/openpose_coco_deploy.prototxt",
    "weights": "./openpose/models/openpose_coco_iter_440000.caffemodel",
    "resize": 368,
    # "resize": [368, 368],
    "num_keypoints": 18,
    "pose_pairs": POSE_PAIRS,
    "pafs": PAFS,
}

THRESHOLD = 0.1

