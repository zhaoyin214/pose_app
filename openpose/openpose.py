#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@file    :   openpose.py
@time    :   2020/01/15 14:45:15
@author  :   XiaoY
@version :   1.0
@contact :   zhaoyin214@qq.com
@license :   (c)copyright XiaoY
@desc    :   openpose for localizing anatomical keypoints
"""

__author__ = "XiaoY"


import cv2
import time
import numpy as np

class OpenPose(object):
    """
    openpose for localizing anatomical keypoints
        - confidence maps for part detection
        - part affinity fields for part association
    """

    def __init__(self, pretrained_model):

        self._net = cv2.dnn.readNetFromCaffe(
            prototxt=pretrained_model["proto"],
            caffeModel=pretrained_model["weights"]
        )
        if pretrained_model.get("resize"):
            self._input_size = pretrained_model["resize"]
        else:
            self._input_size = None

    def predict(self, image):

        image_height, image_width = image.shape[: 2]

        if self._input_size is None:
            input_size = (image_width, image_height)
        else:
            if isinstance(self._input_size, int):
                input_height = self._input_size
                aspect_ratio = image_width / image_height
                input_width = int(aspect_ratio * input_height)
                input_size = (input_width, input_height)
            elif isinstance(self._input_size, list):
                input_size = tuple(self._input_size)
            else:
                input_size = self._input_size

        input_blob = cv2.dnn.blobFromImage(
            image=image,
            scalefactor=1 / 255,
            size=input_size,
            mean=(0, 0, 0),
            swapRB=False,
            crop=False
        )

        t_net = time.time()
        self._net.setInput(input_blob)
        output = self._net.forward()
        t_net = time.time() - t_net

        return output, t_net

    def __call__(self, image):
        return self.predict(image)


def single_person_regressor(image, pretrained_model, threshold, output_path):
    """
    """

    t_total = time.time()

    regressor = OpenPose(pretrained_model=pretrained_model)
    num_keypoints = pretrained_model["num_keypoints"]
    pose_pairs = pretrained_model["pose_pairs"]

    image_height, image_width = image.shape[: 2]
    output, t_net = regressor(image)

    # empty list to store the detected keypoints
    points = []
    for i in range(num_keypoints):
        # confidence map of corresponding hand's part
        conf_map = output[0, i, :, :]

        # cv2.imshow("conf", conf_map)
        # cv2.waitKey(500)

        conf_map = cv2.resize(
            src=conf_map, dsize=(image_width, image_height)
        )

        # cv2.imshow("conf_resize", conf_map)
        # cv2.waitKey(500)

        # find golbal maxima of the confidence map
        _, conf, _, point = cv2.minMaxLoc(src=conf_map)

        if conf > threshold:
            # add the point to the list
            # if the probability is greater than the threshold
            points.append((int(point[0]), int(point[1])))

            cv2.circle(
                img=image,
                center=(int(point[0]), int(point[1])),
                radius=8,
                color=(0, 255, 255),
                thickness=-1,
                lineType=cv2.FILLED
            )
            cv2.putText(
                img=image,
                text="{}".format(i),
                org=(int(point[0]), int(point[1])),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=1,
                color=(0, 0, 255),
                thickness=2,
                lineType=cv2.LINE_AA
            )

        else:
            points.append(None)

    # skeleton
    for pair in pose_pairs:
        root = pair[0]
        child = pair[1]
        if points[root] and points[child]:
            cv2.line(
                img=image,
                pt1=points[root],
                pt2=points[child],
                color=(0, 255, 255),
                thickness=2
            )

    cv2.imwrite(filename=output_path, img=image)

    t_total = time.time() - t_total
    elapsed_times = (t_net, t_total)

    return image, points, elapsed_times


def multi_person_regressor(image, pretrained_model, threshold, output_path):
    """
    """
    image_height, image_width = image.shape[: 2]

