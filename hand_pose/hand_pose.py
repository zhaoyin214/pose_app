import cv2
import time
import numpy as np

class HandPose(object):
    """
    hand pose keypoints regression
    """

    def __init__(self, pretrained_model):
        self._net = cv2.dnn.readNetFromCaffe(
            prototxt=pretrained_model["proto"],
            caffeModel=pretrained_model["weights"]
        )
        self._hand_pose_pairs = pretrained_model["pose_pairs"]
        self._num_keypoints = pretrained_model["num_keypoints"]
        self._input_height = pretrained_model["input_height"]

    def predict(self, image, threshold, output_path):

        # image = cv2.UMat(image)
        image_height, image_width = image.shape[0 : 2]
        aspect_ratio = image_width / image_height
        # input image dimensions for the network
        input_width = int(aspect_ratio * self._input_height)
        input_blob = cv2.dnn.blobFromImage(
            image=image, scalefactor=1 / 255,
            size=(input_width, self._input_height),
            mean=(0, 0, 0), swapRB=False, crop=False
        )


        t_net = time.time()
        self._net.setInput(input_blob)
        output = self._net.forward()
        t_net = time.time() - t_net

        # empty list to store the detected keypoints
        points = []
        for i in range(self._num_keypoints):

            # confidence map of corresponding hand's part.
            heat_map = output[0, i, :, :]
            heat_map = cv2.resize(
                src=heat_map, dsize=(image_width, image_height)
            )

            # find global maxima of the heat_map.
            _, heat, _, point = cv2.minMaxLoc(heat_map)

            if heat > threshold :
                cv2.circle(
                    img=image, center=(int(point[0]), int(point[1])), radius=8,
                    color=(0, 255, 255), thickness=-1, lineType=cv2.FILLED
                )
                cv2.putText(
                    img=image, text="{}".format(i),
                    org=(int(point[0]), int(point[1])),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1,
                    color=(0, 0, 255), thickness=2, lineType=cv2.LINE_AA
                )

                # add the point to the list
                # if the probability is greater than the threshold
                points.append((int(point[0]), int(point[1])))
            else:
                points.append(None)

        # skeleton
        for pair in self._hand_pose_pairs:
            root = pair[0]
            child = pair[1]

            if points[root] and points[child]:
                cv2.line(image, points[root], points[child], (0, 255, 255), 2)

        cv2.imwrite(output_path, image)

        return image, points, t_net

    def __call__(self, image, threshold, output_path):
        return self.predict(
            image=image, threshold=threshold, output_path=output_path
        )


def hand_pose_regressor(image, pretrained_model, threshold, output_path):

    t_total = time.time()
    regressor = HandPose(pretrained_model=pretrained_model)
    image_marked, hand_keypoints, t_net = regressor(
        image=image, threshold=threshold, output_path=output_path
    )
    t_total = time.time() - t_total

    elapsed_times = (t_net, t_total)

    return image_marked, hand_keypoints, elapsed_times

