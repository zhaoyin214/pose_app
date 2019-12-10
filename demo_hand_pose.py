from configs.app_hand_pose_config import \
    PRETRAINED_MODEL, THRESHOLD
from hand_pose import hand_pose_regressor

import cv2

if __name__ == "__main__":

    output_path = "./output/out.jpg"
    image = cv2.imread(filename="./img/hand.jpg")
    image_marked, hand_keypoints, elapsed_times = hand_pose_regressor(
        image=image, pretrained_model=PRETRAINED_MODEL,
        threshold=THRESHOLD,
        output_path=output_path
    )
    print("skeleton: ", hand_keypoints)
    print("elapsed time: ", elapsed_times)

    cv2.imshow("skeleton", image_marked)
    cv2.waitKey(0)
