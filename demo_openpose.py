from configs.app_openpose_config import \
    PRETRAINED_MODEL, THRESHOLD
from openpose import single_person_regressor, multi_person_regressor

import cv2

if __name__ == "__main__":

    output_path = "./output/out.jpg"
    image = cv2.imread(filename="./img/single.jpeg")
    image_marked, keypoints, elapsed_times = single_person_regressor(
        image=image, pretrained_model=PRETRAINED_MODEL,
        threshold=THRESHOLD,
        output_path=output_path
    )
    print("skeleton: ", keypoints)
    print("elapsed time: ", elapsed_times)

    cv2.imshow("skeleton", image_marked)
    cv2.waitKey(0)
