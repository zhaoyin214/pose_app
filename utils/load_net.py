import cv2

class Net(object):

    def __init__(self, pretrained_model):

        if pretrained_model["backend"].lower() == "tf":
            self._net = cv2.dnn.readNetFromTensorflow(
                config=pretrained_model["proto"],
                model=pretrained_model["weights"]
            )
        elif pretrained_model["backend"].lower() == "caffe":
            self._net = cv2.dnn.readNetFromCaffe(
                prototxt=pretrained_model["proto"],
                caffeModel=pretrained_model["weights"]
            )

        self._input_height = pretrained_model.get("input_height")
        self._input_width = pretrained_model.get("input_width")
        self._swap_rb = pretrained_model.get("swap_rb")
        self._crop = pretrained_model.get("crop")
        self._mean = pretrained_model.get("mean")
        self._scale_factor = pretrained_model.get("scale_factor")

        if not self._swap_rb:
            self._swap_rb = False
        if not self._crop:
            self._crop = False
        if not self._mean:
            self._mean =(0, 0, 0)
        if not self._scale_factor:
            self._scale_factor = 1 / 255
