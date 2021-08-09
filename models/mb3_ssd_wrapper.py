import numpy as np
import torch
import cv2

import os, sys

from models.wrapper import Wrapper

sys.path.append(os.path.join(os.getcwd(), 'models', 'mb3_ssd'))
from models.mb3_ssd.vision.ssd.mobilenet_v3_ssd_lite import create_mobilenetv3_ssd_lite, create_mobilenetv3_ssd_lite_predictor


class MobileNetV3SSDLiteWrapper(Wrapper):
    def __init__(self):
        self.predictor = create_mobilenetv3_ssd_lite_predictor(
            create_mobilenetv3_ssd_lite(num_classes=3, width_mult=1.0, is_test=True),
            nms_method='hard', device='cpu'
        )

    def forward(self, frame):
        output = self.predictor.predict(frame)

        return True, output
