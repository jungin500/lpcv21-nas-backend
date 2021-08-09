import numpy as np
import torch
import cv2

import os, sys

from models.wrapper import Wrapper

sys.path.append(os.path.join(os.getcwd(), 'models', 'efficientdet'))
from models.efficientdet.utils.utils import aspectaware_resize_padding
from models.efficientdet.backbone import EfficientDetBackbone


class EfficientDetWrapper(Wrapper):
    def __init__(self):
        self.model = self._load_model()

    def _load_model(self):
        compound_coef = 0

        # replace this part with your project's anchor config
        anchor_ratios = [(1.0, 1.0), (1.4, 0.7), (0.7, 1.4)]
        anchor_scales = [2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)]

        obj_list = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
                    'traffic light',
                    'fire hydrant', '', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
                    'horse', 'sheep',
                    'cow', 'elephant', 'bear', 'zebra', 'giraffe', '', 'backpack', 'umbrella', '', '',
                    'handbag', 'tie',
                    'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',
                    'baseball glove',
                    'skateboard', 'surfboard', 'tennis racket', 'bottle', '', 'wine glass', 'cup', 'fork',
                    'knife', 'spoon',
                    'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog',
                    'pizza', 'donut',
                    'cake', 'chair', 'couch', 'potted plant', 'bed', '', 'dining table', '', '', 'toilet',
                    '', 'tv',
                    'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
                    'sink',
                    'refrigerator', '', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
                    'toothbrush']

        model = EfficientDetBackbone(compound_coef=compound_coef, num_classes=len(obj_list),
                                     ratios=anchor_ratios, scales=anchor_scales)
        model.load_state_dict(
            torch.load(f'models/efficientdet/weights/efficientdet-d{compound_coef}.pth', map_location='cpu'))
        model.requires_grad_(False)
        model.eval()

        return model

    def forward(self, frame):
        max_size = 512
        mean = (0.406, 0.456, 0.485)
        std = (0.225, 0.224, 0.229)

        # image = np.expand_dims(image, 0)

        normalized_imgs = np.array([(img[..., ::-1] / 255 - mean) / std for img in frame])
        framed_imgs = aspectaware_resize_padding(normalized_imgs, max_size, max_size, means=None)[0]
        framed_imgs = np.expand_dims(framed_imgs, 0)
        x = torch.from_numpy(framed_imgs).to(torch.float32).permute(0, 3, 1, 2)

        output = self.model(x)

        return True, output
