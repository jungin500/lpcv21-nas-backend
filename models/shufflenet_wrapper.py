import numpy as np
import torch
import cv2

from models.wrapper import Wrapper


class ShuffleNetWrapper(Wrapper):
    def __init__(self):
        self.model = torch.hub.load('pytorch/vision:v0.8.0', 'shufflenet_v2_x1_0', pretrained=True).eval()
        self.image_size = (300, 300)

    def forward(self, frame):
        frame = cv2.resize(frame, self.image_size)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = np.transpose(frame, (2, 0, 1)).astype(np.float32)
        tensor = torch.from_numpy(frame / 255)
        tensor = torch.unsqueeze(tensor, 0)
        output = self.model(tensor)

        return True, output
