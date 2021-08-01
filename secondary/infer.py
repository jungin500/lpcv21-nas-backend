from controller.util import Timer
import torch
import cv2
import numpy as np


def generate_preprocess_fn(image_size=(300, 300)):
    # Example process function: resize image
    # and return [1, 3, 300, 300] Tensor
    def preprocess_fn(frame):
        frame = cv2.resize(frame, image_size)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = np.transpose(frame, (2, 0, 1)).astype(np.float32)
        tensor = torch.from_numpy(frame / 255)
        tensor = torch.unsqueeze(tensor, 0)
        return tensor

    return preprocess_fn

# Example postprocess function: just returns None
def postprocess_fn(tensor):
    return None


class PiInference(object):
    """ Model inference and metric generator

    Inferences a model given from server, returns metric/execution time
    and send these information to server.

    :Author:
        Jung-In An <ji5489@gmail.com>
        CVMIPLab, Kangwon National University
    """

    def __init__(self, load_model, video_path,
                 process_image=generate_preprocess_fn(), process_output=postprocess_fn):
        self.process_image = process_image
        self.process_output = process_output

        self.model = load_model()
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            print("VideoCapture: can't open video")

        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        self.metrics = []

    def __del__(self):
        try:
            self.cap.release()
            del self.model
        except AttributeError:
            pass

    def warm(self):
        print("Warming up ... ", end='', flush=True)
        for i in range(5):
            ret, metric = self.run_once()
            if not ret:
                break
            print("%d " % i, end='', flush=True)
        print("Done. rewinding video track ...")

        # go to first frame of video file
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    def run(self):
        timer = Timer()
        timer.start()
        while True:
            ret, metric = self.run_once()
            if not ret:
                break
            self.metrics.append(metric)
        elapsed_time = timer.end()

        return elapsed_time, self.metrics

    def run_once(self):
        ret, frame = self.cap.read()
        if not ret:
            return False, None
        tensor = self.process_image(frame)
        output = self.model(tensor)
        metric = self.process_output(output)

        return True, metric

    def get_video_length(self):
        return int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))