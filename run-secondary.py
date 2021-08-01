import cv2

from controller.picontroller import PiControllerClient
from secondary.infer import PiInference, generate_preprocess_fn, Timer
import asyncio
import torch
import base64

import os
import sys

async def main(picontrol):
    """ Raspberry Pi NAS(Network Architecture Search) backend implementation

    Raspberry Pi have responsible to infer with given model, generate
    a metric, and measure inference execution time. Once those jobs are
    finished, Pi sends server that information.

    :Author:
        Jung-In An <ji5489@gmail.com>
        CVMIPLab, Kangwon National University
    """

    while True:
        message = await picontrol.recv_message_async()
        if message.strip() == '':
            continue
        print("<", message)

        message_slice = message.split(' ')
        command = message_slice[0].upper()
        args = message_slice[1:]

        if command == 'SUMMARY':
            model_names = ' '.join(args).split(':')
            print("-- Model summary:", model_names, "--")
        # elif command == 'BINARY':
        #     pass
        #     base64_body = ' '.join(message_slice)
        #     base64.b64
        elif command == 'LOADMODEL':
            subcommand = args[0]
            if subcommand.upper() == 'SHORT':
                model_name = ' '.join(args[1:]).lower().strip()
                short_infer = True
            else:
                model_name = ' '.join(args).lower().strip()
                short_infer = False

            # def load_model():
            #     model = torch.hub.load('pytorch/vision:v0.9.0', 'shufflenet_v2_x1_0', pretrained=True)
            #     model.eval()
            #     return

            if model_name == 'shufflenet':
                load_model = lambda: torch.hub.load('pytorch/vision:v0.9.0', 'shufflenet_v2_x1_0', pretrained=True).eval()
                process_image = generate_preprocess_fn(image_size=(224, 224))
            elif model_name == 'mb3-small-ssd-lite-1.0':
                sys.path.append(os.path.join(os.getcwd(), 'models', 'mb3_ssd'))
                from models.mb3_ssd.vision.ssd.mobilenet_v3_ssd_lite import create_mobilenetv3_ssd_lite, create_mobilenetv3_ssd_lite_predictor
                
                load_model = lambda: create_mobilenetv3_ssd_lite_predictor(
                    create_mobilenetv3_ssd_lite(num_classes=3, width_mult=1.0, is_test=True),
                    nms_method='hard', device='cpu'
                ).predict

                process_image = lambda image: image
            elif model_name == 'efficientdet':
                sys.path.append(os.path.join(os.getcwd(), 'models', 'efficientdet'))
                from models.efficientdet.efficientdet.utils import BBoxTransform, ClipBoxes
                from models.efficientdet.utils.utils import STANDARD_COLORS, standard_to_bgr, aspectaware_resize_padding
                from models.efficientdet.backbone import EfficientDetBackbone

                import numpy as np

                def load_model():
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
                    model.load_state_dict(torch.load(f'models/efficientdet/weights/efficientdet-d{compound_coef}.pth', map_location='cpu'))
                    model.requires_grad_(False)
                    model.eval()

                    return model

                def process_image(image):
                    max_size = 512
                    mean = (0.406, 0.456, 0.485)
                    std = (0.225, 0.224, 0.229)

                    # image = np.expand_dims(image, 0)

                    normalized_imgs = np.array([(img[..., ::-1] / 255 - mean) / std for img in image])
                    framed_imgs = aspectaware_resize_padding(normalized_imgs, max_size, max_size, means=None)[0]
                    framed_imgs = np.expand_dims(framed_imgs, 0)
                    x = torch.from_numpy(framed_imgs).to(torch.float32).permute(0, 3, 1, 2)

                    return x

            else:
                print("ERROR: Model %s not supported!" % model_name)
                await asyncio.sleep(3)
                await picontrol.send_message_async("NOMODEL")
                await asyncio.sleep(1)
                continue

            # do some model loads
            infer = PiInference(
                load_model=load_model,
                video_path='../5p4b_01A2.m4v',
                process_image=process_image
            )

            infer.warm()

            await picontrol.send_message_async(" ".join(["BEGININFER", "%d" % infer.get_video_length()]))

            # if short_infer:
            #     run_frames = 50

            timer = Timer()
            timer.start()
            while True:
                timer.start(key='single')
                ret, metric = infer.run_once()
                if not ret:
                    break
                infer.metrics.append(metric)
                await picontrol.send_message_async(" ".join(["FRAME", "%.4f" % (timer.end(key='single') * 1000, )]))
            total_elapsed_time = timer.end()

            await picontrol.send_message_async(" ".join(["ENDINFER", "%.4f" % total_elapsed_time]))
            # do some model unloads
        elif command == 'BYE':
            try:
                await picontrol.send_message_async("BYE")
            except:
                pass
            return

if __name__ == '__main__':
    print("== PiControlClient v1.1 by LimeOrangePie ==")
    picontrol = PiControllerClient('ws://172.24.90.200:12700')
    while True:
        try:
            print("Trying to connect to server ... ", end='', flush=True)
            picontrol.connect()
            print("Connection success")
            break
        except ConnectionRefusedError:
            print("Connection failed")

    print("Disabling pytorch CUDA ext")
    torch.cuda.is_available = lambda: False

    asyncio.get_event_loop().run_until_complete(main(picontrol))
    exit(0)