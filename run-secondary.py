import cv2

from controller.picontroller import PiControllerClient
from secondary.infer import PiInference, generate_preprocess_fn, Timer
import asyncio
import torch
import base64

import os
import sys

sys.path.append(os.path.join(os.getcwd(), 'models', 'mb3_ssd'))
from models.mb3_ssd.vision.ssd.mobilenet_v3_ssd_lite import create_mobilenetv3_ssd_lite, create_mobilenetv3_ssd_lite_predictor

sys.path.append(os.path.join(os.getcwd(), 'models', 'efficientdet'))
from models.efficientdet.efficientdet.utils import BBoxTransform, ClipBoxes
from models.efficientdet.utils.utils import preprocess, invert_affine, postprocess, STANDARD_COLORS, standard_to_bgr, get_index_label, plot_one_box
from models.efficientdet.backbone import EfficientDetBackbone

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
                load_model = lambda: create_mobilenetv3_ssd_lite_predictor(
                    create_mobilenetv3_ssd_lite(num_classes=3, width_mult=1.0, is_test=True),
                    nms_method='hard', device='cpu'
                ).predict

                process_image = lambda image: image
            elif model_name == 'efficientdet':
                def load_model():
                    compound_coef = 0
                    force_input_size = None  # set None to use default size
                    img_path = 'test/img.png'

                    # replace this part with your project's anchor config
                    anchor_ratios = [(1.0, 1.0), (1.4, 0.7), (0.7, 1.4)]
                    anchor_scales = [2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)]

                    threshold = 0.2
                    iou_threshold = 0.2

                    use_float16 = False

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

                    color_list = standard_to_bgr(STANDARD_COLORS)
                    # tf bilinear interpolation is different from any other's, just make do
                    input_sizes = [512, 640, 768, 896, 1024, 1280, 1280, 1536, 1536]
                    input_size = input_sizes[compound_coef] if force_input_size is None else force_input_size
                    ori_imgs, framed_imgs, framed_metas = preprocess(img_path, max_size=input_size)

                    x = torch.stack([torch.from_numpy(fi) for fi in framed_imgs], 0)
                    x = x.to(torch.float32 if not use_float16 else torch.float16).permute(0, 3, 1, 2)

                    model = EfficientDetBackbone(compound_coef=compound_coef, num_classes=len(obj_list),
                                                 ratios=anchor_ratios, scales=anchor_scales)
                    model.load_state_dict(torch.load(f'weights/efficientdet-d{compound_coef}.pth', map_location='cpu'))
                    model.requires_grad_(False)
                    model.eval()

                    return model

                process_image = lambda image: cv2.resize(image, (512, 512))
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

            if short_infer:
                run_frames = 50

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
    picontrol.connect()

    asyncio.get_event_loop().run_until_complete(main(picontrol))
    exit(0)