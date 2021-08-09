from framework.picontroller import PiControllerClient
from framework.slave.infer import PiInference, generate_preprocess_fn, Timer
from models.efficientdet_wrapper import EfficientDetWrapper
from models.shufflenet_wrapper import ShuffleNetWrapper
from models.mb3_ssd_wrapper import MobileNetV3SSDLiteWrapper
import asyncio
import torch
import numpy as np

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
        try:
            message = await picontrol.recv_message_async()
        except ConnectionResetError:
            print("ERROR: Server connection reset")
            sys.exit(1)
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
                wrapper = ShuffleNetWrapper()
            elif model_name == 'mb3_ssd_pretrained':
                wrapper = MobileNetV3SSDLiteWrapper()
            elif model_name == 'efficientdet':
                wrapper = EfficientDetWrapper()
            else:
                print("ERROR: Model %s not supported!" % model_name)
                await asyncio.sleep(1)
                await picontrol.send_message_async("NOMODEL")
                continue

            # do some model loads
            infer = PiInference(
                wrapper=wrapper,
                video_path='../5p4b_01A2.m4v'
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
                if type(metric) == torch.Tensor:
                    print("Model result tensor shape: ", metric.shape)
                elif type(metric) == np.ndarray:
                    print("Model result ndarray shape: ", metric.shape)
                elif type(metric) == tuple:
                    print("Model results: %d items", len(metric))
                else:
                    print("Model result: ", metric)

                del metric

                # infer.metrics.append(metric)
                await picontrol.send_message_async(" ".join(["FRAME", "%.4f" % (timer.end(key='single') * 1000, )]))
            total_elapsed_time = timer.end()

            await picontrol.send_message_async(" ".join(["ENDINFER", "%.4f" % total_elapsed_time]))

            # do some model unloads
            del wrapper, infer
        elif command == 'BYE':
            try:
                await picontrol.send_message_async("BYE")
            except:
                pass
            return

if __name__ == '__main__':
    print("== PiControlClient v1.2 by LimeOrangePie ==")
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