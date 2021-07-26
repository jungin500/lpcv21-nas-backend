from controller.picontroller import PiControllerClient
from secondary.infer import PiInference, generate_preprocess_fn, Timer
import asyncio
import torch

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
        elif command == 'LOADMODEL':
            model_name = ' '.join(args)

            # def load_model():
            #     model = torch.hub.load('pytorch/vision:v0.9.0', 'shufflenet_v2_x1_0', pretrained=True)
            #     model.eval()
            #     return

            if model_name.lower() == 'shufflenet':
                load_model = lambda: torch.hub.load('pytorch/vision:v0.9.0', 'shufflenet_v2_x1_0', pretrained=True).eval()
                process_image = generate_preprocess_fn(image_size=(224, 224))
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

            timer = Timer()
            timer.start()
            while True:
                ret, metric = infer.run_once()
                if not ret:
                    break
                infer.metrics.append(metric)
            elapsed_time = timer.end()
            picontrol.send_message_async(" ".join(["FRAME", "%.4f" % elapsed_time]))

            await picontrol.send_message_async("ENDINFER")
            # do some model unloads
        elif command == 'BYE':
            try:
                await picontrol.send_message_async("BYE")
            except:
                pass
            return

if __name__ == '__main__':
    print("== PiControlClient v1.0 by LimeOrangePie ==")
    picontrol = PiControllerClient('ws://172.24.90.200:12700')
    picontrol.connect()

    asyncio.get_event_loop().run_until_complete(main(picontrol))
    exit(0)