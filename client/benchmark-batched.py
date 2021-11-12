'''Client

client will run model and check power consumption.
'''

import torch, sys
import numpy as np
from inspector import get_model_metrics
from tqdm.auto import tqdm
import time
from rq import MeasurementRequest
import sys


if __name__ == '__main__':
    if len(sys.argv) != 6:
        print("Usage: python3 {0} <model>.torchscript height width batch_size server_addr".format(sys.argv[0]))
        sys.exit(1)

    batch_size = int(sys.argv[4])
    rq = MeasurementRequest("http://%s:48090" % (sys.argv[5]))
    try:
        rq.hello()
    except:
        print("ERROR: Measurement server connection failed")
        sys.exit(1)

    start_time = time.time()
    t_model = torch.jit.load(sys.argv[1], map_location=torch.device('cpu')).eval()
    end_time = time.time()
    print('Loading model took %.0fms' % ((end_time - start_time) * 1000.0, ))

    input_shape = [batch_size, 3, *[int(i) for i in sys.argv[2:4]]]

    with torch.no_grad():
        rq.ready(model_name=sys.argv[1])

        print("Warming up ...", end=' ', flush=True)
        for _ in range(3):
            input_feature = torch.zeros(input_shape)
            output = t_model(input_feature)
            if type(output) == torch.Tensor or type(output) == np.ndarray:
                output_shape = list(output.shape)
            elif type(output) == list or type(output) == tuple:
                output_shape = len(output)
            else:
                output_shape = str(type(input)) + ": " + str(output)
            del input_feature, output
        print("Done")

        total_frames = max(8, 20 // batch_size)
        rq.start()

        start_time = time.time()
        print("Inference remaining ...", end=' ', flush=True)
        for i in range(total_frames, 0, -1):
            print(str(i), end=' ', flush=True)
            input_feature = torch.zeros(input_shape)
            output = t_model(input_feature)
            del input_feature, output
        print()
        end_time = time.time()
        print('Inspecting model with %d items took %.0fms' % (total_frames, (end_time - start_time) * 1000.0))
        total_energy_consumption_mwh = rq.end(model_name=sys.argv[1], elapsed_time_sec=int(end_time - start_time), total_frames=total_frames)

    metrics = get_model_metrics(t_model, input_shape)
    print("Metric:", metrics)
    one_batch_taken_ms = int((end_time - start_time) * 1000.0 / total_frames)

    print("==== Cut below (InputTotalFrames, InputSizeCHW, ..., ParamSize#MB, EnergymWh) ====\n")
    print(batch_size)
    print(total_frames)
    print(str(input_shape))
    print(str(output_shape))
    print("%d" % (one_batch_taken_ms, ))
    print("%.2f" % (1000 / one_batch_taken_ms, ))
    print("%d" % (one_batch_taken_ms / batch_size, ))
    print("%.2f" % (1000 / (one_batch_taken_ms / batch_size), ))
    print("%.2f" % (metrics['params_mega']))
    print("%.2f" % (metrics['input_size_mb']))
    print("%.2f" % (metrics['param_size_mb']))
    print("%d" % total_energy_consumption_mwh)
    print("\n==== Cut above ====")