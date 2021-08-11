import torch.nn
from torchstat import ModelStat
from torchstat.reporter import round_value
from torchinfo import summary

import pandas as pd


# Changing result metric key will also have responsible
# to change MetricResultDict (TODO: fix this!)
MetricResultDict = [
    'params_mega',
    'input_size_mb',
    'param_size_mb',
    'memory_mb',
    'madds',
    'flops_mega',
    'memrw_mb'
]


def process_nodes(collected_nodes):
    data = list()
    for node in collected_nodes:
        name = node.name
        input_shape = ' '.join(['{:>3d}'] * len(node.input_shape)).format(
            *[e for e in node.input_shape])
        output_shape = ' '.join(['{:>3d}'] * len(node.output_shape)).format(
            *[e for e in node.output_shape])
        parameter_quantity = node.parameter_quantity
        inference_memory = node.inference_memory
        MAdd = node.MAdd
        Flops = node.Flops
        mread, mwrite = [i for i in node.Memory]
        duration = node.duration
        data.append([name, input_shape, output_shape, parameter_quantity,
                     inference_memory, MAdd, duration, Flops, mread,
                     mwrite])
    df = pd.DataFrame(data)
    df.columns = ['module name', 'input shape', 'output shape',
                  'params', 'memory(MB)',
                  'MAdd', 'duration', 'Flops', 'MemRead(B)', 'MemWrite(B)']
    df['duration[%]'] = df['duration'] / (df['duration'].sum() + 1e-7)
    df['MemR+W(B)'] = df['MemRead(B)'] + df['MemWrite(B)']
    total_parameters_quantity = df['params'].sum()
    total_memory = df['memory(MB)'].sum()
    total_operation_quantity = df['MAdd'].sum()
    total_flops = df['Flops'].sum()
    total_duration = df['duration[%]'].sum()
    total_mread = df['MemRead(B)'].sum()
    total_mwrite = df['MemWrite(B)'].sum()
    total_memrw = df['MemR+W(B)'].sum()
    del df['duration']

    # Add Total row
    total_df = pd.Series([total_parameters_quantity, total_memory,
                          total_operation_quantity, total_flops,
                          total_duration, mread, mwrite, total_memrw],
                         index=['params', 'memory(MB)', 'MAdd', 'Flops', 'duration[%]',
                                'MemRead(B)', 'MemWrite(B)', 'MemR+W(B)'],
                         name='total')
    df = df.append(total_df)

    df = df.fillna(' ')
    df['memory(MB)'] = df['memory(MB)'].apply(
        lambda x: '{:.2f}'.format(x))
    df['duration[%]'] = df['duration[%]'].apply(lambda x: '{:.2%}'.format(x))
    df['MAdd'] = df['MAdd'].apply(lambda x: '{:,}'.format(x))
    df['Flops'] = df['Flops'].apply(lambda x: '{:,}'.format(x))

    result = str(df) + '\n'
    result += "=" * len(str(df).split('\n')[0])
    result += '\n'
    result += "Total params: {:,}\n".format(total_parameters_quantity)

    result += "-" * len(str(df).split('\n')[0])
    result += '\n'
    result += "Total memory: {:.2f}MB\n".format(total_memory)
    result += "Total MAdd: {}MAdd\n".format(round_value(total_operation_quantity))
    result += "Total Flops: {}Flops\n".format(round_value(total_flops))
    result += "Total MemR+W: {}B\n".format(round_value(total_memrw, True))

    return (
        total_memory,
        total_operation_quantity / 1000000,
        total_flops / 1000000,
        total_memrw / 1048576
    )


def get_model_metrics(model: torch.nn.Module, input_size=(1, 3, 224, 224)):
    cpu_device = torch.device('cpu')
    model = model.to(cpu_device)

    # torchinfo-related statistics
    model_statistics = summary(model, input_size, verbose=0, device=cpu_device)
    params_mega = model_statistics.total_params / 1048576
    input_size_mb = model_statistics.total_input / 1048576
    param_size_mb = (
        model_statistics.to_megabytes(model_statistics.total_input)
        + model_statistics.float_to_megabytes(model_statistics.total_output + model_statistics.total_params)
    )

    # torchstat-related statistics
    ms = ModelStat(model, list(input_size)[1:], 1)
    collected_nodes = ms._analyze_model()
    memory_mb, madds, flops_mega, memrw_mb = process_nodes(collected_nodes)

    del model_statistics, ms._model,  ms, collected_nodes

    # Changing result metric key will also have responsible
    # to change MetricResultDict (TODO: fix this!)
    return {
        'params_mega': params_mega,
        'input_size_mb': input_size_mb,
        'param_size_mb': param_size_mb,
        'memory_mb': memory_mb,
        'madds': madds,
        'flops_mega': flops_mega,
        'memrw_mb': memrw_mb
    }


if __name__ == '__main__':
    import os, sys

    sys.path.append(os.path.join(os.getcwd(), 'models', 'mb3_ssd'))
    from ..models.mb3_ssd.vision.ssd.mobilenet_v3_ssd_lite import \
        create_mobilenetv3_ssd_lite, create_mobilenetv3_ssd_lite_predictor

    base_model = create_mobilenetv3_ssd_lite(num_classes=3, width_mult=1.0, is_test=True)
    ssd_model = create_mobilenetv3_ssd_lite_predictor(base_model, 'hard', 'cpu')
    prepared_ssd_model = ssd_model.net

    get_model_metrics(prepared_ssd_model, (1, 3, 300, 300))