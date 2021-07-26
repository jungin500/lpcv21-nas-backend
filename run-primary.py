# -*- coding:utf-8 -*-

from controller.picontroller import PiControllerServer
from main.picurrentanalyzer import PiCurrentAnalyzer
from controller.util import Timer
from datetime import datetime
from glob import glob
import os
import pandas as pd


def main():
    """ PC NAS(Network Architecture Search) backend implementation

    PC client is responsible to train NAS network, with Raspberry Pi as a
    metric generator client. PC will choose which model to infer and send these
    information to Pi.

    :Author:
        Jung-In An <ji5489@gmail.com>
        CVMIPLab, Kangwon National University
    """

    print("== PiControlServer v1.0 by LimeOrangePie ==")
    picontrol = PiControllerServer(listen_address='0.0.0.0', listen_port=12700)
    picontrol.serve()

    print("PiControl: Waiting until client connects ... ")
    picontrol.wait_until_client_connects()

    ANALYZER_PATH = 'main/Power-Z.exe'
    analyzer = PiCurrentAnalyzer(ANALYZER_PATH)
    analyzer.open()

    timer = Timer()

    model_list = ['shufflenet']
    picontrol.send_message("SUMMARY %s" % ':'.join(model_list))

    for model_name in model_list:
        # model_path = get_model_path(model_name)
        timer.start()
        begin_date = datetime.now().strftime("%Y%m%d%H%M%S")
        picontrol.send_message("LOADMODEL %s" % model_name)
        if picontrol.wait_until_signals("BEGININFER|NOMODEL") == 'NOMODEL':
            continue

        analyzer.begin()

        # Analyze가 끝날 때까지 기다린다.
        # Wait Time은 0.1초 간격이다.
        picontrol.wait_until_signals("ENDINFER")
        elapsed_time = timer.end()
        end_date = datetime.now().strftime("%Y%m%d%H%M%S")

        model_title = '%s-%s-%s' % (begin_date, end_date, model_name)
        analyzer.end(title=model_title)

        # 출력된 csv 파일에 대한 후처리를 진행하고
        # 파일을 이동한다.
        processed_file_location = os.path.join(".", "csvprocessed")
        if not os.path.isdir(processed_file_location):
            os.mkdir(processed_file_location)

        csv_list = glob(os.path.join(".", "csvout", "*.csv"))
        for i, filename in enumerate(csv_list):
            # 파일을 먼저 읽고 처리한다.
            # Ensure there's no non-latin character in file!
            data = pd.read_csv(filename, encoding='latin1', skiprows=[0]).rename(columns={
                'Vbus(V)': 'vbus',
                'Ibus(A)': 'ibus',
                'Power(W)': 'power',
                'Vd+(V)': 'vdp',
                'Vd-(V)': 'vdm',
                'energy(Wh)': 'wh',
                'temperature(¡É)': 'temp'
            }).interpolate()

            watts = data['vbus'] * data['power']
            power_metric = watts.mean()

            print("Power Metric: %.4f" % power_metric)

            target_filename = 'csvprocessed/%s-%s-frame%dms-total%dsec-item%02d.csv' % (
                datetime.now().strftime("%Y%m%d"), model_name,
                elapsed_time / picontrol.frame_total * 1000, elapsed_time, i)

            print("Moving %s -> %s" % (filename, target_filename))
            os.rename(filename, target_filename)

    print("Analyzer done.")
    analyzer.close()
    picontrol.bye()

if __name__ == '__main__':
    main()