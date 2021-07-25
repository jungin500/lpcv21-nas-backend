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

    model_list = ['mb3-ssd-lite', '<some-other-models>']
    picontrol.send_message("SUMMARY %s" % ':'.join(model_list))

    for model_name in model_list:
        # model_path = get_model_path(model_name)
        timer.start()
        begin_date = datetime.now().strftime("%Y%m%d%H%M%S")
        picontrol.send_message("LOADMODEL %s" % model_name)
        picontrol.wait_until_signals("BEGININFER|NOMODEL")
        if picontrol.poplast() == 'NOMODEL':
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
            data = pd.read_csv(filename, skiprows=[0]).rename(columns={
                'Vbus(V)': 'Vbus',
                'Ibus(A)': 'Ibus',
                'Power(W)': 'Power',
                'Vd+(V)': 'Vdp',
                'Vd-(V)': 'Vdm',
                'energy(Wh)': 'Wh',
                'temperature(��)': 'temp'
            }).interpolate()

            watts = data['Vbus'] * data['Power']
            power_metric = watts.mean()

    analyzer.close()
    picontrol.bye()

if __name__ == '__main__':
    main()