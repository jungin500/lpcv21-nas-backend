# -*- coding:utf-8 -*-

from controller.picontroller import PiControllerServer
from main.picurrentanalyzer import PiCurrentAnalyzer, postprocess_csv
from controller.util import Timer
from datetime import datetime


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

    model_list = ['mb3-small-ssd-lite-1.0']
    picontrol.send_message("SUMMARY %s" % ':'.join(model_list))

    for model_name in model_list:
        # model_path = get_model_path(model_name)
        timer.start()
        begin_date = datetime.now().strftime("%Y%m%d%H%M%S")
        # picontrol.send_message("LOADMODEL %s" % model_name)
        picontrol.send_message("LOADMODEL SHORT_%s" % model_name)
        if picontrol.wait_until_signals("BEGININFER|NOMODEL") == 'NOMODEL':
            continue

        model_title = '%s-%s' % (begin_date, model_name)
        analyzer.begin(title=model_title)

        # Analyze가 끝날 때까지 기다린다.
        # Wait Time은 0.1초 간격이다.
        picontrol.wait_until_signals("ENDINFER")
        elapsed_time = timer.end()
        # end_date = datetime.now().strftime("%Y%m%d-%H%M%S")

        analyzer.end()
        postprocess_csv(model_name, elapsed_time, picontrol.frame_total)

    print("Analyzer done.")
    analyzer.close()
    picontrol.bye()

if __name__ == '__main__':
    main()