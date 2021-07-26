import pywinauto
from pywinauto.application import Application
import psutil
import os
import re


class PiCurrentAnalyzer(object):
    def __init__(self, executable_path):
        self.executable_path = executable_path
        self.app = None
        self.dialog = None
        self.first_run = True

    def open(self):
        # Kill previous running application
        for proc in psutil.process_iter():
            # check whether the process name matches
            if proc.name() == re.split(r'\\|/', self.executable_path).pop():
                print("Killing previous running analyzer %d ..." % proc.pid)
                proc.kill()

        self.app = Application(backend='win32', allow_magic_lookup=False).start(self.executable_path)
        self.dialog = self.app['ChargerLAB POWER-Z Standard Edition']
        self.dialog.wait('visible')

    def close(self):
        self.dialog.close()

    def begin(self):
        self.dialog["High refresh rate"].check()

        if self.first_run:
            btn_text = 'Run Pause'
        else:
            btn_text = '开始记录'
        self.first_run = False

        try:
            # Start current(A) monitoring
            self.dialog[btn_text].click()
            return True
        except pywinauto.findbestmatch.MatchError:
            pass
        return False

    def end(self, title):
        try:
            # Stop current(A) monitoring
            self.dialog['Stop'].click()

            # Set csvout as output filename
            self.dialog['App Setting'].click()
            setting_dialog = self.app['Setting']
            setting_dialog['Edit0'].set_edit_text("output")
            setting_dialog['Edit2'].set_edit_text(title)
            setting_dialog['Edit2'].type_keys('{ENTER}')

            # Save csv file
            self.dialog['導出數據 *.csv'].click()
            child_dialog = self.app['Power-Z']
            child_dialog['확인'].click()
            return True
        except pywinauto.findbestmatch.MatchError:
            pass
        return False