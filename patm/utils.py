from configparser import ConfigParser
from collections import OrderedDict

_section2encoder = {
        'learning': int,
        'regularizers': str,
        'scores': str
    }


def cfg2model_settings(cfg_file):
    config = ConfigParser()
    config.read(cfg_file)
    return OrderedDict([(section.encode('utf-8'), OrderedDict([(setting_name.encode('utf-8'), _section2encoder[section](value)) for setting_name, value in config.items(section) if value])) for section in config.sections()])

import sys
import time
import threading

class Spinner:
    busy = False
    delay = 0.1

    @staticmethod
    def spinning_cursor():
        while 1:
            for cursor in '|/-\\': yield cursor

    def __init__(self, delay=None):
        self.spinner_generator = self.spinning_cursor()
        if delay and float(delay): self.delay = delay

    def spinner_task(self):
        while self.busy:
            sys.stdout.write(next(self.spinner_generator))
            sys.stdout.flush()
            time.sleep(self.delay)
            sys.stdout.write('\b')
            sys.stdout.flush()

    def start(self):
        self.busy = True
        threading.Thread(target=self.spinner_task).start()

    def stop(self):
        self.busy = False
        time.sleep(self.delay)